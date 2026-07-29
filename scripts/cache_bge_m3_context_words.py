from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_keyword_decoding.data.protocol_assets import file_sha256
from eeg_keyword_decoding.text import (
    AGGREGATION_RULE,
    BGE_M3_COLBERT_TOKEN_MAPPING,
    BgeM3ColbertExtractor,
    BgeM3SentenceExtraction,
    BgeM3Spec,
    ContextSentence,
    ContextWordStore,
    load_context_sentences,
    write_context_word_store,
)


CONFIG_SCHEMA_VERSION = "bge_m3_colbert_context_cache_v1"


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported BGE-M3 cache configuration")
    if value["aggregation"]["rule"] != AGGREGATION_RULE:
        raise ValueError("BGE-M3 must reuse the frozen offset aggregation rule")
    if value["inference"]["compute_dtype"] != "float32":
        raise ValueError("BGE-M3 forward computation must use float32")
    if value["model"]["use_fp16"]:
        raise ValueError("BGE-M3 use_fp16 must remain false")
    return value


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _build_spec(
    config: dict[str, Any],
    *,
    batch_size: int,
) -> BgeM3Spec:
    model = config["model"]
    return BgeM3Spec(
        model_id=str(model["model_id"]),
        revision=str(model["revision"]),
        tokenizer_revision=str(model["tokenizer_revision"]),
        max_length=int(model["max_length"]),
        use_fast=bool(model["use_fast"]),
        use_fp16=bool(model["use_fp16"]),
        normalize_embeddings=bool(model["normalize_embeddings"]),
        colbert_dim=int(model["colbert_dim"]),
        batch_size=batch_size,
    )


def _source_metadata(path: Path) -> dict[str, Any]:
    try:
        display_path = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "sha256": file_sha256(path),
        "file_size_bytes": path.stat().st_size,
    }


def _select_sentences(
    sentences: Sequence[ContextSentence],
    requested_indices: Sequence[int],
) -> list[ContextSentence]:
    by_idx = {
        sentence.text_embedding_idx: sentence for sentence in sentences
    }
    missing = sorted(set(requested_indices) - set(by_idx))
    if missing:
        raise ValueError(f"Unknown representative sentence indices: {missing}")
    return [by_idx[index] for index in requested_indices]


def _alignment_example(
    extraction: BgeM3SentenceExtraction,
) -> dict[str, Any]:
    words: list[dict[str, Any]] = []
    for occurrence, alignment in zip(
        extraction.sentence.occurrences,
        extraction.alignments,
        strict=True,
    ):
        total_overlap = alignment.total_overlap
        words.append(
            {
                "word_occurrence_id": occurrence.word_occurrence_id,
                "surface_form": occurrence.surface_form,
                "char_span": [occurrence.char_start, occurrence.char_end],
                "subtokens": [
                    {
                        "colbert_index": token_index,
                        "source_tokenizer_index": extraction.source_token_indices[
                            token_index
                        ],
                        "token": extraction.tokens[token_index],
                        "offset": list(
                            extraction.offset_mapping[token_index]
                        ),
                        "overlap_length": overlap,
                        "aggregation_weight": overlap / total_overlap,
                    }
                    for token_index, overlap in zip(
                        alignment.token_indices,
                        alignment.overlap_lengths,
                        strict=True,
                    )
                ],
                "output_vector_shape": list(
                    extraction.vectors[
                        occurrence.word_position
                    ].shape
                ),
            }
        )
    return {
        "text_embedding_idx": extraction.sentence.text_embedding_idx,
        "text": extraction.sentence.text,
        "colbert_tokens": [
            {
                "colbert_index": index,
                "source_tokenizer_index": source_index,
                "token": token,
                "offset": list(offset),
                "special": bool(special),
                "attended": bool(attended),
            }
            for index, (source_index, token, offset, special, attended) in enumerate(
                zip(
                    extraction.source_token_indices,
                    extraction.tokens,
                    extraction.offset_mapping,
                    extraction.special_tokens_mask,
                    extraction.attention_mask,
                    strict=True,
                )
            )
        ],
        "words": words,
        "output_shape": list(extraction.vectors.shape),
    }


def _repeatability(
    extractor: BgeM3ColbertExtractor,
    sentences: Sequence[ContextSentence],
    *,
    tolerance: float,
) -> tuple[list[BgeM3SentenceExtraction], dict[str, Any]]:
    first = extractor.extract_batch(sentences)
    second = extractor.extract_batch(sentences)
    maximum_error = 0.0
    for first_item, second_item in zip(first, second, strict=True):
        error = float(
            np.max(
                np.abs(first_item.vectors - second_item.vectors),
                initial=0.0,
            )
        )
        maximum_error = max(maximum_error, error)
    if maximum_error > tolerance:
        raise ValueError(
            f"BGE-M3 repeated extraction differs by {maximum_error}; "
            f"tolerance={tolerance}"
        )
    return first, {
        "verified_sentence_count": len(sentences),
        "maximum_absolute_error": maximum_error,
        "tolerance": tolerance,
    }


def _generate_cache(
    *,
    extractor: BgeM3ColbertExtractor,
    sentences: Sequence[ContextSentence],
    batch_size: int,
    output_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    sentence_labels_path: Path,
    occurrences_path: Path,
    representative_indices: Sequence[int],
    overwrite: bool,
) -> dict[str, Any]:
    representative_sentences = _select_sentences(
        sentences,
        representative_indices,
    )
    mapping_verification = (
        extractor.verify_public_mapping_against_direct_forward(
            representative_sentences,
            atol=float(config["audit"]["mapping_atol"]),
        )
    )
    representative_extractions, repeatability = _repeatability(
        extractor,
        representative_sentences,
        tolerance=float(config["audit"]["repeat_atol"]),
    )
    representative_examples = {
        item.sentence.text_embedding_idx: _alignment_example(item)
        for item in representative_extractions
    }

    total_words = sum(sentence.word_count for sentence in sentences)
    model_metadata = extractor.model_metadata()
    hidden_size = int(model_metadata["colbert_dimension"])
    vectors = np.empty((total_words, hidden_size), dtype=np.float32)

    if extractor.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(extractor.device)
    started = time.perf_counter()
    cursor = 0
    maximum_token_length = 0
    total_subtoken_references = 0
    multiple_subtoken_word_count = 0
    shared_token_count = 0
    unknown_token_count = 0
    for batch_start in range(0, len(sentences), batch_size):
        batch = sentences[batch_start : batch_start + batch_size]
        for extraction in extractor.extract_batch(batch):
            word_count = extraction.sentence.word_count
            vectors[cursor : cursor + word_count] = extraction.vectors
            cursor += word_count
            maximum_token_length = max(
                maximum_token_length,
                len(extraction.tokens) + 1,
            )
            total_subtoken_references += sum(
                len(alignment.token_indices)
                for alignment in extraction.alignments
            )
            multiple_subtoken_word_count += sum(
                len(alignment.token_indices) > 1
                for alignment in extraction.alignments
            )
            usage = Counter(
                token_index
                for alignment in extraction.alignments
                for token_index in alignment.token_indices
            )
            shared_token_count += sum(count > 1 for count in usage.values())
            unknown_token_count += sum(
                token == extractor.tokenizer.unk_token
                for token in extraction.tokens
            )
        completed = min(batch_start + batch_size, len(sentences))
        print(
            f"encoded_sentences={completed}/{len(sentences)}",
            file=sys.stderr,
            flush=True,
        )
    elapsed_seconds = time.perf_counter() - started
    if cursor != total_words:
        raise AssertionError(f"Filled {cursor} vectors, expected {total_words}")

    non_finite_vector_count = int(
        np.count_nonzero(~np.isfinite(vectors).all(axis=1))
    )
    zero_vector_count = int(
        np.count_nonzero(np.linalg.norm(vectors, axis=1) == 0)
    )
    if non_finite_vector_count or zero_vector_count:
        raise ValueError(
            "BGE-M3 extraction produced invalid word vectors: "
            f"non_finite={non_finite_vector_count}, zero={zero_vector_count}"
        )

    storage_dtype = np.dtype(config["inference"]["storage_dtype"])
    if storage_dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError(f"Unsupported storage dtype: {storage_dtype}")
    stored_vectors = vectors.astype(storage_dtype)
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(extractor.device))
        if extractor.device.type == "cuda"
        else 0
    )
    created_at = datetime.now(timezone.utc).isoformat()
    metadata_fields = {
        "model": model_metadata,
        "tokenizer": extractor.tokenizer_metadata(),
        "extraction": {
            "backend_name": "bge_m3_colbert",
            "input_scope": "full_normalized_sentence",
            "representation": "ColBERT multi-vector token representations",
            "dense_sentence_vectors_used": False,
            "sparse_lexical_weights_used": False,
            "aggregation_rule": AGGREGATION_RULE,
            "offset_unit": config["aggregation"]["offset_unit"],
            "normalize_text_rule": config["aggregation"][
                "normalize_text_rule"
            ],
            "normalize_text_rule_version": config["aggregation"][
                "normalize_text_rule_version"
            ],
            "special_token_policy": config["aggregation"][
                "special_token_policy"
            ],
            "padding_token_policy": config["aggregation"][
                "padding_token_policy"
            ],
            "colbert_token_mapping": BGE_M3_COLBERT_TOKEN_MAPPING,
            "compute_dtype": "float32",
            "storage_dtype": str(storage_dtype),
            "output_axis_order": ["word", "hidden"],
            "model_eval": True,
            "inference_mode": True,
            "split_or_eeg_information_included": False,
        },
        "sources": {
            "sentence_labels": _source_metadata(sentence_labels_path),
            "word_occurrences": _source_metadata(occurrences_path),
            "configuration": _source_metadata(config_path),
            "implementation": {
                "cache_script": _source_metadata(Path(__file__).resolve()),
                "schema": _source_metadata(
                    PROJECT_ROOT
                    / "src"
                    / "eeg_keyword_decoding"
                    / "text"
                    / "schema.py"
                ),
                "offsets": _source_metadata(
                    PROJECT_ROOT
                    / "src"
                    / "eeg_keyword_decoding"
                    / "text"
                    / "offsets.py"
                ),
                "context_store": _source_metadata(
                    PROJECT_ROOT
                    / "src"
                    / "eeg_keyword_decoding"
                    / "text"
                    / "context_store.py"
                ),
                "bge_m3": _source_metadata(
                    PROJECT_ROOT
                    / "src"
                    / "eeg_keyword_decoding"
                    / "text"
                    / "bge_m3.py"
                ),
            },
        },
        "generation": {
            "created_at_utc": created_at,
            "batch_size": batch_size,
            "sentence_count": len(sentences),
            "word_occurrence_count": total_words,
            "offset_alignment_failures": 0,
            "special_token_overlap_failures": 0,
            "empty_vector_count": zero_vector_count,
            "non_finite_vector_count": non_finite_vector_count,
            "maximum_attended_token_length": maximum_token_length,
            "total_subtoken_references": total_subtoken_references,
            "multiple_subtoken_word_count": multiple_subtoken_word_count,
            "multiple_subtoken_word_fraction": (
                multiple_subtoken_word_count / total_words
            ),
            "tokens_shared_across_frozen_words": shared_token_count,
            "unknown_token_count": unknown_token_count,
            "representative_text_embedding_indices": list(
                representative_indices
            ),
            "representative_alignments": [
                representative_examples[index]
                for index in representative_indices
            ],
            "public_mapping_verification": mapping_verification,
            "repeatability": repeatability,
            "extraction_elapsed_seconds": elapsed_seconds,
            "average_sentence_extraction_seconds": (
                elapsed_seconds / len(sentences)
            ),
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        },
        "runtime": extractor.runtime_metadata(),
    }
    metadata = write_context_word_store(
        output_dir,
        vectors=stored_vectors,
        sentences=sentences,
        metadata_fields=metadata_fields,
        overwrite=overwrite,
    )

    store = ContextWordStore(output_dir, verify_hashes=True)
    store.audit_against(sentences)
    roundtrip = np.asarray(store.context_vectors)
    roundtrip_max_error = float(
        np.max(np.abs(roundtrip - stored_vectors), initial=0.0)
    )
    if roundtrip_max_error != 0:
        raise ValueError(
            f"Cache write/read round trip changed values by "
            f"{roundtrip_max_error}"
        )
    sample = store.get_sentence(
        sentences[0].text_embedding_idx,
        vector_dtype="float32",
    )
    if sample.vectors.dtype != np.float32:
        raise ValueError("ContextWordStore failed float32 restoration")

    return {
        "cache_dir": str(output_dir),
        "schema_version": metadata.schema_version,
        "model_revision": metadata.model["resolved_revision"],
        "tokenizer_revision": metadata.tokenizer["resolved_revision"],
        "context_vectors": metadata.arrays["context_vectors"].to_dict(),
        "sentence_count": len(sentences),
        "word_occurrence_count": total_words,
        "offset_alignment_failures": 0,
        "non_finite_vector_count": non_finite_vector_count,
        "empty_vector_count": zero_vector_count,
        "multiple_subtoken_word_count": multiple_subtoken_word_count,
        "unknown_token_count": unknown_token_count,
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "mapping_verification": mapping_verification,
        "repeatability": repeatability,
        "cache_roundtrip_max_absolute_error": roundtrip_max_error,
        "metadata_sha256": file_sha256(output_dir / "metadata.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache contextual BGE-M3 ColBERT word representations."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "text"
        / "bge_m3_colbert.yaml",
    )
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _load_config(config_path)
    sentence_labels_path = _project_path(
        config["inputs"]["sentence_labels"]
    )
    occurrences_path = _project_path(config["inputs"]["word_occurrences"])
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else _project_path(config["output"]["cache_dir"])
    )
    sentences = load_context_sentences(
        sentence_labels_path,
        occurrences_path,
    )
    representative_indices = [
        int(value)
        for value in config["audit"][
            "representative_text_embedding_indices"
        ]
    ]
    inference = config["inference"]
    batch_size = args.batch_size or int(inference["batch_size"])
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = args.device or str(inference["device"])
    local_files_only = args.local_files_only or bool(
        inference["local_files_only"]
    )
    extractor = BgeM3ColbertExtractor(
        _build_spec(config, batch_size=batch_size),
        device=device,
        local_files_only=local_files_only,
    )

    if args.smoke_only:
        selected = _select_sentences(sentences, representative_indices)
        mapping = extractor.verify_public_mapping_against_direct_forward(
            selected,
            atol=float(config["audit"]["mapping_atol"]),
        )
        extractions, repeatability = _repeatability(
            extractor,
            selected,
            tolerance=float(config["audit"]["repeat_atol"]),
        )
        result = {
            "mode": "smoke_only",
            "model": extractor.model_metadata(),
            "tokenizer": extractor.tokenizer_metadata(),
            "runtime": extractor.runtime_metadata(),
            "offset_alignment_failures": 0,
            "mapping_verification": mapping,
            "repeatability": repeatability,
            "examples": [
                _alignment_example(extraction)
                for extraction in extractions
            ],
        }
    else:
        result = _generate_cache(
            extractor=extractor,
            sentences=sentences,
            batch_size=batch_size,
            output_dir=output_dir,
            config=config,
            config_path=config_path,
            sentence_labels_path=sentence_labels_path,
            occurrences_path=occurrences_path,
            representative_indices=representative_indices,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
