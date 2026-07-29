from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_keyword_decoding.data.protocol_assets import file_sha256
from eeg_keyword_decoding.text import (
    AGGREGATION_RULE,
    ContextSentence,
    MacBertContextExtractor,
    MacBertSpec,
    SentenceExtraction,
    load_context_sentences,
    write_context_word_store,
)


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    if value.get("schema_version") != "macbert_context_cache_v1":
        raise ValueError("Unsupported MacBERT cache configuration")
    return value


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _build_spec(config: dict[str, Any]) -> MacBertSpec:
    model = config["model"]
    return MacBertSpec(
        model_id=str(model["model_id"]),
        revision=str(model["revision"]),
        tokenizer_revision=str(model["tokenizer_revision"]),
        hidden_state_layer_indices=tuple(
            int(value) for value in model["hidden_state_layer_indices"]
        ),
        max_length=int(model["max_length"]),
        use_fast=bool(model["use_fast"]),
        use_safetensors=bool(model["use_safetensors"]),
    )


def _alignment_example(extraction: SentenceExtraction) -> dict[str, Any]:
    words = []
    for occurrence, alignment in zip(
        extraction.sentence.occurrences,
        extraction.alignments,
        strict=True,
    ):
        words.append(
            {
                "word_occurrence_id": occurrence.word_occurrence_id,
                "surface_form": occurrence.surface_form,
                "char_span": [occurrence.char_start, occurrence.char_end],
                "subtokens": [
                    {
                        "token_index": token_index,
                        "token": extraction.tokens[token_index],
                        "offset": list(extraction.offset_mapping[token_index]),
                        "overlap_length": overlap,
                    }
                    for token_index, overlap in zip(
                        alignment.token_indices,
                        alignment.overlap_lengths,
                        strict=True,
                    )
                ],
            }
        )
    return {
        "text_embedding_idx": extraction.sentence.text_embedding_idx,
        "text": extraction.sentence.text,
        "model_input_tokens": [
            {
                "token": token,
                "offset": list(offset),
                "special": bool(special),
                "attended": bool(attended),
            }
            for token, offset, special, attended in zip(
                extraction.tokens,
                extraction.offset_mapping,
                extraction.special_tokens_mask,
                extraction.attention_mask,
                strict=True,
            )
            if attended
        ],
        "words": words,
        "output_shape": list(extraction.vectors.shape),
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


def _multiword_token_count(extraction: SentenceExtraction) -> int:
    usage = Counter(
        token_index
        for alignment in extraction.alignments
        for token_index in alignment.token_indices
    )
    return sum(count > 1 for count in usage.values())


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


def _generate_cache(
    *,
    extractor: MacBertContextExtractor,
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
    total_words = sum(sentence.word_count for sentence in sentences)
    layer_count = len(extractor.spec.hidden_state_layer_indices)
    hidden_size = int(extractor.model_metadata()["hidden_size"])
    vectors = np.empty(
        (total_words, layer_count, hidden_size),
        dtype=np.float32,
    )

    cursor = 0
    maximum_token_length = 0
    total_subtoken_references = 0
    multiword_token_count = 0
    examples: dict[int, dict[str, Any]] = {}
    representative_set = set(representative_indices)
    for batch_start in range(0, len(sentences), batch_size):
        batch = sentences[batch_start : batch_start + batch_size]
        for extraction in extractor.extract_batch(batch):
            word_count = extraction.sentence.word_count
            vectors[cursor : cursor + word_count] = extraction.vectors
            cursor += word_count
            maximum_token_length = max(
                maximum_token_length,
                sum(extraction.attention_mask),
            )
            total_subtoken_references += sum(
                len(alignment.token_indices)
                for alignment in extraction.alignments
            )
            multiword_token_count += _multiword_token_count(extraction)
            idx = extraction.sentence.text_embedding_idx
            if idx in representative_set:
                examples[idx] = _alignment_example(extraction)
        completed = min(batch_start + batch_size, len(sentences))
        print(
            f"encoded_sentences={completed}/{len(sentences)}",
            file=sys.stderr,
            flush=True,
        )
    if cursor != total_words:
        raise AssertionError(f"Filled {cursor} vectors, expected {total_words}")
    if set(examples) != representative_set:
        raise AssertionError("Not all representative examples were extracted")

    metadata_fields = {
        "model": extractor.model_metadata(),
        "tokenizer": extractor.tokenizer_metadata(),
        "extraction": {
            "input_scope": "full_normalized_sentence",
            "hidden_state_layer_indices": list(
                extractor.spec.hidden_state_layer_indices
            ),
            "aggregation_rule": AGGREGATION_RULE,
            "offset_unit": config["aggregation"]["offset_unit"],
            "special_token_policy": config["aggregation"][
                "special_token_policy"
            ],
            "padding_token_policy": config["aggregation"][
                "padding_token_policy"
            ],
            "storage_dtype": "float32",
            "output_axis_order": ["word", "encoder_layer", "hidden"],
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
                "macbert": _source_metadata(
                    PROJECT_ROOT
                    / "src"
                    / "eeg_keyword_decoding"
                    / "text"
                    / "macbert.py"
                ),
            },
        },
        "generation": {
            "sentence_count": len(sentences),
            "word_occurrence_count": total_words,
            "offset_alignment_failures": 0,
            "special_token_overlap_failures": 0,
            "maximum_attended_token_length": maximum_token_length,
            "total_subtoken_references": total_subtoken_references,
            "tokens_shared_across_frozen_words": multiword_token_count,
            "representative_text_embedding_indices": list(
                representative_indices
            ),
            "representative_alignments": [
                examples[index] for index in representative_indices
            ],
        },
        "runtime": extractor.runtime_metadata(),
    }
    metadata = write_context_word_store(
        output_dir,
        vectors=vectors,
        sentences=sentences,
        metadata_fields=metadata_fields,
        overwrite=overwrite,
    )
    return {
        "cache_dir": str(output_dir),
        "schema_version": metadata.schema_version,
        "model_revision": metadata.model["resolved_revision"],
        "tokenizer_revision": metadata.tokenizer["resolved_revision"],
        "context_vectors": metadata.arrays["context_vectors"].to_dict(),
        "sentence_count": len(sentences),
        "word_occurrence_count": total_words,
        "offset_alignment_failures": 0,
        "representative_alignments": [
            examples[index] for index in representative_indices
        ],
        "metadata_sha256": file_sha256(output_dir / "metadata.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache contextual MacBERT word representations."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "text" / "macbert_base.yaml",
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
    device = args.device or str(inference["device"])
    batch_size = args.batch_size or int(inference["batch_size"])
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    local_files_only = args.local_files_only or bool(
        inference["local_files_only"]
    )
    extractor = MacBertContextExtractor(
        _build_spec(config),
        device=device,
        local_files_only=local_files_only,
    )

    if args.smoke_only:
        selected = _select_sentences(sentences, representative_indices)
        result = {
            "mode": "smoke_only",
            "model": extractor.model_metadata(),
            "tokenizer": extractor.tokenizer_metadata(),
            "runtime": extractor.runtime_metadata(),
            "offset_alignment_failures": 0,
            "examples": [
                _alignment_example(extraction)
                for extraction in extractor.extract_batch(selected)
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
