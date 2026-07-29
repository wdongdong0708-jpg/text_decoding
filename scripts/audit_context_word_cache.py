from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_keyword_decoding.data.protocol_assets import file_sha256
from eeg_keyword_decoding.text import (
    ContextWordStore,
    OffsetAlignmentError,
    align_sentence_to_tokens,
    load_context_sentences,
)


def _source_path(value: dict[str, Any]) -> Path:
    path = Path(str(value["path"]))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _source_records(
    value: Any,
    *,
    prefix: str = "sources",
) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(
            value.get("sha256"), str
        ):
            records.append((prefix, value))
        else:
            for name, nested in value.items():
                records.extend(
                    _source_records(nested, prefix=f"{prefix}.{name}")
                )
    return records


def _audit_example(
    sentence: Any,
    tokens: list[str],
    offsets: list[tuple[int, int]],
    special: list[int],
    attention: list[int],
    alignments: Any,
) -> dict[str, Any]:
    return {
        "text_embedding_idx": sentence.text_embedding_idx,
        "text": sentence.text,
        "words": [
            {
                "surface_form": occurrence.surface_form,
                "char_span": [occurrence.char_start, occurrence.char_end],
                "subtokens": [
                    {
                        "token": tokens[token_index],
                        "offset": list(offsets[token_index]),
                        "overlap_length": overlap,
                    }
                    for token_index, overlap in zip(
                        alignment.token_indices,
                        alignment.overlap_lengths,
                        strict=True,
                    )
                ],
            }
            for occurrence, alignment in zip(
                sentence.occurrences,
                alignments,
                strict=True,
            )
        ],
        "special_tokens": [
            {
                "token": token,
                "offset": list(offset),
                "attended": bool(attended),
            }
            for token, offset, is_special, attended in zip(
                tokens,
                offsets,
                special,
                attention,
                strict=True,
            )
            if is_special
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a ContextWordStore and all tokenizer offsets."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "cache"
            / "context_words"
            / "macbert_v1"
        ),
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow tokenizer files to be fetched if absent locally.",
    )
    parser.add_argument(
        "--compare-cache-dir",
        type=Path,
        help="Compare sentence and frozen word ordering with another store.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    store = ContextWordStore(cache_dir, verify_hashes=True)
    sources = store.metadata.sources
    source_hash_audit: dict[str, dict[str, Any]] = {}
    for name, record in _source_records(sources):
        path = _source_path(record)
        actual_sha256 = file_sha256(path)
        matches = actual_sha256 == record["sha256"]
        source_hash_audit[name] = {
            "path": str(path),
            "expected_sha256": record["sha256"],
            "actual_sha256": actual_sha256,
            "matches": matches,
        }
        if not matches:
            raise ValueError(f"Recorded source hash changed: {path}")
    sentence_labels_path = _source_path(sources["sentence_labels"])
    occurrences_path = _source_path(sources["word_occurrences"])
    sentences = load_context_sentences(
        sentence_labels_path,
        occurrences_path,
    )
    store.audit_against(sentences)

    from transformers import AutoTokenizer

    tokenizer_metadata = store.metadata.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_metadata["tokenizer_id"],
        revision=tokenizer_metadata["resolved_revision"],
        use_fast=True,
        local_files_only=not args.allow_network,
    )
    if not tokenizer.is_fast:
        raise ValueError("Offset audit requires a fast tokenizer")

    representative_indices = set(
        int(value)
        for value in store.metadata.generation[
            "representative_text_embedding_indices"
        ]
    )
    failures: list[dict[str, Any]] = []
    examples: dict[int, dict[str, Any]] = {}
    maximum_token_length = 0
    total_subtoken_references = 0
    shared_token_count = 0
    multiple_subtoken_word_count = 0
    unknown_token_count = 0
    for sentence in sentences:
        encoded = tokenizer(
            sentence.text,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_attention_mask=True,
        )
        offsets = [
            (int(start), int(stop))
            for start, stop in encoded["offset_mapping"]
        ]
        special = [int(value) for value in encoded["special_tokens_mask"]]
        attention = [int(value) for value in encoded["attention_mask"]]
        try:
            alignments = align_sentence_to_tokens(
                sentence.occurrences,
                offsets,
                special,
                attention,
            )
        except OffsetAlignmentError as error:
            failures.append(
                {
                    "text_embedding_idx": sentence.text_embedding_idx,
                    "error": str(error),
                }
            )
            continue
        maximum_token_length = max(maximum_token_length, sum(attention))
        total_subtoken_references += sum(
            len(alignment.token_indices) for alignment in alignments
        )
        multiple_subtoken_word_count += sum(
            len(alignment.token_indices) > 1 for alignment in alignments
        )
        usage = Counter(
            token_index
            for alignment in alignments
            for token_index in alignment.token_indices
        )
        shared_token_count += sum(count > 1 for count in usage.values())
        if sentence.text_embedding_idx in representative_indices:
            tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
            unknown_token_count += sum(
                token == tokenizer.unk_token for token in tokens
            )
            examples[sentence.text_embedding_idx] = _audit_example(
                sentence,
                tokens,
                offsets,
                special,
                attention,
                alignments,
            )
        else:
            tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
            unknown_token_count += sum(
                token == tokenizer.unk_token for token in tokens
            )

    vector_descriptor = store.metadata.arrays["context_vectors"]
    vector_rows = np.asarray(store.context_vectors, dtype=np.float32).reshape(
        store.context_vectors.shape[0],
        -1,
    )
    non_finite_vector_count = int(
        np.count_nonzero(~np.isfinite(vector_rows).all(axis=1))
    )
    empty_vector_count = int(
        np.count_nonzero(np.linalg.norm(vector_rows, axis=1) == 0)
    )
    comparison: dict[str, Any] | None = None
    if args.compare_cache_dir is not None:
        other = ContextWordStore(
            args.compare_cache_dir.resolve(),
            verify_hashes=True,
        )
        comparison_arrays = (
            "text_embedding_indices",
            "sentence_offsets",
            "word_occurrence_ids",
            "word_positions",
            "surface_forms",
            "char_spans",
            "keyword_ids",
        )
        array_matches = {
            name: bool(
                np.array_equal(
                    getattr(store, name),
                    getattr(other, name),
                )
            )
            for name in comparison_arrays
        }
        comparison = {
            "cache_dir": str(other.root),
            "array_matches": array_matches,
            "all_frozen_order_arrays_match": all(array_matches.values()),
        }
        if not comparison["all_frozen_order_arrays_match"]:
            raise ValueError("Context stores have incompatible frozen ordering")

    result = {
        "schema_version": store.metadata.schema_version,
        "cache_dir": str(cache_dir),
        "metadata_sha256": file_sha256(cache_dir / "metadata.json"),
        "model_revision": store.metadata.model["resolved_revision"],
        "tokenizer_revision": tokenizer_metadata["resolved_revision"],
        "sentence_count": len(sentences),
        "word_occurrence_count": sum(
            sentence.word_count for sentence in sentences
        ),
        "context_vectors_shape": list(vector_descriptor.shape),
        "context_vectors_dtype": vector_descriptor.dtype,
        "context_vectors_file_size_bytes": vector_descriptor.file_size_bytes,
        "offset_alignment_failures": len(failures),
        "special_token_overlap_failures": sum(
            "Special token overlaps" in item["error"] for item in failures
        ),
        "maximum_attended_token_length": maximum_token_length,
        "total_subtoken_references": total_subtoken_references,
        "multiple_subtoken_word_count": multiple_subtoken_word_count,
        "multiple_subtoken_word_fraction": (
            multiple_subtoken_word_count
            / sum(sentence.word_count for sentence in sentences)
        ),
        "tokens_shared_across_frozen_words": shared_token_count,
        "unknown_token_count": unknown_token_count,
        "non_finite_vector_count": non_finite_vector_count,
        "empty_vector_count": empty_vector_count,
        "source_hash_audit": source_hash_audit,
        "comparison": comparison,
        "array_sha256": {
            name: descriptor.sha256
            for name, descriptor in sorted(store.metadata.arrays.items())
        },
        "representative_alignments": [
            examples[index] for index in sorted(examples)
        ],
        "failures": failures[:20],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)
    if non_finite_vector_count or empty_vector_count:
        raise ValueError(
            "Context store contains invalid vectors: "
            f"non_finite={non_finite_vector_count}, empty={empty_vector_count}"
        )
    expected = store.metadata.generation
    for key in (
        "maximum_attended_token_length",
        "total_subtoken_references",
        "tokens_shared_across_frozen_words",
        "multiple_subtoken_word_count",
        "unknown_token_count",
    ):
        if key in expected and result[key] != expected[key]:
            raise ValueError(
                f"Independent audit disagrees for {key}: "
                f"{result[key]} != {expected[key]}"
            )


if __name__ == "__main__":
    main()
