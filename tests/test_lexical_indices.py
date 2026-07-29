from __future__ import annotations

import json
from pathlib import Path

from eeg_keyword_decoding.data import build_lexical_identity_index

from conftest import PROTOCOL_ROOT


def test_lexical_identity_mapping_is_complete_and_deterministic(split_index):
    path = PROTOCOL_ROOT / "littleprince_word_occurrences_v1.csv"
    first = build_lexical_identity_index(path, split_index)
    second = build_lexical_identity_index(path, split_index)

    assert first.mapping_sha256 == second.mapping_sha256
    assert first.occurrence_count == 14034
    assert first.surface_type_count == 2757
    assert first.context_token_group_count > 0
    assert len(first.sentence_group_ids) == 2575
    assert first.surface_types == tuple(sorted(first.surface_types))
    assert set(first.occurrence_to_surface_index) == set(
        first.occurrence_to_context_token_group_index
    )


def test_context_token_group_shares_duplicate_context_position(split_index):
    index = build_lexical_identity_index(
        PROTOCOL_ROOT / "littleprince_word_occurrences_v1.csv",
        split_index,
    )
    occurrences_by_key: dict[tuple[str, int, str], list[str]] = {}
    for occurrence_id, context_index in (
        index.occurrence_to_context_token_group_index.items()
    ):
        key = index.context_token_keys[context_index]
        occurrences_by_key.setdefault(
            (key.sentence_group_id, key.word_position, key.surface_form),
            [],
        ).append(occurrence_id)
    shared = [values for values in occurrences_by_key.values() if len(values) > 1]
    assert shared
    for occurrence_ids in shared[:10]:
        observed = {
            index.occurrence_to_context_token_group_index[item]
            for item in occurrence_ids
        }
        assert len(observed) == 1


def test_context_token_key_never_merges_different_positions(split_index):
    index = build_lexical_identity_index(
        PROTOCOL_ROOT / "littleprince_word_occurrences_v1.csv",
        split_index,
    )
    assert len(index.context_token_keys) == len(set(index.context_token_keys))
    for context_index, key in enumerate(index.context_token_keys):
        occurrence_ids = [
            occurrence_id
            for occurrence_id, value in (
                index.occurrence_to_context_token_group_index.items()
            )
            if value == context_index
        ]
        assert occurrence_ids
        assert all(
            occurrence_id.endswith(f":word:{key.word_position}")
            for occurrence_id in occurrence_ids
        )


def test_lexical_mapping_is_serializable_and_source_hash_sensitive(
    tmp_path: Path,
    split_index,
):
    source = PROTOCOL_ROOT / "littleprince_word_occurrences_v1.csv"
    index = build_lexical_identity_index(source, split_index)
    output = tmp_path / "lexical_index.json"
    index.write(output)
    metadata = json.loads(output.read_text(encoding="utf-8"))
    assert metadata["mapping_sha256"] == index.mapping_sha256
    assert metadata["surface_type_count"] == 2757

    modified = tmp_path / "occurrences.csv"
    modified.write_bytes(source.read_bytes() + b"\n")
    changed = build_lexical_identity_index(modified, split_index)
    assert changed.surface_types == index.surface_types
    assert changed.source_word_occurrences_sha256 != (
        index.source_word_occurrences_sha256
    )
    assert changed.mapping_sha256 != index.mapping_sha256
