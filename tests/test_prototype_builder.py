from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from eeg_keyword_decoding.data import build_lexical_identity_index
from eeg_keyword_decoding.models import build_context_text_projector
from eeg_keyword_decoding.prototypes import (
    PrototypeBuilderConfig,
    TrainOnlyPrototypeBuilder,
    aggregate_sentence_group_balanced,
    aggregate_within_sentence_keyword,
    load_prototype_bank,
    module_state_sha256,
    save_prototype_bank,
)

from conftest import PROJECT_ROOT, PROTOCOL_ROOT


TEXT_CONFIGS = {
    "macbert": PROJECT_ROOT / "configs" / "text" / "macbert_projection_v1.yaml",
    "bge_m3": PROJECT_ROOT / "configs" / "text" / "bge_m3_projection_v1.yaml",
}
CACHE_DIRS = {
    "macbert": PROJECT_ROOT / "data" / "cache" / "context_words" / "macbert_v1",
    "bge_m3": (
        PROJECT_ROOT
        / "data"
        / "cache"
        / "context_words"
        / "bge_m3_colbert_v1"
    ),
}


@pytest.fixture(scope="module")
def real_prototype_banks(split_index, keyword_index):
    lexical = build_lexical_identity_index(
        PROTOCOL_ROOT / "littleprince_word_occurrences_v1.csv",
        split_index,
    )
    config = PrototypeBuilderConfig.from_yaml(
        PROJECT_ROOT
        / "configs"
        / "prototypes"
        / "train_only_group_balanced_v1.yaml"
    )
    values = {}
    for backend in ("macbert", "bge_m3"):
        torch.manual_seed(42)
        projector = build_context_text_projector(
            TEXT_CONFIGS[backend],
            cache_metadata_path=CACHE_DIRS[backend] / "metadata.json",
        )
        builder = TrainOnlyPrototypeBuilder(
            config=config,
            split_index=split_index,
            outer_fold=0,
            keyword_index=keyword_index,
            lexical_identity_index=lexical,
            sentence_labels_path=(
                PROTOCOL_ROOT
                / "littleprince_sentence_keyword_labels_v1.csv"
            ),
            word_occurrences_path=(
                PROTOCOL_ROOT / "littleprince_word_occurrences_v1.csv"
            ),
            eligibility_path=(
                PROTOCOL_ROOT
                / "littleprince_keyword_fold_eligibility_v1.csv"
            ),
            context_store_root=CACHE_DIRS[backend],
            text_backend=backend,
            projection_batch_size=128,
        )
        values[backend] = (builder.build(projector), builder, projector)
    return values


def test_group_balanced_aggregation_equalizes_repeated_sentence_group():
    vectors = {
        1: torch.tensor([1.0, 0.0]),
        2: torch.tensor([3.0, 0.0]),
        3: torch.tensor([0.0, 2.0]),
    }
    groups = {1: "same", 2: "same", 3: "other"}
    observed = aggregate_sentence_group_balanced(vectors, groups)
    assert torch.allclose(observed, torch.tensor([1.0, 1.0]))


def test_repeated_keyword_is_averaged_inside_sentence_first():
    projected = torch.tensor(
        [[1.0, 0.0], [9.0, 9.0], [3.0, 2.0]]
    )
    observed = aggregate_within_sentence_keyword(projected, [0, 2])
    assert torch.equal(observed, torch.tensor([2.0, 1.0]))


@pytest.mark.parametrize("backend", ["macbert", "bge_m3"])
def test_real_builder_retains_fixed_master_and_matches_frozen_df(
    real_prototype_banks,
    keyword_index,
    backend,
):
    bank, _, _ = real_prototype_banks[backend]
    assert bank.vectors.shape == (247, 256)
    assert bank.vectors.dtype == torch.float32
    assert bank.available_count == 165
    assert bank.keyword_ids == keyword_index.keyword_ids
    assert len(keyword_index.core_indices & set(
        torch.nonzero(bank.available_mask).flatten().tolist()
    )) == 33
    assert bool(
        (bank.train_group_df[bank.available_mask] >= 10).all()
    )
    assert not bool(bank.available_mask[bank.train_group_df < 10].any())
    assert bank.metadata["contributors"]["validation_intersection"] == []
    assert bank.metadata["contributors"]["test_intersection"] == []
    assert bank.metadata["contributors"]["eeg_views_used"] is False
    assert (
        bank.metadata["contributors"]["unique_text_embedding_indices"]
        == 1967
    )


def test_builder_hard_rejects_non_train_role(real_prototype_banks):
    _, builder, projector = real_prototype_banks["bge_m3"]
    with pytest.raises(ValueError, match="only role='train'"):
        builder.build(projector, role="validation")


def test_eeg_view_repetition_cannot_change_prototype(
    real_prototype_banks,
):
    bank, builder, projector = real_prototype_banks["bge_m3"]
    repeated_views = builder.split_index.valid_manifest_records * 2
    inflated_split = replace(
        builder.split_index,
        manifest_records=repeated_views,
        valid_manifest_records=repeated_views,
    )
    inflated_builder = TrainOnlyPrototypeBuilder(
        config=builder.config,
        split_index=inflated_split,
        outer_fold=builder.outer_fold,
        keyword_index=builder.keyword_index,
        lexical_identity_index=builder.lexical_identity_index,
        sentence_labels_path=builder.sentence_labels_path,
        word_occurrences_path=builder.word_occurrences_path,
        eligibility_path=builder.eligibility_path,
        context_store_root=builder.context_store_root,
        text_backend=builder.text_backend,
        projection_batch_size=128,
    )
    repeated_bank = inflated_builder.build(projector)
    assert torch.equal(repeated_bank.vectors, bank.vectors)
    assert torch.equal(repeated_bank.train_group_df, bank.train_group_df)


def test_prototype_store_round_trip_and_guard_failures(
    tmp_path: Path,
    real_prototype_banks,
):
    bank, _, _ = real_prototype_banks["bge_m3"]
    save_prototype_bank(bank, tmp_path)
    kwargs = {
        "expected_outer_fold": bank.outer_fold,
        "expected_text_backend": bank.text_backend,
        "expected_projector_state_hash": bank.projector_state_hash,
        "expected_source_cache_hash": bank.source_cache_hash,
        "expected_fold_hash": bank.fold_hash,
        "expected_keyword_ids": bank.keyword_ids,
        "expected_eligibility_hash": bank.eligibility_hash,
        "expected_lexical_mapping_hash": bank.lexical_mapping_hash,
    }
    restored = load_prototype_bank(tmp_path, **kwargs)
    assert torch.equal(restored.vectors, bank.vectors)
    assert torch.equal(restored.available_mask, bank.available_mask)
    assert torch.equal(restored.train_group_df, bank.train_group_df)

    for key, bad_value in (
        ("expected_outer_fold", 1),
        ("expected_text_backend", "macbert"),
        ("expected_projector_state_hash", "0" * 64),
        ("expected_source_cache_hash", "1" * 64),
        ("expected_fold_hash", "2" * 64),
        ("expected_eligibility_hash", "3" * 64),
        ("expected_lexical_mapping_hash", "4" * 64),
    ):
        bad = dict(kwargs)
        bad[key] = bad_value
        with pytest.raises(ValueError, match="mismatch"):
            load_prototype_bank(tmp_path, **bad)

    vector_path = tmp_path / "vectors.npy"
    corrupted = bytearray(vector_path.read_bytes())
    corrupted[-1] ^= 1
    vector_path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_prototype_bank(tmp_path, **kwargs)


def test_projector_state_hash_changes_after_parameter_update(
    real_prototype_banks,
):
    bank, _, projector = real_prototype_banks["bge_m3"]
    before = module_state_sha256(projector)
    assert before == bank.projector_state_hash
    with torch.no_grad():
        next(projector.parameters()).add_(0.01)
    assert module_state_sha256(projector) != before
