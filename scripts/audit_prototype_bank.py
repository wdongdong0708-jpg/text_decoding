from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_keyword_decoding.data import (
    build_lexical_identity_index,
    build_master_keyword_index,
    build_split_view_index,
)
from eeg_keyword_decoding.models import build_context_text_projector
from eeg_keyword_decoding.prototypes import (
    file_sha256,
    load_prototype_bank,
    module_state_sha256,
)


PROTOCOL = PROJECT_ROOT / "data" / "protocols" / "littleprince_hf_v1"
MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "littleprince_pl_all_clean_manifest.csv"
)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a saved train-only prototype bank."
    )
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument(
        "--text-backend",
        choices=("macbert", "bge_m3"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bank-dir",
        type=Path,
        help="Defaults to data/cache/prototypes/<backend>/outer_fold_<n>.",
    )
    args = parser.parse_args()
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must lie in [0,4]")

    split_index = build_split_view_index(
        MANIFEST,
        PROTOCOL / "littleprince_sentence_folds_v1.csv",
    )
    keyword_index = build_master_keyword_index(
        PROTOCOL / "littleprince_hf_lexicon_v1.csv"
    )
    lexical_index = build_lexical_identity_index(
        PROTOCOL / "littleprince_word_occurrences_v1.csv",
        split_index,
    )
    torch.manual_seed(args.seed)
    projector = build_context_text_projector(
        TEXT_CONFIGS[args.text_backend],
        cache_metadata_path=(
            CACHE_DIRS[args.text_backend] / "metadata.json"
        ),
    )
    metadata = json.loads(
        (
            CACHE_DIRS[args.text_backend] / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    cache_hash = metadata["arrays"]["context_vectors"]["sha256"]
    eligibility_hash = file_sha256(
        PROTOCOL / "littleprince_keyword_fold_eligibility_v1.csv"
    )
    bank_dir = args.bank_dir or (
        PROJECT_ROOT
        / "data"
        / "cache"
        / "prototypes"
        / args.text_backend
        / f"outer_fold_{args.outer_fold}"
    )
    bank = load_prototype_bank(
        bank_dir,
        expected_outer_fold=args.outer_fold,
        expected_text_backend=args.text_backend,
        expected_projector_state_hash=module_state_sha256(projector),
        expected_source_cache_hash=cache_hash,
        expected_fold_hash=split_index.fold_source_sha256,
        expected_keyword_ids=keyword_index.keyword_ids,
        expected_eligibility_hash=eligibility_hash,
        expected_lexical_mapping_hash=lexical_index.mapping_sha256,
    )
    contributors = set(
        bank.metadata["contributors"]["text_embedding_indices"]
    )
    validation = set(
        split_index.sentence_indices_for(args.outer_fold, "validation")
    )
    test = set(split_index.sentence_indices_for(args.outer_fold, "test"))
    result = {
        "outer_fold": bank.outer_fold,
        "text_backend": bank.text_backend,
        "vectors_shape": list(bank.vectors.shape),
        "vectors_dtype": str(bank.vectors.dtype),
        "available_count": bank.available_count,
        "available_norm_range": [
            float(bank.vectors[bank.available_mask].norm(dim=1).min()),
            float(bank.vectors[bank.available_mask].norm(dim=1).max()),
        ],
        "unavailable_nonzero_count": int(
            torch.count_nonzero(bank.vectors[~bank.available_mask])
        ),
        "train_contributor_count": len(contributors),
        "validation_contributor_intersection": sorted(
            contributors & validation
        ),
        "test_contributor_intersection": sorted(contributors & test),
        "eeg_views_used": bank.metadata["contributors"]["eeg_views_used"],
        "hashes": {
            "projector_state": bank.projector_state_hash,
            "context_vectors": bank.source_cache_hash,
            "context_cache_metadata": bank.source_cache_metadata_hash,
            "fold": bank.fold_hash,
            "eligibility": bank.eligibility_hash,
            "lexical_mapping": bank.lexical_mapping_hash,
            "bank_metadata": file_sha256(bank_dir / "metadata.json"),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
