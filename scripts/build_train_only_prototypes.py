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
    PrototypeBuilderConfig,
    TrainOnlyPrototypeBuilder,
    save_prototype_bank,
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
        description="Build detached inner-train-only group-balanced prototypes."
    )
    parser.add_argument(
        "--outer-fold",
        type=int,
        action="append",
        help="Repeat to select folds; default builds all five.",
    )
    parser.add_argument(
        "--text-backend",
        choices=("macbert", "bge_m3"),
        action="append",
        help="Repeat to select backends; default builds both.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection-batch-size", type=int, default=64)
    parser.add_argument(
        "--prototype-config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "prototypes"
            / "train_only_group_balanced_v1.yaml"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "prototypes",
    )
    args = parser.parse_args()
    folds = args.outer_fold or list(range(5))
    backends = args.text_backend or ["macbert", "bge_m3"]
    if any(fold not in range(5) for fold in folds):
        raise ValueError("outer-fold must lie in [0,4]")
    if args.projection_batch_size <= 0:
        raise ValueError("projection-batch-size must be positive")

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
    prototype_config = PrototypeBuilderConfig.from_yaml(
        args.prototype_config
    )
    device = torch.device(args.device)
    results = []
    for backend in backends:
        torch.manual_seed(args.seed)
        projector = build_context_text_projector(
            TEXT_CONFIGS[backend],
            cache_metadata_path=CACHE_DIRS[backend] / "metadata.json",
        ).to(device)
        for fold in folds:
            builder = TrainOnlyPrototypeBuilder(
                config=prototype_config,
                split_index=split_index,
                outer_fold=fold,
                keyword_index=keyword_index,
                lexical_identity_index=lexical_index,
                sentence_labels_path=(
                    PROTOCOL
                    / "littleprince_sentence_keyword_labels_v1.csv"
                ),
                word_occurrences_path=(
                    PROTOCOL / "littleprince_word_occurrences_v1.csv"
                ),
                eligibility_path=(
                    PROTOCOL
                    / "littleprince_keyword_fold_eligibility_v1.csv"
                ),
                context_store_root=CACHE_DIRS[backend],
                text_backend=backend,
                projection_batch_size=args.projection_batch_size,
            )
            bank = builder.build(projector)
            directory = (
                args.output_root / backend / f"outer_fold_{fold}"
            )
            artifact = save_prototype_bank(bank, directory)
            results.append(
                {
                    "outer_fold": fold,
                    "text_backend": backend,
                    "vectors_shape": list(bank.vectors.shape),
                    "vectors_dtype": str(bank.vectors.dtype),
                    "available_count": bank.available_count,
                    "core_available": len(
                        keyword_index.core_indices
                        & frozenset(
                            torch.nonzero(bank.available_mask)
                            .flatten()
                            .tolist()
                        )
                    ),
                    "main_available": len(
                        keyword_index.main_indices
                        & frozenset(
                            torch.nonzero(bank.available_mask)
                            .flatten()
                            .tolist()
                        )
                    ),
                    "extended_available": len(
                        keyword_index.extended_indices
                        & frozenset(
                            torch.nonzero(bank.available_mask)
                            .flatten()
                            .tolist()
                        )
                    ),
                    "train_group_df_range_available": [
                        int(bank.train_group_df[bank.available_mask].min()),
                        int(bank.train_group_df[bank.available_mask].max()),
                    ],
                    "projector_state_hash": bank.projector_state_hash,
                    "source_cache_hash": bank.source_cache_hash,
                    "fold_hash": bank.fold_hash,
                    "artifact": artifact,
                }
            )
    print(
        json.dumps(
            {
                "seed": args.seed,
                "device": str(device),
                "role": "train",
                "lexical_mapping": lexical_index.to_metadata(
                    include_mappings=False
                ),
                "banks": results,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
