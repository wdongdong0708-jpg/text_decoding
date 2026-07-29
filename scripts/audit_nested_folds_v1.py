from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = (
    PROJECT_ROOT / "data" / "protocols" / "littleprince_hf_v1"
)
FOLD_PATH = PROTOCOL_ROOT / "littleprince_sentence_folds_v1.csv"
ELIGIBILITY_PATH = (
    PROTOCOL_ROOT / "littleprince_keyword_fold_eligibility_v1.csv"
)
EEG_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "littleprince_pl_all_clean_manifest.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    fold_rows = read_csv(FOLD_PATH)
    eligibility_rows = read_csv(ELIGIBILITY_PATH)
    eeg_rows = read_csv(EEG_MANIFEST_PATH)

    by_fold: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in fold_rows:
        by_fold[int(row["outer_fold"])].append(row)

    test_counts: Counter[int] = Counter()
    role_counts: dict[str, dict[str, int]] = {}
    fold_distribution: dict[str, dict[str, object]] = {}
    for outer_fold, rows in sorted(by_fold.items()):
        group_roles: dict[str, set[str]] = defaultdict(set)
        role_by_idx: dict[int, str] = {}
        for row in rows:
            group_roles[row["sentence_group_id"]].add(row["role"])
            role_by_idx[int(row["text_embedding_idx"])] = row["role"]
            if row["role"] == "test":
                test_counts[int(row["text_embedding_idx"])] += 1
        if any(len(roles) != 1 for roles in group_roles.values()):
            raise AssertionError(
                f"Sentence-group leakage detected in outer fold {outer_fold}"
            )
        role_counts[str(outer_fold)] = dict(
            sorted(Counter(row["role"] for row in rows).items())
        )

        distribution: dict[str, object] = {}
        for role in ("train", "validation", "test"):
            sentence_rows = [row for row in rows if row["role"] == role]
            token_counts = [int(row["token_count"]) for row in sentence_rows]
            eeg_durations = [
                int(row["n_samples"])
                for row in eeg_rows
                if role_by_idx.get(int(row["text_embedding_idx"])) == role
            ]
            quartiles = Counter(
                int(row["story_quartile"]) for row in sentence_rows
            )
            distribution[role] = {
                "sentences": len(sentence_rows),
                "eeg_views": len(eeg_durations),
                "mean_token_count": round(
                    statistics.fmean(token_counts), 4
                ),
                "mean_eeg_samples": round(
                    statistics.fmean(eeg_durations), 4
                ),
                "median_eeg_samples": statistics.median(eeg_durations),
                "min_eeg_samples": min(eeg_durations),
                "max_eeg_samples": max(eeg_durations),
                "story_quartile_sentences": {
                    str(key): quartiles[key] for key in sorted(quartiles)
                },
            }
        fold_distribution[str(outer_fold)] = distribution

    if set(test_counts.values()) != {1}:
        raise AssertionError(
            "Every valid sentence must be outer-test exactly once"
        )
    eeg_views_per_sentence = Counter(
        int(row["text_embedding_idx"]) for row in eeg_rows
    )
    if not set(test_counts).issubset(eeg_views_per_sentence):
        raise AssertionError("Some valid fold sentences have no EEG view")

    eligibility: dict[str, dict[str, dict[str, int]]] = {}
    for outer_fold in sorted(by_fold):
        fold_result: dict[str, dict[str, int]] = {}
        rows = [
            row
            for row in eligibility_rows
            if int(row["outer_fold"]) == outer_fold
        ]
        for view in ("core", "main", "extended"):
            view_rows = [
                row for row in rows if row[f"include_{view}"] == "true"
            ]
            fold_result[view] = {
                "keywords": len(view_rows),
                "eligible_1_1_1": sum(
                    row["eligible_1_1_1"] == "true" for row in view_rows
                ),
                "eligible_10_3_3": sum(
                    row["eligible_10_3_3"] == "true" for row in view_rows
                ),
                "eligible_20_5_5": sum(
                    row["eligible_20_5_5"] == "true" for row in view_rows
                ),
                "eligible_group_20_5_5": sum(
                    row["eligible_group_20_5_5"] == "true"
                    for row in view_rows
                ),
            }
        eligibility[str(outer_fold)] = fold_result

    result = {
        "valid_sentences": len(test_counts),
        "outer_test_exactly_once": True,
        "sentence_groups_are_role_atomic": True,
        "role_counts": role_counts,
        "fold_distribution": fold_distribution,
        "eeg_manifest_join": {
            "valid_sentences_with_eeg": len(test_counts),
            "min_views_per_valid_sentence": min(
                eeg_views_per_sentence[idx] for idx in test_counts
            ),
            "max_views_per_valid_sentence": max(
                eeg_views_per_sentence[idx] for idx in test_counts
            ),
            "all_views_inherit_role_by_text_embedding_idx": True,
        },
        "eligibility": eligibility,
        "df_note": (
            "*_df counts sentence occurrences once; *_group_df counts "
            "distinct normalized sentence groups once"
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
