import csv
from collections import Counter
from pathlib import Path

from eeg_keyword_decoding.data import align_segmented_words


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = PROJECT_ROOT / "data" / "protocols" / "littleprince_hf_v1"


def test_alignment_preserves_decomposition_repetition_and_punctuation():
    assert align_segmented_words(
        "我的小王子,我",
        ["我", "的", "小王子", "我"],
    ) == [(0, 1), (1, 2), (2, 5), (6, 7)]


def test_generated_occurrence_table_matches_frozen_sentence_tokens():
    occurrence_path = PROTOCOL_ROOT / "littleprince_word_occurrences_v1.csv"
    labels_path = PROTOCOL_ROOT / "littleprince_sentence_keyword_labels_v1.csv"
    with occurrence_path.open("r", encoding="utf-8", newline="") as handle:
        occurrences = list(csv.DictReader(handle))
    with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle))

    counts = Counter(int(row["text_embedding_idx"]) for row in occurrences)
    valid_labels = [
        row
        for row in labels
        if row["is_chapter_heading"] != "true" and int(row["token_count"]) > 0
    ]

    assert len(occurrences) == 14034
    assert len(counts) == 2809
    assert {
        int(row["text_embedding_idx"]): int(row["token_count"])
        for row in valid_labels
    } == counts
    assert all(int(row["char_start"]) < int(row["char_end"]) for row in occurrences)

