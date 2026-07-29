from pathlib import Path

from eeg_keyword_decoding.data import audit_littleprince_hf_v1


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_littleprince_protocol_assets_match_v1_contract():
    summary = audit_littleprince_hf_v1(
        PROJECT_ROOT / "data" / "protocols" / "littleprince_hf_v1",
        PROJECT_ROOT
        / "data"
        / "manifests"
        / "littleprince_pl_all_clean_manifest.csv",
    )

    assert summary["master_words"] == 247
    assert summary["core_words"] == 33
    assert summary["main_words"] == 64
    assert summary["extended_words"] == 100
    assert summary["story_local_words"] == 32
    assert summary["sentence_rows"] == 2837
    assert summary["valid_word_sequences"] == 2809
    assert summary["word_occurrences"] == 14034
    assert summary["eeg_rows"] == 21110
    assert summary["eeg_view_counts"] == {6: 167, 7: 1252, 8: 1418}

