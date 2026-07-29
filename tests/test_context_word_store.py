from pathlib import Path

import numpy as np
import pytest

from eeg_keyword_decoding.text import (
    ContextSentence,
    ContextWordOccurrence,
    ContextWordStore,
    load_context_sentences,
    write_context_word_store,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = PROJECT_ROOT / "data" / "protocols" / "littleprince_hf_v1"


def _occurrence(
    sentence_idx: int,
    position: int,
    surface: str,
    start: int,
    stop: int,
) -> ContextWordOccurrence:
    return ContextWordOccurrence(
        word_occurrence_id=f"lp:{sentence_idx}:word:{position}",
        text_embedding_idx=sentence_idx,
        word_position=position,
        surface_form=surface,
        char_start=start,
        char_end=stop,
        keyword_id="" if position else "kw-1",
    )


def _sentences() -> list[ContextSentence]:
    return [
        ContextSentence(
            text_embedding_idx=10,
            text="甲乙",
            occurrences=(
                _occurrence(10, 0, "甲", 0, 1),
                _occurrence(10, 1, "乙", 1, 2),
            ),
        ),
        ContextSentence(
            text_embedding_idx=20,
            text="丙",
            occurrences=(_occurrence(20, 0, "丙", 0, 1),),
        ),
    ]


def _metadata_fields() -> dict[str, object]:
    return {
        "model": {"model_id": "fake", "resolved_revision": "abc"},
        "tokenizer": {"tokenizer_id": "fake", "resolved_revision": "abc"},
        "extraction": {"aggregation_rule": "test"},
        "sources": {},
        "generation": {"sentence_count": 2, "word_occurrence_count": 3},
        "runtime": {"python": "test"},
    }


def test_context_word_store_round_trip_and_sentence_lookup(tmp_path: Path):
    sentences = _sentences()
    vectors = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
    write_context_word_store(
        tmp_path,
        vectors=vectors,
        sentences=sentences,
        metadata_fields=_metadata_fields(),
    )

    store = ContextWordStore(tmp_path, verify_hashes=True)
    store.audit_against(sentences)
    assert store.context_vectors.shape == (3, 4, 2)
    assert store.context_vectors.dtype == np.float32
    assert store.sentence_offsets.tolist() == [0, 2, 3]

    sentence = store.get_sentence(10)
    assert sentence.vectors.shape == (2, 4, 2)
    assert sentence.surface_forms.tolist() == ["甲", "乙"]
    assert sentence.char_spans.tolist() == [[0, 1], [1, 2]]
    assert sentence.keyword_ids.tolist() == ["kw-1", ""]
    np.testing.assert_array_equal(sentence.vectors, vectors[:2])

    with pytest.raises(KeyError, match="Unknown text_embedding_idx"):
        store.get_sentence(999)
    with pytest.raises(FileExistsError):
        write_context_word_store(
            tmp_path,
            vectors=vectors,
            sentences=sentences,
            metadata_fields=_metadata_fields(),
        )


def test_frozen_context_sentence_order_and_counts_match_protocol():
    sentences = load_context_sentences(
        PROTOCOL_ROOT / "littleprince_sentence_keyword_labels_v1.csv",
        PROTOCOL_ROOT / "littleprince_word_occurrences_v1.csv",
    )
    assert len(sentences) == 2809
    assert sum(sentence.word_count for sentence in sentences) == 14034
    assert sentences[0].text_embedding_idx == 17
    assert sentences[0].text == "我六岁那年,"
    assert [item.surface_form for item in sentences[0].occurrences] == [
        "我",
        "六岁",
        "那",
        "年",
    ]


def test_context_store_contract_accepts_model_agnostic_2d_vectors(
    tmp_path: Path,
):
    sentences = _sentences()
    vectors = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
    write_context_word_store(
        tmp_path,
        vectors=vectors,
        sentences=sentences,
        metadata_fields=_metadata_fields(),
    )
    store = ContextWordStore(tmp_path)
    assert store.context_vectors.shape == (3, 5)
    assert store.get_sentence(20).vectors.shape == (1, 5)


def test_context_store_can_restore_float16_vectors_as_float32(tmp_path: Path):
    sentences = _sentences()
    vectors = np.arange(3 * 5, dtype=np.float16).reshape(3, 5)
    write_context_word_store(
        tmp_path,
        vectors=vectors,
        sentences=sentences,
        metadata_fields=_metadata_fields(),
    )
    store = ContextWordStore(tmp_path)
    stored = store.get_sentence(10)
    restored = store.get_sentence(10, vector_dtype="float32")
    assert stored.vectors.dtype == np.float16
    assert restored.vectors.dtype == np.float32
    np.testing.assert_array_equal(restored.vectors, vectors[:2].astype(np.float32))
