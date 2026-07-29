from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eeg_keyword_decoding.data import (
    build_master_keyword_index,
    build_split_view_index,
    build_subject_index,
)
from eeg_keyword_decoding.text import (
    load_context_sentences,
    write_context_word_store,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = (
    PROJECT_ROOT / "data" / "protocols" / "littleprince_hf_v1"
)
MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "littleprince_pl_all_clean_manifest.csv"
)


@pytest.fixture(scope="session")
def split_index():
    return build_split_view_index(
        MANIFEST_PATH,
        PROTOCOL_ROOT / "littleprince_sentence_folds_v1.csv",
    )


@pytest.fixture(scope="session")
def subject_index(split_index):
    return build_subject_index(split_index.manifest_records)


@pytest.fixture(scope="session")
def keyword_index():
    return build_master_keyword_index(
        PROTOCOL_ROOT / "littleprince_hf_lexicon_v1.csv"
    )


@pytest.fixture(scope="session")
def synthetic_context_stores(tmp_path_factory):
    sentences = load_context_sentences(
        PROTOCOL_ROOT / "littleprince_sentence_keyword_labels_v1.csv",
        PROTOCOL_ROOT / "littleprince_word_occurrences_v1.csv",
    )
    total_words = sum(sentence.word_count for sentence in sentences)
    root = tmp_path_factory.mktemp("context_stores")
    stores = {}
    for backend, tail_shape in {
        "macbert": (4, 3),
        "bge_m3": (5,),
    }.items():
        output = root / backend
        vectors = np.arange(
            total_words * int(np.prod(tail_shape)),
            dtype=np.float32,
        ).reshape(total_words, *tail_shape)
        write_context_word_store(
            output,
            vectors=vectors,
            sentences=sentences,
            metadata_fields={
                "model": {
                    "model_id": f"fake-{backend}",
                    "resolved_revision": f"{backend}-model-sha",
                },
                "tokenizer": {
                    "tokenizer_id": f"fake-{backend}",
                    "resolved_revision": f"{backend}-tokenizer-sha",
                },
                "extraction": {"backend": backend},
                "sources": {},
                "generation": {
                    "sentence_count": len(sentences),
                    "word_occurrence_count": total_words,
                },
                "runtime": {"python": "pytest"},
            },
        )
        stores[backend] = output
    return stores
