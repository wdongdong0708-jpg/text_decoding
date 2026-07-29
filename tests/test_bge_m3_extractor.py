from types import SimpleNamespace

import numpy as np
import torch

from eeg_keyword_decoding.text import (
    BgeM3ColbertExtractor,
    BgeM3Spec,
    ContextSentence,
    ContextWordOccurrence,
)


class FakeBgeTokenizer:
    is_fast = True
    vocab_size = 8
    padding_side = "right"
    unk_token = "<unk>"

    def __call__(self, texts, **_kwargs):
        assert texts == ["甲乙"]
        return {
            "input_ids": [[0, 11, 12, 2]],
            "attention_mask": [[1, 1, 1, 1]],
            "offset_mapping": [[(0, 0), (0, 1), (1, 2), (0, 0)]],
            "special_tokens_mask": [[1, 0, 0, 1]],
        }

    def convert_ids_to_tokens(self, input_ids):
        mapping = {0: "<s>", 11: "▁甲", 12: "乙", 2: "</s>"}
        return [mapping[value] for value in input_ids]


class FakeFlagEmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            hidden_size=4,
            num_hidden_layers=2,
        )
        self.colbert_linear = torch.nn.Linear(4, 4)


class FakeOfficialEncoder:
    def __init__(self):
        self.model = FakeFlagEmbeddingModel()
        self.forward_grad_enabled = None

    def encode(self, sentences, **kwargs):
        assert sentences == ["甲乙"]
        assert kwargs["return_dense"] is False
        assert kwargs["return_sparse"] is False
        assert kwargs["return_colbert_vecs"] is True
        self.forward_grad_enabled = torch.is_grad_enabled()
        return {
            "dense_vecs": None,
            "lexical_weights": None,
            "colbert_vecs": [
                np.asarray(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                )
            ],
        }


def _sentence() -> ContextSentence:
    return ContextSentence(
        text_embedding_idx=1,
        text="甲乙",
        occurrences=(
            ContextWordOccurrence(
                word_occurrence_id="lp:1:word:0",
                text_embedding_idx=1,
                word_position=0,
                surface_form="甲",
                char_start=0,
                char_end=1,
                keyword_id="",
            ),
            ContextWordOccurrence(
                word_occurrence_id="lp:1:word:1",
                text_embedding_idx=1,
                word_position=1,
                surface_form="乙",
                char_start=1,
                char_end=2,
                keyword_id="",
            ),
        ),
    )


def test_bge_extractor_uses_public_colbert_vectors_with_explicit_token_mapping():
    encoder = FakeOfficialEncoder()
    extractor = BgeM3ColbertExtractor.from_components(
        BgeM3Spec(
            model_id="fake",
            revision="model-sha",
            tokenizer_revision="model-sha",
            max_length=16,
            batch_size=2,
        ),
        tokenizer=FakeBgeTokenizer(),
        encoder=encoder,
    )
    result = extractor.extract_batch([_sentence()])[0]

    assert result.vectors.shape == (2, 4)
    assert result.vectors.dtype == np.float32
    np.testing.assert_array_equal(
        result.vectors,
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert result.tokens == ("▁甲", "乙", "</s>")
    assert result.source_token_indices == (1, 2, 3)
    assert [item.token_indices for item in result.alignments] == [(0,), (1,)]
    assert encoder.model.training is False
    assert encoder.forward_grad_enabled is False
