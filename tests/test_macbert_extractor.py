from types import SimpleNamespace

import numpy as np
import torch

from eeg_keyword_decoding.text import (
    ContextSentence,
    ContextWordOccurrence,
    MacBertContextExtractor,
    MacBertSpec,
)


class FakeFastTokenizer:
    is_fast = True
    vocab_size = 4
    init_kwargs = {"_commit_hash": "tokenizer-sha"}

    def __call__(self, texts, **_kwargs):
        assert texts == ["甲乙"]
        return {
            "input_ids": torch.tensor([[101, 11, 12, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "token_type_ids": torch.tensor([[0, 0, 0, 0]]),
            "offset_mapping": torch.tensor(
                [[[0, 0], [0, 1], [1, 2], [0, 0]]]
            ),
            "special_tokens_mask": torch.tensor([[1, 0, 0, 1]]),
        }

    def convert_ids_to_tokens(self, input_ids):
        mapping = {101: "[CLS]", 11: "甲", 12: "乙", 102: "[SEP]"}
        return [mapping[value] for value in input_ids]


class FakeMacBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            hidden_size=2,
            num_hidden_layers=4,
            _commit_hash="model-sha",
        )
        self.forward_grad_enabled = None

    def forward(self, input_ids, **_kwargs):
        self.forward_grad_enabled = torch.is_grad_enabled()
        base = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 2)
        hidden_states = tuple(base + layer * 100 for layer in range(5))
        return SimpleNamespace(hidden_states=hidden_states)


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


def test_macbert_extractor_uses_full_sentence_last_layers_and_inference_mode():
    model = FakeMacBert()
    extractor = MacBertContextExtractor.from_components(
        MacBertSpec(
            model_id="fake",
            revision="model-sha",
            tokenizer_revision="tokenizer-sha",
            hidden_state_layer_indices=(1, 2, 3, 4),
        ),
        tokenizer=FakeFastTokenizer(),
        model=model,
    )
    result = extractor.extract_batch([_sentence()])[0]

    assert result.vectors.shape == (2, 4, 2)
    assert result.vectors.dtype == np.float32
    np.testing.assert_array_equal(
        result.vectors[0, :, 0],
        np.asarray([111.0, 211.0, 311.0, 411.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.vectors[1, :, 0],
        np.asarray([112.0, 212.0, 312.0, 412.0], dtype=np.float32),
    )
    assert model.training is False
    assert model.forward_grad_enabled is False
    assert result.tokens == ("[CLS]", "甲", "乙", "[SEP]")
    assert [item.token_indices for item in result.alignments] == [(1,), (2,)]
