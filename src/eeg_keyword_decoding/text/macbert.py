from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any, Sequence

import numpy as np
import torch

from .offsets import (
    TokenAlignment,
    aggregate_token_features,
    align_sentence_to_tokens,
)
from .schema import ContextSentence


@dataclass(frozen=True)
class MacBertSpec:
    model_id: str
    revision: str
    tokenizer_revision: str
    hidden_state_layer_indices: tuple[int, ...] = (9, 10, 11, 12)
    max_length: int = 512
    use_fast: bool = True
    use_safetensors: bool = False


@dataclass(frozen=True)
class SentenceExtraction:
    sentence: ContextSentence
    vectors: np.ndarray
    alignments: tuple[TokenAlignment, ...]
    tokens: tuple[str, ...]
    offset_mapping: tuple[tuple[int, int], ...]
    special_tokens_mask: tuple[int, ...]
    attention_mask: tuple[int, ...]


class MacBertContextExtractor:
    def __init__(
        self,
        spec: MacBertSpec,
        *,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        # Transformers 5.x otherwise starts a background Hub PR lookup for
        # legacy .bin checkpoints even when use_safetensors=False. The pinned
        # MacBERT revision intentionally uses its original PyTorch checkpoint.
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id,
            revision=spec.tokenizer_revision,
            use_fast=spec.use_fast,
            local_files_only=local_files_only,
        )
        model = AutoModel.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            use_safetensors=spec.use_safetensors,
            local_files_only=local_files_only,
        )
        self._initialize(
            spec,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

    @classmethod
    def from_components(
        cls,
        spec: MacBertSpec,
        *,
        tokenizer: Any,
        model: torch.nn.Module,
        device: str = "cpu",
    ) -> MacBertContextExtractor:
        instance = cls.__new__(cls)
        instance._initialize(
            spec,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
        return instance

    def _initialize(
        self,
        spec: MacBertSpec,
        *,
        tokenizer: Any,
        model: torch.nn.Module,
        device: str,
    ) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("MacBERT extraction requires a fast tokenizer")
        resolved_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if resolved_device == "auto":
            resolved_device = "cpu"
        self.spec = spec
        self.tokenizer = tokenizer
        self.model = model.to(torch.device(resolved_device))
        self.device = torch.device(resolved_device)
        self.model.eval()

        model_config = getattr(self.model, "config", None)
        hidden_layers = getattr(model_config, "num_hidden_layers", None)
        if hidden_layers is not None and max(
            spec.hidden_state_layer_indices
        ) > int(hidden_layers):
            raise ValueError(
                "Requested hidden-state layer exceeds model depth: "
                f"{spec.hidden_state_layer_indices} vs {hidden_layers}"
            )

    def extract_batch(
        self,
        sentences: Sequence[ContextSentence],
    ) -> list[SentenceExtraction]:
        if not sentences:
            return []
        texts = [sentence.text for sentence in sentences]
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        offset_mapping = encoded.pop("offset_mapping")
        special_tokens_mask = encoded.pop("special_tokens_mask")
        if int(encoded["input_ids"].shape[1]) > self.spec.max_length:
            raise ValueError(
                f"Tokenized batch length {encoded['input_ids'].shape[1]} "
                f"exceeds configured maximum {self.spec.max_length}"
            )
        attention_mask = encoded["attention_mask"]
        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        with torch.inference_mode():
            output = self.model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states")
        try:
            selected = [
                hidden_states[index]
                for index in self.spec.hidden_state_layer_indices
            ]
        except IndexError as error:
            raise ValueError(
                f"Model returned {len(hidden_states)} hidden-state tensors, "
                f"cannot select {self.spec.hidden_state_layer_indices}"
            ) from error
        token_features = (
            torch.stack(selected, dim=2).detach().to("cpu", torch.float32).numpy()
        )

        results: list[SentenceExtraction] = []
        for batch_index, sentence in enumerate(sentences):
            offsets = tuple(
                (int(start), int(stop))
                for start, stop in offset_mapping[batch_index].tolist()
            )
            special = tuple(
                int(value) for value in special_tokens_mask[batch_index].tolist()
            )
            attended = tuple(
                int(value) for value in attention_mask[batch_index].tolist()
            )
            alignments = align_sentence_to_tokens(
                sentence.occurrences,
                offsets,
                special,
                attended,
            )
            vectors = aggregate_token_features(
                token_features[batch_index],
                alignments,
            ).astype(np.float32, copy=False)
            input_ids = encoded["input_ids"][batch_index].tolist()
            tokens = tuple(self.tokenizer.convert_ids_to_tokens(input_ids))
            results.append(
                SentenceExtraction(
                    sentence=sentence,
                    vectors=vectors,
                    alignments=alignments,
                    tokens=tokens,
                    offset_mapping=offsets,
                    special_tokens_mask=special,
                    attention_mask=attended,
                )
            )
        return results

    def model_metadata(self) -> dict[str, Any]:
        config = getattr(self.model, "config", None)
        return {
            "model_id": self.spec.model_id,
            "requested_revision": self.spec.revision,
            "resolved_revision": getattr(config, "_commit_hash", None)
            or self.spec.revision,
            "model_class": type(self.model).__name__,
            "hidden_size": int(getattr(config, "hidden_size")),
            "num_hidden_layers": int(getattr(config, "num_hidden_layers")),
            "parameter_count": sum(
                parameter.numel() for parameter in self.model.parameters()
            ),
            "source_weight_format": (
                "safetensors" if self.spec.use_safetensors else "pytorch_bin"
            ),
        }

    def tokenizer_metadata(self) -> dict[str, Any]:
        init_kwargs = getattr(self.tokenizer, "init_kwargs", {})
        return {
            "tokenizer_id": self.spec.model_id,
            "requested_revision": self.spec.tokenizer_revision,
            "resolved_revision": init_kwargs.get("_commit_hash")
            or self.spec.tokenizer_revision,
            "tokenizer_class": type(self.tokenizer).__name__,
            "is_fast": bool(self.tokenizer.is_fast),
            "vocab_size": int(self.tokenizer.vocab_size),
            "max_length": self.spec.max_length,
        }

    def runtime_metadata(self) -> dict[str, Any]:
        def version(package: str) -> str:
            return importlib_metadata.version(package)

        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "tokenizers": version("tokenizers"),
            "huggingface_hub": version("huggingface-hub"),
            "safetensors": version("safetensors"),
            "device": str(self.device),
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else None
            ),
            "compute_dtype": "float32",
            "safetensors_auto_conversion_disabled": (
                os.environ.get("DISABLE_SAFETENSORS_CONVERSION") == "1"
            ),
        }
