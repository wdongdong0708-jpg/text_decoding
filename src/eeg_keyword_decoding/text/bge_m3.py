from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .offsets import (
    TokenAlignment,
    aggregate_token_features,
    align_sentence_to_tokens,
)
from .schema import ContextSentence


BGE_M3_COLBERT_TOKEN_MAPPING = (
    "FlagEmbedding colbert output index j maps to tokenizer index j+1: "
    "the model removes the first CLS token, retains the final SEP token, "
    "and the public encode interface strips padding"
)


@dataclass(frozen=True)
class BgeM3Spec:
    model_id: str
    revision: str
    tokenizer_revision: str
    max_length: int = 512
    use_fast: bool = True
    use_fp16: bool = False
    normalize_embeddings: bool = True
    colbert_dim: int = -1
    batch_size: int = 16


@dataclass(frozen=True)
class BgeM3SentenceExtraction:
    sentence: ContextSentence
    vectors: np.ndarray
    alignments: tuple[TokenAlignment, ...]
    tokens: tuple[str, ...]
    offset_mapping: tuple[tuple[int, int], ...]
    special_tokens_mask: tuple[int, ...]
    attention_mask: tuple[int, ...]
    source_token_indices: tuple[int, ...]


def colbert_source_token_indices(
    *,
    attention_mask: Sequence[int],
    special_tokens_mask: Sequence[int],
    colbert_row_count: int,
) -> tuple[int, ...]:
    """Map public FlagEmbedding ColBERT rows to tokenizer token positions.

    FlagEmbedding applies ``colbert_linear(last_hidden_state[:, 1:])`` and
    subsequently strips padding from the public ``encode`` result. Therefore
    row zero is tokenizer position one, not the CLS token. The terminal SEP
    token remains present and is excluded from word aggregation by its
    special-token mask and zero-length offset.
    """

    if len(attention_mask) != len(special_tokens_mask):
        raise ValueError(
            "attention_mask and special_tokens_mask length mismatch"
        )
    attended_indices = [
        index for index, attended in enumerate(attention_mask) if int(attended)
    ]
    if not attended_indices:
        raise ValueError("BGE-M3 tokenization produced no attended tokens")
    first_index = attended_indices[0]
    if int(special_tokens_mask[first_index]) != 1:
        raise ValueError("First attended BGE-M3 token must be special CLS")
    source_indices = tuple(attended_indices[1:])
    if len(source_indices) != int(colbert_row_count):
        raise ValueError(
            "FlagEmbedding ColBERT row count does not match tokenizer mapping: "
            f"{colbert_row_count} != {len(source_indices)}"
        )
    return source_indices


class BgeM3ColbertExtractor:
    def __init__(
        self,
        spec: BgeM3Spec,
        *,
        device: str = "auto",
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        if spec.revision != spec.tokenizer_revision:
            raise ValueError(
                "BGE-M3 model and tokenizer revisions must identify the same "
                "pinned snapshot"
            )
        if spec.use_fp16:
            raise ValueError(
                "This project requires float32 BGE-M3 forward computation"
            )

        from FlagEmbedding import BGEM3FlagModel
        from huggingface_hub import snapshot_download

        snapshot_path = snapshot_download(
            repo_id=spec.model_id,
            revision=spec.revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
            ignore_patterns=[
                "flax_model.msgpack",
                "rust_model.ot",
                "tf_model.h5",
            ],
        )
        resolved_device = self._resolve_device(device)
        encoder = BGEM3FlagModel(
            snapshot_path,
            use_fp16=False,
            use_bf16=False,
            devices=resolved_device,
            normalize_embeddings=spec.normalize_embeddings,
            colbert_dim=spec.colbert_dim,
            batch_size=spec.batch_size,
            passage_max_length=spec.max_length,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        self._initialize(
            spec,
            tokenizer=encoder.tokenizer,
            encoder=encoder,
            device=resolved_device,
            snapshot_path=Path(snapshot_path),
        )

    @classmethod
    def from_components(
        cls,
        spec: BgeM3Spec,
        *,
        tokenizer: Any,
        encoder: Any,
        device: str = "cpu",
        snapshot_path: str | Path | None = None,
    ) -> BgeM3ColbertExtractor:
        instance = cls.__new__(cls)
        instance._initialize(
            spec,
            tokenizer=tokenizer,
            encoder=encoder,
            device=instance._resolve_device(device),
            snapshot_path=(
                Path(snapshot_path) if snapshot_path is not None else None
            ),
        )
        return instance

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            return "cuda:0"
        return device

    def _initialize(
        self,
        spec: BgeM3Spec,
        *,
        tokenizer: Any,
        encoder: Any,
        device: str,
        snapshot_path: Path | None,
    ) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("BGE-M3 extraction requires a fast tokenizer")
        if spec.use_fp16:
            raise ValueError(
                "This project requires float32 BGE-M3 forward computation"
            )
        self.spec = spec
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.device = torch.device(device)
        self.snapshot_path = snapshot_path
        model = getattr(self.encoder, "model", None)
        if model is not None and hasattr(model, "eval"):
            model.eval()

    def _tokenize_with_offsets(
        self,
        sentences: Sequence[ContextSentence],
    ) -> Any:
        texts = [sentence.text for sentence in sentences]
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        too_long = [
            (sentence.text_embedding_idx, len(input_ids))
            for sentence, input_ids in zip(
                sentences,
                encoded["input_ids"],
                strict=True,
            )
            if len(input_ids) > self.spec.max_length
        ]
        if too_long:
            raise ValueError(
                "BGE-M3 input exceeds configured maximum without truncation: "
                f"{too_long[:5]}"
            )
        return encoded

    def _encode_public(self, texts: Sequence[str]) -> list[np.ndarray]:
        with torch.inference_mode():
            output = self.encoder.encode(
                list(texts),
                batch_size=self.spec.batch_size,
                max_length=self.spec.max_length,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )
        raw_vectors = output["colbert_vecs"]
        if not isinstance(raw_vectors, list):
            raise TypeError(
                "FlagEmbedding must return one ColBERT array per input sentence"
            )
        if len(raw_vectors) != len(texts):
            raise ValueError(
                f"FlagEmbedding returned {len(raw_vectors)} sentences for "
                f"{len(texts)} inputs"
            )
        arrays: list[np.ndarray] = []
        for value in raw_vectors:
            array = np.asarray(value)
            if array.dtype != np.float32:
                raise ValueError(
                    "BGE-M3 ColBERT forward must return float32, got "
                    f"{array.dtype}"
                )
            if array.ndim != 2:
                raise ValueError(
                    f"Expected [token, hidden] ColBERT output, got {array.shape}"
                )
            arrays.append(array)
        return arrays

    def extract_batch(
        self,
        sentences: Sequence[ContextSentence],
    ) -> list[BgeM3SentenceExtraction]:
        if not sentences:
            return []
        encoded = self._tokenize_with_offsets(sentences)
        raw_vectors = self._encode_public(
            [sentence.text for sentence in sentences]
        )

        results: list[BgeM3SentenceExtraction] = []
        for batch_index, (sentence, token_features) in enumerate(
            zip(sentences, raw_vectors, strict=True)
        ):
            input_ids = [int(value) for value in encoded["input_ids"][batch_index]]
            full_offsets = [
                (int(start), int(stop))
                for start, stop in encoded["offset_mapping"][batch_index]
            ]
            full_special = [
                int(value)
                for value in encoded["special_tokens_mask"][batch_index]
            ]
            full_attention = [
                int(value) for value in encoded["attention_mask"][batch_index]
            ]
            source_indices = colbert_source_token_indices(
                attention_mask=full_attention,
                special_tokens_mask=full_special,
                colbert_row_count=token_features.shape[0],
            )
            offsets = tuple(full_offsets[index] for index in source_indices)
            special = tuple(full_special[index] for index in source_indices)
            attended = tuple(full_attention[index] for index in source_indices)
            tokens = tuple(
                self.tokenizer.convert_ids_to_tokens(
                    [input_ids[index] for index in source_indices]
                )
            )
            alignments = align_sentence_to_tokens(
                sentence.occurrences,
                offsets,
                special,
                attended,
            )
            vectors = aggregate_token_features(
                token_features.astype(np.float32, copy=False),
                alignments,
            ).astype(np.float32, copy=False)
            if not np.isfinite(vectors).all():
                raise ValueError(
                    f"Non-finite BGE-M3 word vector for "
                    f"{sentence.text_embedding_idx}"
                )
            if np.any(np.linalg.norm(vectors, axis=-1) == 0):
                raise ValueError(
                    f"Zero BGE-M3 word vector for "
                    f"{sentence.text_embedding_idx}"
                )
            results.append(
                BgeM3SentenceExtraction(
                    sentence=sentence,
                    vectors=vectors,
                    alignments=alignments,
                    tokens=tokens,
                    offset_mapping=offsets,
                    special_tokens_mask=special,
                    attention_mask=attended,
                    source_token_indices=source_indices,
                )
            )
        return results

    def verify_public_mapping_against_direct_forward(
        self,
        sentences: Sequence[ContextSentence],
        *,
        atol: float = 1e-6,
    ) -> dict[str, Any]:
        """Prove the public-output/tokenizer mapping against direct forward."""

        if not sentences:
            raise ValueError("At least one sentence is required")
        if not hasattr(self.encoder, "model"):
            raise TypeError("Direct mapping verification requires FlagEmbedding")
        texts = [sentence.text for sentence in sentences]
        public_vectors = self._encode_public(texts)
        batch = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if int(batch["input_ids"].shape[1]) > self.spec.max_length:
            raise ValueError("Direct verification batch exceeds max_length")
        model_inputs = {
            key: value.to(self.device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        model = self.encoder.model.to(self.device)
        model.eval()
        with torch.inference_mode():
            direct = model(
                model_inputs,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )["colbert_vecs"].detach().to("cpu", torch.float32).numpy()

        maximum_error = 0.0
        row_counts: list[int] = []
        for batch_index, public in enumerate(public_vectors):
            row_count = int(batch["attention_mask"][batch_index].sum()) - 1
            direct_rows = direct[batch_index, :row_count]
            if direct_rows.shape != public.shape:
                raise ValueError(
                    "Direct/public ColBERT shape mismatch: "
                    f"{direct_rows.shape} != {public.shape}"
                )
            error = float(np.max(np.abs(direct_rows - public), initial=0.0))
            maximum_error = max(maximum_error, error)
            row_counts.append(row_count)
        if maximum_error > atol:
            raise ValueError(
                f"Direct/public ColBERT outputs differ by {maximum_error} "
                f"(tolerance {atol})"
            )
        return {
            "mapping_rule": BGE_M3_COLBERT_TOKEN_MAPPING,
            "verified_sentence_count": len(sentences),
            "colbert_row_counts": row_counts,
            "maximum_absolute_error": maximum_error,
            "tolerance": atol,
        }

    def model_metadata(self) -> dict[str, Any]:
        model_wrapper = getattr(self.encoder, "model", None)
        config = getattr(model_wrapper, "config", None)
        if config is None and model_wrapper is not None:
            config = getattr(getattr(model_wrapper, "model", None), "config", None)
        hidden_size = int(getattr(config, "hidden_size"))
        colbert_layer = getattr(model_wrapper, "colbert_linear", None)
        colbert_dimension = (
            int(colbert_layer.out_features)
            if colbert_layer is not None
            else hidden_size
        )
        parameter_count = (
            sum(parameter.numel() for parameter in model_wrapper.parameters())
            if model_wrapper is not None
            else None
        )
        dtypes = (
            sorted({str(parameter.dtype) for parameter in model_wrapper.parameters()})
            if model_wrapper is not None
            else []
        )
        return {
            "backend": "FlagEmbedding.BGEM3FlagModel",
            "interface": (
                "BGEM3FlagModel.encode(return_dense=False, "
                "return_sparse=False, return_colbert_vecs=True)"
            ),
            "model_id": self.spec.model_id,
            "requested_revision": self.spec.revision,
            "resolved_revision": self.spec.revision,
            "snapshot_path": (
                str(self.snapshot_path) if self.snapshot_path is not None else None
            ),
            "model_class": (
                type(getattr(model_wrapper, "model", model_wrapper)).__name__
                if model_wrapper is not None
                else type(self.encoder).__name__
            ),
            "hidden_size": hidden_size,
            "colbert_dimension": colbert_dimension,
            "num_hidden_layers": int(getattr(config, "num_hidden_layers")),
            "parameter_count": parameter_count,
            "parameter_dtypes": dtypes,
            "normalize_colbert_vectors": self.spec.normalize_embeddings,
            "source_weight_format": "pytorch_bin",
        }

    def tokenizer_metadata(self) -> dict[str, Any]:
        return {
            "tokenizer_id": self.spec.model_id,
            "requested_revision": self.spec.tokenizer_revision,
            "resolved_revision": self.spec.tokenizer_revision,
            "tokenizer_class": type(self.tokenizer).__name__,
            "is_fast": bool(self.tokenizer.is_fast),
            "vocab_size": int(self.tokenizer.vocab_size),
            "max_length": self.spec.max_length,
            "padding_side": str(self.tokenizer.padding_side),
        }

    def runtime_metadata(self) -> dict[str, Any]:
        def version(package: str) -> str:
            return importlib_metadata.version(package)

        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "transformers": version("transformers"),
            "tokenizers": version("tokenizers"),
            "huggingface_hub": version("huggingface-hub"),
            "safetensors": version("safetensors"),
            "FlagEmbedding": version("FlagEmbedding"),
            "device": str(self.device),
            "gpu": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else None
            ),
            "compute_dtype": "float32",
        }
