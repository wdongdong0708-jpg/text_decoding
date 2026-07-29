from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_keyword_decoding.data import (  # noqa: E402
    build_split_view_index,
    build_subject_index,
)
from eeg_keyword_decoding.data.protocol_assets import file_sha256  # noqa: E402
from eeg_keyword_decoding.models import (  # noqa: E402
    build_eeg_sequence_encoder,
)


DEFAULT_CONFIGS = (
    PROJECT_ROOT / "configs" / "models" / "eeg_sequence_conv_v1.yaml",
    PROJECT_ROOT
    / "configs"
    / "models"
    / "eeg_sequence_conv_no_subject_v1.yaml",
)


def _prefix_mask(lengths: list[int], maximum: int) -> torch.Tensor:
    values = torch.tensor(lengths, dtype=torch.int64)
    return torch.arange(maximum).unsqueeze(0) < values.unsqueeze(1)


def _padding_audit(model: nn.Module) -> dict[str, Any]:
    torch.manual_seed(29)
    model.eval()
    short = torch.randn(1, 128, 33)
    subject = torch.tensor([2], dtype=torch.int64)
    with torch.inference_mode():
        single = model(
            eeg=short,
            eeg_mask=torch.ones(1, 33, dtype=torch.bool),
            subject_indices=subject,
        )
        padded = torch.zeros(1, 128, 101)
        padded[:, :, :33] = short
        padded[:, :, 33:] = torch.randn(1, 128, 68) * 100_000
        mixed = model(
            eeg=padded,
            eeg_mask=_prefix_mask([33], 101),
            subject_indices=subject,
        )
    valid = int(single.lengths[0])
    difference = (
        single.sequence[:, :valid] - mixed.sequence[:, :valid]
    ).abs()
    invalid_values = mixed.sequence.masked_select(~mixed.mask.unsqueeze(-1))
    return {
        "single_vs_extra_padding_max_abs_error": float(
            difference.max().item()
        ),
        "padding_region_max_abs_value": (
            float(invalid_values.abs().max().item())
            if invalid_values.numel()
            else 0.0
        ),
        "invalid_padding_input_value_scale": 100_000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit EEG sequence encoder configuration and mask rules."
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        dest="configs",
        help="Repeat to audit multiple configs; defaults to both v1 configs.",
    )
    args = parser.parse_args()
    configs = tuple(args.configs) if args.configs else DEFAULT_CONFIGS

    protocol = (
        PROJECT_ROOT / "data" / "protocols" / "littleprince_hf_v1"
    )
    split_index = build_split_view_index(
        PROJECT_ROOT
        / "data"
        / "manifests"
        / "littleprince_pl_all_clean_manifest.csv",
        protocol / "littleprince_sentence_folds_v1.csv",
    )
    subject_index = build_subject_index(split_index.manifest_records)
    input_channels = 128
    all_lengths = [
        record.n_samples for record in split_index.valid_manifest_records
    ]
    audited_lengths = [7, 31, 32, 33, 100, 101, max(all_lengths)]

    model_results: list[dict[str, Any]] = []
    for config_path in configs:
        resolved = config_path.resolve()
        model = build_eeg_sequence_encoder(
            resolved,
            actual_input_channels=input_channels,
            actual_num_subjects=len(subject_index.subjects),
        )
        length_input = max(audited_lengths)
        with torch.inference_mode():
            output = model.eval()(
                eeg=torch.zeros(
                    len(audited_lengths),
                    input_channels,
                    length_input,
                ),
                eeg_mask=_prefix_mask(audited_lengths, length_input),
                subject_indices=torch.arange(
                    len(audited_lengths),
                    dtype=torch.int64,
                ),
            )
        forbidden_norms = [
            type(module).__name__
            for module in model.modules()
            if isinstance(module, (nn.BatchNorm1d, nn.GroupNorm))
        ]
        model_results.append(
            {
                "config_path": str(resolved),
                "config_file_sha256": file_sha256(resolved),
                "canonical_config_sha256": (
                    model.config.canonical_sha256
                ),
                "config": model.to_config(),
                "total_parameters": model.total_parameter_count,
                "trainable_parameters": model.trainable_parameter_count,
                "forbidden_normalization_modules": forbidden_norms,
                "length_audit": [
                    {
                        "input": input_length,
                        "observed_output": int(output.lengths[index]),
                        "expected_ceil_divide_by_four": (
                            input_length + 3
                        )
                        // 4,
                    }
                    for index, input_length in enumerate(audited_lengths)
                ],
                "maximum_output_time": output.sequence.shape[1],
                "padding_audit": _padding_audit(model),
            }
        )

    result = {
        "audited_data_contract": {
            "input_channels": input_channels,
            "num_subjects": len(subject_index.subjects),
            "subjects": list(subject_index.subjects),
            "minimum_real_inner_train_length_fold_0": min(
                record.n_samples
                for record in split_index.records_for(0, "train")
            ),
            "maximum_real_inner_train_length_fold_0": max(
                record.n_samples
                for record in split_index.records_for(0, "train")
            ),
            "global_valid_length_range": [
                min(all_lengths),
                max(all_lengths),
            ],
        },
        "models": model_results,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
