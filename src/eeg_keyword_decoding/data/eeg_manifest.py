from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EEGManifestRecord:
    """One EEG view of one sentence occurrence.

    The new project deliberately does not derive task identity from a text
    embedding hash. ``text_embedding_idx`` is the stable sentence-occurrence
    join key; cross-validation groups are supplied by a separate protocol file.
    """

    subject: str
    session: str
    task: str
    run: int
    local_row_idx: int
    global_row_idx: int
    text_embedding_idx: int
    start_time: float
    stop_time: float
    sfreq: float
    start_sample: int
    stop_sample: int
    n_samples: int
    eeg_vhdr_path: Path
    events_tsv_path: Path

    @property
    def sentence_occurrence_id(self) -> str:
        return f"littleprince:{self.text_embedding_idx}"

    @property
    def eeg_view_id(self) -> str:
        return (
            f"{self.subject}/{self.session}/{self.task}/"
            f"run-{self.run}/row-{self.local_row_idx}"
        )


def _record_from_row(row: dict[str, str]) -> EEGManifestRecord:
    return EEGManifestRecord(
        subject=row["subject"],
        session=row["session"],
        task=row["task"],
        run=int(row["run"]),
        local_row_idx=int(row["local_row_idx"]),
        global_row_idx=int(row["global_row_idx"]),
        text_embedding_idx=int(row["text_embedding_idx"]),
        start_time=float(row["start_time"]),
        stop_time=float(row["stop_time"]),
        sfreq=float(row["sfreq"]),
        start_sample=int(row["start_sample"]),
        stop_sample=int(row["stop_sample"]),
        n_samples=int(row["n_samples"]),
        eeg_vhdr_path=Path(row["eeg_vhdr_path"]),
        events_tsv_path=Path(row["events_tsv_path"]),
    )


def load_eeg_manifest(path: str | Path) -> list[EEGManifestRecord]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        records = [_record_from_row(row) for row in csv.DictReader(handle)]
    if not records:
        raise ValueError(f"EEG manifest is empty: {manifest_path}")
    return records


def validate_eeg_manifest(
    records: list[EEGManifestRecord],
    *,
    check_files: bool = False,
) -> None:
    seen_views: set[str] = set()
    for record in records:
        if record.stop_sample <= record.start_sample:
            raise ValueError(f"Invalid EEG window for {record.eeg_view_id}")
        if record.n_samples != record.stop_sample - record.start_sample:
            raise ValueError(f"n_samples mismatch for {record.eeg_view_id}")
        if record.sfreq <= 0:
            raise ValueError(f"Invalid sampling frequency for {record.eeg_view_id}")
        if record.eeg_view_id in seen_views:
            raise ValueError(f"Duplicate EEG view ID: {record.eeg_view_id}")
        seen_views.add(record.eeg_view_id)
        if check_files:
            if not record.eeg_vhdr_path.exists():
                raise FileNotFoundError(record.eeg_vhdr_path)
            if not record.events_tsv_path.exists():
                raise FileNotFoundError(record.events_tsv_path)


def group_records_by_sentence(
    records: list[EEGManifestRecord],
) -> dict[int, list[EEGManifestRecord]]:
    grouped: dict[int, list[EEGManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.text_embedding_idx].append(record)
    return dict(grouped)

