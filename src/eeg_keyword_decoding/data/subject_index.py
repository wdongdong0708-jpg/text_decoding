from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .eeg_manifest import EEGManifestRecord


@dataclass(frozen=True)
class SubjectIndex:
    subjects: tuple[str, ...]
    subject_to_index: dict[str, int]

    def index(self, subject: str) -> int:
        try:
            return self.subject_to_index[subject]
        except KeyError as error:
            raise KeyError(f"Unknown subject: {subject!r}") from error

    def to_metadata(self) -> dict[str, object]:
        return {
            "ordering_rule": "ascending Unicode code-point order",
            "subjects": list(self.subjects),
            "subject_to_index": dict(self.subject_to_index),
        }


def build_subject_index(
    records: Iterable[EEGManifestRecord],
) -> SubjectIndex:
    subjects = tuple(sorted({record.subject for record in records}))
    if not subjects:
        raise ValueError("Cannot build a subject index from no EEG records")
    return SubjectIndex(
        subjects=subjects,
        subject_to_index={
            subject: index for index, subject in enumerate(subjects)
        },
    )
