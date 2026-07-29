from __future__ import annotations

import csv
import json
import unicodedata
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from .protocol_assets import file_sha256


WORD_OCCURRENCE_FIELDS = (
    "word_occurrence_id",
    "text_embedding_idx",
    "word_position",
    "surface_form",
    "char_start",
    "char_end",
    "keyword_id",
    "canonical_concept",
    "semantic_categories",
    "story_local_flag",
    "include_core",
    "include_main",
    "include_extended",
)


@dataclass(frozen=True)
class WordOccurrence:
    word_occurrence_id: str
    text_embedding_idx: int
    word_position: int
    surface_form: str
    char_start: int
    char_end: int
    keyword_id: str
    canonical_concept: str
    semantic_categories: str
    story_local_flag: str
    include_core: str
    include_main: str
    include_extended: str


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\u00a0", " ").strip()


def normalized_text_sha256(text: str) -> str:
    return sha256(normalize_text(text).encode("utf-8")).hexdigest()


def align_segmented_words(
    text: str,
    segmented_words: list[str],
) -> list[tuple[int, int]]:
    """Align the frozen v1 word sequence to normalized sentence characters.

    The v1 segmentation is the source of truth. Sequential exact matching
    preserves repeated words and deterministic decompositions such as
    ``我的 -> 我 | 的`` while skipping punctuation and excluded numerals.
    Offsets use Python Unicode code-point indices, matching fast-tokenizer
    offset mappings for the Chinese BMP text in this corpus.
    """

    normalized = normalize_text(text)
    cursor = 0
    spans: list[tuple[int, int]] = []
    for position, word in enumerate(segmented_words):
        start = normalized.find(word, cursor)
        if start < 0:
            raise ValueError(
                f"Cannot align word {word!r} at position {position} in "
                f"{normalized!r} after character {cursor}"
            )
        stop = start + len(word)
        spans.append((start, stop))
        cursor = stop
    return spans


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def build_word_occurrences(
    lexicon_path: str | Path,
    sentence_labels_path: str | Path,
) -> tuple[list[WordOccurrence], list[int]]:
    lexicon_file = Path(lexicon_path)
    labels_file = Path(sentence_labels_path)
    lexicon_rows = _read_csv(lexicon_file)
    sentence_rows = _read_csv(labels_file)
    lexicon_by_surface = {row["surface_form"]: row for row in lexicon_rows}

    occurrences: list[WordOccurrence] = []
    excluded_sentence_indices: list[int] = []
    for sentence in sentence_rows:
        text_embedding_idx = int(sentence["text_embedding_idx"])
        words = [word for word in sentence["segmented_words"].split("|") if word]
        declared_count = int(sentence["token_count"])
        if len(words) != declared_count:
            raise ValueError(
                f"token_count mismatch for text_embedding_idx={text_embedding_idx}: "
                f"{declared_count} != {len(words)}"
            )

        is_heading = sentence["is_chapter_heading"] == "true"
        if is_heading or not words:
            excluded_sentence_indices.append(text_embedding_idx)
            continue

        spans = align_segmented_words(sentence["text"], words)
        for position, (word, (char_start, char_end)) in enumerate(
            zip(words, spans, strict=True)
        ):
            keyword = lexicon_by_surface.get(word)
            occurrences.append(
                WordOccurrence(
                    word_occurrence_id=(
                        f"lp:{text_embedding_idx}:word:{position}"
                    ),
                    text_embedding_idx=text_embedding_idx,
                    word_position=position,
                    surface_form=word,
                    char_start=char_start,
                    char_end=char_end,
                    keyword_id=keyword["keyword_id"] if keyword else "",
                    canonical_concept=(
                        keyword["canonical_concept"] if keyword else word
                    ),
                    semantic_categories=(
                        keyword["semantic_categories"] if keyword else ""
                    ),
                    story_local_flag=(
                        keyword["story_local_flag"] if keyword else ""
                    ),
                    include_core=keyword["include_core"] if keyword else "false",
                    include_main=keyword["include_main"] if keyword else "false",
                    include_extended=(
                        keyword["include_extended"] if keyword else "false"
                    ),
                )
            )
    return occurrences, excluded_sentence_indices


def write_word_occurrence_artifacts(
    *,
    lexicon_path: str | Path,
    sentence_labels_path: str | Path,
    output_path: str | Path,
    provenance_path: str | Path,
) -> dict[str, object]:
    lexicon_file = Path(lexicon_path)
    labels_file = Path(sentence_labels_path)
    output_file = Path(output_path)
    provenance_file = Path(provenance_path)
    occurrences, excluded = build_word_occurrences(lexicon_file, labels_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WORD_OCCURRENCE_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(occurrence) for occurrence in occurrences)

    sentence_count = len({row.text_embedding_idx for row in occurrences})
    summary: dict[str, object] = {
        "schema_version": "littleprince_word_occurrences_v1",
        "source_lexicon_sha256": file_sha256(lexicon_file),
        "source_sentence_labels_sha256": file_sha256(labels_file),
        "output_sha256": file_sha256(output_file),
        "word_occurrences": len(occurrences),
        "valid_sentences": sentence_count,
        "excluded_sentence_count": len(excluded),
        "excluded_text_embedding_indices": excluded,
        "offset_unit": "normalized Unicode code-point index; end-exclusive",
    }
    provenance_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary

