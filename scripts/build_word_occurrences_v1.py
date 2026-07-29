from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_keyword_decoding.data import write_word_occurrence_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen Little Prince v1 word-occurrence table."
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "protocols" / "littleprince_hf_v1",
    )
    args = parser.parse_args()
    protocol_dir = args.protocol_dir
    summary = write_word_occurrence_artifacts(
        lexicon_path=protocol_dir / "littleprince_hf_lexicon_v1.csv",
        sentence_labels_path=(
            protocol_dir / "littleprince_sentence_keyword_labels_v1.csv"
        ),
        output_path=protocol_dir / "littleprince_word_occurrences_v1.csv",
        provenance_path=protocol_dir / "WORD_OCCURRENCES_PROVENANCE.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

