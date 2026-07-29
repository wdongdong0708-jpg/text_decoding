from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_keyword_decoding.training import inspect_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate checkpoint schema, config, prototype and projector hashes "
            "without constructing any EEG Dataset."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    result = inspect_checkpoint(args.checkpoint)
    if result["integrity_validated"] is not True:
        raise RuntimeError("Checkpoint integrity validation failed")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
