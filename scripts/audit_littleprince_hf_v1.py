from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_keyword_decoding.data import audit_littleprince_hf_v1


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen Little Prince v1 assets.")
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "protocols" / "littleprince_hf_v1",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "manifests"
        / "littleprince_pl_all_clean_manifest.csv",
    )
    args = parser.parse_args()
    summary = audit_littleprince_hf_v1(args.protocol_dir, args.manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

