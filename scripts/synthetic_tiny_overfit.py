from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_keyword_decoding.training.synthetic import run_synthetic_tiny_overfit


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6A synthetic tiny overfit.")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_synthetic_tiny_overfit(steps=args.steps, seed=args.seed)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
