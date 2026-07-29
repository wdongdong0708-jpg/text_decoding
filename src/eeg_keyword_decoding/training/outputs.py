from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def create_unique_run_directory(
    root: str | Path,
    *,
    backend: str,
    outer_fold: int,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    path = Path(root) / backend / f"outer_fold_{outer_fold}" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path
