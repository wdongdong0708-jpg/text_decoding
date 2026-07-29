from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, event: dict[str, Any]) -> None:
        encoded = json.dumps(event, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
