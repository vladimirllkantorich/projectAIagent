from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import PROJECT_ROOT


HISTORY_PATH = PROJECT_ROOT / "data" / "index_history.json"


def load_index_history(limit: int = 12) -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []

    try:
        records = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(records, list):
        return []

    return records[-limit:]


def add_index_history_record(
    source_type: str,
    source: str,
    chunks: int,
    status: str = "indexed",
    message: str = "",
) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = load_index_history(limit=500)
    records.append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_type": source_type,
            "source": source,
            "chunks": chunks,
            "status": status,
            "message": message,
        }
    )
    HISTORY_PATH.write_text(json.dumps(records[-500:], indent=2), encoding="utf-8")


def clear_index_history() -> None:
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
