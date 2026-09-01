"""Small, dependency-light helpers for OPBench files and output paths."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    """Write JSON atomically so an interrupted run does not corrupt a result."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def slug(value: str) -> str:
    """Return a filesystem-safe label for models, methods, and experiment IDs."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unnamed"

