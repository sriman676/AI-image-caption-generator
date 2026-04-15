from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, path: Path) -> None:
    ensure_parent(path)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_text(value: str, path: Path) -> None:
    ensure_parent(path)
    path.write_text(value.strip() + "\n", encoding="utf-8")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()
