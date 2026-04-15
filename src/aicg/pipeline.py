from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_supported_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists() or not images_dir.is_dir():
        return []
    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    return sorted(images)


def preflight_validation(
    images_dir: Path,
    captions_file: Path,
    train_images_file: Path,
    features_path: Path,
    model_out: Path,
    tokenizer_out: Path,
    max_length_out: Path,
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []

    if not images_dir.exists() or not images_dir.is_dir():
        issues.append(f"Images directory missing: {images_dir}")

    image_count = len(list_supported_images(images_dir)) if images_dir.exists() else 0
    if image_count == 0:
        issues.append(f"No supported images found in: {images_dir}")

    if not captions_file.exists():
        issues.append(f"Captions file missing: {captions_file}")

    if not train_images_file.exists():
        issues.append(f"Train image list missing: {train_images_file}")

    train_image_count = 0
    train_overlap_count = 0
    if train_images_file.exists() and images_dir.exists() and images_dir.is_dir():
        train_entries = [line.strip() for line in train_images_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        train_image_count = len(train_entries)

        train_names = {Path(item).name for item in train_entries}
        available_names = {p.name for p in list_supported_images(images_dir)}
        train_overlap_count = len(train_names & available_names)

        if train_image_count == 0:
            issues.append(f"Train image list is empty: {train_images_file}")
        elif train_overlap_count == 0:
            issues.append(
                "No overlap between train image list and available images in the images directory. "
                "Check that filenames and paths match."
            )

    for out_path in (features_path, model_out, tokenizer_out, max_length_out):
        parent = out_path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Cannot create output directory {parent}: {exc}")
            continue
        if not parent.is_dir():
            issues.append(f"Output parent is not a directory: {parent}")
            continue
        if not os.access(parent, os.W_OK):
            issues.append(f"Output directory is not writable: {parent}")

    details = {
        "image_count": image_count,
        "train_image_count": train_image_count,
        "train_overlap_count": train_overlap_count,
        "paths": {
            "images_dir": str(images_dir),
            "captions_file": str(captions_file),
            "train_images_file": str(train_images_file),
            "features_path": str(features_path),
            "model_out": str(model_out),
            "tokenizer_out": str(tokenizer_out),
            "max_length_out": str(max_length_out),
        },
    }
    return issues, details


def _file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"


def dataset_fingerprint(
    images_dir: Path,
    captions_file: Path,
    train_images_file: Path,
    image_size: int,
    epochs: int,
    batch_size: int,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(int(image_size)).encode("utf-8"))
    hasher.update(str(int(epochs)).encode("utf-8"))
    hasher.update(str(int(batch_size)).encode("utf-8"))

    for file_path in [captions_file, train_images_file]:
        hasher.update(_file_signature(file_path).encode("utf-8"))

    for image_path in list_supported_images(images_dir):
        hasher.update(_file_signature(image_path).encode("utf-8"))

    return hasher.hexdigest()


def read_cache_metadata(cache_file: Path) -> dict[str, Any] | None:
    if not cache_file.exists():
        return None
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_cache_metadata(cache_file: Path, payload: dict[str, Any]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def cache_hit(cache_file: Path, fingerprint: str, required_outputs: list[Path]) -> bool:
    metadata = read_cache_metadata(cache_file)
    if metadata is None:
        return False
    if metadata.get("fingerprint") != fingerprint:
        return False
    return all(path.exists() for path in required_outputs)
