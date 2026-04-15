from __future__ import annotations

import re
import string
from collections import defaultdict
from pathlib import Path


def _clean_caption(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if len(w) > 1 and w.isalpha()]
    return " ".join(words)


def load_captions(captions_file: Path) -> dict[str, list[str]]:
    """Load Flickr8k token file into image_id -> cleaned captions mapping."""
    mapping: dict[str, list[str]] = defaultdict(list)
    with captions_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if "\t" in line:
                image_caption_id, caption = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                image_caption_id, caption = parts

            image_id = image_caption_id.split("#", 1)[0]
            cleaned = _clean_caption(caption)
            if not cleaned:
                continue
            mapping[image_id].append(f"startseq {cleaned} endseq")
    return dict(mapping)


def load_image_list(image_list_file: Path) -> set[str]:
    with image_list_file.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def filter_captions_by_images(
    captions: dict[str, list[str]],
    image_names: set[str],
) -> dict[str, list[str]]:
    return {img: captions[img] for img in image_names if img in captions}
