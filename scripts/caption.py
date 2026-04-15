#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from aicg.inference import extract_single_image_feature, generate_caption


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate caption for an image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/caption_model.keras"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.pkl"))
    parser.add_argument("--max-length", type=Path, default=Path("artifacts/max_length.txt"))
    args = parser.parse_args()

    feature = extract_single_image_feature(args.image)
    caption = generate_caption(
        image_feature=feature,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        max_length_path=args.max_length,
    )

    print(caption if caption else "[empty caption]")


if __name__ == "__main__":
    main()
