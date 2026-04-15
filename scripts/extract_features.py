#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image


def build_extractor() -> tf.keras.Model:
    return keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )


def image_to_array(path: Path, target_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(target_size)
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return keras.applications.mobilenet_v2.preprocess_input(arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CNN features for Flickr8k images.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Path to images directory")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz feature file")
    parser.add_argument("--image-size", type=int, default=224, help="Square image resize")
    args = parser.parse_args()

    model = build_extractor()
    image_files = sorted([p for p in args.images_dir.glob("*.jpg")])

    if not image_files:
        raise FileNotFoundError(f"No .jpg images found in {args.images_dir}")

    feature_map: dict[str, np.ndarray] = {}

    for idx, image_path in enumerate(image_files, start=1):
        arr = image_to_array(image_path, (args.image_size, args.image_size))
        feature = model.predict(arr, verbose=0)[0].astype(np.float32)
        feature_map[image_path.name] = feature

        if idx % 500 == 0:
            print(f"Processed {idx}/{len(image_files)} images")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **feature_map)
    print(f"Saved features to {args.output}")


if __name__ == "__main__":
    main()
