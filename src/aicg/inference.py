from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from aicg.utils.io import load_pickle, load_text


@lru_cache(maxsize=1)
def _build_feature_extractor() -> tf.keras.Model:
    base = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    return base


@lru_cache(maxsize=8)
def _load_captioning_artifacts(
    model_path: str,
    tokenizer_path: str,
    max_length_path: str,
) -> tuple[tf.keras.Model, object, int]:
    model = tf.keras.models.load_model(Path(model_path))
    tokenizer = load_pickle(Path(tokenizer_path))
    max_length = int(load_text(Path(max_length_path)))
    return model, tokenizer, max_length


def extract_single_image_feature(image_path: Path, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    model = _build_feature_extractor()
    image = Image.open(image_path).convert("RGB").resize(target_size)
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = keras.applications.mobilenet_v2.preprocess_input(arr)
    feat = model.predict(arr, verbose=0)[0]
    return feat.astype(np.float32)


def _id_to_word(index: int, tokenizer) -> str | None:
    return tokenizer.index_word.get(index)


def generate_caption(
    image_feature: np.ndarray,
    model_path: Path,
    tokenizer_path: Path,
    max_length_path: Path,
) -> str:
    model, tokenizer, max_length = _load_captioning_artifacts(
        str(model_path.resolve()),
        str(tokenizer_path.resolve()),
        str(max_length_path.resolve()),
    )

    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = keras.preprocessing.sequence.pad_sequences([seq], maxlen=max_length, padding="post")

        yhat = model.predict([
            np.expand_dims(image_feature, axis=0),
            seq,
        ], verbose=0)

        next_index = int(np.argmax(yhat, axis=-1)[0])
        word = _id_to_word(next_index, tokenizer)
        if word is None:
            break

        in_text += f" {word}"
        if word == "endseq":
            break

    words = [w for w in in_text.split() if w not in {"startseq", "endseq"}]
    return " ".join(words).strip()
