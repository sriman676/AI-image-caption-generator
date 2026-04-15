from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf
from tensorflow import keras

from aicg.data.flickr8k import filter_captions_by_images, load_captions, load_image_list
from aicg.model.caption_model import build_caption_model
from aicg.utils.io import save_pickle, save_text


def build_tokenizer(
    captions_map: dict[str, list[str]],
    num_words: int | None = None,
) -> keras.preprocessing.text.Tokenizer:
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="<unk>")
    all_captions = [caption for caps in captions_map.values() for caption in caps]
    tokenizer.fit_on_texts(all_captions)
    return tokenizer


def max_caption_length(captions_map: dict[str, list[str]]) -> int:
    return max(len(caption.split()) for caps in captions_map.values() for caption in caps)


def _pair_count(captions_map: dict[str, list[str]], tokenizer: keras.preprocessing.text.Tokenizer) -> int:
    total = 0
    for caps in captions_map.values():
        for caption in caps:
            seq = tokenizer.texts_to_sequences([caption])[0]
            total += max(0, len(seq) - 1)
    return total


def _sequence_generator(
    captions_map: dict[str, list[str]],
    features: dict[str, np.ndarray],
    tokenizer: keras.preprocessing.text.Tokenizer,
    max_length: int,
    batch_size: int,
) -> Iterator[tuple[tuple[np.ndarray, np.ndarray], np.ndarray]]:
    image_batch: list[np.ndarray] = []
    input_seq_batch: list[np.ndarray] = []
    target_batch: list[int] = []

    while True:
        for image_id, captions in captions_map.items():
            if image_id not in features:
                continue
            feature_vec = features[image_id]

            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_word = seq[i]

                    in_seq = keras.preprocessing.sequence.pad_sequences(
                        [in_seq], maxlen=max_length, padding="post"
                    )[0]

                    image_batch.append(feature_vec)
                    input_seq_batch.append(in_seq)
                    target_batch.append(out_word)

                    if len(image_batch) == batch_size:
                        yield (
                            np.array(image_batch, dtype=np.float32),
                            np.array(input_seq_batch, dtype=np.int32),
                        ), np.array(target_batch, dtype=np.int32)
                        image_batch, input_seq_batch, target_batch = [], [], []

        if image_batch:
            yield (
                np.array(image_batch, dtype=np.float32),
                np.array(input_seq_batch, dtype=np.int32),
            ), np.array(target_batch, dtype=np.int32)
            image_batch, input_seq_batch, target_batch = [], [], []


def train_model(
    captions_file: Path,
    train_images_file: Path,
    features_path: Path,
    model_path: Path,
    tokenizer_path: Path,
    max_length_path: Path,
    epochs: int = 20,
    batch_size: int = 64,
) -> None:
    captions = load_captions(captions_file)
    train_images = load_image_list(train_images_file)
    train_captions = filter_captions_by_images(captions, train_images)

    if not train_captions:
        raise ValueError("No matching captions found for the provided training image list.")

    loaded = np.load(features_path)
    features = {k: loaded[k] for k in loaded.files}

    if not features:
        raise ValueError("Feature file is empty. Run feature extraction before training.")

    # Train only on images that exist in both caption mapping and extracted features.
    available_ids = set(features.keys())
    train_captions = {img: caps for img, caps in train_captions.items() if img in available_ids}

    if not train_captions:
        raise ValueError(
            "No overlap between training captions and extracted image features. "
            "Check --train-images-file and --features inputs."
        )

    tokenizer = build_tokenizer(train_captions)
    max_length = max_caption_length(train_captions)

    vocab_size = len(tokenizer.word_index) + 1
    sample_feature = next(iter(features.values()))
    feature_dim = int(sample_feature.shape[-1])

    model = build_caption_model(vocab_size=vocab_size, max_length=max_length, feature_dim=feature_dim)

    pairs = _pair_count(train_captions, tokenizer)
    steps_per_epoch = max(1, math.ceil(pairs / batch_size))

    generator = _sequence_generator(train_captions, features, tokenizer, max_length, batch_size)

    model.fit(generator, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    save_pickle(tokenizer, tokenizer_path)
    save_text(str(max_length), max_length_path)
