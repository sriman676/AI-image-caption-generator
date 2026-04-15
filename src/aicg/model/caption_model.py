from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


def build_caption_model(vocab_size: int, max_length: int, feature_dim: int) -> tf.keras.Model:
    """CNN feature vector + LSTM decoder captioning model."""
    image_input = keras.layers.Input(shape=(feature_dim,), name="image_features")
    image_branch = keras.layers.Dropout(0.5)(image_input)
    image_branch = keras.layers.Dense(256, activation="relu")(image_branch)

    text_input = keras.layers.Input(shape=(max_length,), name="text_sequence")
    text_branch = keras.layers.Embedding(vocab_size, 256, mask_zero=True)(text_input)
    text_branch = keras.layers.Dropout(0.5)(text_branch)
    text_branch = keras.layers.LSTM(256)(text_branch)

    merged = keras.layers.Add()([image_branch, text_branch])
    merged = keras.layers.Dense(256, activation="relu")(merged)
    output = keras.layers.Dense(vocab_size, activation="softmax")(merged)

    model = keras.Model(inputs=[image_input, text_input], outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model
