from __future__ import annotations

from pathlib import Path

import numpy as np

from aicg import inference


class DummyTokenizer:
    def __init__(self) -> None:
        self.word_index = {"startseq": 1, "cat": 2, "sits": 3, "endseq": 4}
        self.index_word = {v: k for k, v in self.word_index.items()}

    def texts_to_sequences(self, texts: list[str]) -> list[list[int]]:
        seqs: list[list[int]] = []
        for text in texts:
            seqs.append([self.word_index.get(token, 0) for token in text.split()])
        return seqs


class DummyModel:
    def predict(self, inputs, verbose=0):  # noqa: ANN001
        _img, seq = inputs
        seq_vals = [int(v) for v in seq[0] if int(v) != 0]
        if not seq_vals:
            probs = np.array([0.0, 0.9, 0.08, 0.01, 0.01], dtype=np.float32)
        elif seq_vals[-1] == 1:
            probs = np.array([0.0, 0.05, 0.8, 0.1, 0.05], dtype=np.float32)
        elif seq_vals[-1] == 2:
            probs = np.array([0.0, 0.05, 0.05, 0.8, 0.1], dtype=np.float32)
        else:
            probs = np.array([0.0, 0.05, 0.05, 0.1, 0.8], dtype=np.float32)
        return np.expand_dims(probs, axis=0)


def _patch_loader(monkeypatch):
    def fake_loader(model_path: str, tokenizer_path: str, max_length_path: str):  # noqa: ARG001
        return DummyModel(), DummyTokenizer(), 8

    monkeypatch.setattr(inference, "_load_captioning_artifacts", fake_loader)


def test_generate_caption_greedy(monkeypatch) -> None:
    _patch_loader(monkeypatch)
    caption = inference.generate_caption(
        image_feature=np.zeros((1280,), dtype=np.float32),
        model_path=Path("m.keras"),
        tokenizer_path=Path("t.pkl"),
        max_length_path=Path("ml.txt"),
        strategy="greedy",
    )
    assert "cat" in caption


def test_generate_caption_beam(monkeypatch) -> None:
    _patch_loader(monkeypatch)
    caption = inference.generate_caption(
        image_feature=np.zeros((1280,), dtype=np.float32),
        model_path=Path("m.keras"),
        tokenizer_path=Path("t.pkl"),
        max_length_path=Path("ml.txt"),
        strategy="beam",
        beam_width=3,
    )
    assert len(caption.split()) >= 1
