#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from aicg.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train image captioning model.")
    parser.add_argument("--captions-file", type=Path, required=True)
    parser.add_argument("--train-images-file", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/caption_model.keras"))
    parser.add_argument("--tokenizer-out", type=Path, default=Path("artifacts/tokenizer.pkl"))
    parser.add_argument("--max-length-out", type=Path, default=Path("artifacts/max_length.txt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    args = parser.parse_args()

    metrics = train_model(
        captions_file=args.captions_file,
        train_images_file=args.train_images_file,
        features_path=args.features,
        model_path=args.model_out,
        tokenizer_path=args.tokenizer_out,
        max_length_path=args.max_length_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
        resume_from_checkpoint=args.resume,
        early_stopping_patience=args.early_stopping_patience or None,
    )

    print(
        "Training metrics: "
        f"epochs_ran={metrics['epochs_ran']} "
        f"final_loss={metrics['final_loss']:.4f} "
        f"perplexity={metrics['perplexity']:.2f}"
    )


if __name__ == "__main__":
    main()
