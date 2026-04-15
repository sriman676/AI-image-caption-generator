from __future__ import annotations

from pathlib import Path

from aicg.pipeline import cache_hit, dataset_fingerprint, preflight_validation, write_cache_metadata


def test_preflight_validation_passes_on_sample_set() -> None:
    root = Path("web/sample_sets/basic_set")
    issues, details = preflight_validation(
        images_dir=root / "images",
        captions_file=root / "Flickr8k.token.txt",
        train_images_file=root / "Flickr_8k.trainImages.txt",
        features_path=Path("artifacts/features_test.npz"),
        model_out=Path("artifacts/model_test.keras"),
        tokenizer_out=Path("artifacts/tokenizer_test.pkl"),
        max_length_out=Path("artifacts/max_length_test.txt"),
    )

    assert issues == []
    assert details["image_count"] > 0


def test_dataset_fingerprint_changes_with_training_params() -> None:
    root = Path("web/sample_sets/basic_set")
    fp1 = dataset_fingerprint(
        images_dir=root / "images",
        captions_file=root / "Flickr8k.token.txt",
        train_images_file=root / "Flickr_8k.trainImages.txt",
        image_size=224,
        epochs=1,
        batch_size=2,
    )
    fp2 = dataset_fingerprint(
        images_dir=root / "images",
        captions_file=root / "Flickr8k.token.txt",
        train_images_file=root / "Flickr_8k.trainImages.txt",
        image_size=224,
        epochs=2,
        batch_size=2,
    )

    assert fp1 != fp2


def test_cache_hit_checks_metadata_and_outputs(tmp_path: Path) -> None:
    cache_file = tmp_path / "cache.json"
    out1 = tmp_path / "out1.bin"
    out2 = tmp_path / "out2.bin"
    out1.write_text("a", encoding="utf-8")
    out2.write_text("b", encoding="utf-8")

    write_cache_metadata(cache_file, {"fingerprint": "abc"})

    assert cache_hit(cache_file, "abc", [out1, out2])
    assert not cache_hit(cache_file, "xyz", [out1, out2])

    out2.unlink()
    assert not cache_hit(cache_file, "abc", [out1, out2])


def test_preflight_reports_no_overlap_between_train_list_and_images(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "img_a.jpg").write_bytes(b"x")

    captions_file = tmp_path / "caps.txt"
    captions_file.write_text("img_a.jpg#0 startseq test endseq\n", encoding="utf-8")

    train_images_file = tmp_path / "train.txt"
    train_images_file.write_text("other_name.jpg\n", encoding="utf-8")

    issues, details = preflight_validation(
        images_dir=images_dir,
        captions_file=captions_file,
        train_images_file=train_images_file,
        features_path=tmp_path / "artifacts" / "features.npz",
        model_out=tmp_path / "artifacts" / "model.keras",
        tokenizer_out=tmp_path / "artifacts" / "tokenizer.pkl",
        max_length_out=tmp_path / "artifacts" / "max_length.txt",
    )

    assert any("No overlap between train image list" in issue for issue in issues)
    assert details["train_image_count"] == 1
    assert details["train_overlap_count"] == 0


def test_preflight_creates_missing_output_directory(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "img_a.jpg").write_bytes(b"x")

    captions_file = tmp_path / "caps.txt"
    captions_file.write_text("img_a.jpg#0 startseq test endseq\n", encoding="utf-8")

    train_images_file = tmp_path / "train.txt"
    train_images_file.write_text("img_a.jpg\n", encoding="utf-8")

    out_dir = tmp_path / "new_outputs"
    assert not out_dir.exists()

    issues, _ = preflight_validation(
        images_dir=images_dir,
        captions_file=captions_file,
        train_images_file=train_images_file,
        features_path=out_dir / "features.npz",
        model_out=out_dir / "model.keras",
        tokenizer_out=out_dir / "tokenizer.pkl",
        max_length_out=out_dir / "max_length.txt",
    )

    assert out_dir.exists()
    assert issues == []
