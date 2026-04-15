from __future__ import annotations

import queue
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

from aicg.inference import extract_single_image_feature, generate_caption
from aicg.pipeline import cache_hit, dataset_fingerprint, preflight_validation, write_cache_metadata
from aicg.training import train_model


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SAMPLE_SETS_DIR = Path(__file__).resolve().parent / "sample_sets"
_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _available_sample_sets() -> list[str]:
    if not SAMPLE_SETS_DIR.exists():
        return []
    return sorted([p.name for p in SAMPLE_SETS_DIR.iterdir() if p.is_dir()])


def _sample_set_paths(sample_set_name: str) -> dict[str, str]:
    root = SAMPLE_SETS_DIR / sample_set_name
    return {
        "images_dir": str(root / "images"),
        "captions_file": str(root / "Flickr8k.token.txt"),
        "train_images_file": str(root / "Flickr_8k.trainImages.txt"),
        "features_path": f"artifacts/features_{sample_set_name}.npz",
        "model_out": f"artifacts/caption_model_{sample_set_name}.keras",
        "tokenizer_out": f"artifacts/tokenizer_{sample_set_name}.pkl",
        "max_length_out": f"artifacts/max_length_{sample_set_name}.txt",
        "checkpoint_path": f"artifacts/checkpoint_{sample_set_name}.keras",
    }


def _ready(path_value: str) -> bool:
    return Path(path_value).exists()


def _apply_sample_set(sample_set_name: str) -> None:
    paths = _sample_set_paths(sample_set_name)
    for key, value in paths.items():
        st.session_state[key] = value
    st.session_state["model_path"] = paths["model_out"]
    st.session_state["tokenizer_path"] = paths["tokenizer_out"]
    st.session_state["max_length_path"] = paths["max_length_out"]


def _build_runtime_config() -> dict[str, Any]:
    return {
        "images_dir": st.session_state["images_dir"],
        "captions_file": st.session_state["captions_file"],
        "train_images_file": st.session_state["train_images_file"],
        "features_path": st.session_state["features_path"],
        "model_out": st.session_state["model_out"],
        "tokenizer_out": st.session_state["tokenizer_out"],
        "max_length_out": st.session_state["max_length_out"],
        "checkpoint_path": st.session_state["checkpoint_path"],
        "image_size": int(st.session_state["image_size"]),
        "epochs": int(st.session_state["train_epochs"]),
        "batch_size": int(st.session_state["train_batch_size"]),
        "resume_from_checkpoint": bool(st.session_state["resume_from_checkpoint"]),
        "early_stopping_patience": int(st.session_state["early_stopping_patience"]),
    }


@lru_cache(maxsize=1)
def _extractor_model():
    return keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )


def extract_features_from_dir(
    images_dir: Path,
    output_path: Path,
    image_size: int = 224,
    progress_callback: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    model = _extractor_model()
    patterns = ("*.jpg", "*.jpeg", "*.png")
    image_files = sorted({p for pattern in patterns for p in images_dir.glob(pattern)})
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}. Supported: .jpg, .jpeg, .png")

    features: dict[str, np.ndarray] = {}
    total = len(image_files)
    for idx, image_path in enumerate(image_files, start=1):
        if should_stop is not None and should_stop():
            raise RuntimeError("Pipeline cancelled by user.")
        arr = np.array(Image.open(image_path).convert("RGB").resize((image_size, image_size)), dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        arr = keras.applications.mobilenet_v2.preprocess_input(arr)
        feat = model.predict(arr, verbose=0)[0].astype(np.float32)
        features[image_path.name] = feat
        if progress_callback is not None and (idx % 10 == 0 or idx == total):
            progress_callback(f"Extracted {idx}/{total} images")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **features)
    return len(features)


def _first_image_in_dir(images_dir: Path) -> Path:
    patterns = ("*.jpg", "*.jpeg", "*.png")
    image_files = sorted({p for pattern in patterns for p in images_dir.glob(pattern)})
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}. Supported: .jpg, .jpeg, .png")
    return image_files[0]


def _generate_caption_for_image(
    image_path: Path,
    model_path: Path,
    tokenizer_path: Path,
    max_length_path: Path,
    strategy: str,
    beam_width: int,
    temperature: float,
    top_k: int,
) -> str:
    feature = extract_single_image_feature(image_path)
    return generate_caption(
        image_feature=feature,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        max_length_path=max_length_path,
        strategy=strategy,
        beam_width=beam_width,
        temperature=temperature,
        top_k=top_k,
    )


def _cleanup_temp_preview_file() -> None:
    temp_file = st.session_state.get("pending_preview_temp_file")
    if not temp_file:
        return
    try:
        Path(str(temp_file)).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
    finally:
        st.session_state["pending_preview_temp_file"] = None


def _drain_job_log_queue() -> None:
    log_queue = st.session_state["job_log_queue"]
    while True:
        try:
            msg = log_queue.get_nowait()
            st.session_state["job_logs"].append(msg)
        except queue.Empty:
            break


def _pipeline_worker(
    config: dict[str, Any],
    preview_image: str | None,
    run_rounds: int,
    decode_options: dict[str, Any],
    cancel_event: threading.Event,
    log_queue: queue.Queue,
) -> dict[str, Any]:
    def log(msg: str) -> None:
        log_queue.put(msg)

    images_dir = Path(config["images_dir"])
    captions_file = Path(config["captions_file"])
    train_images_file = Path(config["train_images_file"])
    features_path = Path(config["features_path"])
    model_out = Path(config["model_out"])
    tokenizer_out = Path(config["tokenizer_out"])
    max_length_out = Path(config["max_length_out"])
    checkpoint_path = Path(config["checkpoint_path"])

    issues, details = preflight_validation(
        images_dir=images_dir,
        captions_file=captions_file,
        train_images_file=train_images_file,
        features_path=features_path,
        model_out=model_out,
        tokenizer_out=tokenizer_out,
        max_length_out=max_length_out,
    )
    if issues:
        raise ValueError("Preflight failed:\n- " + "\n- ".join(issues))

    log(f"Preflight passed with {details['image_count']} images.")

    metrics_list: list[dict[str, float | int]] = []
    for round_idx in range(1, max(1, run_rounds) + 1):
        if cancel_event.is_set():
            raise RuntimeError("Pipeline cancelled by user.")

        log(f"Round {round_idx}/{run_rounds}: computing fingerprint...")
        fingerprint = dataset_fingerprint(
            images_dir=images_dir,
            captions_file=captions_file,
            train_images_file=train_images_file,
            image_size=int(config["image_size"]),
            epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"]),
        )
        cache_file = model_out.with_suffix(".cache.json")
        skip = cache_hit(
            cache_file=cache_file,
            fingerprint=fingerprint,
            required_outputs=[features_path, model_out, tokenizer_out, max_length_out],
        )

        if skip:
            log(f"Round {round_idx}/{run_rounds}: cache hit, skipping extraction and training.")
            metrics_list.append({"pairs": 0, "steps_per_epoch": 0, "epochs_ran": 0, "final_loss": 0.0, "perplexity": 0.0})
            continue

        log(f"Round {round_idx}/{run_rounds}: extracting features...")
        image_count = extract_features_from_dir(
            images_dir=images_dir,
            output_path=features_path,
            image_size=int(config["image_size"]),
            progress_callback=log,
            should_stop=cancel_event.is_set,
        )
        log(f"Round {round_idx}/{run_rounds}: extracted {image_count} images.")

        log(f"Round {round_idx}/{run_rounds}: training model...")

        def training_progress(epoch: int, loss: float) -> None:
            log(f"Round {round_idx}/{run_rounds}: epoch {epoch} loss={loss:.4f}")

        metrics = train_model(
            captions_file=captions_file,
            train_images_file=train_images_file,
            features_path=features_path,
            model_path=model_out,
            tokenizer_path=tokenizer_out,
            max_length_path=max_length_out,
            epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"]),
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=bool(config["resume_from_checkpoint"]),
            early_stopping_patience=int(config["early_stopping_patience"]) or None,
            progress_callback=training_progress,
            should_stop=cancel_event.is_set,
        )
        metrics_list.append(metrics)
        log(
            f"Round {round_idx}/{run_rounds}: training done "
            f"(epochs={metrics['epochs_ran']}, loss={float(metrics['final_loss']):.4f}, ppl={float(metrics['perplexity']):.2f})."
        )

        write_cache_metadata(
            cache_file,
            {
                "fingerprint": fingerprint,
                "image_count": image_count,
                "epochs": int(config["epochs"]),
                "batch_size": int(config["batch_size"]),
            },
        )

    preview_path = Path(preview_image) if preview_image else _first_image_in_dir(images_dir)
    log("Generating final caption preview...")
    caption = _generate_caption_for_image(
        image_path=preview_path,
        model_path=model_out,
        tokenizer_path=tokenizer_out,
        max_length_path=max_length_out,
        strategy=str(decode_options["strategy"]),
        beam_width=int(decode_options["beam_width"]),
        temperature=float(decode_options["temperature"]),
        top_k=int(decode_options["top_k"]),
    )

    return {
        "preview_image": str(preview_path),
        "preview_caption": caption,
        "model_path": str(model_out),
        "tokenizer_path": str(tokenizer_out),
        "max_length_path": str(max_length_out),
        "metrics": metrics_list,
    }


def _start_pipeline_job(preview_image: str | None = None) -> None:
    if st.session_state["job_future"] is not None and not st.session_state["job_future"].done():
        st.warning("A job is already running.")
        return

    st.session_state["job_logs"] = []
    st.session_state["job_error"] = None
    st.session_state["job_result"] = None
    st.session_state["job_cancel_event"] = threading.Event()

    config = _build_runtime_config()
    decode = {
        "strategy": st.session_state["decode_strategy"],
        "beam_width": st.session_state["beam_width"],
        "temperature": st.session_state["temperature"],
        "top_k": st.session_state["top_k"],
    }
    rounds = int(st.session_state["run_rounds"])
    st.session_state["job_future"] = _EXECUTOR.submit(
        _pipeline_worker,
        config,
        preview_image,
        rounds,
        decode,
        st.session_state["job_cancel_event"],
        st.session_state["job_log_queue"],
    )


def _render_job_status() -> None:
    _drain_job_log_queue()
    future: Future | None = st.session_state.get("job_future")
    if future is None:
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("Background Job")

    if not future.done():
        st.sidebar.info("Job running...")
        if st.sidebar.button("Cancel Running Job", use_container_width=True):
            st.session_state["job_cancel_event"].set()
        st.sidebar.caption("Click any control to refresh status.")
        return

    if st.session_state["job_result"] is None and st.session_state["job_error"] is None:
        try:
            st.session_state["job_result"] = future.result()
        except Exception as exc:  # noqa: BLE001
            st.session_state["job_error"] = str(exc)
        finally:
            _cleanup_temp_preview_file()

    if st.session_state["job_error"]:
        st.sidebar.error(st.session_state["job_error"])
        return

    result = st.session_state["job_result"]
    if result:
        st.session_state["auto_preview_image"] = result["preview_image"]
        st.session_state["auto_preview_caption"] = result["preview_caption"]
        st.session_state["model_path"] = result["model_path"]
        st.session_state["tokenizer_path"] = result["tokenizer_path"]
        st.session_state["max_length_path"] = result["max_length_path"]
        st.session_state["latest_metrics"] = result["metrics"]
        st.sidebar.success("Job completed.")


def _render_header() -> None:
    st.title("AI Image Caption Generator")
    st.caption("One-click workflow: Validate -> Extract -> Train -> Preview Caption")


def _render_feature_checklist() -> None:
    st.info(
        "Features available: 1) Preflight validation  2) Automatic extraction+training  "
        "3) Caption generation (greedy/sample/beam)  4) Background run with cancel/resume"
    )


def _render_beginner_wizard() -> None:
    st.markdown("### Beginner Wizard")
    c1, c2, c3 = st.columns(3)

    features_ok = _ready(st.session_state.get("features_path", ""))
    model_ok = _ready(st.session_state.get("model_path", ""))
    tokenizer_ok = _ready(st.session_state.get("tokenizer_path", ""))
    max_length_ok = _ready(st.session_state.get("max_length_path", ""))

    with c1:
        st.metric("Step 1: Extract", "Ready" if features_ok else "Pending")
    with c2:
        st.metric("Step 2: Train", "Ready" if model_ok and tokenizer_ok and max_length_ok else "Pending")
    with c3:
        st.metric("Step 3: Caption", "Ready" if model_ok and tokenizer_ok and max_length_ok else "Pending")


def _render_preflight_check() -> None:
    issues, details = preflight_validation(
        images_dir=Path(st.session_state["images_dir"]),
        captions_file=Path(st.session_state["captions_file"]),
        train_images_file=Path(st.session_state["train_images_file"]),
        features_path=Path(st.session_state["features_path"]),
        model_out=Path(st.session_state["model_out"]),
        tokenizer_out=Path(st.session_state["tokenizer_out"]),
        max_length_out=Path(st.session_state["max_length_out"]),
    )
    st.markdown("### Preflight")
    st.write(f"Detected images: {details['image_count']}")
    if issues:
        st.error("Preflight checks failed:")
        for issue in issues:
            st.write(f"- {issue}")
    else:
        st.success("Preflight checks passed.")


def _extract_tab() -> None:
    st.subheader("1) Extract Features")

    c1, c2 = st.columns([2, 1])
    with c1:
        images_dir = st.text_input("Images directory", key="images_dir")
    with c2:
        image_size = st.selectbox("Image size", [160, 192, 224, 256], index=2, key="image_size")

    output_path = st.text_input("Feature output path", key="features_path")

    images_dir_path = Path(images_dir)
    if images_dir_path.exists():
        patterns = ("*.jpg", "*.jpeg", "*.png")
        count = len({p for pattern in patterns for p in images_dir_path.glob(pattern)})
        st.caption(f"Detected {count} supported images in this folder.")

    if st.button("Run Feature Extraction", use_container_width=True):
        with st.spinner("Extracting features..."):
            try:
                count = extract_features_from_dir(Path(images_dir), Path(output_path), int(image_size))
                st.success(f"Done. Extracted features for {count} images into {output_path}.")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))


def _train_tab() -> None:
    st.subheader("2) Train Model")
    st.text_input("Captions file", key="captions_file")
    st.text_input("Train image list", key="train_images_file")
    st.text_input("Features path", key="features_path")

    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Model output", key="model_out")
        st.text_input("Tokenizer output", key="tokenizer_out")
        st.text_input("Max length output", key="max_length_out")
        st.text_input("Checkpoint path", key="checkpoint_path")
    with c2:
        st.slider("Epochs", min_value=1, max_value=100, value=20, key="train_epochs")
        st.selectbox("Batch size", [16, 32, 64, 128], index=2, key="train_batch_size")
        st.checkbox("Resume from checkpoint", key="resume_from_checkpoint")
        st.slider("Early stopping patience", min_value=0, max_value=10, value=2, key="early_stopping_patience")

    if st.button("Run Training", use_container_width=True):
        with st.spinner("Training model..."):
            try:
                metrics = train_model(
                    captions_file=Path(st.session_state["captions_file"]),
                    train_images_file=Path(st.session_state["train_images_file"]),
                    features_path=Path(st.session_state["features_path"]),
                    model_path=Path(st.session_state["model_out"]),
                    tokenizer_path=Path(st.session_state["tokenizer_out"]),
                    max_length_path=Path(st.session_state["max_length_out"]),
                    epochs=int(st.session_state["train_epochs"]),
                    batch_size=int(st.session_state["train_batch_size"]),
                    checkpoint_path=Path(st.session_state["checkpoint_path"]),
                    resume_from_checkpoint=bool(st.session_state["resume_from_checkpoint"]),
                    early_stopping_patience=int(st.session_state["early_stopping_patience"]) or None,
                )
                st.success("Training completed and artifacts saved.")
                st.session_state["model_path"] = st.session_state["model_out"]
                st.session_state["tokenizer_path"] = st.session_state["tokenizer_out"]
                st.session_state["max_length_path"] = st.session_state["max_length_out"]
                st.session_state["latest_metrics"] = [metrics]
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    metrics = st.session_state.get("latest_metrics") or []
    if metrics:
        last = metrics[-1]
        st.markdown("### Quality Metrics")
        st.write(f"- epochs_ran: {last['epochs_ran']}")
        st.write(f"- final_loss: {float(last['final_loss']):.4f}")
        st.write(f"- perplexity: {float(last['perplexity']):.2f}")


def _render_caption_explanation(caption: str, model_path: Path, tokenizer_path: Path, max_length_path: Path) -> None:
    st.markdown("### Why this output?")
    if not caption.strip():
        st.info("Model returned an empty caption. This usually means training was too short or dataset is too small.")
        return

    words = caption.split()
    st.write(f"- Caption length: {len(words)} words")
    st.write(f"- Model used: {model_path}")
    st.write(f"- Tokenizer used: {tokenizer_path}")
    st.write(f"- Max length config: {max_length_path}")


def _caption_tab() -> None:
    st.subheader("3) Generate Caption")
    model_path = Path(st.text_input("Model path", key="model_path"))
    tokenizer_path = Path(st.text_input("Tokenizer path", key="tokenizer_path"))
    max_length_path = Path(st.text_input("Max length path", key="max_length_path"))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.selectbox("Decoding strategy", ["greedy", "sample", "beam"], key="decode_strategy")
    with c2:
        st.number_input("Beam width", min_value=1, max_value=10, value=3, step=1, key="beam_width")
    with c3:
        st.number_input("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="temperature")
    with c4:
        st.number_input("Top-k", min_value=0, max_value=50, value=0, step=1, key="top_k")

    auto_preview_image = st.session_state.get("auto_preview_image")
    auto_preview_caption = st.session_state.get("auto_preview_caption")
    if auto_preview_image and auto_preview_caption is not None:
        st.markdown("### Automatic Preview")
        st.image(auto_preview_image, caption="Auto preview image", use_container_width=True)
        st.success(f"Caption: {auto_preview_caption if auto_preview_caption else '[empty caption]'}")
        _render_caption_explanation(auto_preview_caption, model_path, tokenizer_path, max_length_path)

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)

    if st.button("One-Click Full Run (Upload -> Extract -> Train -> Caption)", use_container_width=True):
        preview_path: str | None = None
        _cleanup_temp_preview_file()
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                preview_path = tmp.name
            st.session_state["pending_preview_temp_file"] = preview_path
        _start_pipeline_job(preview_image=preview_path)
        st.info("Started full pipeline in background.")

    if uploaded is None:
        st.caption("Upload an image and use one-click run, or run auto setup from sidebar.")
        return

    if not (_ready(str(model_path)) and _ready(str(tokenizer_path)) and _ready(str(max_length_path))):
        st.warning("Model/tokenizer/max-length file missing. Run one-click pipeline first.")
        return

    upload_signature = (
        f"{uploaded.name}:{uploaded.size}:"
        f"{st.session_state['decode_strategy']}:{st.session_state['beam_width']}:"
        f"{st.session_state['temperature']}:{st.session_state['top_k']}"
    )
    if st.session_state.get("last_upload_signature") != upload_signature:
        temp_image_path: Path | None = None
        with st.spinner("Generating caption..."):
            try:
                suffix = Path(uploaded.name).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded.getbuffer())
                    temp_image_path = Path(tmp.name)

                caption = _generate_caption_for_image(
                    image_path=temp_image_path,
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    max_length_path=max_length_path,
                    strategy=str(st.session_state["decode_strategy"]),
                    beam_width=int(st.session_state["beam_width"]),
                    temperature=float(st.session_state["temperature"]),
                    top_k=int(st.session_state["top_k"]),
                )
                st.session_state["last_upload_signature"] = upload_signature
                st.session_state["last_upload_caption"] = caption
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return
            finally:
                if temp_image_path is not None:
                    try:
                        temp_image_path.unlink(missing_ok=True)
                    except Exception:  # noqa: BLE001
                        pass

    last_caption = st.session_state.get("last_upload_caption", "")
    st.success(f"Caption: {last_caption if last_caption else '[empty caption]'}")
    _render_caption_explanation(last_caption, model_path, tokenizer_path, max_length_path)


def _help_tab() -> None:
    st.subheader("Help")
    st.write("Use this page to understand the full automatic flow and controls.")

    for image_name, caption in [
        ("workflow_help.png", "End-to-end workflow"),
        ("quickstart_help.png", "Quick start guide"),
        ("extract_settings_help.png", "How to configure extraction"),
        ("output_change_help.png", "How changing settings changes output"),
        ("wizard_help.png", "Beginner wizard and sample set flow"),
        ("output_explain_help.png", "How output explanation is shown"),
    ]:
        image_path = ASSETS_DIR / image_name
        if image_path.exists():
            st.image(str(image_path), caption=caption, use_container_width=True)
        else:
            st.warning(f"Missing help image: {image_name}")


def main() -> None:
    st.set_page_config(page_title="AI Caption Generator", page_icon="AI", layout="wide")

    st.session_state.setdefault("images_dir", "data/Flickr8k_Dataset")
    st.session_state.setdefault("features_path", "artifacts/features.npz")
    st.session_state.setdefault("captions_file", "data/Flickr8k.token.txt")
    st.session_state.setdefault("train_images_file", "data/Flickr_8k.trainImages.txt")
    st.session_state.setdefault("model_out", "artifacts/caption_model.keras")
    st.session_state.setdefault("tokenizer_out", "artifacts/tokenizer.pkl")
    st.session_state.setdefault("max_length_out", "artifacts/max_length.txt")
    st.session_state.setdefault("checkpoint_path", "artifacts/caption_checkpoint.keras")
    st.session_state.setdefault("model_path", "artifacts/caption_model.keras")
    st.session_state.setdefault("tokenizer_path", "artifacts/tokenizer.pkl")
    st.session_state.setdefault("max_length_path", "artifacts/max_length.txt")
    st.session_state.setdefault("image_size", 224)
    st.session_state.setdefault("train_epochs", 20)
    st.session_state.setdefault("train_batch_size", 64)
    st.session_state.setdefault("resume_from_checkpoint", False)
    st.session_state.setdefault("early_stopping_patience", 2)
    st.session_state.setdefault("decode_strategy", "greedy")
    st.session_state.setdefault("beam_width", 3)
    st.session_state.setdefault("temperature", 1.0)
    st.session_state.setdefault("top_k", 0)
    st.session_state.setdefault("run_rounds", 1)
    st.session_state.setdefault("page", "Extract")
    st.session_state.setdefault("auto_preview_image", None)
    st.session_state.setdefault("auto_preview_caption", None)
    st.session_state.setdefault("last_upload_signature", None)
    st.session_state.setdefault("last_upload_caption", None)
    st.session_state.setdefault("job_future", None)
    st.session_state.setdefault("job_cancel_event", None)
    st.session_state.setdefault("job_log_queue", queue.Queue())
    st.session_state.setdefault("job_logs", [])
    st.session_state.setdefault("job_result", None)
    st.session_state.setdefault("job_error", None)
    st.session_state.setdefault("pending_preview_temp_file", None)
    st.session_state.setdefault("latest_metrics", [])

    _render_header()
    _render_feature_checklist()
    _render_beginner_wizard()
    _render_preflight_check()

    st.sidebar.header("Navigation")
    st.sidebar.radio("Go to", ["Extract", "Train", "Caption", "Help"], key="page")
    page = st.session_state["page"]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Auto Pipeline")
    st.sidebar.number_input("Rounds", min_value=1, max_value=10, value=1, step=1, key="run_rounds")
    if st.sidebar.button("Auto Run Extract + Train", use_container_width=True):
        _start_pipeline_job(preview_image=None)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Try Sample Test Set")
    sample_sets = _available_sample_sets()
    if sample_sets:
        sample_set = st.sidebar.selectbox("Select sample", sample_sets, index=0)
        if st.sidebar.button("Run Full Auto Setup", use_container_width=True):
            _apply_sample_set(sample_set)
            st.session_state["run_rounds"] = 1
            _start_pipeline_job(preview_image=None)
            st.session_state["page"] = "Caption"
            st.rerun()
        if st.sidebar.button("Load Paths Only", use_container_width=True):
            _apply_sample_set(sample_set)
            st.sidebar.success(f"Loaded {sample_set} paths")

    _render_job_status()
    if st.session_state.get("job_logs"):
        with st.expander("Pipeline logs", expanded=False):
            for line in st.session_state["job_logs"][-80:]:
                st.write(f"- {line}")

    if page == "Extract":
        _extract_tab()
    elif page == "Train":
        _train_tab()
    elif page == "Caption":
        _caption_tab()
    else:
        _help_tab()


if __name__ == "__main__":
    main()
