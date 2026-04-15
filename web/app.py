from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

from aicg.inference import extract_single_image_feature, generate_caption
from aicg.training import train_model


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SAMPLE_SETS_DIR = Path(__file__).resolve().parent / "sample_sets"


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
    }


def _ready(path_value: str) -> bool:
    return Path(path_value).exists()


def _apply_sample_set(sample_set_name: str) -> None:
    paths = _sample_set_paths(sample_set_name)
    st.session_state["images_dir"] = paths["images_dir"]
    st.session_state["captions_file"] = paths["captions_file"]
    st.session_state["train_images_file"] = paths["train_images_file"]
    st.session_state["features_path"] = paths["features_path"]
    st.session_state["model_out"] = paths["model_out"]
    st.session_state["tokenizer_out"] = paths["tokenizer_out"]
    st.session_state["max_length_out"] = paths["max_length_out"]
    st.session_state["model_path"] = paths["model_out"]
    st.session_state["tokenizer_path"] = paths["tokenizer_out"]
    st.session_state["max_length_path"] = paths["max_length_out"]


def _complete_sample_setup(sample_set_name: str) -> tuple[int, str]:
    """Load sample paths, run extraction and a quick training cycle.

    Returns:
        (image_count, model_path)
    """
    _apply_sample_set(sample_set_name)
    paths = _sample_set_paths(sample_set_name)

    image_count = extract_features_from_dir(
        Path(paths["images_dir"]),
        Path(paths["features_path"]),
        image_size=224,
    )

    train_model(
        captions_file=Path(paths["captions_file"]),
        train_images_file=Path(paths["train_images_file"]),
        features_path=Path(paths["features_path"]),
        model_path=Path(paths["model_out"]),
        tokenizer_path=Path(paths["tokenizer_out"]),
        max_length_path=Path(paths["max_length_out"]),
        epochs=1,
        batch_size=2,
    )

    return image_count, paths["model_out"]


@lru_cache(maxsize=1)
def _extractor_model():
    return keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )


def extract_features_from_dir(images_dir: Path, output_path: Path, image_size: int = 224) -> int:
    model = _extractor_model()
    patterns = ("*.jpg", "*.jpeg", "*.png")
    image_files = sorted({p for pattern in patterns for p in images_dir.glob(pattern)})
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}. Supported: .jpg, .jpeg, .png")

    features: dict[str, np.ndarray] = {}
    for image_path in image_files:
        arr = np.array(Image.open(image_path).convert("RGB").resize((image_size, image_size)), dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        arr = keras.applications.mobilenet_v2.preprocess_input(arr)
        feat = model.predict(arr, verbose=0)[0].astype(np.float32)
        features[image_path.name] = feat

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **features)
    return len(features)


def _render_header() -> None:
    st.title("AI Image Caption Generator")
    st.caption("Simple workflow: Extract features -> Train model -> Generate captions")


def _render_feature_checklist() -> None:
    st.info(
        "Features available: 1) Feature Extraction  2) Model Training  3) Caption Generation  4) Help"
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

    st.caption("Tip: For best results, run pages in order: Extract -> Train -> Caption")


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
    else:
        st.caption("Folder not found yet. Check path or use sample set from sidebar.")

    if st.button("Run Feature Extraction", use_container_width=True):
        with st.spinner("Extracting features..."):
            try:
                count = extract_features_from_dir(Path(images_dir), Path(output_path), int(image_size))
                st.success(f"Done. Extracted features for {count} images into {output_path}.")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with st.expander("Beginner tips"):
        st.write("- Use a folder with JPG/JPEG/PNG images.")
        st.write("- Keep image size at 224 for speed and compatibility.")
        st.write("- The output feature file is required for training.")


def _train_tab() -> None:
    st.subheader("2) Train Model")
    captions_file = st.text_input("Captions file", key="captions_file")
    train_images_file = st.text_input("Train image list", key="train_images_file")
    features_path = st.text_input("Features path", key="features_path")

    c1, c2 = st.columns(2)
    with c1:
        model_out = st.text_input("Model output", key="model_out")
        tokenizer_out = st.text_input("Tokenizer output", key="tokenizer_out")
        max_length_out = st.text_input("Max length output", key="max_length_out")
    with c2:
        epochs = st.slider("Epochs", min_value=1, max_value=100, value=20)
        batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=2)

    if not _ready(features_path):
        st.warning("Features file not found yet. Run Extract first or verify path.")

    if st.button("Run Training", use_container_width=True):
        with st.spinner("Training model..."):
            try:
                train_model(
                    captions_file=Path(captions_file),
                    train_images_file=Path(train_images_file),
                    features_path=Path(features_path),
                    model_path=Path(model_out),
                    tokenizer_path=Path(tokenizer_out),
                    max_length_path=Path(max_length_out),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                )
                st.success("Training completed and artifacts saved.")
                st.session_state["model_path"] = model_out
                st.session_state["tokenizer_path"] = tokenizer_out
                st.session_state["max_length_path"] = max_length_out
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with st.expander("Beginner tips"):
        st.write("- Start with 10 to 20 epochs to test quickly.")
        st.write("- Smaller batch size can improve fit but is slower.")
        st.write("- After training, caption page paths auto-update from outputs.")


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
    st.write("- To change output quality, retrain with more epochs or a larger dataset.")


def _caption_tab() -> None:
    st.subheader("3) Generate Caption")
    model_path = Path(st.text_input("Model path", key="model_path"))
    tokenizer_path = Path(st.text_input("Tokenizer path", key="tokenizer_path"))
    max_length_path = Path(st.text_input("Max length path", key="max_length_path"))

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)

    if not (_ready(str(model_path)) and _ready(str(tokenizer_path)) and _ready(str(max_length_path))):
        st.warning("Model/tokenizer/max-length file missing. Train first or verify paths.")

    if st.button("Generate Caption", use_container_width=True):
        if uploaded is None:
            st.warning("Upload an image first.")
            return

        temp_image_path: Path | None = None
        with st.spinner("Generating caption..."):
            try:
                suffix = Path(uploaded.name).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded.getbuffer())
                    temp_image_path = Path(tmp.name)

                feature = extract_single_image_feature(temp_image_path)
                caption = generate_caption(
                    image_feature=feature,
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    max_length_path=max_length_path,
                )
                st.success(f"Caption: {caption if caption else '[empty caption]'}")
                _render_caption_explanation(caption, model_path, tokenizer_path, max_length_path)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
            finally:
                if temp_image_path is not None:
                    try:
                        temp_image_path.unlink(missing_ok=True)
                    except Exception:  # noqa: BLE001
                        pass


def _help_tab() -> None:
    st.subheader("Help")
    st.write("Use this page to understand the flow and how to use each feature.")

    workflow_image = ASSETS_DIR / "workflow_help.png"
    quickstart_image = ASSETS_DIR / "quickstart_help.png"
    extract_settings_image = ASSETS_DIR / "extract_settings_help.png"
    output_change_image = ASSETS_DIR / "output_change_help.png"
    wizard_help_image = ASSETS_DIR / "wizard_help.png"
    output_help_image = ASSETS_DIR / "output_explain_help.png"

    if workflow_image.exists():
        st.image(str(workflow_image), caption="End-to-end workflow", use_container_width=True)
    else:
        st.warning("Missing help image: workflow_help.png")

    if quickstart_image.exists():
        st.image(str(quickstart_image), caption="Quick start guide", use_container_width=True)
    else:
        st.warning("Missing help image: quickstart_help.png")

    if extract_settings_image.exists():
        st.image(str(extract_settings_image), caption="How to configure extraction", use_container_width=True)
    else:
        st.warning("Missing help image: extract_settings_help.png")

    if output_change_image.exists():
        st.image(str(output_change_image), caption="How changing settings changes output", use_container_width=True)
    else:
        st.warning("Missing help image: output_change_help.png")

    if wizard_help_image.exists():
        st.image(str(wizard_help_image), caption="Beginner wizard and sample set flow", use_container_width=True)
    else:
        st.warning("Missing help image: wizard_help.png")

    if output_help_image.exists():
        st.image(str(output_help_image), caption="How output explanation is shown", use_container_width=True)
    else:
        st.warning("Missing help image: output_explain_help.png")

    st.markdown("### Built-in test sets")
    st.write("Try these ready-made datasets from sidebar:")
    st.write("- basic_set")
    st.write("- alt_set")

    st.markdown("### Common checks")
    st.write("- Ensure feature file exists before training.")
    st.write("- Ensure model/tokenizer/max-length files exist before caption generation.")
    st.write("- Use matching paths across Extract, Train, and Caption pages.")


def main() -> None:
    st.set_page_config(page_title="AI Caption Generator", page_icon="AI", layout="wide")

    # Default field values (can be overwritten by sample set selector).
    st.session_state.setdefault("images_dir", "data/Flickr8k_Dataset")
    st.session_state.setdefault("features_path", "artifacts/features.npz")
    st.session_state.setdefault("captions_file", "data/Flickr8k.token.txt")
    st.session_state.setdefault("train_images_file", "data/Flickr_8k.trainImages.txt")
    st.session_state.setdefault("model_out", "artifacts/caption_model.keras")
    st.session_state.setdefault("tokenizer_out", "artifacts/tokenizer.pkl")
    st.session_state.setdefault("max_length_out", "artifacts/max_length.txt")
    st.session_state.setdefault("model_path", "artifacts/caption_model.keras")
    st.session_state.setdefault("tokenizer_path", "artifacts/tokenizer.pkl")
    st.session_state.setdefault("max_length_path", "artifacts/max_length.txt")
    st.session_state.setdefault("image_size", 224)

    _render_header()
    _render_feature_checklist()
    _render_beginner_wizard()

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Extract", "Train", "Caption", "Help"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Try Sample Test Set")
    sample_sets = _available_sample_sets()
    if sample_sets:
        sample_set = st.sidebar.selectbox("Select sample", sample_sets, index=0)
        if st.sidebar.button("Load Sample Paths", use_container_width=True):
            with st.spinner("Running sample setup (extract + train)..."):
                try:
                    image_count, model_path = _complete_sample_setup(sample_set)
                    st.sidebar.success(
                        f"Sample setup complete. Extracted {image_count} images and trained model at {model_path}."
                    )
                    st.sidebar.info("Go to Caption page and click Generate Caption.")
                except Exception as exc:  # noqa: BLE001
                    st.sidebar.error(str(exc))

        if st.sidebar.button("Load Paths Only", use_container_width=True):
            _apply_sample_set(sample_set)
            st.sidebar.success(f"Loaded {sample_set} paths")
    else:
        st.sidebar.caption("No sample sets found.")

    st.sidebar.markdown("---")
    st.sidebar.write("Keep this order for best results:")
    st.sidebar.write("1. Extract")
    st.sidebar.write("2. Train")
    st.sidebar.write("3. Caption")
    st.sidebar.write("4. Help (reference)")

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
