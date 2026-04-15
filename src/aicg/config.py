from pathlib import Path

# Default workspace directories.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Default model artifact paths.
FEATURES_PATH = ARTIFACTS_DIR / "features.npz"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.pkl"
MAX_LENGTH_PATH = ARTIFACTS_DIR / "max_length.txt"
MODEL_PATH = ARTIFACTS_DIR / "caption_model.keras"
