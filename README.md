# AI Image Caption Generator

Python project that generates natural language captions for images using a CNN encoder and LSTM decoder.

## UI Section (Easy Understanding)

### Workflow at a glance

```text
Input Image Folder
	|
	v
[Feature Extraction: MobileNetV2]
	|
	v
[Caption Model Training: LSTM Decoder]
	|
	v
Saved Model + Tokenizer + Max Length
	|
	v
New Image -> Generated Caption
```

### Quick action cards

| Action | Command | Output |
|---|---|---|
| Extract Features | `python scripts/extract_features.py --images-dir data/Flickr8k_Dataset --output artifacts/features.npz` | `artifacts/features.npz` |
| Train Model | `PYTHONPATH=src python scripts/train.py --captions-file data/Flickr8k.token.txt --train-images-file data/Flickr_8k.trainImages.txt --features artifacts/features.npz --model-out artifacts/caption_model.keras --tokenizer-out artifacts/tokenizer.pkl --max-length-out artifacts/max_length.txt --epochs 20 --batch-size 64` | Model artifacts in `artifacts/` |
| Generate Caption | `PYTHONPATH=src python scripts/caption.py --image data/Flickr8k_Dataset/1000268201_693b08cb0e.jpg --model artifacts/caption_model.keras --tokenizer artifacts/tokenizer.pkl --max-length artifacts/max_length.txt` | Caption text in terminal |

### Input and output view

- Input:
  - Training images: `data/Flickr8k_Dataset/*.jpg`
  - Captions file: `data/Flickr8k.token.txt`
  - Train split file: `data/Flickr_8k.trainImages.txt`
- Output:
  - Features: `artifacts/features.npz`
  - Model: `artifacts/caption_model.keras`
  - Tokenizer: `artifacts/tokenizer.pkl`
  - Max sequence length: `artifacts/max_length.txt`
  - Final prediction: generated sentence printed in terminal

## Overview

This implementation uses:

- `MobileNetV2` (pre-trained on ImageNet) to extract image feature vectors
- `LSTM` language decoder to generate captions word-by-word
- Flickr8k caption format (`Flickr8k.token.txt`) for supervised training

The project provides a CLI workflow for:

- feature extraction
- model training
- caption generation on new images

## Project Structure

```text
.
├── requirements.txt
├── scripts/
│   ├── extract_features.py
│   ├── train.py
│   └── caption.py
└── src/
		└── aicg/
				├── config.py
				├── training.py
				├── inference.py
				├── data/
				│   └── flickr8k.py
				└── model/
						└── caption_model.py
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure Python can find the local package:

```bash
export PYTHONPATH=src
```

## Webpage UI

Run the webpage locally:

```bash
PYTHONPATH=src streamlit run web/app.py
```

The webpage includes 3 tabs:

- `Extract`: build `features.npz` from an image folder.
- `Train`: train the caption model and save artifacts.
- `Caption`: upload a new image and generate a caption.

It also includes sidebar automation:
- `Auto Run Extract + Train`: runs extraction and training automatically for the currently configured paths.
- `Run Full Auto Setup` (sample sets): applies sample paths, runs extraction + training, then opens Caption with an automatic preview.

It also includes a `Help` page with visual screenshots for:

- extraction setup
- quick start flow
- how changing training/model paths affects caption output

### Built-in Sample Test Sets

You can try the app quickly using sample sets in:

- `web/sample_sets/basic_set`
- `web/sample_sets/alt_set`
- `web/sample_sets/scenic_set`
- `web/sample_sets/people_set`

Use the sidebar in the webpage (`Try Sample Test Set` -> `Load Sample Paths`) to auto-fill all paths.

For a full beginner flow, use `Run Full Auto Setup` in the sidebar. It will automatically:
- load sample paths
- extract features
- train the model
- switch to the Caption page
- show an automatic caption preview when setup is finished

The webpage is beginner-friendly with:

- a `Beginner Wizard` readiness section
- extraction tips and image count detection
- caption output explanation (why this output appears)
- automatic caption generation right after image upload
- Help screenshots for setup and output-change guidance

## Dataset Layout (Flickr8k)

Expected files:

- `data/Flickr8k_Dataset/*.jpg`
- `data/Flickr8k.token.txt`
- `data/Flickr_8k.trainImages.txt`

You can use different paths if you pass them in CLI arguments.

## Run Pipeline

### 1) Extract image features

```bash
python scripts/extract_features.py \
	--images-dir data/Flickr8k_Dataset \
	--output artifacts/features.npz
```

### 2) Train caption model

```bash
PYTHONPATH=src python scripts/train.py \
	--captions-file data/Flickr8k.token.txt \
	--train-images-file data/Flickr_8k.trainImages.txt \
	--features artifacts/features.npz \
	--model-out artifacts/caption_model.keras \
	--tokenizer-out artifacts/tokenizer.pkl \
	--max-length-out artifacts/max_length.txt \
	--epochs 20 \
	--batch-size 64
```

### 3) Generate caption for a new image

```bash
PYTHONPATH=src python scripts/caption.py \
	--image data/Flickr8k_Dataset/1000268201_693b08cb0e.jpg \
	--model artifacts/caption_model.keras \
	--tokenizer artifacts/tokenizer.pkl \
	--max-length artifacts/max_length.txt
```

## Notes

- This is a baseline architecture focused on clarity and reproducibility.
- Greedy decoding is used for inference; beam search can improve caption quality.
- Training on CPU can be slow. GPU is recommended.