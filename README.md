# Skin Image Classification — CNN/Flask Prototype

A small Flask web application that accepts a skin image and passes it to a separately supplied TensorFlow/Keras model. The configured output labels are No Acne, Mild Acne, Moderate Acne, and Severe Acne.

> **Important:** This is an educational prototype, not a medical diagnostic tool. The model's training dataset, label definitions, held-out evaluation, and clinical reliability have not been established by the files in this repository. Do not use the output to diagnose a condition or make treatment decisions.

## Current functionality

- Flask serves an HTML upload page.
- The app accepts JPG/JPEG and PNG uploads up to 5 MB.
- Pillow verifies that the upload is a decodable image before preprocessing.
- The image is converted to RGB, padded to a square, resized to 126 × 126, and normalized.
- TensorFlow/Keras runs inference using the model file placed next to `app.py`.
- The page displays the predicted configured label and the selected model output score.

## Project files

- `app.py` — Flask routes, model loading, image validation, preprocessing, and inference.
- `templates/index.html` — upload page and output display.
- `requirements.txt` — core application dependencies.
- `acne_cnn_model.h5` — required model file, downloaded separately and not included in this repository.

## Setup

Use a compatible Python environment (Python 3.10–3.12 is a reasonable starting point for the listed TensorFlow range).

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the model file from the existing project link below and place `acne_cnn_model.h5` in the same folder as `app.py`:

[Model download (Google Drive)](https://drive.google.com/file/d/19H6aYr4m2POV7okmZzXwi4JOJQ7Dv5Pn/view?usp=sharing)

Only use the model file if you trust its source. The repository code alone does not verify the file's provenance, training process, or performance.

## Run

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser. Debug mode is disabled by default.

## Model assumptions that still need verification

The current code assumes that the model accepts 126 × 126 RGB images and produces four outputs in the order listed in `CLASSES`. It also retains ImageNet mean/std normalization from the original code. **These assumptions must be checked against the exact model architecture and training preprocessing.** If training used different scaling or class ordering, predictions may be invalid even when inference runs without errors.

The displayed output score is not a calibrated probability or a measure of medical certainty.

## Limitations and responsible next steps

- No training script, labelled dataset, class mapping file, or reproducible evaluation report is included here.
- No accuracy, precision, recall, F1-score, confusion matrix, or external validation results are claimed.
- The uploaded images are stored under `static/uploads` so the result page can display them. Avoid using identifiable or sensitive images; add retention/deletion controls before any public deployment.
- Before making claims about model performance, verify the training pipeline, evaluate on a properly separated test set, document class balance and metrics, and seek appropriate clinical validation.
