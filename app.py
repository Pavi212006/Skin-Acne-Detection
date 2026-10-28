"""Flask prototype for running a supplied acne-image CNN model.

This is an educational demonstration, not a medical diagnostic tool. Model
weights and the original training/preprocessing pipeline must be verified
before predictions can be interpreted as meaningful.
"""
from pathlib import Path
import uuid

import numpy as np
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "acne_cnn_model.h5"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
IMG_HEIGHT = 126
IMG_WIDTH = 126
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASSES = ["No Acne", "Mild Acne", "Moderate Acne", "Severe Acne"]

# These normalization values are preserved from the original code. Confirm
# them against the exact preprocessing used when this model was trained.
IMAGENET_MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

model = None
try:
    if MODEL_PATH.is_file():
        model = load_model(str(MODEL_PATH), compile=False)
        print(f"CNN model loaded from {MODEL_PATH.name}.")
    else:
        print(
            f"Model file not found: {MODEL_PATH.name}. "
            "Download the model described in README.md and place it beside app.py."
        )
except Exception as exc:
    # Log startup detail to the server console, not to the public response.
    app.logger.exception("Could not load the CNN model: %s", exc)
    model = None

def prepare_image(image_file):
    """Validate and preprocess an uploaded image for the expected model input."""
    try:
        image_file.seek(0)
        with Image.open(image_file) as source:
            source.verify()

        image_file.seek(0)
        with Image.open(image_file) as source:
            img = source.convert("RGB")

        # Preserve aspect ratio by centering the image on a square canvas.
        width, height = img.size
        if width < 1 or height < 1:
            raise ValueError("Image has invalid dimensions.")
        max_dim = max(width, height)
        square = Image.new("RGB", (max_dim, max_dim), color=(0, 0, 0))
        square.paste(img, ((max_dim - width) // 2, (max_dim - height) // 2))
        resized = square.resize(TARGET_SIZE)

        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - IMAGENET_MEANS) / IMAGENET_STDS
        return np.expand_dims(array, axis=0)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Please upload a valid JPG or PNG image.") from exc

@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        prediction="Image is too large. Please upload an image smaller than 5 MB.",
        error=True,
    ), 413

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction=None, error=False, image_url=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "index.html",
            prediction="The model is unavailable. Check the server setup instructions.",
            error=True,
            image_url=None,
        ), 503

    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return render_template(
            "index.html",
            prediction="Choose a JPG or PNG image before submitting.",
            error=True,
            image_url=None,
        ), 400

    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return render_template(
            "index.html",
            prediction="Unsupported file type. Upload a JPG or PNG image.",
            error=True,
            image_url=None,
        ), 400

    try:
        processed = prepare_image(uploaded.stream)
        output = np.asarray(model.predict(processed, verbose=0))
        if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] != len(CLASSES):
            app.logger.error("Unexpected model output shape: %s", output.shape)
            raise RuntimeError("The model output does not match the four configured classes.")

        scores = output[0]
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("The model returned invalid numeric scores.")

        predicted_index = int(np.argmax(scores))
        score = float(scores[predicted_index])

        # The model's output is shown as a score, not a calibrated probability.
        result_text = (
            f"Model output label: {CLASSES[predicted_index]}. "
            f"Model output score: {score:.4f}. "
            "This is an experimental result, not a medical diagnosis."
        )

        # Save only after the file has been decoded and processed successfully.
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        safe_filename = f"{uuid.uuid4().hex}.jpg"
        saved_path = UPLOAD_FOLDER / safe_filename
        with Image.open(uploaded.stream) as source:
            source.convert("RGB").save(saved_path, format="JPEG", quality=90)

        from flask import url_for
        return render_template(
            "index.html",
            prediction=result_text,
            error=False,
            image_url=url_for("static", filename=f"uploads/{safe_filename}"),
        )
    except ValueError as exc:
        return render_template(
            "index.html", prediction=str(exc), error=True, image_url=None
        ), 400
    except Exception:
        app.logger.exception("Prediction failed.")
        return render_template(
            "index.html",
            prediction="Prediction failed. Check the model file and server logs.",
            error=True,
            image_url=None,
        ), 500

if __name__ == "__main__":
    # Debug mode is disabled by default; do not expose Flask's debugger publicly.
    app.run(debug=False)
