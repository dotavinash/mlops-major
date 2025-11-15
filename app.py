from flask import Flask, render_template, request
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
import io

# Paths
MODEL_PATH = Path("models") / "savedmodel.pth"

app = Flask(__name__)

# Load model once at startup
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. "
                            f"Run train.py on the dev branch first.")
clf = joblib.load(MODEL_PATH)

# Simple label names (Olivetti faces classes are 0–39)
LABELS = [f"Person {i}" for i in range(40)]


def preprocess_image(file_storage):
    """
    Read an uploaded image, convert to 64x64 grayscale like Olivetti faces,
    flatten to 1D vector, and return as shape (1, -1) for prediction.
    """
    # Read bytes
    img_bytes = file_storage.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale

    # Resize to 64x64 (Olivetti resolution)
    img = img.resize((64, 64))

    # Convert to numpy array in [0, 1]
    arr = np.array(img, dtype="float32") / 255.0

    # Flatten to 1D and reshape as (1, -1)
    arr_flat = arr.reshape(1, -1)
    return arr_flat


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        if "image" not in request.files or request.files["image"].filename == "":
            error = "Please choose an image file before submitting."
        else:
            try:
                x = preprocess_image(request.files["image"])
                y_pred = clf.predict(x)[0]
                label = LABELS[int(y_pred)] if int(y_pred) < len(LABELS) else str(y_pred)
                prediction = f"Predicted class: {label}"
            except Exception as e:
                error = f"Error while predicting: {e}"

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    # Debug=True is fine for assignment demo
    app.run(host="0.0.0.0", port=5000, debug=True)
