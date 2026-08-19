from io import BytesIO
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

app = FastAPI(
    title="Lion or Tiger Classifier",
    description="Upload an image to classify it as lion or tiger.",
)

BASE_DIR = Path(__file__).resolve().parent

# Load the finished model once when the API starts.
model = tf.keras.models.load_model(BASE_DIR / "lion_tiger_classifier_finetuned.keras")

# Load the correct label order: 0 = lion, 1 = tiger.
with open(BASE_DIR / "class_names.json", "r") as file:
    class_names = json.load(file)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Reject files that are not images.
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, or other image file.",
        )

    image_bytes = await file.read()

    try:
        # Open the image and make sure it has three colour channels: red, green, blue.
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as an image.",
        )

    # The model was trained on 224 × 224 images.
    image = image.resize((224, 224))

    # Convert the image into numbers the model can process.
    image_array = np.array(image, dtype=np.float32)

    # Add one outer dimension because the model expects a batch of images.
    image_array = np.expand_dims(image_array, axis=0)

    # Ask the model for lion and tiger probabilities.
    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    return {
        "prediction": class_names[predicted_index],
        "confidence": round(confidence, 4),
        "probabilities": {
            class_names[0]: round(float(probabilities[0]), 4),
            class_names[1]: round(float(probabilities[1]), 4),
        },
    }
