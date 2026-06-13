"""
app.py — FastAPI REST API for chest X-ray pneumonia detection.

Wraps the trained model as an HTTP endpoint so any app, website, or
mobile device can send an X-ray image and receive a prediction.

Usage:
    uvicorn app:app --reload --port 8000

Endpoints:
    GET  /health   → check the server is running
    POST /predict  → upload an image, get NORMAL/PNEUMONIA + confidence

Example with curl:
    curl -X POST http://localhost:8000/predict \
         -F "file=@data/test/PNEUMONIA/person1_virus_1.jpeg"
"""

import io
import os

import torch
import torchvision.transforms as transforms
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# ── Configuration ─────────────────────────────────────────────────────────────

# The TorchScript model is preferred (no Python class needed to load it).
# If it does not exist yet, we fall back to the regular .pth checkpoint.
from config import TORCHSCRIPT_PATH, CLASS_NAMES, IMAGE_SIZE, MEAN, STD, API_HOST, API_PORT
from config import CHECKPOINT as CHECKPOINT_PATH

# ── Device ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Load model at startup ──────────────────────────────────────────────────────

def load_inference_model():
    """
    Load the model for inference.

    Tries to load the TorchScript model first (faster, no class needed).
    Falls back to the regular .pth checkpoint if TorchScript is not found.
    """
    if os.path.isfile(TORCHSCRIPT_PATH):
        print(f"Loading TorchScript model from: {TORCHSCRIPT_PATH}")
        model = torch.jit.load(TORCHSCRIPT_PATH, map_location=device)
    elif os.path.isfile(CHECKPOINT_PATH):
        print(f"TorchScript not found. Loading checkpoint from: {CHECKPOINT_PATH}")
        from model import load_model
        model = load_model(CHECKPOINT_PATH, device)
    else:
        raise RuntimeError(
            f"No model found. Expected '{TORCHSCRIPT_PATH}' or '{CHECKPOINT_PATH}'. "
            "Please run train.py or export.py first."
        )

    model.eval()
    print(f"Model ready on device: {device}")
    return model


# Load once when the server starts — not on every request
model = load_inference_model()


# ── Image preprocessing ────────────────────────────────────────────────────────

# This must match exactly what was used during training (see dataset.py)
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def prepare_image(image_bytes: bytes) -> torch.Tensor:
    """
    Read raw image bytes and convert to a model-ready tensor.

    Steps:
      1. Open the image with PIL
      2. Convert to RGB (handles grayscale X-rays automatically)
      3. Resize and normalize
      4. Add a batch dimension: (3, 224, 224) → (1, 3, 224, 224)

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        A float tensor of shape (1, 3, 224, 224) on the correct device.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not open the uploaded file as an image. "
                   "Please upload a valid JPEG or PNG."
        )

    tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension
    return tensor


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    description=(
        "Upload a chest X-ray image to receive a prediction: "
        "NORMAL or PNEUMONIA, with a confidence score."
    ),
    version="1.0.0",
)


# ── Response models ────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    prediction: str   # "NORMAL" or "PNEUMONIA"
    confidence: float  # 0.0 – 1.0, e.g. 0.95 means 95% confident

class HealthResponse(BaseModel):
    status:     str  # "ok"
    model:      str  # which model file is loaded
    device:     str  # "cpu" or "cuda"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check that the server is running and the model is loaded.",
)
def health():
    """
    Returns the server status and which model is currently loaded.

    Use this to verify the API is up before sending predictions.
    """
    model_file = TORCHSCRIPT_PATH if os.path.isfile(TORCHSCRIPT_PATH) else CHECKPOINT_PATH
    return HealthResponse(
        status="ok",
        model=model_file,
        device=str(device),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict NORMAL or PNEUMONIA",
    description=(
        "Upload a chest X-ray image (JPEG or PNG). "
        "Returns the predicted class and a confidence score between 0 and 1."
    ),
)
async def predict(file: UploadFile = File(..., description="Chest X-ray image (JPEG or PNG)")):
    """
    Classify a chest X-ray as NORMAL or PNEUMONIA.

    Steps:
      1. Read the uploaded image
      2. Preprocess it (resize + normalize)
      3. Run the model
      4. Return the predicted label and confidence score

    Returns:
        {
            "prediction": "PNEUMONIA",
            "confidence": 0.97
        }
    """
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: '{file.content_type}'. "
                   "Please upload a JPEG or PNG image."
        )

    # Read the uploaded bytes
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Preprocess the image into a tensor
    input_tensor = prepare_image(image_bytes)

    # Run inference (no gradients needed for prediction)
    with torch.no_grad():
        output     = model(input_tensor)         # Raw scores (logits)
        probs      = torch.softmax(output, dim=1) # Convert to probabilities
        pred_idx   = probs.argmax(dim=1).item()
        confidence = probs[0][pred_idx].item()

    prediction = CLASS_NAMES[pred_idx]

    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence, 4),  # e.g. 0.9731
    )


# ── Run directly ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
