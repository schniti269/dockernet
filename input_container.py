import numpy as np
import requests
import uuid
import json
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict, Any
from PIL import Image
from io import BytesIO
import base64

# Initialize FastAPI app
app = FastAPI(title="CNN Input Container")

# Global variables
next_layer_hosts = []
prediction_results = {}  # Store prediction results


@app.on_event("startup")
async def setup():
    """Configure input container on startup."""
    global next_layer_hosts

    # Get first layer hosts from environment variable
    hosts_str = os.getenv("NEXT_LAYER_HOSTS", "")
    if hosts_str:
        next_layer_hosts = hosts_str.split(",")

    print(f"Input container configured with next layer hosts: {next_layer_hosts}")


@app.get("/")
async def root():
    """Basic health check endpoint."""
    return {"status": "running", "type": "input_container"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Process an uploaded MNIST image and distribute to first layer neurons.

    Expects a 28x28 grayscale image file upload.
    """
    if not next_layer_hosts:
        raise HTTPException(status_code=500, detail="No next layer hosts configured")

    # Generate a unique ID for this prediction
    prediction_id = str(uuid.uuid4())

    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("L")  # Convert to grayscale

        # Resize to 28x28 if needed
        if image.size != (28, 28):
            image = image.resize((28, 28))

        # Normalize pixel values similar to MNIST preprocessing
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array - 0.1307) / 0.3081  # MNIST normalization

        # Flatten the image
        flat_img = img_array.flatten().tolist()

        # Distribute to all first layer neurons (conv1)
        for host in next_layer_hosts:
            try:
                data = {"data": flat_img, "prediction_id": prediction_id}
                requests.post(f"http://{host}/forward", json=data, timeout=5)
            except Exception as e:
                print(f"Error sending to {host}: {e}")

        return {
            "status": "processing",
            "prediction_id": prediction_id,
            "message": "Image sent to neural network for processing",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/collect_result")
async def collect_result(data: Dict[str, Any]):
    """
    Collect results from output neurons.

    Expected format:
    {
        "neuron_index": 5,
        "result": 0.75,
        "prediction_id": "uuid-string"
    }
    """
    prediction_id = data.get("prediction_id")
    neuron_index = data.get("neuron_index")
    result = data.get("result")

    if not all([prediction_id, neuron_index is not None, result is not None]):
        raise HTTPException(status_code=400, detail="Missing required fields")

    # Store the result
    if prediction_id not in prediction_results:
        prediction_results[prediction_id] = {}

    prediction_results[prediction_id][neuron_index] = result

    return {"status": "result_collected"}


@app.get("/result/{prediction_id}")
async def get_result(prediction_id: str):
    """Get the prediction result for a given ID."""
    if prediction_id not in prediction_results:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Check if we have all 10 outputs (0-9 digits)
    results = prediction_results[prediction_id]
    if len(results) < 10:
        return {
            "status": "processing",
            "message": f"Received {len(results)} of 10 outputs so far",
            "partial_results": results,
        }

    # Find the predicted digit (highest activation)
    predicted_digit = max(results.items(), key=lambda x: x[1])[0]

    return {
        "status": "complete",
        "prediction": predicted_digit,
        "confidence_scores": results,
    }


def start_input_service():
    """Start the input container service."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start_input_service()
