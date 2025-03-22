import os
import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any, Optional, List

# Initialize FastAPI app
app = FastAPI(title="CNN Output Container")

# Global configuration
input_container_host = os.getenv("INPUT_CONTAINER_HOST", "input-container:8000")
results_cache = {}  # Cache for collected results


class NeuronOutput(BaseModel):
    """Model for output neuron data."""

    result: float
    prediction_id: str


@app.get("/")
async def root():
    """Basic health check endpoint."""
    return {"status": "running", "type": "output_container"}


@app.post("/collect/{neuron_index}")
async def collect_output(neuron_index: int, output_data: NeuronOutput):
    """
    Collect output from a final layer neuron.

    Each output neuron (0-9 for MNIST) reports to this endpoint.
    """
    # Store the result in the local cache
    prediction_id = output_data.prediction_id

    if prediction_id not in results_cache:
        results_cache[prediction_id] = {}

    results_cache[prediction_id][neuron_index] = output_data.result

    # Forward to the input container
    try:
        data = {
            "neuron_index": neuron_index,
            "result": output_data.result,
            "prediction_id": prediction_id,
        }
        response = requests.post(
            f"http://{input_container_host}/collect_result", json=data, timeout=5
        )

        if response.status_code != 200:
            print(f"Error sending to input container: {response.text}")
    except Exception as e:
        print(f"Error forwarding result to input container: {e}")

    return {"status": "collected"}


@app.get("/results/{prediction_id}")
async def get_results(prediction_id: str):
    """Get the current collected results for a prediction."""
    if prediction_id not in results_cache:
        raise HTTPException(status_code=404, detail="Prediction not found")

    results = results_cache[prediction_id]
    expected_neurons = 10  # MNIST has 10 output neurons (0-9)

    return {
        "prediction_id": prediction_id,
        "completed": len(results) == expected_neurons,
        "collected_count": len(results),
        "expected_count": expected_neurons,
        "results": results,
    }


@app.get("/status")
async def get_status():
    """Get the status of the output container."""
    return {"active_predictions": len(results_cache), "status": "running"}


def start_output_service():
    """Start the output container service."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start_output_service()
