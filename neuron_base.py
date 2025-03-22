import json
import os
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any, Optional

# Initialize FastAPI app
app = FastAPI(title="Neuron API")


class NeuronInput(BaseModel):
    """Model for neuron input data."""

    data: List[float]
    prediction_id: str


class NeuronConfig:
    """Configuration for a neuron."""

    def __init__(self, layer_type, neuron_index, weights, bias, next_layer_hosts=None):
        self.layer_type = layer_type
        self.neuron_index = neuron_index
        self.weights = np.array(weights)
        self.bias = bias
        self.next_layer_hosts = next_layer_hosts or []

    @classmethod
    def from_json_file(cls, file_path):
        """Load neuron configuration from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)

        return cls(
            layer_type=data["layer"],
            neuron_index=data["index"],
            weights=data["weights"],
            bias=data["bias"],
        )


# Global neuron configuration
neuron_config = None
# Store last activation for monitoring
last_activation = 0.0
# Track metrics
request_count = 0
error_count = 0


@app.on_event("startup")
async def load_config():
    """Load neuron configuration on startup."""
    global neuron_config

    # Get configuration file path from environment
    config_file = os.getenv("NEURON_CONFIG_FILE")
    if not config_file:
        raise ValueError("NEURON_CONFIG_FILE environment variable not set")

    # Load configuration
    neuron_config = NeuronConfig.from_json_file(config_file)

    # Get next layer hosts from environment
    next_layer_hosts_str = os.getenv("NEXT_LAYER_HOSTS", "")
    if next_layer_hosts_str:
        neuron_config.next_layer_hosts = next_layer_hosts_str.split(",")

    print(
        f"Loaded configuration for {neuron_config.layer_type} neuron {neuron_config.neuron_index}"
    )
    print(f"Next layer hosts: {neuron_config.next_layer_hosts}")


@app.get("/")
async def root():
    """Basic endpoint to check if the neuron is alive."""
    return {
        "status": "running",
        "layer": neuron_config.layer_type if neuron_config else "not configured",
        "index": neuron_config.neuron_index if neuron_config else -1,
        "metrics": {
            "requests": request_count,
            "errors": error_count,
            "last_activation": float(last_activation),
        },
    }


@app.post("/forward")
async def forward(input_data: NeuronInput):
    """Process input and forward to the next layer."""
    global request_count, error_count, last_activation

    request_count += 1

    if neuron_config is None:
        error_count += 1
        raise HTTPException(status_code=500, detail="Neuron not configured")

    try:
        # For convolutional layers, the input is expected to be in the right format already
        # For FC layers, just do a dot product
        input_array = np.array(input_data.data)

        # Compute the neuron's activation
        # For simplicity, we're just doing a dot product for all neuron types
        # In a real system, conv layers would do convolutions
        activation = np.dot(input_array, neuron_config.weights) + neuron_config.bias

        # Apply ReLU activation function
        if neuron_config.layer_type != "fc2":  # No ReLU for output layer
            activation = max(0, activation)

        result = float(activation)
        last_activation = result

        # Report activation to monitoring service if configured
        monitoring_host = os.getenv("MONITORING_HOST")
        if monitoring_host:
            try:
                requests.post(
                    f"http://{monitoring_host}/report_activation",
                    json={
                        "layer": neuron_config.layer_type,
                        "neuron_id": neuron_config.neuron_index,
                        "activation": result,
                        "prediction_id": input_data.prediction_id,
                    },
                    timeout=0.5,
                )
            except Exception as e:
                print(f"Error reporting to monitoring: {e}")

        # If this is the output layer, we don't forward to anyone
        if neuron_config.layer_type == "fc2":
            # Store result in a way that can be retrieved by the output collector
            # In a real system, you'd use a database or message queue
            # For this demo, we'll just print it
            print(
                f"OUTPUT NEURON {neuron_config.neuron_index}: {result} for prediction {input_data.prediction_id}"
            )
            return {"result": result, "prediction_id": input_data.prediction_id}

        # Forward to next layer
        for host in neuron_config.next_layer_hosts:
            try:
                forward_data = {
                    "data": [result],
                    "prediction_id": input_data.prediction_id,
                }
                requests.post(f"http://{host}/forward", json=forward_data, timeout=5)
            except Exception as e:
                print(f"Error forwarding to {host}: {e}")
                error_count += 1

        return {"status": "processed"}
    except Exception as e:
        error_count += 1
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Return metrics for monitoring."""
    if neuron_config is None:
        return {
            "status": "not_configured",
            "metrics": {"requests": request_count, "errors": error_count},
        }

    return {
        "status": "running",
        "layer": neuron_config.layer_type,
        "index": neuron_config.neuron_index,
        "metrics": {
            "requests": request_count,
            "errors": error_count,
            "last_activation": float(last_activation),
        },
    }


def start_neuron_service():
    """Start the neuron service."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start_neuron_service()
