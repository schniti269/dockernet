import os
import time
import json
import requests
import threading
from prometheus_client import start_http_server, Counter, Gauge, Histogram
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cnn-monitoring")

# Prometheus metrics
PREDICTIONS_TOTAL = Counter("predictions_total", "Total number of predictions made")
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds"
)
ACTIVE_PREDICTIONS = Gauge("active_predictions", "Number of active predictions")
NEURON_ACTIVATIONS = Gauge(
    "neuron_activations", "Neuron activation values", ["layer", "neuron_id"]
)
NEURON_REQUESTS = Counter(
    "neuron_requests_total", "Total requests to neurons", ["layer", "neuron_id"]
)
NEURON_ERRORS = Counter(
    "neuron_errors_total", "Total errors from neurons", ["layer", "neuron_id"]
)

# Global state
neuron_states = {}
prediction_times = {}
neuron_hosts = {}


def discover_neurons():
    """Discover all neuron containers from environment variables."""
    logger.info("Discovering neurons from docker-compose...")

    try:
        # Read network info
        if os.path.exists("neuron_params/network_info.json"):
            with open("neuron_params/network_info.json", "r") as f:
                network_info = json.load(f)
                structure = network_info.get("structure", {})

                # Add all neurons by layer
                for layer in ["conv1", "conv2", "fc1", "fc2"]:
                    if layer in structure:
                        neurons = structure[layer].get("neurons", 0)
                        logger.info(f"Discovered {neurons} neurons in layer {layer}")

                        for i in range(neurons):
                            neuron_key = f"{layer}-{i}"
                            host = f"{layer}-neuron-{i}:8000"
                            neuron_hosts[neuron_key] = host
        else:
            logger.warning(
                "network_info.json not found, using environment for discovery"
            )

            # Fallback to checking a few sample neurons
            for layer in ["conv1", "conv2", "fc1", "fc2"]:
                for i in range(10):  # Check up to 10 neurons per layer
                    neuron_key = f"{layer}-{i}"
                    host = f"{layer}-neuron-{i}:8000"
                    neuron_hosts[neuron_key] = host

        logger.info(f"Discovered {len(neuron_hosts)} neurons")
    except Exception as e:
        logger.error(f"Error discovering neurons: {e}")


def collect_metrics():
    """Continuously collect metrics from all components."""
    while True:
        try:
            # Check input container
            try:
                response = requests.get("http://input-container:8000/", timeout=2)
                if response.status_code == 200:
                    # Get active predictions
                    active = len(prediction_times)
                    ACTIVE_PREDICTIONS.set(active)
            except Exception as e:
                logger.error(f"Error checking input container: {e}")

            # Check output container
            try:
                response = requests.get(
                    "http://output-container:8000/status", timeout=2
                )
                if response.status_code == 200:
                    data = response.json()
                    active = data.get("active_predictions", 0)
                    ACTIVE_PREDICTIONS.set(active)
            except Exception as e:
                logger.error(f"Error checking output container: {e}")

            # Check random subset of neurons to avoid too many requests
            sample_neurons = list(neuron_hosts.items())
            if len(sample_neurons) > 10:
                import random

                sample_neurons = random.sample(sample_neurons, 10)

            for neuron_key, host in sample_neurons:
                layer, neuron_id = neuron_key.split("-")

                try:
                    response = requests.get(f"http://{host}/", timeout=1)
                    if response.status_code == 200:
                        neuron_states[neuron_key] = "healthy"
                    else:
                        neuron_states[neuron_key] = "unhealthy"
                        NEURON_ERRORS.labels(layer=layer, neuron_id=neuron_id).inc()
                except Exception as e:
                    neuron_states[neuron_key] = "unreachable"
                    NEURON_ERRORS.labels(layer=layer, neuron_id=neuron_id).inc()

            # Sleep before next collection
            time.sleep(5)
        except Exception as e:
            logger.error(f"Error in metrics collection: {e}")
            time.sleep(5)


def track_prediction(prediction_id):
    """Start tracking a prediction."""
    prediction_times[prediction_id] = time.time()
    PREDICTIONS_TOTAL.inc()


def end_prediction(prediction_id):
    """End tracking a prediction and record latency."""
    if prediction_id in prediction_times:
        start_time = prediction_times[prediction_id]
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        del prediction_times[prediction_id]


def record_neuron_activation(layer, neuron_id, activation):
    """Record a neuron's activation value."""
    NEURON_ACTIVATIONS.labels(layer=layer, neuron_id=neuron_id).set(activation)
    NEURON_REQUESTS.labels(layer=layer, neuron_id=neuron_id).inc()


def start_monitoring(port=9090):
    """Start the monitoring service."""
    # Start Prometheus HTTP server
    start_http_server(port)
    logger.info(f"Prometheus metrics server started on port {port}")

    # Discover neurons
    discover_neurons()

    # Start metrics collection in a background thread
    metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
    metrics_thread.start()

    return {
        "track_prediction": track_prediction,
        "end_prediction": end_prediction,
        "record_neuron_activation": record_neuron_activation,
    }


if __name__ == "__main__":
    # This script can be run standalone for testing
    start_monitoring()
    while True:
        time.sleep(1)
