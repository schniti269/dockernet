import torch
import torch.nn as nn
import json
import os
import numpy as np
from train import SimpleCNN


def extract_model_parameters():
    """Extracts parameters from the trained model and saves them in a format suitable for neuron containers."""
    # Create output directory
    os.makedirs("neuron_params", exist_ok=True)

    # Load model
    model = SimpleCNN()
    model.load_state_dict(
        torch.load("models/mnist_cnn.pth", map_location=torch.device("cpu"))
    )
    model.eval()

    # Dictionary to store layer information and neuron parameters
    network_info = {}

    # Extract convolutional layer 1 parameters
    conv1_weights = model.conv1.weight.detach().numpy()
    conv1_bias = model.conv1.bias.detach().numpy()

    # Extract convolutional layer 2 parameters
    conv2_weights = model.conv2.weight.detach().numpy()
    conv2_bias = model.conv2.bias.detach().numpy()

    # Extract fully connected layer 1 parameters
    fc1_weights = model.fc1.weight.detach().numpy()
    fc1_bias = model.fc1.bias.detach().numpy()

    # Extract fully connected layer 2 parameters
    fc2_weights = model.fc2.weight.detach().numpy()
    fc2_bias = model.fc2.bias.detach().numpy()

    # Save network structure
    network_info["structure"] = {
        "conv1": {
            "shape": list(conv1_weights.shape),
            "neurons": conv1_weights.shape[0],
        },
        "conv2": {
            "shape": list(conv2_weights.shape),
            "neurons": conv2_weights.shape[0],
        },
        "fc1": {"shape": list(fc1_weights.shape), "neurons": fc1_weights.shape[0]},
        "fc2": {"shape": list(fc2_weights.shape), "neurons": fc2_weights.shape[0]},
    }

    # Save overall network architecture
    with open("neuron_params/network_info.json", "w") as f:
        json.dump(network_info, f, indent=4, cls=NumpyEncoder)

    # Save individual neuron parameters

    # Conv1 neurons
    for i in range(conv1_weights.shape[0]):
        neuron_data = {
            "layer": "conv1",
            "index": i,
            "weights": conv1_weights[i].tolist(),
            "bias": float(conv1_bias[i]),
        }
        with open(f"neuron_params/conv1_neuron_{i}.json", "w") as f:
            json.dump(neuron_data, f, cls=NumpyEncoder)

    # Conv2 neurons
    for i in range(conv2_weights.shape[0]):
        neuron_data = {
            "layer": "conv2",
            "index": i,
            "weights": conv2_weights[i].tolist(),
            "bias": float(conv2_bias[i]),
        }
        with open(f"neuron_params/conv2_neuron_{i}.json", "w") as f:
            json.dump(neuron_data, f, cls=NumpyEncoder)

    # FC1 neurons
    for i in range(fc1_weights.shape[0]):
        neuron_data = {
            "layer": "fc1",
            "index": i,
            "weights": fc1_weights[i].tolist(),
            "bias": float(fc1_bias[i]),
        }
        with open(f"neuron_params/fc1_neuron_{i}.json", "w") as f:
            json.dump(neuron_data, f, cls=NumpyEncoder)

    # FC2 neurons (output layer)
    for i in range(fc2_weights.shape[0]):
        neuron_data = {
            "layer": "fc2",
            "index": i,
            "weights": fc2_weights[i].tolist(),
            "bias": float(fc2_bias[i]),
        }
        with open(f"neuron_params/fc2_neuron_{i}.json", "w") as f:
            json.dump(neuron_data, f, cls=NumpyEncoder)

    print(
        f"Extracted parameters for {conv1_weights.shape[0]} conv1 neurons, "
        f"{conv2_weights.shape[0]} conv2 neurons, "
        f"{fc1_weights.shape[0]} fc1 neurons, and "
        f"{fc2_weights.shape[0]} fc2 (output) neurons."
    )


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    extract_model_parameters()
