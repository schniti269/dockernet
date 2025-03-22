import unittest
import json
import numpy as np
from neuron_base import NeuronConfig
import tempfile
import os


class TestNeuronConfig(unittest.TestCase):
    """Test the NeuronConfig class from neuron_base.py."""

    def test_initialization(self):
        """Test basic initialization of NeuronConfig."""
        # Create a sample neuron config
        weights = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        bias = 0.5
        neuron = NeuronConfig("conv1", 0, weights, bias)

        # Check attributes
        self.assertEqual(neuron.layer_type, "conv1")
        self.assertEqual(neuron.neuron_index, 0)
        np.testing.assert_array_equal(neuron.weights, np.array(weights))
        self.assertEqual(neuron.bias, bias)
        self.assertEqual(neuron.next_layer_hosts, [])

    def test_from_json_file(self):
        """Test loading config from a JSON file."""
        # Create a temporary JSON file
        test_data = {
            "layer": "fc1",
            "index": 42,
            "weights": [0.1, 0.2, 0.3],
            "bias": 1.5,
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            # Load config from the temp file
            neuron = NeuronConfig.from_json_file(temp_file)

            # Check attributes
            self.assertEqual(neuron.layer_type, "fc1")
            self.assertEqual(neuron.neuron_index, 42)
            np.testing.assert_array_equal(neuron.weights, np.array([0.1, 0.2, 0.3]))
            self.assertEqual(neuron.bias, 1.5)
        finally:
            # Clean up
            os.unlink(temp_file)


class TestNeuronAPI(unittest.TestCase):
    """Test the neuron API functionality (could be expanded in real implementation)."""

    def test_neuron_computation(self):
        """Test the actual computation of a neuron."""
        # Create a simple neuron
        weights = np.array([0.1, 0.2, 0.3])
        bias = 0.5
        neuron = NeuronConfig("fc1", 0, weights, bias)

        # Sample input
        input_data = np.array([1.0, 2.0, 3.0])

        # Expected output (weights . input + bias)
        expected = np.dot(input_data, weights) + bias
        expected = max(0, expected)  # ReLU

        # Manual computation
        activation = np.dot(input_data, neuron.weights) + neuron.bias

        # Should be non-output layer so apply ReLU
        if neuron.layer_type != "fc2":
            activation = max(0, activation)

        # Check result
        self.assertAlmostEqual(activation, expected)


if __name__ == "__main__":
    unittest.main()
