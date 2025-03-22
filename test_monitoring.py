import unittest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
import monitoring
from prometheus_client import REGISTRY


class TestMonitoring(unittest.TestCase):
    """Test the monitoring functionality."""

    def setUp(self):
        """Set up the test environment."""
        # Create a mock network structure
        self.test_structure = {
            "structure": {
                "conv1": {"shape": [32, 1, 3, 3], "neurons": 32},
                "conv2": {"shape": [64, 32, 3, 3], "neurons": 64},
                "fc1": {"shape": [128, 3136], "neurons": 128},
                "fc2": {"shape": [10, 128], "neurons": 10},
            }
        }

        # Create a temporary file for the network structure
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            json.dump(self.test_structure, f)
            self.temp_file = f.name

        # Reset global state
        monitoring.neuron_states = {}
        monitoring.prediction_times = {}
        monitoring.neuron_hosts = {}

    def tearDown(self):
        """Clean up after the test."""
        if hasattr(self, "temp_file") and os.path.exists(self.temp_file):
            os.unlink(self.temp_file)

    @patch("os.path.exists")
    @patch("builtins.open", create=True)
    def test_discover_neurons(self, mock_open, mock_exists):
        """Test discovering neurons from the network structure."""
        # Mock the file exists check
        mock_exists.return_value = True

        # Mock the file open to return our test data
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = json.dumps(self.test_structure)
        mock_open.return_value = mock_file

        # Call the function
        monitoring.discover_neurons()

        # Check that the neuron hosts were discovered correctly
        self.assertEqual(len(monitoring.neuron_hosts), 32 + 64 + 128 + 10)

        # Check a few sample neurons
        self.assertEqual(monitoring.neuron_hosts.get("conv1-0"), "conv1-neuron-0:8000")
        self.assertEqual(
            monitoring.neuron_hosts.get("conv2-10"), "conv2-neuron-10:8000"
        )
        self.assertEqual(monitoring.neuron_hosts.get("fc1-50"), "fc1-neuron-50:8000")
        self.assertEqual(monitoring.neuron_hosts.get("fc2-5"), "fc2-neuron-5:8000")

    def test_track_prediction(self):
        """Test tracking predictions."""
        # Track a prediction
        monitoring.track_prediction("test-123")

        # Check that the prediction was tracked
        self.assertIn("test-123", monitoring.prediction_times)

        # Check that the counter was incremented
        counter_value = REGISTRY.get_sample_value("predictions_total_total")
        self.assertIsNotNone(counter_value)
        self.assertGreater(counter_value, 0)

    def test_end_prediction(self):
        """Test ending prediction tracking."""
        # First track a prediction
        import time

        monitoring.track_prediction("test-456")
        time.sleep(0.1)  # Ensure some time passes

        # End the prediction
        monitoring.end_prediction("test-456")

        # Check that the prediction was removed from tracking
        self.assertNotIn("test-456", monitoring.prediction_times)

        # Check that the histogram was updated
        histogram_count = REGISTRY.get_sample_value("prediction_latency_seconds_count")
        self.assertIsNotNone(histogram_count)
        self.assertGreater(histogram_count, 0)

    def test_record_neuron_activation(self):
        """Test recording neuron activations."""
        # Record an activation
        monitoring.record_neuron_activation("conv1", "5", 0.75)

        # Check that the gauge was updated
        activation_value = REGISTRY.get_sample_value(
            "neuron_activations", {"layer": "conv1", "neuron_id": "5"}
        )
        self.assertIsNotNone(activation_value)
        self.assertEqual(activation_value, 0.75)

        # Check that the counter was incremented
        requests_count = REGISTRY.get_sample_value(
            "neuron_requests_total", {"layer": "conv1", "neuron_id": "5"}
        )
        self.assertIsNotNone(requests_count)
        self.assertEqual(requests_count, 1)


if __name__ == "__main__":
    unittest.main()
