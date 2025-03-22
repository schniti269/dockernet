import unittest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
import web_ui


class TestDashboard(unittest.TestCase):
    """Test the dashboard functionality."""

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

    def tearDown(self):
        """Clean up after the test."""
        if hasattr(self, "temp_file") and os.path.exists(self.temp_file):
            os.unlink(self.temp_file)

    @patch("os.path.exists")
    @patch("builtins.open", create=True)
    def test_load_network_structure(self, mock_open, mock_exists):
        """Test loading the network structure."""
        # Mock the file exists check
        mock_exists.return_value = True

        # Mock the file open to return our test data
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = json.dumps(self.test_structure)
        mock_open.return_value = mock_file

        # Reset the layer sizes
        web_ui.layer_sizes = {"conv1": 0, "conv2": 0, "fc1": 0, "fc2": 0}

        # Call the function
        web_ui.load_network_structure()

        # Check that the layer sizes were set correctly
        self.assertEqual(web_ui.layer_sizes["conv1"], 32)
        self.assertEqual(web_ui.layer_sizes["conv2"], 64)
        self.assertEqual(web_ui.layer_sizes["fc1"], 128)
        self.assertEqual(web_ui.layer_sizes["fc2"], 10)

    def test_update_metrics(self):
        """Test the update_metrics function."""
        # Set up test data
        web_ui.prediction_history = [
            {"time_ms": 100},
            {"time_ms": 200},
            {"time_ms": 300},
        ]

        web_ui.neuron_states = {
            "conv1-0": "healthy",
            "conv1-1": "unhealthy",
            "conv2-0": "healthy",
            "fc1-0": "unreachable",
        }

        web_ui.layer_sizes = {"conv1": 32, "conv2": 64, "fc1": 128, "fc2": 10}

        # Call the function
        count, active, avg_time = web_ui.update_metrics(1)

        # Check the results
        self.assertEqual(count, "3")  # 3 predictions
        self.assertEqual(active, "2 / 234")  # 2 healthy out of 234 total neurons
        self.assertEqual(avg_time, "200 ms")  # Average of 100, 200, 300

    def test_render_history(self):
        """Test the render_history function."""
        # Set up test data
        web_ui.prediction_history = [
            {
                "id": "123",
                "time": "12:34:56",
                "time_ms": 100,
                "image": "data:image/png;base64,abc123",
                "result": 5,
            },
            {
                "id": "456",
                "time": "12:35:00",
                "time_ms": 150,
                "image": "data:image/png;base64,def456",
                "result": None,
            },
        ]

        # Call the function
        result = web_ui.render_history()

        # Check that the function returned a div
        self.assertTrue(isinstance(result, web_ui.html.Div))

        # Check that there are two rows (one for each prediction)
        self.assertEqual(len(result.children), 2)

        # Check the content of the first row
        first_row = result.children[0]
        self.assertIn("Prediction: 5", str(first_row))
        self.assertIn("12:34:56", str(first_row))
        self.assertIn("100 ms", str(first_row))

        # Check the content of the second row
        second_row = result.children[1]
        self.assertIn("Prediction: N/A", str(second_row))
        self.assertIn("12:35:00", str(second_row))
        self.assertIn("150 ms", str(second_row))


if __name__ == "__main__":
    unittest.main()
