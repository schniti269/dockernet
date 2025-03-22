# Containerized CNN for MNIST

This project implements a CNN trained on MNIST where each neuron runs in its own Docker container.

## Architecture
- Each neuron lives in a separate container
- Communication between neurons happens via API calls
- Input and output containers handle external communication
- Real-time monitoring with Prometheus metrics
- Interactive web dashboard to visualize network activity

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the training script: `python train.py`
3. Extract parameters: `python extract_params.py`
4. Generate Docker Compose: `python generate_docker_compose.py`
5. Start the containerized network: `python run_network.py`

## API Usage
- Send images to the network at: `POST /predict`
- Receive predictions at: `GET /result/{prediction_id}`

## Web UI and Monitoring
- Access the web dashboard at: `http://localhost:8050`
- Prometheus metrics available at: `http://localhost:9090`

## Features
- **Network Visualization**: Interactive visualization of neuron connections and states
- **Real-time Metrics**: Monitor neuron health, activation values, and prediction latency
- **Interactive Testing**: Upload and test MNIST images directly from the UI
- **Neuron Activation Visualization**: View activation distributions across layers 