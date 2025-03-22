import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import json
import os
import time
from datetime import datetime, timedelta
import threading
import uuid
from io import BytesIO
from PIL import Image
import base64
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cnn-web-ui")

# Global variables
neuron_states = {}
prediction_history = []
layer_sizes = {"conv1": 0, "conv2": 0, "fc1": 0, "fc2": 0}
current_activations = {}


# Load network structure
def load_network_structure():
    """Load the network structure from the JSON file."""
    global layer_sizes

    try:
        if os.path.exists("neuron_params/network_info.json"):
            with open("neuron_params/network_info.json", "r") as f:
                network_info = json.load(f)
                structure = network_info.get("structure", {})

                for layer in ["conv1", "conv2", "fc1", "fc2"]:
                    if layer in structure:
                        layer_sizes[layer] = structure[layer].get("neurons", 0)
                        logger.info(f"Layer {layer} has {layer_sizes[layer]} neurons")
    except Exception as e:
        logger.error(f"Error loading network structure: {e}")


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css"
    ],
    title="CNN Neural Network Visualizer",
)

# Create layout
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "Containerized CNN Network Dashboard",
                    className="text-center mt-3 mb-4",
                ),
                html.P(
                    "A visualization of neural network neurons running in Docker containers",
                    className="text-center mb-4",
                ),
            ],
            className="container",
        ),
        html.Div(
            [
                # Top row with metrics
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H5("Network Status"),
                                        html.Div(
                                            id="status-indicator",
                                            className="status-light status-active",
                                        ),
                                        html.P(id="status-text", children="Active"),
                                    ],
                                    className="card-body",
                                )
                            ],
                            className="card shadow-sm h-100",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H5("Predictions"),
                                        html.H2(id="predictions-count", children="0"),
                                        html.P("Total predictions processed"),
                                    ],
                                    className="card-body",
                                )
                            ],
                            className="card shadow-sm h-100",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H5("Active Neurons"),
                                        html.H2(id="active-neurons", children="0 / 0"),
                                        html.P("Healthy / Total neurons"),
                                    ],
                                    className="card-body",
                                )
                            ],
                            className="card shadow-sm h-100",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H5("Avg. Processing Time"),
                                        html.H2(id="avg-time", children="0 ms"),
                                        html.P("Average prediction time"),
                                    ],
                                    className="card-body",
                                )
                            ],
                            className="card shadow-sm h-100",
                        ),
                    ],
                    className="row m-3",
                ),
                # Middle row with network visualization and prediction tool
                html.Div(
                    [
                        # Network visualization
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H4("Neural Network Visualization"),
                                        dcc.Graph(
                                            id="network-graph",
                                            figure={},
                                            style={"height": "500px"},
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                    className="card-body",
                                )
                            ],
                            className="card shadow-sm h-100",
                        ),
                        # Prediction input
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H4("Make a Prediction"),
                                        html.Div(
                                            [
                                                html.Div(id="upload-display"),
                                                dcc.Upload(
                                                    id="upload-image",
                                                    children=html.Div(
                                                        [
                                                            "Drag and Drop or ",
                                                            html.A("Select an Image"),
                                                        ]
                                                    ),
                                                    style={
                                                        "width": "100%",
                                                        "height": "100px",
                                                        "lineHeight": "100px",
                                                        "borderWidth": "1px",
                                                        "borderStyle": "dashed",
                                                        "borderRadius": "5px",
                                                        "textAlign": "center",
                                                        "margin": "10px 0",
                                                    },
                                                    multiple=False,
                                                ),
                                                html.Button(
                                                    "Predict",
                                                    id="predict-button",
                                                    className="btn btn-primary mt-2",
                                                ),
                                                html.Div(
                                                    id="prediction-result",
                                                    className="mt-3",
                                                ),
                                            ]
                                        ),
                                        html.Hr(),
                                        html.H5("Recent Predictions"),
                                        html.Div(
                                            id="prediction-history",
                                            className="prediction-history",
                                        ),
                                    ],
                                    className="card-body",
                                )
                            ],
                            className="card shadow-sm h-100",
                        ),
                    ],
                    className="row m-3",
                ),
                # Bottom row with neuron activations visualization
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H4("Neuron Activations"),
                                        html.Div(
                                            [
                                                dcc.Dropdown(
                                                    id="layer-dropdown",
                                                    options=[
                                                        {
                                                            "label": "Conv Layer 1",
                                                            "value": "conv1",
                                                        },
                                                        {
                                                            "label": "Conv Layer 2",
                                                            "value": "conv2",
                                                        },
                                                        {
                                                            "label": "FC Layer 1",
                                                            "value": "fc1",
                                                        },
                                                        {
                                                            "label": "Output Layer",
                                                            "value": "fc2",
                                                        },
                                                    ],
                                                    value="fc2",
                                                    className="mb-2",
                                                ),
                                                dcc.Graph(
                                                    id="activations-graph",
                                                    figure={},
                                                    style={"height": "400px"},
                                                    config={"displayModeBar": False},
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="card-body",
                                )
                            ],
                            className="card shadow-sm h-100",
                        ),
                    ],
                    className="row m-3",
                ),
            ],
            className="container-fluid",
        ),
        # Refresh interval
        dcc.Interval(
            id="interval-component", interval=2000, n_intervals=0  # 2 seconds
        ),
        # Hidden div for storing the current image
        html.Div(id="hidden-image-storage", style={"display": "none"}),
        # Footer
        html.Footer(
            [
                html.Div(
                    [
                        html.P(
                            "CNN Neuron Container Framework | Built with FastAPI, Docker, and Dash"
                        )
                    ],
                    className="container text-center py-3",
                )
            ],
            className="bg-light mt-5",
        ),
    ],
    className="bg-light min-vh-100",
)


# Callbacks for interactivity
@callback(
    Output("status-indicator", "className"),
    Output("status-text", "children"),
    Input("interval-component", "n_intervals"),
)
def update_status(n):
    """Update network status indicator."""
    try:
        # Check input container
        input_resp = requests.get("http://input-container:8000/", timeout=1)

        # Check output container
        output_resp = requests.get("http://output-container:8000/status", timeout=1)

        if input_resp.status_code == 200 and output_resp.status_code == 200:
            return "status-light status-active", "Active"
        else:
            return "status-light status-warning", "Partially Available"
    except:
        return "status-light status-inactive", "Inactive"


@callback(
    Output("predictions-count", "children"),
    Output("active-neurons", "children"),
    Output("avg-time", "children"),
    Input("interval-component", "n_intervals"),
)
def update_metrics(n):
    """Update dashboard metrics."""
    # Count predictions
    prediction_count = len(prediction_history)

    # Count active neurons
    active = sum(1 for state in neuron_states.values() if state == "healthy")
    total = sum(layer_sizes.values())

    # Calculate average time
    if prediction_history:
        times = [p.get("time_ms", 0) for p in prediction_history if "time_ms" in p]
        avg_time = f"{int(sum(times) / len(times)) if times else 0} ms"
    else:
        avg_time = "0 ms"

    return str(prediction_count), f"{active} / {total}", avg_time


@callback(Output("network-graph", "figure"), Input("interval-component", "n_intervals"))
def update_network_graph(n):
    """Generate neural network visualization graph."""
    # Create positions for layers
    layers = ["Input", "conv1", "conv2", "fc1", "fc2", "Output"]
    layer_x = {layer: i for i, layer in enumerate(layers)}

    # Count nodes in each layer
    counts = {
        "Input": 784,  # 28x28 MNIST image
        "conv1": layer_sizes["conv1"],
        "conv2": layer_sizes["conv2"],
        "fc1": layer_sizes["fc1"],
        "fc2": layer_sizes["fc2"],
        "Output": 10,  # 10 digit classes
    }

    # Generate node positions
    nodes_x = []
    nodes_y = []
    node_text = []
    node_color = []
    node_size = []

    for layer in layers:
        count = min(counts[layer], 100)  # Limit display to 100 nodes
        sample_step = max(1, counts[layer] // 100)

        for i in range(0, counts[layer], sample_step):
            if len(nodes_x) >= 100 * len(layers):
                break

            nodes_x.append(layer_x[layer])

            # Space nodes vertically
            if count > 1:
                nodes_y.append((i / sample_step) / (count - 1) * 2 - 1)
            else:
                nodes_y.append(0)

            # Add text and color
            if layer in ["Input", "Output"]:
                node_text.append(f"{layer} {i}")
                node_color.append("#bbbbbb")
            else:
                # Check neuron state
                state = neuron_states.get(f"{layer}-{i}", "unknown")
                node_text.append(f"{layer}-{i}: {state}")

                if state == "healthy":
                    node_color.append("#4CAF50")  # Green
                elif state == "unhealthy":
                    node_color.append("#FF9800")  # Orange
                else:
                    node_color.append("#9E9E9E")  # Gray

            # Smaller nodes for input/conv layers
            if layer in ["Input", "conv1", "conv2"]:
                node_size.append(5)
            else:
                node_size.append(10)

    # Create scatter plot for nodes
    fig = go.Figure()

    # Add nodes
    fig.add_trace(
        go.Scatter(
            x=nodes_x,
            y=nodes_y,
            mode="markers",
            marker=dict(size=node_size, color=node_color),
            text=node_text,
            hoverinfo="text",
            name="Neurons",
        )
    )

    # Add layer labels
    for layer, x in layer_x.items():
        fig.add_annotation(x=x, y=1.1, text=layer, showarrow=False, font=dict(size=14))

    # Update layout
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, len(layers) - 0.5],
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[-1.1, 1.1]
        ),
        plot_bgcolor="#f8f9fa",
    )

    return fig


@callback(
    Output("activations-graph", "figure"),
    Input("layer-dropdown", "value"),
    Input("interval-component", "n_intervals"),
)
def update_activations_graph(layer, n):
    """Update the neuron activations visualization for the selected layer."""
    # Get activations for the selected layer
    layer_activations = {
        neuron_id: activation
        for (layer_name, neuron_id), activation in current_activations.items()
        if layer_name == layer
    }

    if not layer_activations:
        # Create empty placeholder
        fig = px.bar(
            x=list(range(min(10, layer_sizes.get(layer, 10)))),
            y=[0] * min(10, layer_sizes.get(layer, 10)),
            labels={"x": "Neuron ID", "y": "Activation"},
        )
        fig.update_layout(
            title=f"No activation data for {layer}", plot_bgcolor="#f8f9fa"
        )
        return fig

    # Create dataframe
    df = pd.DataFrame(
        [
            {"neuron": neuron_id, "activation": activation}
            for neuron_id, activation in layer_activations.items()
        ]
    )

    # Create visualization based on layer type
    if layer == "fc2":  # Output layer - show as probabilities
        fig = px.bar(
            df,
            x="neuron",
            y="activation",
            labels={"neuron": "Digit", "activation": "Confidence"},
            color="activation",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            title="Output Layer Activations (Digit Probabilities)",
            xaxis_title="Digit Class (0-9)",
            yaxis_title="Confidence Score",
            plot_bgcolor="#f8f9fa",
        )
    else:
        fig = px.histogram(
            df,
            x="activation",
            nbins=30,
            labels={"activation": "Activation Value"},
            marginal="box",
        )
        fig.update_layout(
            title=f"{layer} Layer Activation Distribution",
            xaxis_title="Activation Value",
            yaxis_title="Count",
            plot_bgcolor="#f8f9fa",
        )

    return fig


@callback(
    Output("upload-display", "children"),
    Output("hidden-image-storage", "children"),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
)
def display_uploaded_image(contents, filename):
    """Display the uploaded image and store it."""
    if contents is None:
        return html.Div(), ""

    # Parse the contents
    content_type, content_string = contents.split(",")

    # Store the image
    return (
        html.Img(
            src=contents,
            style={"width": "100%", "max-height": "200px", "object-fit": "contain"},
        ),
        contents,
    )


@callback(
    Output("prediction-result", "children"),
    Output("prediction-history", "children"),
    Input("predict-button", "n_clicks"),
    State("hidden-image-storage", "children"),
    prevent_initial_call=True,
)
def make_prediction(n_clicks, contents):
    """Send the image to the neural network for prediction."""
    if not n_clicks or not contents:
        return html.Div(), html.Div()

    try:
        # Convert base64 string to file
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        # Send to neural network API
        start_time = time.time()

        files = {"file": ("image.png", BytesIO(decoded), "image/png")}
        response = requests.post("http://input-container:8000/predict", files=files)

        if response.status_code == 200:
            data = response.json()
            prediction_id = data.get("prediction_id")

            # Wait for the result (in a real app, you might want to poll instead)
            max_attempts = 10
            result = None

            for _ in range(max_attempts):
                time.sleep(0.5)
                result_response = requests.get(
                    f"http://input-container:8000/result/{prediction_id}"
                )

                if result_response.status_code == 200:
                    result_data = result_response.json()

                    if result_data.get("status") == "complete":
                        result = result_data
                        break

            # Calculate time
            time_ms = int((time.time() - start_time) * 1000)

            # Add to history
            prediction_obj = {
                "id": prediction_id,
                "time": datetime.now().strftime("%H:%M:%S"),
                "time_ms": time_ms,
                "image": contents,
                "result": result.get("prediction") if result else None,
            }
            prediction_history.insert(0, prediction_obj)

            # Limit history
            if len(prediction_history) > 10:
                prediction_history.pop()

            # Display result
            if result:
                return (
                    html.Div(
                        [
                            html.H3(
                                f"Predicted: {result.get('prediction')}",
                                className="text-center",
                            ),
                            html.P(
                                f"Processed in {time_ms} ms", className="text-center"
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [f"{digit}: {score:.2f}"],
                                        className="confidence-bar",
                                    )
                                    for digit, score in result.get(
                                        "confidence_scores", {}
                                    ).items()
                                ]
                            ),
                        ]
                    ),
                    render_history(),
                )
            else:
                return (
                    html.Div(
                        [
                            html.H3("Processing timed out", className="text-danger"),
                            html.P(
                                "The network took too long to respond. Check the status."
                            ),
                        ]
                    ),
                    render_history(),
                )
        else:
            return (
                html.Div(
                    [
                        html.H3("Error", className="text-danger"),
                        html.P(f"Status code: {response.status_code}"),
                    ]
                ),
                render_history(),
            )

    except Exception as e:
        return (
            html.Div([html.H3("Error", className="text-danger"), html.P(str(e))]),
            render_history(),
        )


def render_history():
    """Render the prediction history."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src=item["image"],
                                style={"width": "50px", "height": "50px"},
                            ),
                        ],
                        className="col-3",
                    ),
                    html.Div(
                        [
                            html.Strong(
                                f"Prediction: {item['result'] if item['result'] is not None else 'N/A'}"
                            ),
                            html.Br(),
                            html.Small(f"Time: {item['time']} ({item['time_ms']} ms)"),
                        ],
                        className="col-9",
                    ),
                ],
                className="row mb-2 border-bottom pb-2",
            )
            for item in prediction_history
        ]
    )


# Function to poll neuron status
def poll_neuron_status():
    """Background thread to poll neuron status."""
    global neuron_states, current_activations

    while True:
        try:
            # Poll a sample of neurons
            for layer in ["conv1", "conv2", "fc1", "fc2"]:
                for i in range(min(10, layer_sizes.get(layer, 0))):
                    neuron_key = f"{layer}-{i}"
                    host = f"{layer}-neuron-{i}:8000"

                    try:
                        response = requests.get(f"http://{host}/", timeout=0.5)
                        if response.status_code == 200:
                            neuron_states[neuron_key] = "healthy"

                            # Simulate activation data (in a real app, this would come from the neuron)
                            # In each layer, one neuron would have higher activation to show a pattern
                            activation = abs(np.random.normal(0, 0.1))
                            if i == (int(time.time()) % 10):
                                activation += 0.5

                            current_activations[(layer, i)] = activation
                        else:
                            neuron_states[neuron_key] = "unhealthy"
                    except:
                        neuron_states[neuron_key] = "unreachable"
        except Exception as e:
            logger.error(f"Error polling neuron status: {e}")

        time.sleep(2)


# On server start
@app.callback(
    Output("interval-component", "disabled"), Input("interval-component", "n_intervals")
)
def on_start(n):
    """Handle initialization."""
    if n == 0:
        # Load network structure
        load_network_structure()

        # Start background polling
        polling_thread = threading.Thread(target=poll_neuron_status, daemon=True)
        polling_thread.start()

    return False


# Custom CSS
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .status-light {
                width: 20px;
                height: 20px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 10px;
            }
            .status-active {
                background-color: #4CAF50;
                box-shadow: 0 0 10px 3px rgba(76, 175, 80, 0.5);
            }
            .status-warning {
                background-color: #FF9800;
                box-shadow: 0 0 10px 3px rgba(255, 152, 0, 0.5);
            }
            .status-inactive {
                background-color: #F44336;
                box-shadow: 0 0 10px 3px rgba(244, 67, 54, 0.5);
            }
            .card {
                margin-bottom: 20px;
            }
            .confidence-bar {
                background-color: #e9ecef;
                padding: 5px;
                margin-bottom: 5px;
                border-radius: 3px;
            }
            .prediction-history {
                max-height: 300px;
                overflow-y: auto;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# Start the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
