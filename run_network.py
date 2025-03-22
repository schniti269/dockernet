import os
import subprocess
import time
import argparse
import json
from pathlib import Path


def check_prerequisites():
    """Check if all prerequisites are met."""
    # Check if model is trained
    if not os.path.exists("models/mnist_cnn.pth"):
        print("Model not found. Training the model first...")
        subprocess.run(["python", "train.py"], check=True)

    # Check if parameters are extracted
    if not os.path.exists("neuron_params/network_info.json"):
        print("Neuron parameters not found. Extracting parameters...")
        subprocess.run(["python", "extract_params.py"], check=True)

    # Check if docker-compose.yml exists or generate it
    if not os.path.exists("docker-compose.yml"):
        print("docker-compose.yml not found. Generating it...")
        subprocess.run(["python", "generate_docker_compose.py"], check=True)


def start_network(limit_neurons=None, debug=False):
    """Start the containerized neural network."""
    # If limited number of neurons requested, modify the docker-compose file
    if limit_neurons:
        modify_docker_compose(limit_neurons)

    # Start docker-compose - use newer Docker Compose V2 syntax
    print(f"Starting the containerized neural network...")

    # Check if Docker is running
    try:
        docker_info_result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, check=False
        )
        if docker_info_result.returncode != 0:
            print(
                "ERROR: Docker doesn't appear to be running. Please start Docker Desktop first."
            )
            print(docker_info_result.stderr)
            return
    except Exception as e:
        print(f"Error checking Docker status: {e}")
        print("Please ensure Docker is installed and running.")
        return

    try:
        # Try with the newer 'docker compose' command (V2)
        cmd = ["docker", "compose"]

        if debug:
            # In debug mode, don't run in detached mode so we can see logs
            cmd.extend(["up", "--build"])
            print("Debug mode: Running in foreground with logs visible")
        else:
            cmd.extend(["up", "-d"])

        result = subprocess.run(cmd, capture_output=debug, text=True)

        if result.returncode != 0:
            print(f"Error starting containers (code {result.returncode}):")
            print(
                result.stderr
                if hasattr(result, "stderr")
                else "No error details available"
            )

            if debug:
                # In debug mode, print more details
                print("\nDOCKER-COMPOSE FILE CONTENTS:")
                with open("docker-compose.yml", "r") as f:
                    print(f.read())

                print("\nDOCKER VERSION INFO:")
                subprocess.run(["docker", "version"], check=False)

                print("\nTrying with fewer neuron containers...")
                limit = 10  # Just try with a few neurons
                modify_docker_compose(limit)
                print(
                    f"Modified docker-compose.yml to include only {limit} neuron containers"
                )

                retry_cmd = ["docker", "compose", "up", "-d"]
                print(f"Retrying with command: {' '.join(retry_cmd)}")
                retry_result = subprocess.run(retry_cmd, capture_output=True, text=True)

                if retry_result.returncode != 0:
                    print(f"Still failed with error code {retry_result.returncode}:")
                    print(retry_result.stderr)
                    print(
                        "\nConsider trying with docker-compose instead of docker compose"
                    )
                    subprocess.run(["docker-compose", "up", "-d"], check=False)

            print(
                "If problems persist, try running individual containers or manually checking Docker logs."
            )
            return
    except subprocess.CalledProcessError:
        # Fall back to the older 'docker-compose' command if needed
        print("Falling back to older docker-compose command...")
        try:
            subprocess.run(["docker-compose", "up", "-d"], check=True)
        except Exception as e:
            print(f"Error with docker-compose fallback: {e}")
            return

    # Wait for services to start
    print("Waiting for services to start...")
    time.sleep(10)

    # Print connection information
    print("\n" + "=" * 50)
    print("Containerized Neural Network is running!")
    print("=" * 50)
    print("Input API (for sending images): http://localhost:8000/predict")
    print(
        "Results API (for checking results): http://localhost:8000/result/{prediction_id}"
    )
    print("Web Dashboard: http://localhost:8050")
    print("Prometheus Metrics: http://localhost:9090")
    print("=" * 50)


def modify_docker_compose(limit):
    """Modify docker-compose to limit the number of neurons (for testing)."""
    # This is a simple way to limit the system for testing
    # A real system would have a more sophisticated approach

    with open("docker-compose.yml", "r") as f:
        content = f.read()

    lines = content.split("\n")
    service_count = 0
    included_lines = []

    for line in lines:
        if "neuron-" in line and "service_name" in line:
            service_count += 1
            if service_count > limit:
                # Skip this service and related lines
                in_service_def = True
                continue

        included_lines.append(line)

    with open("docker-compose.yml", "w") as f:
        f.write("\n".join(included_lines))


def stop_network():
    """Stop the containerized neural network."""
    try:
        # Try with the newer 'docker compose' command (V2)
        subprocess.run(["docker", "compose", "down"], check=True)
    except subprocess.CalledProcessError:
        # Fall back to the older 'docker-compose' command if needed
        print("Falling back to older docker-compose command...")
        subprocess.run(["docker-compose", "down"], check=True)
    print("Neural network containers stopped.")


def test_network():
    """Run a simple test on the network."""
    # Check if test MNIST images exist
    test_image = Path("data/test_images/test_digit.png")
    if not test_image.exists():
        print("Test image not found. Skipping test.")
        return

    print("Running a test prediction...")

    # Use curl to send an image to the network
    result = subprocess.run(
        [
            "curl",
            "-X",
            "POST",
            "http://localhost:8000/predict",
            "-F",
            f"file=@{test_image}",
        ],
        capture_output=True,
        text=True,
    )

    # Extract prediction ID from response
    try:
        response = json.loads(result.stdout)
        prediction_id = response.get("prediction_id")

        if prediction_id:
            print(f"Test image sent. Prediction ID: {prediction_id}")
            print(f"Check result at: http://localhost:8000/result/{prediction_id}")
            print(f"Or view in dashboard: http://localhost:8050")
        else:
            print("Error: Could not get prediction ID from response.")
            print(f"Response: {result.stdout}")
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {result.stdout}")


def open_dashboard():
    """Open the web dashboard in the default browser."""
    import webbrowser

    print("Opening web dashboard...")
    webbrowser.open("http://localhost:8050")


def check_docker_status():
    """Check if Docker is running correctly."""
    print("Checking Docker status...")

    try:
        # Check docker version
        print("\nDocker Version:")
        subprocess.run(["docker", "--version"], check=True)

        # Check docker info
        print("\nDocker Info:")
        subprocess.run(["docker", "info"], check=True)

        # Check available images
        print("\nDocker Images:")
        subprocess.run(["docker", "images"], check=True)

        # Check running containers
        print("\nRunning Containers:")
        subprocess.run(["docker", "ps"], check=True)

        print("\nDocker appears to be running correctly.")
    except subprocess.CalledProcessError as e:
        print(f"\nError checking Docker status: {e}")
        print("Please ensure Docker is properly installed and running.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a containerized CNN for MNIST")
    parser.add_argument(
        "--action",
        choices=["start", "stop", "test", "dashboard", "check"],
        default="start",
        help="Action to perform: start, stop, test the network, open dashboard, or check Docker status",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit the number of neuron containers (for testing)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode for troubleshooting"
    )

    args = parser.parse_args()

    if args.action == "start":
        check_prerequisites()
        start_network(args.limit, args.debug)
    elif args.action == "stop":
        stop_network()
    elif args.action == "test":
        test_network()
    elif args.action == "dashboard":
        open_dashboard()
    elif args.action == "check":
        check_docker_status()
