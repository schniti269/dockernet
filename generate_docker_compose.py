import json
import os
import glob


def generate_docker_compose():
    """Generate a docker-compose.yml file for the containerized neural network."""
    # Check if network_info.json exists
    if not os.path.exists("neuron_params/network_info.json"):
        print(
            "Error: neuron_params/network_info.json not found. Run extract_params.py first."
        )
        return

    # Load network structure
    with open("neuron_params/network_info.json", "r") as f:
        network_info = json.load(f)

    # Start building docker-compose file
    docker_compose = """services:
  input-container:
    build:
      context: .
      dockerfile: Dockerfile.input
    ports:
      - "8000:8000"
    environment:
      - NEXT_LAYER_HOSTS=${CONV1_HOSTS}
      - MONITORING_HOST=monitoring-service:9090
    volumes:
      - ./data:/app/data
    networks:
      - neural-network

  output-container:
    build:
      context: .
      dockerfile: Dockerfile.output
    ports:
      - "8001:8000"
    environment:
      - INPUT_CONTAINER_HOST=input-container:8000
      - MONITORING_HOST=monitoring-service:9090
    networks:
      - neural-network
      
  monitoring-service:
    build:
      context: .
      dockerfile: Dockerfile.monitoring
    ports:
      - "9090:9090"
    volumes:
      - ./neuron_params:/app/neuron_params
    networks:
      - neural-network
  
  web-ui:
    build:
      context: .
      dockerfile: Dockerfile.webui
    ports:
      - "8050:8050"
    volumes:
      - ./neuron_params:/app/neuron_params
    networks:
      - neural-network
    depends_on:
      - monitoring-service
      - input-container
      - output-container
"""

    # Get structure for each layer
    structure = network_info["structure"]

    # Track all services for each layer
    layer_services = {"conv1": [], "conv2": [], "fc1": [], "fc2": []}

    # Generate service definitions for each neuron in each layer
    for layer in ["conv1", "conv2", "fc1", "fc2"]:
        if layer not in structure:
            continue

        neuron_count = structure[layer]["neurons"]

        for i in range(neuron_count):
            service_name = f"{layer}-neuron-{i}"
            layer_services[layer].append(service_name)

            # Determine next layer hosts
            next_layer = None
            if layer == "conv1":
                next_layer = "conv2"
            elif layer == "conv2":
                next_layer = "fc1"
            elif layer == "fc1":
                next_layer = "fc2"

            next_hosts = ""
            if layer == "fc2":  # Output layer connects to output container
                next_hosts = f"output-container:8000/collect/{i}"
            elif next_layer:
                # For demonstration, we connect to a subset of the next layer's neurons
                # In a real system, you'd have more sophisticated connection patterns
                subset_size = min(
                    5,
                    len(layer_services.get(next_layer, []))
                    or structure.get(next_layer, {}).get("neurons", 0),
                )
                next_hosts = ",".join(
                    [f"{next_layer}-neuron-{j}:8000" for j in range(subset_size)]
                )

            # Add service definition
            docker_compose += f"""
  {service_name}:
    build:
      context: .
      dockerfile: Dockerfile.neuron
    volumes:
      - ./neuron_params:/app/config
    environment:
      - NEURON_CONFIG_FILE=/app/config/{layer}_neuron_{i}.json
      - NEXT_LAYER_HOSTS={next_hosts}
      - MONITORING_HOST=monitoring-service:9090
    networks:
      - neural-network
"""

    # Add network definition
    docker_compose += """
networks:
  neural-network:
    driver: bridge
"""

    # Write to file
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose)

    # Generate .env file with connection strings
    with open(".env", "w") as f:
        # Get all conv1 neuron hosts for input container
        conv1_hosts = ",".join([f"{name}:8000" for name in layer_services["conv1"]])
        f.write(f"CONV1_HOSTS={conv1_hosts}\n")

    print(
        f"Generated docker-compose.yml with {sum(len(neurons) for neurons in layer_services.values())} neuron containers"
    )


if __name__ == "__main__":
    generate_docker_compose()
