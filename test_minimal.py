import subprocess
import os
import time


def test_minimal_docker():
    """Test minimal Docker functionality with a single container."""
    print("Testing minimal Docker functionality...")

    # Create minimal docker-compose.yml
    minimal_compose = """services:
  test-container:
    image: hello-world
    container_name: test-minimal-container
"""

    # Write the file
    with open("docker-compose.minimal.yml", "w") as f:
        f.write(minimal_compose)

    try:
        # Check Docker status
        print("Checking Docker status...")
        docker_info = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, check=False
        )

        if docker_info.returncode != 0:
            print("ERROR: Docker doesn't appear to be running!")
            print(docker_info.stderr)
            return False

        print("Docker is running. Testing with a minimal container...")

        # Try with newer Docker Compose V2
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                "docker-compose.minimal.yml",
                "up",
                "--no-log-prefix",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print("Failed with Docker Compose V2, trying older docker-compose...")
            result = subprocess.run(
                ["docker-compose", "-f", "docker-compose.minimal.yml", "up"],
                capture_output=True,
                text=True,
                check=False,
            )

        # Print the output from hello-world container
        print("\nContainer output:")
        print(result.stdout)

        # Check if container ran successfully
        if "Hello from Docker!" in result.stdout:
            print("SUCCESS: Docker container ran correctly!")
            success = True
        else:
            print("ERROR: Docker container did not run as expected.")
            print(f"Return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            success = False

        # Clean up
        print("\nCleaning up...")
        cleanup = subprocess.run(
            ["docker", "compose", "-f", "docker-compose.minimal.yml", "down"],
            capture_output=True,
            text=True,
            check=False,
        )

        return success

    except Exception as e:
        print(f"Error during test: {e}")
        return False
    finally:
        # Clean up the file
        if os.path.exists("docker-compose.minimal.yml"):
            os.unlink("docker-compose.minimal.yml")


def suggest_fixes(success):
    """Suggest fixes based on test result."""
    if success:
        print("\n===== DOCKER IS WORKING CORRECTLY =====")
        print("Suggestions for the full app:")
        print(
            "1. Try running with fewer containers first: python run_network.py --limit 10"
        )
        print("2. Use debug mode: python run_network.py --debug")
        print("3. Check system resources (memory, disk space)")
    else:
        print("\n===== DOCKER NEEDS TROUBLESHOOTING =====")
        print("Suggestions:")
        print("1. Restart Docker Desktop completely")
        print("2. Check Docker Desktop settings (memory, CPU allocation)")
        print("3. Look for error messages in Docker Desktop")
        print("4. Make sure you have enough disk space")
        print("5. Check if any firewall/antivirus is blocking Docker")
        print("6. Reinstall Docker Desktop if problems persist")


if __name__ == "__main__":
    success = test_minimal_docker()
    suggest_fixes(success)
