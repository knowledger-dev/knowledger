# fine_tune_rag.py

import os
import subprocess
import time
import requests
import json
import csv
import logging
from itertools import product
from typing import Dict, Any, List
import socket

# Configure Logging
logging.basicConfig(
    filename='fine_tune_rag.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def frange(start: float, stop: float, step: float):
    """
    A generator for floating point ranges.
    """
    while start <= stop:
        yield round(start, 4)
        start += step

def is_port_free(host: str, port: int) -> bool:
    """
    Checks if a given port is free on the specified host.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        return result != 0  # True if port is free

def wait_until_port_free(host: str, port: int, timeout: int = 30) -> bool:
    """
    Waits until the specified port is free or until timeout.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_free(host, port):
            return True
        time.sleep(1)
    return False

def wait_until_port_open(host: str, port: int, timeout: int = 30) -> bool:
    """
    Waits until the specified port is open (app is running) or until timeout.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not is_port_free(host, port):
            return True
        time.sleep(1)
    return False

def set_environment_variables(config: Dict[str, Any]):
    """
    Sets the environment variables based on the provided configuration dictionary.
    """
    for key, value in config.items():
        os.environ[key] = str(value)
    logging.debug(f"Environment variables set: {config}")

def start_app():
    """
    Starts the FastAPI app using uvicorn in a subprocess.
    """
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))

    # Check if port is free before starting
    if not is_port_free(host, port):
        logging.warning(f"Port {port} is not free. Waiting for it to be free...")
        if not wait_until_port_free(host, port, timeout=30):
            logging.error(f"Port {port} is still in use after waiting. Cannot start the app.")
            return None

    app_process = subprocess.Popen(
        [
            "uvicorn",
            "app:app",
            "--host",
            host,
            "--port",
            str(port)
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logging.info("Started FastAPI app.")

    # Wait for the server to start
    if wait_until_port_open(host, port, timeout=30):
        logging.info("FastAPI app is running.")
        return app_process
    else:
        logging.error("FastAPI app failed to start within the timeout period.")
        app_process.terminate()
        return None

def stop_app(app_process):
    """
    Terminates the FastAPI app subprocess.
    """
    if app_process:
        app_process.terminate()
        try:
            app_process.wait(timeout=10)
            logging.info("Stopped FastAPI app.")
        except subprocess.TimeoutExpired:
            app_process.kill()
            logging.warning("Force killed FastAPI app after timeout.")

        # Ensure port is free before proceeding
        host = os.getenv("APP_HOST", "127.0.0.1")
        port = int(os.getenv("APP_PORT", "8000"))
        if not wait_until_port_free(host, port, timeout=30):
            logging.error(f"Port {port} is still in use after stopping the app.")

def test_rag_endpoint(query: str, max_tokens: int) -> Dict[str, Any]:
    """
    Sends a RAG query to the /rag_query endpoint and returns the response.
    """
    host = os.getenv('APP_HOST', '127.0.0.1')
    port = os.getenv('APP_PORT', '8000')
    base_url = f"http://{host}:{port}"
    endpoint = f"{base_url}/rag_query"
    payload = {
        "query": query,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(endpoint, json=payload, timeout=60)  # Increased timeout for longer responses
        response.raise_for_status()
        logging.debug(f"RAG query successful: {payload}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"RAG query failed: {payload} | Error: {e}")
        return {"error": str(e)}

import random

def generate_parameter_configs(max_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Generates a list of parameter configurations by randomly sampling from the defined parameter ranges.
    The total number of configurations is limited to 'max_samples'.

    Args:
        max_samples (int): Maximum number of parameter configurations to generate.

    Returns:
        List[Dict[str, Any]]: A list of parameter configuration dictionaries.
    """
    parameter_configs = []

    # Define fine-grained increments for each parameter
    similarity_thresholds_recalc = [round(x, 4) for x in frange(0.35, 0.425, 0.005)]  # 0.35 to 0.425 step 0.005 (16 values)
    similarity_thresholds_update = [round(x, 4) for x in frange(0.35, 0.425, 0.005)]  # 0.35 to 0.425 step 0.005 (16 values)
    dbscan_eps_values = [round(x, 4) for x in frange(1.10, 1.25, 0.005)]               # 1.10 to 1.25 step 0.005 (31 values)
    dbscan_min_samples = [1, 2, 3, 4, 5, 6]                                         # 1 to 6 (6 values)
    pagerank_alpha = [round(x, 4) for x in frange(0.80, 0.90, 0.005)]                # 0.80 to 0.90 step 0.005 (21 values)
    similarity_threshold_rag = [round(x, 4) for x in frange(0.05, 0.15, 0.005)]      # 0.05 to 0.15 step 0.005 (21 values)

    # Log the parameter ranges
    logging.info("Generating parameter lists:")
    logging.info(f"similarity_thresholds_recalc: {similarity_thresholds_recalc}")
    logging.info(f"similarity_thresholds_update: {similarity_thresholds_update}")
    logging.info(f"dbscan_eps_values: {dbscan_eps_values}")
    logging.info(f"dbscan_min_samples: {dbscan_min_samples}")
    logging.info(f"pagerank_alpha: {pagerank_alpha}")
    logging.info(f"similarity_threshold_rag: {similarity_threshold_rag}")

    # Calculate total possible combinations
    total_combinations = (
        len(similarity_thresholds_recalc) *
        len(similarity_thresholds_update) *
        len(dbscan_eps_values) *
        len(dbscan_min_samples) *
        len(pagerank_alpha) *
        len(similarity_threshold_rag)
    )
    logging.info(f"Total possible parameter combinations: {total_combinations}")

    # Set a seed for reproducibility
    random.seed(42)

    # If total combinations <= max_samples, return all combinations
    if total_combinations <= max_samples:
        logging.info("Total combinations less than or equal to max_samples. Generating all possible configurations.")
        for st_recalc, st_update, eps, min_samples, alpha, rag_threshold in product(
            similarity_thresholds_recalc,
            similarity_thresholds_update,
            dbscan_eps_values,
            dbscan_min_samples,
            pagerank_alpha,
            similarity_threshold_rag
        ):
            config = {
                "SIMILARITY_THRESHOLD_RECALCULATE_ALL": st_recalc,
                "SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS": st_update,
                "DBSCAN_EPS": eps,
                "DBSCAN_MIN_SAMPLES": min_samples,
                "PAGERANK_ALPHA": alpha,
                "SIMILARITY_THRESHOLD_RAG": rag_threshold
            }
            parameter_configs.append(config)
        logging.info(f"Generated {len(parameter_configs)} configurations.")
        return parameter_configs

    # Otherwise, randomly sample configurations
    logging.info(f"Sampling {max_samples} configurations out of {total_combinations} possible.")
    sampled_indices = set()
    attempts = 0
    max_attempts = max_samples * 10  # Prevent infinite loops

    while len(parameter_configs) < max_samples and attempts < max_attempts:
        st_recalc = random.choice(similarity_thresholds_recalc)
        st_update = random.choice(similarity_thresholds_update)
        eps = random.choice(dbscan_eps_values)
        min_samples = random.choice(dbscan_min_samples)
        alpha = random.choice(pagerank_alpha)
        rag_threshold = random.choice(similarity_threshold_rag)

        config = {
            "SIMILARITY_THRESHOLD_RECALCULATE_ALL": st_recalc,
            "SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS": st_update,
            "DBSCAN_EPS": eps,
            "DBSCAN_MIN_SAMPLES": min_samples,
            "PAGERANK_ALPHA": alpha,
            "SIMILARITY_THRESHOLD_RAG": rag_threshold
        }

        # Convert config to a tuple of sorted items for uniqueness
        config_tuple = tuple(sorted(config.items()))
        if config_tuple not in sampled_indices:
            sampled_indices.add(config_tuple)
            parameter_configs.append(config)
            logging.debug(f"Sampled config: {config}")
        attempts += 1

    logging.info(f"Generated {len(parameter_configs)} sampled configurations.")

    if len(parameter_configs) < max_samples:
        logging.warning(f"Only generated {len(parameter_configs)} unique configurations after {attempts} attempts.")

    return parameter_configs

def initialize_csv(filename: str, fieldnames: List[str]):
    """
    Initializes the CSV file with headers.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    logging.info(f"Initialized CSV file: {filename}")

def append_to_csv(filename: str, data: Dict[str, Any], fieldnames: List[str]):
    """
    Appends a row of data to the CSV file.
    """
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(data)
    logging.debug(f"Appended data to CSV: {data}")

def main(iteration_num: int):
    # Define CSV filename and headers
    csv_filename = f"rag_test_results_simulation_{iteration_num}.csv"
    fieldnames = [
        "Test_ID",
        "SIMILARITY_THRESHOLD_RECALCULATE_ALL",
        "SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS",
        "DBSCAN_EPS",
        "DBSCAN_MIN_SAMPLES",
        "PAGERANK_ALPHA",
        "SIMILARITY_THRESHOLD_RAG",
        "Answer",
        "Referenced_Note_IDs",
        "Error"
    ]

    # Initialize CSV
    initialize_csv(csv_filename, fieldnames)

    # Generate all parameter configurations
    parameter_configs = generate_parameter_configs()

    logging.info("Starting RAG endpoint tests...")

    # Define the RAG query to test
    test_query = "Explain the progress for our machine-learning based medical solutions."

    # Buffer settings
    buffer_interval = 3  # Number of tests after which to insert a buffer
    buffer_duration = 30  # Duration of the buffer in seconds

    # Iterate over each configuration
    for idx, config in enumerate(parameter_configs, 1):
        logging.info(f"Testing configuration {idx}/{len(parameter_configs)}: {config}")

        # Set environment variables
        set_environment_variables(config)

        # Start the FastAPI app
        app_process = start_app()

        if app_process is None:
            # App failed to start
            csv_data = {
                "Test_ID": idx,
                "SIMILARITY_THRESHOLD_RECALCULATE_ALL": config["SIMILARITY_THRESHOLD_RECALCULATE_ALL"],
                "SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS": config["SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS"],
                "DBSCAN_EPS": config["DBSCAN_EPS"],
                "DBSCAN_MIN_SAMPLES": config["DBSCAN_MIN_SAMPLES"],
                "PAGERANK_ALPHA": config["PAGERANK_ALPHA"],
                "SIMILARITY_THRESHOLD_RAG": config["SIMILARITY_THRESHOLD_RAG"],
                "Answer": "",
                "Referenced_Note_IDs": "",
                "Error": "Failed to start FastAPI app"
            }
            append_to_csv(csv_filename, csv_data, fieldnames)
            logging.error(f"Configuration {idx} failed to start FastAPI app.")
            continue

        try:
            # Perform the RAG query
            response = test_rag_endpoint(query=test_query, max_tokens=32768)

            # Prepare data for CSV
            csv_data = {
                "Test_ID": idx,
                "SIMILARITY_THRESHOLD_RECALCULATE_ALL": config["SIMILARITY_THRESHOLD_RECALCULATE_ALL"],
                "SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS": config["SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS"],
                "DBSCAN_EPS": config["DBSCAN_EPS"],
                "DBSCAN_MIN_SAMPLES": config["DBSCAN_MIN_SAMPLES"],
                "PAGERANK_ALPHA": config["PAGERANK_ALPHA"],
                "SIMILARITY_THRESHOLD_RAG": config["SIMILARITY_THRESHOLD_RAG"],
                "Answer": response.get("answer", ""),
                "Referenced_Note_IDs": json.dumps(response.get("referenced_note_ids", [])),
                "Error": response.get("error", "")
            }

            # Append results to CSV
            append_to_csv(csv_filename, csv_data, fieldnames)

            # Log success or error
            if "error" not in response or response["error"] == "":
                logging.info(f"Configuration {idx} passed.")
            else:
                logging.warning(f"Configuration {idx} returned an error: {response['error']}")

        except Exception as e:
            # In case of unexpected errors, log and record the error
            logging.error(f"Configuration {idx} failed with exception: {e}")
            csv_data = {
                "Test_ID": idx,
                "SIMILARITY_THRESHOLD_RECALCULATE_ALL": config["SIMILARITY_THRESHOLD_RECALCULATE_ALL"],
                "SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS": config["SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS"],
                "DBSCAN_EPS": config["DBSCAN_EPS"],
                "DBSCAN_MIN_SAMPLES": config["DBSCAN_MIN_SAMPLES"],
                "PAGERANK_ALPHA": config["PAGERANK_ALPHA"],
                "SIMILARITY_THRESHOLD_RAG": config["SIMILARITY_THRESHOLD_RAG"],
                "Answer": "",
                "Referenced_Note_IDs": "",
                "Error": str(e)
            }
            append_to_csv(csv_filename, csv_data, fieldnames)

        finally:
            # Stop the FastAPI app
            stop_app(app_process)

            # Buffer logic: Add a pause every 'buffer_interval' tests
            if idx % buffer_interval == 0:
                logging.info(f"Buffering for {buffer_duration} seconds after {idx} tests to ensure app stability.")
                time.sleep(buffer_duration)
                logging.info("Buffer period ended. Resuming tests.")

    logging.info("All RAG endpoint tests completed. Results saved to 'rag_test_results.csv'.")

if __name__ == "__main__":
    main(1)
