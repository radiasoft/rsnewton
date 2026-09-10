import os
import re
import subprocess
import concurrent.futures
import queue

def discover_cavities(template_file="template.i"):
    """Extracts a list of all cavity names from the template."""
    cavities = []
    if not os.path.exists(template_file):
        print(f"[!] Error: {template_file} not found.")
        return cavities

    with open(template_file, 'r') as file:
        for line in file:
            match = re.match(r'^\s*"?([A-Z0-9_]+)"?\s*:', line)
            if match and match.group(1).upper().startswith("CAV"):
                cavities.append(match.group(1).upper())
    return cavities

def optimize_worker(cavity, host_queue):
    """Worker thread function that claims a host ID and runs the optimizer."""
    # Block until a host ID becomes available
    host_id = host_queue.get()

    print(f"[Node {host_id}] Starting optimization campaign for {cavity}...")

    try:
        # Call the PyBOBYQA script with the specific cavity and host ID
        subprocess.run(
            ["python", "optimize_rf_recovery.py", cavity, str(host_id)],
            check=True
        )
        print(f"[Node {host_id}] Successfully finished {cavity}.")
    except subprocess.CalledProcessError:
        print(f"[!] PyBOBYQA script failed for {cavity} on Node {host_id}.")
    finally:
        # Always release the host ID back to the queue for the next thread
        host_queue.put(host_id)

def main():
    NUM_NODES = 3

    cavities = discover_cavities()
    if not cavities:
        return

    print(f"Discovered {len(cavities)} cavities. Initiating {NUM_NODES}-node parallel optimization.\n")

    # Initialize the queue with available host IDs (1 through 8)
    host_queue = queue.Queue()
    for i in range(1, NUM_NODES + 1):
        host_queue.put(i)

    # Launch the thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_NODES) as executor:
        futures = [executor.submit(optimize_worker, cav, host_queue) for cav in cavities]

        # Wait for all to complete
        concurrent.futures.wait(futures)

    print("\n=== All cavity optimization campaigns completed ===")

if __name__ == "__main__":
    main()
