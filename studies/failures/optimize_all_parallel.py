import os
import sys
import re
import subprocess
import concurrent.futures
import queue
import argparse
import time
import random

# --- CONFIGURATION ---
TEMPLATE_FILE = "template.i"   # The base linac lattice
END_ELEMENT = "D234"           # Your final linac element for rsnewton
CHECKPOINT_INTERVAL = 999       # Generate a checkpoint every X cavities
N_ADJACENT = 5                 # Must match optimize_rf_recovery.py!
NUM_NODES = 6
# ---------------------

def discover_cavities(template_file):
    """Extracts an ordered list of all cavity names from the template."""
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


def generate_checkpoints(cavities, host_queue, skip=False):
    """
    Runs rsnewton to pre-compute tracking up to specific checkpoints in PARALLEL.
    Uses sys.executable to bypass Pyenv/bash remote non-interactive shell traps.
    """
    checkpoints = {0: TEMPLATE_FILE}

    tasks = []
    for i in range(CHECKPOINT_INTERVAL - 1, len(cavities), CHECKPOINT_INTERVAL):
        cav = cavities[i]
        out_dir = f"{cav}_to_end"
        trimmed_template = os.path.join(out_dir, os.path.basename(TEMPLATE_FILE))

        checkpoints[i] = trimmed_template
        if not skip:
            tasks.append((i, cav, out_dir, trimmed_template))

    if skip or not tasks:
        return checkpoints

    print(f"\n[Checkpointing] Pre-computing {len(tasks)} checkpoints across compute nodes...")

    def checkpoint_worker(task):
        idx, cav, out_dir, trimmed_template = task

        host_id = host_queue.get()
        print(f"  -> [Node {host_id}] Starting checkpoint build for {cav}...")

        os.makedirs(out_dir, exist_ok=True)

        # 1. The exact Python interpreter currently running this master script
        current_python = sys.executable

        # 2. Ask pyenv (or the shell) where the real rsnewton file actually lives
        try:
            rsnewton_script = subprocess.check_output(["pyenv", "which", "rsnewton"], text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            import shutil
            rsnewton_script = shutil.which("rsnewton")

        if not rsnewton_script:
            print(f"  [!] [Node {host_id}] Error: Cannot locate the rsnewton script on the filesystem.")
            host_queue.put(host_id)
            return idx, False

        # 3. Explicitly run THAT Python interpreter against THAT script file
        cmd = [
            #"rsmpi", "-n", "1", "-h", str(host_id),
            current_python, rsnewton_script, "lattice", "trim-lattice",
            TEMPLATE_FILE, cav, END_ELEMENT,
            "--output-dir", out_dir,
            "--opal-bin", "opal",
            "--mpi-ranks", "16"
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            print(f"  -> [Node {host_id}] [Done] Checkpoint {cav} ready.")
            return idx, True
        except subprocess.CalledProcessError as e:
            print(f"  [!] [Node {host_id}] Failed to generate checkpoint for {cav}.")
            print(f"      --- Error Output ---")
            print("\n".join(e.stdout.strip().split("\n")[-15:]))
            print(f"      --------------------")
            return idx, False
        finally:
            host_queue.put(host_id)

    # Dispatch tasks to the compute nodes
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_NODES) as executor:
        futures = [executor.submit(checkpoint_worker, task) for task in tasks]

        for future in concurrent.futures.as_completed(futures):
            idx, success = future.result()
            if not success:
                # If a checkpoint fails, remove it from the pool so we safely fallback
                del checkpoints[idx]

    return checkpoints

def get_best_template(broken_idx, checkpoints):
    """
    Finds the latest possible checkpoint that doesn't trim off upstream tuning cavities.
    Floors at 0 to ensure the base template is safely returned for the first few cavities.
    """
    safe_start_idx = max(0, broken_idx - N_ADJACENT)
    valid_indices = [idx for idx in checkpoints.keys() if idx <= safe_start_idx]
    best_idx = max(valid_indices)
    return checkpoints[best_idx]


def optimize_worker(cavity, cavity_idx, template_to_use, host_queue):
    """Worker thread function that claims a host ID and runs the optimizer."""
    host_id = host_queue.get()

    # --- I/O De-synchronization ---
    # Apply a random 2 to 8 second jitter before launching. This staggers
    # the heavy initial HDF5 writes across the nodes so they don't hit the disk at once.
    jitter = random.uniform(0.0, 300.0)
    time.sleep(jitter)

    print(f"[Node {host_id}] Starting {cavity} (using {template_to_use})...")

    try:
        # Use sys.executable to ensure we use the same Pyenv python for the worker script
        subprocess.run(
            [sys.executable, "optimize_rf_recovery.py", cavity, str(host_id), template_to_use],
            check=True
        )
        print(f"[Node {host_id}] Successfully finished {cavity}.")
    except subprocess.CalledProcessError:
        print(f"[!] PyBOBYQA script failed for {cavity} on Node {host_id}.")
    finally:
        host_queue.put(host_id)


def main():
    parser = argparse.ArgumentParser(description="Parallel Linac Optimization")
    parser.add_argument('--skip-checkpoints', action='store_true', help="Skip rsnewton checkpoint generation")
    args = parser.parse_args()

    cavities = discover_cavities(TEMPLATE_FILE)
    if not cavities:
        return

    print(f"Discovered {len(cavities)} cavities.")

    # 1. Initialize the host queue first
    host_queue = queue.Queue()
    for i in range(1, NUM_NODES + 1):
        host_queue.put(i)

    # 2. Pass the queue to the checkpoint generator
    checkpoints = generate_checkpoints(cavities, host_queue, skip=args.skip_checkpoints)

    print(f"\nAvailable Checkpoint Indices: {list(checkpoints.keys())}")
    print(f"Initiating {NUM_NODES}-node parallel PyBOBYQA optimization...\n")

    # 3. Launch threads using the most efficient template for each cavity
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_NODES) as executor:
        futures = []
        for idx, cav in enumerate(cavities):
            best_template = get_best_template(idx, checkpoints)

            # Optional check: skip already completed optimizations
            if os.path.exists(f"optimal_{cav}.h5"):
                print(f"Skipping {cav} (optimal output already exists).")
                continue

            futures.append(executor.submit(optimize_worker, cav, idx, best_template, host_queue))

        concurrent.futures.wait(futures)

    print("\n=== All cavity optimization campaigns completed ===")


if __name__ == "__main__":
    main()
