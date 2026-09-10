import os
import re
import subprocess
import csv
import concurrent.futures
import queue

def discover_cavities(template_file):
    """Scans the template and returns a list of all cavity names."""
    cavities = []
    with open(template_file, 'r') as file:
        for line in file:
            match = re.match(r'^\s*"?([A-Z0-9_]+)"?\s*:', line)
            if match:
                element = match.group(1).upper()
                if element.startswith("CAV"):
                    cavities.append(element)
    return cavities

def modify_for_failure(template_file, output_file, broken_cavity):
    """Creates an OPAL input file where only the target cavity is set to 1e-6."""
    modified_lines = []
    with open(template_file, 'r') as file:
        for line in file:
            match = re.match(r'^\s*"?([A-Z0-9_]+)"?\s*:', line)
            if match and match.group(1).upper() == broken_cavity:
                line = re.sub(r'(VOLT\s*=\s*)[^,;]+', r'\g<1>1.0e-6', line, flags=re.IGNORECASE)
            modified_lines.append(line)

    with open(output_file, 'w') as file:
        file.writelines(modified_lines)

def run_opal(input_file, opal_bin='opal', host_id=1, timeout_sec=2000):
    """Executes OPAL concurrently using rsmpi."""
    # Substituted the command to use rsmpi with the dynamically assigned host_id
    command = ["rsmpi", "-n", "16", "-h", str(host_id), opal_bin, input_file]
    try:
        subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, check=True, timeout=timeout_sec
        )
        return True
    except subprocess.TimeoutExpired:
        print(f"  [!] TIMEOUT on {input_file} (Host {host_id}).")
        return False
    except subprocess.CalledProcessError:
        print(f"  [!] CRASH on {input_file} (Host {host_id}).")
        return False
    except FileNotFoundError:
        print(f"  [!] Executable '{command[0]}' not found.")
        return False

def analyze_stat_file(stat_file, target_zstop=148.345, z_tolerance=0.5):
    """Parses the .stat file, treating early termination as 100% loss."""
    if not os.path.exists(stat_file):
        return {"status": "NO_STAT_FILE"}

    col_names = []
    in_column = False

    with open(stat_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith('&column'):
            in_column = True
        elif in_column and line_stripped.startswith('name='):
            name = line_stripped.split('=')[1].strip(',"\'').lower()
            col_names.append(name)
        elif line_stripped.startswith('&end') and in_column:
            in_column = False

    expected_cols = len(col_names)
    data_rows = []

    for line in lines:
        parts = line.split()
        if len(parts) == expected_cols:
            try:
                data_rows.append([float(x) for x in parts])
            except ValueError:
                pass

    if not data_rows:
        return {"status": "EMPTY_STAT_FILE"}

    try:
        idx_s = col_names.index('s')
        idx_particles = col_names.index('numparticles')
        idx_eps_x = col_names.index('rms_x')
        idx_eps_y = col_names.index('rms_y')
        idx_eps_z = col_names.index('rms_s')
    except ValueError:
        return {"status": "MISSING_COLUMNS"}

    initial_particles = data_rows[0][idx_particles]
    final_particles = data_rows[-1][idx_particles]
    final_s = data_rows[-1][idx_s]

    if final_s < (target_zstop - z_tolerance):
        total_loss = initial_particles
        fatal_crash = True
    else:
        total_loss = initial_particles - final_particles
        fatal_crash = False

    final_eps_x = data_rows[-1][idx_eps_x]
    final_eps_y = data_rows[-1][idx_eps_y]
    final_eps_z = data_rows[-1][idx_eps_z]

    loss_start_s = None
    for row in data_rows:
        if row[idx_particles] < initial_particles:
            loss_start_s = row[idx_s]
            break

    if fatal_crash and loss_start_s is None:
        loss_start_s = final_s

    return {
        "status": "SUCCESS",
        "initial_particles": initial_particles,
        "final_particles": 0 if fatal_crash else final_particles,
        "total_lost": total_loss,
        "loss_start_s": loss_start_s if loss_start_s is not None else "No Loss",
        "final_s": final_s,
        "reached_end": not fatal_crash,
        "final_eps_x": final_eps_x,
        "final_eps_y": final_eps_y,
        "final_eps_z": final_eps_z,
        "emit_sum": final_eps_x + final_eps_y
    }

def process_cavity(cavity, template_file, host_queue):
    """Worker function to process a single cavity using an available host."""
    # 1. Wait for an available host ID
    host_id = host_queue.get()

    try:
        print(f"--- [Host {host_id}] Starting {cavity} ---")
        run_file = f"fail_{cavity}.i"
        stat_file = f"fail_{cavity}.stat"

        # Prepare and Run
        modify_for_failure(template_file, run_file, cavity)
        success = run_opal(run_file, host_id=host_id)

        # Analyze
        if success:
            data = analyze_stat_file(stat_file)
        else:
            data = {"status": "SIMULATION_FAILED"}

        result_row = {
            "Broken_Cavity": cavity,
            "Status": data.get("status"),
            "Reached_End": data.get("reached_end", "N/A"),
            "Final_S": data.get("final_s", "N/A"),
            "Total_Lost": data.get("total_lost", "N/A"),
            "Loss_Location_s": data.get("loss_start_s", "N/A"),
            "Final_Eps_X": data.get("final_eps_x", "N/A"),
            "Final_Eps_Y": data.get("final_eps_y", "N/A"),
            "Final_Eps_Z": data.get("final_eps_z", "N/A"),
            "Emit_Sum": data.get("emit_sum", "N/A")
        }

        if data.get("status") == "SUCCESS":
            status_msg = f"OK" if result_row['Reached_End'] else f"CRASH at {result_row['Final_S']:.2f}m"
            print(f"--- [Host {host_id}] Finished {cavity} | Status: {status_msg} | Lost: {result_row['Total_Lost']} ---")

        # Cleanup temporary input file
#        if os.path.exists(run_file):
#            os.remove(run_file)

        return result_row

    finally:
        # Always return the host ID to the queue so the next simulation can use it
        host_queue.put(host_id)

def main():
    template_file = "template.i"
    if not os.path.exists(template_file):
        print(f"Error: {template_file} not found.")
        return

    cavities = discover_cavities(template_file)
    print(f"Found {len(cavities)} cavities to test.")

    # -------------------------------------------------------------
    # PARALLEL EXECUTION SETUP
    # Change this number to the maximum concurrent hosts you have
    # -------------------------------------------------------------
    NUM_CONCURRENT_HOSTS = 8

    # Pre-fill the queue with host IDs (e.g., 1, 2, 3, 4)
    host_queue = queue.Queue()
    for i in range(1, NUM_CONCURRENT_HOSTS + 1):
        host_queue.put(i)

    results = []

    print(f"Launching {NUM_CONCURRENT_HOSTS} parallel workers...\n")

    # Create a ThreadPoolExecutor with exactly as many workers as hosts
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CONCURRENT_HOSTS) as executor:
        # Submit all tasks to the executor
        futures = {executor.submit(process_cavity, cav, template_file, host_queue): cav for cav in cavities}

        # As each task completes, append its result to our list
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"An error occurred: {e}")

    # Sort results to ensure the CSV is nicely ordered (CAV001, CAV002, etc.)
    results = sorted(results, key=lambda x: x["Broken_Cavity"])

    # Write Catalog to CSV
    csv_file = "cavity_failure_catalog.csv"
    if results:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\nAll parallel runs complete! Results saved to '{csv_file}'.")

if __name__ == "__main__":
    main()
