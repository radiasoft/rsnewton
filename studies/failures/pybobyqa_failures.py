import re
import os
import subprocess
import sys
import numpy as np
import pybobyqa
import csv

# ---------------------------------------------------------
# 1. Helper Functions (Modification & Execution)
# ---------------------------------------------------------

def modify_opal_file(input_file, output_file, parameter_changes):
    """
    Reads an OPAL input file and modifies parameters based on a dictionary of changes.
    """
    if not os.path.exists(input_file):
        print(f"Error: Could not find '{input_file}'.")
        sys.exit(1)

    modified_lines = []
    with open(input_file, 'r') as file:
        for line in file:
            match = re.match(r'^"([A-Z0-9]+)"\s*:', line)
            if match:
                element_name = match.group(1)
                if element_name in parameter_changes:
                    changes = parameter_changes[element_name]
                    for param, new_value in changes.items():
                        pattern = rf'({param}\s*=\s*)[^,;]+'
                        replacement = rf'\g<1>{new_value}'
                        line = re.sub(pattern, replacement, line, flags=re.IGNORECASE)
            modified_lines.append(line)

    with open(output_file, 'w') as file:
        file.writelines(modified_lines)

def run_opal_simulation(input_file, opal_bin='opal'):
    """
    Executes the OPAL simulation silently.
    """
    command = ["rsmpi","-n","16","-h","1",opal_bin, input_file]
    #command = [opal_bin, input_file]  #Serial mode
    try:
        # check=True forces an exception if OPAL returns a non-zero exit code
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )
        return True

    except subprocess.CalledProcessError as e:
        # Intercept the crash (e.g., exit status 156)
        print(f"\n[!] OPAL CRASH DETECTED on {input_file}")
        print(f"[!] Exit Status: {e.returncode}")
        print("-" * 40)
        print("OPAL ERROR LOG:")
        print(e.output)  # This contains the actual text output from OPAL explaining the failure
        print("-" * 40 + "\n")

        return False

    except FileNotFoundError:
        print(f"[!] Error: Could not find the executable '{opal_bin}'. Is it in your PATH?")
        return False
    return True

# ---------------------------------------------------------
# 2. Penalty Extraction (.stat parsing)
# ---------------------------------------------------------

import os

def extract_penalty(stat_file, loss_weight=1e3, emit_weight=1):
    """
    Parses an OPAL .stat file and returns a single optimization penalty float.

    Args:
        stat_file (str): Path to the .stat file.
        loss_weight (float): Multiplier for the fraction of particles lost.
        emit_weight (float): Multiplier for the summed 6D emittance.

    Returns:
        float: The calculated penalty (lower is better).
    """
    # Massive "Death Penalty" if simulation failed to even write a file
    if not os.path.exists(stat_file):
        print("Simulation failed to write")
        return 1e6

    with open(stat_file, 'r') as file:
        lines = file.readlines()

    # --- Robust Stat File Parser ---
    col_names = []
    in_column = False
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith('&column'):
            in_column = True
        elif in_column and line_stripped.startswith('name='):
            col_names.append(line_stripped.split('=')[1].strip(',"\'').lower())
        elif line_stripped.startswith('&end') and in_column:
            in_column = False

    try:
        idx_s = col_names.index('s')
        idx_particles = col_names.index('numparticles')
        idx_eps_x = col_names.index('rms_x')
        idx_eps_y = col_names.index('rms_y')
        idx_eps_z = col_names.index('rms_s')
    except ValueError:
        print("Failed to read stat file")
        return 1e6  # Missing critical columns

    # Extract first and last data rows
    expected_cols = len(col_names)
    data_rows = [ [float(x) for x in line.split()] for line in lines if len(line.split()) == expected_cols ]

    if not data_rows:
        print("No data rows")
        return 1e6

    initial_particles = data_rows[0][idx_particles]
    final_row = data_rows[-1]
    final_s = final_row[idx_s]

    # --- Constraint 1: Fatal Crashing ---
    target_s = 148.0 # Adjust to your exact ZSTOP if needed
    if final_s < target_s:
        print("Didn't reach end of beamline")
        # Base penalty + penalty for how early the beam crashed
        distance_short = target_s - final_s
        return 1e5 + (distance_short * loss_weight)

    # --- Extract Final Values ---
    final_particles = final_row[idx_particles]

    # Scale emittances by 1e6 (to mm-mrad) so they aren't trivially small floats
    eps_x = final_row[idx_eps_x] * 1e6
    eps_y = final_row[idx_eps_y] * 1e6
    eps_z = final_row[idx_eps_z] * 1e6

    # --- Calculate Transmission ---
    transmission = final_particles / initial_particles
    loss_fraction = 1.0 - transmission

    # --- Formulate the Penalty ---

    # 1. Loss Penalty
    # Linearly scaled by your independent weight (e.g., if weight is 10,000 and you lose 10%, adds 1,000)
    loss_penalty = loss_weight * loss_fraction

    # 2. Emittance Penalty
    # Sum the emittances. We divide by transmission to severely punish "scraper cheating"
    raw_emit_sum = eps_x + eps_y + eps_z
    emit_penalty = emit_weight * (raw_emit_sum / max(transmission, 0.01))

    total_penalty = loss_penalty + emit_penalty

    return total_penalty


# ---------------------------------------------------------
# 3. The BOBYQA Objective & Setup
# ---------------------------------------------------------

# The cavity we are forcing to 0 (NOT optimized by BOBYQA)
FIXED_ZERO_ELEMENT = "CAV005"
FIXED_ZERO_PARAM = "volt"

    # The parameters BOBYQA WILL optimize to compensate
TARGETS = [
    ("CAV001","lag"),
    ("CAV001", "volt"),
    ("CAV002", "lag"),
    ("CAV002", "volt"),
    ("CAV003","lag"),
    ("CAV003", "volt"),
    ("CAV004","lag"),
    ("CAV004", "volt"),
    ("CAV006","lag"),
    ("CAV006", "volt"),
    ("CAV007","lag"),
    ("CAV007", "volt"),
]

iteration_count = 0

def objective_function(x):
    global iteration_count
    iteration_count += 1

    # 1. Map the flat array 'x' back into the dictionary structure
    changes = {}
    for value, (element, param) in zip(x, TARGETS):
        if element not in changes:
            changes[element] = {}
        changes[element][param] = round(value, 6)

    # 2. Inject the fixed zero parameter so it always applies
    if FIXED_ZERO_ELEMENT not in changes:
        changes[FIXED_ZERO_ELEMENT] = {}
    changes[FIXED_ZERO_ELEMENT][FIXED_ZERO_PARAM] = 0.000001

    # 3. Write and run
    template_file = "template.i"
    run_file = "opt_run.i"
    modify_opal_file(template_file, run_file, changes)

    success = run_opal_simulation(run_file)
    if not success:
        penalty = 1e6
        print(f"Iter {iteration_count:03d} | Crash! Returning large penalty.")
    else:
        # 4. Calculate Penalty
        stat_file = run_file.replace('.i', '.stat')
        penalty = extract_penalty(stat_file, loss_weight=1e3, emit_weight=1.0)
        print(f"Iter {iteration_count:03d} | Params: {x} | Penalty: {penalty:.4e}")

    # 5. Write iteration data to CSV immediately (Appends per run)
    with open("optimization_history.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        row_data = [iteration_count, penalty] + list(x)
        writer.writerow(row_data)

    return penalty

if __name__ == "__main__":
    # ---------------------------------------------------------
    # Custom Logger for File and Console Output
    # ---------------------------------------------------------
    class DualLogger:
        """Writes print statements to both the terminal and a log file."""
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    # Redirect all print() statements to our dual logger
    log_filename = "optimization_log.txt"
    sys.stdout = DualLogger(log_filename)

    # ---------------------------------------------------------
    # 4. Run BOBYQA
    # ---------------------------------------------------------
    if not os.path.exists("template.i"):
        print("Error: Please save your original OPAL script as 'template.i' in this folder.")
        sys.exit(1)

    # Initial warm-start guesses for TARGETS [lag, volt] * 7, [ks] * 6
    # Grab these directly from your previously optimized values
    x0 = np.array([-0.67631509,1.35,  #CAV001
                   -0.52429691, 1.7,  #CAV002
                   -0.52482051, 2.05, #CAV003
                   -0.54558992, 2.05,  #CAV004
                   -0.60580378, 2.75,  #CAV005
                   -0.44121923, 2.75,  #CAV007
])

    # Define physical limits (Lower Bounds, Upper Bounds)
#    lower_bounds = np.array([0.0, -1.0, 0.0, 0.0])
#    upper_bounds = np.array([15.0, 1.0, 10.0, 10.0])
    lower_bounds = np.array([-1.0, 0.0]*6)
    upper_bounds = np.array([+1.0, 4]*6)
    print(x0)
    print(lower_bounds)
    print(upper_bounds)
    print(f"Forcing {FIXED_ZERO_ELEMENT} {FIXED_ZERO_PARAM} to 0.0")
    print(f"Starting BOBYQA Optimization... (Logging to {log_filename})")
    print("-" * 50)

    # ---------------------------------------------------------
    # CSV Logging Setup
    # ---------------------------------------------------------
    csv_filename = "optimization_history.csv"

    # Build dynamic header: Iteration, Penalty, then names of all target parameters
    csv_header = ["Iteration", "Penalty"] + [f"{elem}_{param}" for elem, param in TARGETS]

    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    # Execute PyBobyqa
    result = pybobyqa.solve(
        objective_function,
        x0,
        bounds=(lower_bounds, upper_bounds),
        maxfun=200,
        rhobeg=0.1,
        rhoend=1e-5,
        print_progress=False
    )

    print("-" * 50)
    print("Optimization Complete!")
    print(f"Best Penalty Achieved: {result.f}")
    print("Optimal Compensating Parameters:")
    for value, (element, param) in zip(result.x, TARGETS):
        print(f"  {element} {param} = {value:.6f}")

    # Close the log file gracefully
    sys.stdout.log.close()
    # Restore standard output
    sys.stdout = sys.stdout.terminal
