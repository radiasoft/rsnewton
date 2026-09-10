import os
import re
import subprocess
import pybobyqa
import numpy as np
import math
import csv

# --- 1. Core Functions ---

def extract_penalty(stat_file, loss_weight=2000.0, emit_weight=1.0):
    """Calculates the penalty from a .stat file (Lower is better)."""
    if not os.path.exists(stat_file): return 1e6
    with open(stat_file, 'r') as file: lines = file.readlines()

    col_names = []
    in_column = False
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith('&column'): in_column = True
        elif in_column and line_stripped.startswith('name='): col_names.append(line_stripped.split('=')[1].strip(',"\'').lower())
        elif line_stripped.startswith('&end') and in_column: in_column = False

    try:
        idx_s = col_names.index('s')
        idx_particles = col_names.index('numparticles')
        idx_eps_x = col_names.index('rms_x')
        idx_eps_y = col_names.index('rms_y')
        idx_eps_z = col_names.index('rms_s')
    except ValueError: return 1e6

    data_rows = [ [float(x) for x in line.split()] for line in lines if len(line.split()) == len(col_names) ]
    if not data_rows: return 1e6

    initial_particles = data_rows[0][idx_particles]
    final_row = data_rows[-1]
    final_s = final_row[idx_s]

    target_s = 148.0
    if final_s < target_s:
        return 1e5 + ((target_s - final_s) * loss_weight)

    transmission = final_row[idx_particles] / initial_particles
    loss_fraction = 1.0 - transmission

    eps_x = final_row[idx_eps_x] * 1e6
    eps_y = final_row[idx_eps_y] * 1e6
    eps_z = final_row[idx_eps_z] * 1e6

    loss_penalty = loss_weight * (math.exp(loss_fraction * 10) - 1.0)
    emit_penalty = emit_weight * ((eps_x + eps_y + eps_z * 2.0) / max(transmission, 0.01))

    return loss_penalty + emit_penalty


def discover_cavities(template_file):
    """Extracts an ordered list of all cavity names from the template."""
    cavities = []
    with open(template_file, 'r') as file:
        for line in file:
            match = re.match(r'^\s*"?([A-Z0-9_]+)"?\s*:', line)
            if match and match.group(1).upper().startswith("CAV"):
                cavities.append(match.group(1).upper())
    return cavities


def extract_nominal_rf_params(template_file, cavity_name):
    """Finds the nominal VOLT and LAG for a specific cavity."""
    volt, lag = None, None
    num_pattern = r'([+-]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'

    with open(template_file, 'r') as file:
        for line in file:
            match = re.match(r'^\s*"?([A-Z0-9_]+)"?\s*:', line)
            if match and match.group(1).upper() == cavity_name:
                v_match = re.search(rf'VOLT\s*=\s*{num_pattern}', line, re.IGNORECASE)
                l_match = re.search(rf'LAG\s*=\s*{num_pattern}', line, re.IGNORECASE)
                if v_match: volt = float(v_match.group(1))
                if l_match: lag = float(l_match.group(1))
                break
    return volt, lag


def run_opal(input_file, host_id=1, timeout_sec=2000):
    command = ["rsmpi", "-n", "16", "-h", str(host_id), "opal", input_file]
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, check=True, timeout=timeout_sec)
        return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


def generate_run_file(template_file, output_file, broken_cavity, psdumpfreq, rf_updates=None):
    """
    Updates psdumpfreq, destroys the broken cavity, and applies VOLT/LAG updates.
    rf_updates = {'CAV004': {'VOLT': 12.1, 'LAG': 0.5}, ...}
    """
    modified_lines = []
    with open(template_file, 'r') as file:
        for line in file:
            if "psdumpfreq" in line.lower():
                line = re.sub(r'(psdumpfreq\s*=\s*)\d+', rf'\g<1>{psdumpfreq}', line, flags=re.IGNORECASE)

            match_cav = re.match(r'^\s*"?([A-Z0-9_]+)"?\s*:', line)
            if match_cav:
                element_name = match_cav.group(1).upper()
                if element_name == broken_cavity.upper():
                    line = re.sub(r'(VOLT\s*=\s*)[^,;]+', r'\g<1>1.0e-6', line, flags=re.IGNORECASE)
                elif rf_updates and element_name in rf_updates:
                    new_volt = rf_updates[element_name]['VOLT']
                    new_lag = rf_updates[element_name]['LAG']
                    line = re.sub(r'(VOLT\s*=\s*)[^,;]+', rf'\g<1>{new_volt:.6f}', line, flags=re.IGNORECASE)
                    line = re.sub(r'(LAG\s*=\s*)[^,;]+', rf'\g<1>{new_lag:.6f}', line, flags=re.IGNORECASE)

            modified_lines.append(line)

    with open(output_file, 'w') as file:
        file.writelines(modified_lines)


# --- 2. Main Execution ---

def main(broken_cavity, host_id):
    template_file = "template.i"

    # --- CONFIGURATION ---
    N_ADJACENT = 5            # Number of cavities to tune BEFORE and AFTER
    VOLT_BOUND_PCT = 0.2     # Allow +/- 15% change in voltage
    LAG_BOUND_ABS = 0.1       # Allow +/- 0.1 radian shift in phase
    # ---------------------

    all_cavities = discover_cavities(template_file)
    if broken_cavity not in all_cavities:
        print(f"[!] Error: {broken_cavity} not found in template.")
        return

    # Identify the N adjacent cavities safely (handle edges of the linac)
    broken_idx = all_cavities.index(broken_cavity)
    start_idx = max(0, broken_idx - N_ADJACENT)
    end_idx = min(len(all_cavities), broken_idx + N_ADJACENT + 1)

    target_cavs = [all_cavities[i] for i in range(start_idx, end_idx) if i != broken_idx]

    print(f"=== Starting RF Recovery Optimization ===")
    print(f"Broken Cavity: {broken_cavity}")
    print(f"Tuning {len(target_cavs)} neighboring cavities: {', '.join(target_cavs)}")

    # Extract nominal values to build initial guess array x0
    x0_list = []
    lower_bounds_list = []
    upper_bounds_list = []

    for cav in target_cavs:
        nom_volt, nom_lag = extract_nominal_rf_params(template_file, cav)
        if nom_volt is None or nom_lag is None:
            print(f"[!] Error: Could not parse VOLT/LAG for {cav}.")
            return

        # Append VOLT
        x0_list.append(nom_volt)
        lower_bounds_list.append(nom_volt * (1.0 - VOLT_BOUND_PCT))
        upper_bounds_list.append(nom_volt * (1.0 + VOLT_BOUND_PCT))

        # Append LAG
        x0_list.append(nom_lag)
        lower_bounds_list.append(nom_lag - LAG_BOUND_ABS)
        upper_bounds_list.append(nom_lag + LAG_BOUND_ABS)

    x0 = np.array(x0_list)
    bounds = (np.array(lower_bounds_list), np.array(upper_bounds_list))

    # --- PHASE 1: Baseline Run ---
    print("\n[Phase 1] Evaluating unoptimized failure (psdumpfreq=100)...")
    init_file = f"initial_{broken_cavity}.i"
    init_stat = f"initial_{broken_cavity}.stat"
    generate_run_file(template_file, init_file, broken_cavity, 100)
    run_opal(init_file, host_id=host_id)
    print(f"Initial Penalty: {extract_penalty(init_stat):.2f}")


    # --- PHASE 2: PyBOBYQA Optimization ---
    print("\n[Phase 2] Optimizing neighboring VOLT and LAG (psdumpfreq=0)...")
    iteration_counter = 0

    # 1. Setup the CSV file and write the dynamic headers
    history_file = f"optimization_history_{broken_cavity}.csv"
    csv_headers = ['Iteration', 'Penalty']
    for cav in target_cavs:
        csv_headers.extend([f"{cav}_VOLT", f"{cav}_LAG"])

    with open(history_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    def objective_function(x):
        nonlocal iteration_counter
        iteration_counter += 1

        # Reconstruct updates dict from the flat array x
        rf_updates = {}
        for i, cav in enumerate(target_cavs):
            rf_updates[cav] = {
                'VOLT': x[2*i],
                'LAG': x[2*i + 1]
            }

        # Make the temporary files strictly unique to this thread's cavity
        run_file = f"opt_iter_{broken_cavity}.i"
        stat_file = f"opt_iter_{broken_cavity}.stat"


        generate_run_file(template_file, run_file, broken_cavity, 0, rf_updates)
        success = run_opal(run_file, host_id=host_id)
        penalty = extract_penalty(stat_file) if success else 1e6

        if iteration_counter % 5 == 0 or iteration_counter == 1:
            print(f"  Iter {iteration_counter:03d} | Penalty: {penalty:8.2f}")

        # 2. Log this iteration to the CSV
        with open(history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row_data = [iteration_counter, penalty] + list(x)
            writer.writerow(row_data)

        # Clean temp files
        for ext in ['.i', '.stat', '.lbal', '.h5']:
            if os.path.exists(run_file.replace('.i', ext)):
                os.remove(run_file.replace('.i', ext))

        return penalty

    res = pybobyqa.solve(objective_function, x0, bounds=bounds, maxfun=150, rhobeg=0.05)

    print("\n--- Optimization Finished ---")
    print(f"Best Penalty Found: {res.f:.2f} (from initial guesses)")

    # --- PHASE 3: Final Run ---
    print("\n[Phase 3] Generating final high-res output (psdumpfreq=100)...")

    final_rf_updates = {}
    for i, cav in enumerate(target_cavs):
        final_rf_updates[cav] = {
            'VOLT': res.x[2*i],
            'LAG': res.x[2*i + 1]
        }

    optimal_file = f"optimal_{broken_cavity}.i"
    generate_run_file(template_file, optimal_file, broken_cavity, 100, final_rf_updates)
    run_opal(optimal_file, host_id=host_id)

    print("\n[Success] Complete! You can compare the phase spaces of:")
    print(f"  initial_{broken_cavity}.h5")
    print(f"  optimal_{broken_cavity}.h5")

if __name__ == "__main__":
    import sys
    # Accept arguments from the command line, defaulting to CAV050 and host 1 if none provided
    cav = sys.argv[1] if len(sys.argv) > 1 else "CAV050"
    host = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    main(cav, host)
