import os
import re
import subprocess
import pybobyqa
import numpy as np
import math
import csv
import pandas as pd


# Global cache so we only read the nominal file from disk once
_NOMINAL_CACHE = None

def extract_penalty(stat_file_path):
    global _NOMINAL_CACHE

    # --- CONFIGURATION WEIGHTS ---
    loss_weight = 100.0
    emit_weight = 50.0
    energy_weight = 100.0
    # -----------------------------

    # 1. Load the trial data using the parser
    df_trial = read_opal_stat(stat_file_path)

    # 2. Load and cache the nominal baseline (if not already loaded)
    if _NOMINAL_CACHE is None:
        df_nom = read_opal_stat("nominal.stat")
        nom_last = df_nom.iloc[-1]
        
        _NOMINAL_CACHE = {
            'emit_x': nom_last['emit_x'],
            'emit_y': nom_last['emit_y'],
            'emit_s': nom_last['emit_s'],
            'energy': nom_last['energy']
        }

    nom = _NOMINAL_CACHE

    # 3. Extract current trial's starting and ending states
    initial_particles = df_trial.iloc[0]['numParticles']
    last_row = df_trial.iloc[-1]
    final_particles = last_row['numParticles']

    transmission = final_particles / initial_particles if initial_particles > 0 else 0.0

    # Instant failure constraint: if beam dies, return max penalty immediately
    if transmission < 0.05:
        return 1e6
    # Did it reach the end? ---
    # Using 147.5m to allow for discrete OPAL integration step boundaries
    if last_row['s'] < 147.5:
        return 1e6
        

    eps_x = last_row['emit_x']
    eps_y = last_row['emit_y']
    eps_z = last_row['emit_s']
    final_energy = last_row['energy']

    # 4. Calculate Penalties

    # Loss Penalty (Exponential wall)
    loss_fraction = 1.0 - transmission
    loss_penalty = loss_weight * (math.exp(loss_fraction * 10) - 1.0)

    # Emittance Penalty (Normalized relative percentage growth)
    # max(0, ...) prevents PyBOBYQA from being rewarded for shrinking the beam below nominal
    dx = max(0, (eps_x - nom['emit_x']) / nom['emit_x'])
    dy = max(0, (eps_y - nom['emit_y']) / nom['emit_y'])
    dz = max(0, (eps_z - nom['emit_s']) / nom['emit_s'])

    #emit_penalty = emit_weight * (dx + dy + dz * 2.0)
    # The squared formulation (L2 norm-style) heavily punishes severe blowups
    # in a single plane, forcing the optimizer to balance X and Y distortion.
    emit_penalty = emit_weight * (dx**2 + dy**2 + (dz * 2.0)**2)

    # Energy Penalty (Percentage difference from target energy)
    energy_error_fraction = abs(final_energy - nom['energy']) / nom['energy']
    energy_penalty = energy_weight * energy_error_fraction

    return loss_penalty + emit_penalty + energy_penalty

import pandas as pd

def read_opal_stat(filepath):
    """
    SDDS-compliant OPAL .stat file parser.
    Reads SDDS &column blocks to extract headers dynamically,
    then parses the ASCII data block.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    headers = []
    data_start_idx = -1
    
    in_column_block = False
    in_data_block = False
    
    for i, line in enumerate(lines):
        clean_line = line.strip()
        
        # 1. Look for column definitions
        if clean_line.startswith('&column'):
            in_column_block = True
            continue
            
        if in_column_block:
            # Extract the variable name (e.g., 'name=s,' -> 's')
            if clean_line.startswith('name='):
                col_name = clean_line.split('=')[1]
                # Strip out commas, quotes, and whitespace
                col_name = col_name.strip(',"\' ')
                headers.append(col_name)
            elif clean_line == '&end':
                in_column_block = False
            continue
            
        # 2. Look for the start of the actual data
        if clean_line.startswith('&data'):
            in_data_block = True
            continue
            
        # The data block ends with '&end'. 
        # Everything after this specific '&end' is the numerical text.
        if in_data_block and clean_line == '&end':
            data_start_idx = i + 1
            break

    if not headers or data_start_idx == -1:
        raise ValueError(f"Could not parse SDDS headers or data in {filepath}")

    num_cols = len(headers)
    valid_data = []
    
    # 3. Read the numerical data safely
    for line in lines[data_start_idx:]:
        clean_line = line.strip()
        if not clean_line:
            continue
            
        parts = clean_line.split()
        
        # Enforce column geometry to skip any mid-file garbage
        if len(parts) == num_cols:
            try:
                float(parts[0])
                valid_data.append(parts)
            except ValueError:
                pass

    # Build DataFrame
    df = pd.DataFrame(valid_data, columns=headers)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    return df
    
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

def main(broken_cavity, host_id, template_file):
    # Remove the hardcoded template_file = "template.in" from here

    # --- CONFIGURATION ---
    N_ADJACENT = 5            # Number of cavities to tune BEFORE and AFTER
    VOLT_BOUND_PCT = 0.2     # Allow +/- 20% change in voltage
    LAG_BOUND_ABS =  0.4 # 0.1       # Allow +/- 0.1 radian shift in phase
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

    initial_h5 = f"initial_{broken_cavity}.h5"
    initial_stat = f"initial_{broken_cavity}.stat"

    if os.path.exists(initial_h5):
        print(f"  -> [Skip] {initial_h5} already exists. Skipping baseline tracking.")
    else:
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
    # Accept arguments: python optimize_rf_recovery.py CAV050 1 checkpoint_CAV040/template.in
    cav = sys.argv[1] if len(sys.argv) > 1 else "CAV050"
    host = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    template = sys.argv[3] if len(sys.argv) > 3 else "template.in"

    main(cav, host, template)
