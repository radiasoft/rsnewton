import concurrent.futures
import glob
import os
import queue
import shutil
import subprocess
import threading
import re
import math
import numpy as np
import json
import pathlib

from pykern.pkcollections import PKDict
import pykern.pkjson
import sirepo.lib
import sdds

# --- Absolute Paths & Threading ---
_BASE_DIR = os.path.abspath(os.getcwd())
_RUN_DIR = pathlib.Path(_BASE_DIR, 'work')
_SIM_DIR = pathlib.Path(_BASE_DIR, 'sim')
_OUT_DIR = pathlib.Path(_BASE_DIR, 'out')

_sirepo_lock = threading.Lock()
_counter_lock = threading.Lock()
_global_run_counter = 0

# --- Cluster Configuration ---
_AVAILABLE_HOSTS = [1, 2, 3, 4, 5, 6, 7, 8]
_CORES_PER_NODE = 16

_host_queue = queue.Queue()
for host in _AVAILABLE_HOSTS:
    _host_queue.put(host)

# --- Unified Tuning Configuration ---
_TARGET_CAV = "CAV001"  
_NUM_NEIGHBORS = 5      
_RAD_CONV = math.pi / 180.0

# Voltage Profile (Tapered localized compensation)
_V_BOOSTS = {
    "CAV002": 0.30,  # +30% to immediately capture the beam
    "CAV003": 0.25,  # +25% to continue the hard squeeze
    "CAV004": 0.20,  # +20% tapering off
    "CAV005": 0.20,  # +20% 
    "CAV006": 0.20   # +20% back to nominal tuning
}

# --- State Control ---
_USE_PREVIOUS_ANCHORS = False  # False = Fresh 360-degree sweep | True = Fine sweep around anchors

_BEST_PHASES_RAD = {
    "CAV002": 2.8798,
    "CAV003": 0.2921,
    "CAV004": 3.0024,
    "CAV005": 1.4894,
    "CAV006": 1.8629
}

# Sweep Parameters
_FINE_SCAN_OFFSETS_DEG = np.arange(-7, 8, 1)  # Used if _USE_PREVIOUS_ANCHORS == True
_COARSE_SCAN_DEG = np.arange(0, 360, 15)      # Used if _USE_PREVIOUS_ANCHORS == False

def parse_cavity_specs(lattice_file='opal.in'):
    cav_specs = {}
    cav_pattern = re.compile(r'^"(CAV\d+)"\s*:\s*RFCAVITY.*freq=([0-9.]+)\s*\*\s*beam_frequency_mhz.*volt=([0-9.]+)')
    manual_clusters = {f"CAV{i:03d}": 2.75 for i in range(1, 8)}
    
    with open(lattice_file, 'r') as f:
        for line in f:
            match = cav_pattern.search(line)
            if match:
                cav_name = match.group(1)
                freq_mult = float(match.group(2))
                volt = float(match.group(3))
                family_volt = manual_clusters.get(cav_name, volt)
                cav_specs[cav_name] = {
                    'freq_mult': freq_mult, 'volt': volt, 'family_volt': family_volt
                }
    return cav_specs

def build_cryomodules(cav_specs):
    cavities = sorted(cav_specs.keys())
    modules, current_mod, current_sig = [], [], None
    for cav in cavities:
        sig = (cav_specs[cav]['freq_mult'], cav_specs[cav]['family_volt'])
        if sig != current_sig:
            if current_mod: modules.append(current_mod)
            current_mod = [cav]
            current_sig = sig
        else:
            current_mod.append(cav)
    if current_mod: modules.append(current_mod)
    return modules

def get_neighbors(cavity, modules, max_neighbors):
    for mod in modules:
        if cavity in mod:
            candidates = [c for c in mod if c != cavity]
            cav_idx = int(cavity.replace('CAV', ''))
            candidates.sort(key=lambda c: abs(int(c.replace('CAV', '')) - cav_idx))
            return candidates[:max_neighbors]
    return []

def _get_next_run_id():
    global _global_run_counter
    with _counter_lock:
        _global_run_counter += 1
        return _global_run_counter

def _extract_foms(stat_file):
    if not os.path.exists(stat_file): return None
    sdds_index = 0
    foms = PKDict()
    try:
        sdds.sddsdata.InitializeInput(sdds_index, stat_file)
        sdds.sddsdata.ReadPage(sdds_index)
        names = sdds.sddsdata.GetColumnNames(sdds_index)
        
        def get_val(col_names):
            col = next((c for c in col_names if c in names), None)
            if col:
                return float(sdds.sddsdata.GetColumn(sdds_index, names.index(col))[-1])
            return np.nan

        foms.np = get_val(['np', 'numParticles', 'N_part', 'NP'])
        foms.emit_s = get_val(['emit_s', 'eps_z', 'emit_z'])
        
    finally:
        sdds.sddsdata.Terminate(sdds_index)
    return foms

def run_opal_1d_step(failed_cav, neighbors, target_cav, test_phase_deg, locked_phases):
    """Evaluates a specific phase, recording the exact machine state."""
    run_id = _get_next_run_id()
    run_str = f"run_{run_id:04d}"
    run_dir = f"{_RUN_DIR}/{run_str}"
    
    os.makedirs(run_dir, exist_ok=True)
    host_id = _host_queue.get()
    
    applied_volts = {}
    applied_phases = {}
    
    try:
        for filepath in glob.glob(str(_SIM_DIR / '*.txt')):
            shutil.copy(filepath, os.path.join(run_dir, os.path.basename(filepath)))
            
        with _sirepo_lock:
            d = sirepo.lib.Importer('opal').parse_file(str(_SIM_DIR / 'opal.in'))
            for e in d.models.elements:
                if e.name == failed_cav:
                    e['volt'] = 0.0  
                elif e.name in neighbors:
                    # Voltage Application
                    cav_boost = _V_BOOSTS.get(e.name, 0.0)
                    e['volt'] = e['volt'] * (1.0 + cav_boost)
                    applied_volts[e.name] = e['volt']
                    
                    # Phase Application
                    if e.name == target_cav:
                        e['lag'] = test_phase_deg * _RAD_CONV
                    elif e.name in locked_phases:
                        e['lag'] = locked_phases[e.name]
                    else:
                        if _USE_PREVIOUS_ANCHORS:
                            e['lag'] = _BEST_PHASES_RAD.get(e.name, e.get('lag', 0.0))
                        else:
                            e['lag'] = e.get('lag', 0.0)
                            
                    applied_phases[e.name] = e['lag']
            d.write_files(run_dir)
            
        dist_src = os.path.join(run_dir, 'TRACK.initial.particles.txt')
        dist_dst = os.path.join(run_dir, 'command_distribution-fname.ini_dis_opal.txt')
        if os.path.exists(dist_src): shutil.copy(dist_src, dist_dst)

        cmd = ["rsmpi", "-h", str(host_id), "-n", str(_CORES_PER_NODE), "opal", "opal.in"]
        out_file_path = os.path.join(run_dir, f"out{_CORES_PER_NODE}.txt")
        
        with open(out_file_path, "w") as out_f:
            subprocess.run(cmd, cwd=run_dir, stdout=out_f, stderr=subprocess.STDOUT, check=True)
            
        stat_src = os.path.join(run_dir, 'opal.stat')
        foms = _extract_foms(stat_src)
        
        return PKDict(
            run_dir=run_str,
            target_cavity=target_cav,
            test_phase_deg=test_phase_deg,
            applied_volts=applied_volts,
            applied_phases=applied_phases,
            np=foms.np if foms else np.nan,
            emit_s=foms.emit_s if foms else np.nan
        )
        
    except Exception:
        return PKDict(run_dir=run_str, target_cavity=target_cav, test_phase_deg=test_phase_deg, np=np.nan, emit_s=np.nan)
    finally:
        _host_queue.put(host_id)

def execute_scan():
    os.makedirs(_RUN_DIR, exist_ok=True)
    os.makedirs(_OUT_DIR, exist_ok=True)
    json_log_path = os.path.join(_OUT_DIR, "tuning_run_library.json")
    
    cav_specs = parse_cavity_specs(str(_SIM_DIR / 'opal.in'))
    modules = build_cryomodules(cav_specs)
    neighbors = get_neighbors(_TARGET_CAV, modules, _NUM_NEIGHBORS)
    
    scan_type = "FINE" if _USE_PREVIOUS_ANCHORS else "FRESH COARSE"
    print(f"Initiating {scan_type} Sequential Tuning for {_TARGET_CAV} failure.")
    print(f"Tuning path: {neighbors}\n")
    
    locked_phases = {}
    run_library = []
    
    for current_cav in neighbors:
        print(f"--- Sweeping {current_cav} ---")
        
        if _USE_PREVIOUS_ANCHORS:
            center_deg = _BEST_PHASES_RAD.get(current_cav, 0.0) / _RAD_CONV
            sweep_array = [(center_deg + offset) % 360.0 for offset in _FINE_SCAN_OFFSETS_DEG]
        else:
            sweep_array = _COARSE_SCAN_DEG
        
        results = []
        
        # Parallelize the sweep tasks across the hosts
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(_AVAILABLE_HOSTS)) as executor:
            future_to_phase = {
                executor.submit(run_opal_1d_step, _TARGET_CAV, neighbors, current_cav, p, locked_phases): p
                for p in sweep_array
            }
            
            for future in concurrent.futures.as_completed(future_to_phase):
                res = future.result()
                results.append(res)

                # Sanitize numpy types to standard Python types for JSON serialization
                survival_val = res.np
                survival_clean = int(survival_val) if not np.isnan(survival_val) else None
                
                emit_val = res.emit_s
                emit_clean = float(emit_val) if not np.isnan(emit_val) else None

                # Append to the data library
                run_library.append({
                    "run_id": str(res.run_dir),
                    "target_cavity": str(res.target_cavity),
                    "test_phase_deg": float(round(res.test_phase_deg, 2)),
                    "applied_voltages_MV_m": {str(k): float(round(v, 4)) for k, v in res.get('applied_volts', {}).items()},
                    "applied_phases_rad": {str(k): float(round(v, 4)) for k, v in res.get('applied_phases', {}).items()},
                    "survival": survival_clean,
                    "emit_s": emit_clean
                })
                
                # Live-write to JSON to protect data
                with open(json_log_path, 'w') as f:
                    json.dump(run_library, f, indent=4)
        
        # Filter crashed runs
        valid_results = [r for r in results if not np.isnan(r.np)]
        if not valid_results:
            print(f"CRITICAL ERROR: All runs for {current_cav} crashed. Tuning cannot proceed.")
            return
            
        # Select the optimal phase natively (Primary: Max Survival, Secondary: Min Emittance)
        valid_results.sort(key=lambda x: (x.np, -x.emit_s), reverse=True)
        best_run = valid_results[0]
        
        print(f"[*] SECURED: {current_cav} locked at {best_run.test_phase_deg:.2f} degrees (Survival: {best_run.np})\n")
        
        # Lock it in the dictionary for the next iteration
        locked_phases[current_cav] = best_run.test_phase_deg * _RAD_CONV
        
    print("=========================================")
    print(f"Tuning Complete for {_TARGET_CAV}. Data logged to {json_log_path}")
    print("Final Tuned Focusing Phases (rad):")
    print(json.dumps(locked_phases, indent=4))
    print("=========================================")

if __name__ == '__main__':
    execute_scan()