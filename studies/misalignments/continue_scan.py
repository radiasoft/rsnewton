import concurrent.futures
import glob
import math
import os
import queue
import random
import shutil
import subprocess
import threading
import numpy as np
import json

from pykern.pkcollections import PKDict
from sirepo.template.lattice import LatticeUtil
import pykern.pkjson
import sirepo.lib
import sdds

# --- 1. Anchor to Absolute Paths ---
_BASE_DIR = os.path.abspath(os.getcwd())
_RUN_DIR = os.path.join(_BASE_DIR, 'work')
_SIM_DIR = os.path.join(_BASE_DIR, 'sim')
_OUT_DIR = os.path.join(_BASE_DIR, 'out')
_JSON_RESULTS_FILE = 'alignment_results.json'

# --- Misalignment & Tolerance Bounds ---
_ALTER_POS_MAX = 5e-4 # 0.5 mm maximum offset in meters
_ALTER_ROT_MAX = 0.1 * math.pi / 180 # 0.1 degree maximum tilt in radians

# High-level tolerances for a "passing" run
_MAX_EMITTANCE_GROWTH_PCT = 25.0  # Maximum allowable emittance growth in percent
_MAX_PARTICLE_LOSS = 10            # Maximum allowable number of lost particles

# Only apply physical misalignments to cavities and solenoids
_TARGET_ELEMENTS = ['SOLENOID', 'RFCAVITY']

# --- Cluster Configuration ---
_AVAILABLE_HOSTS = [1, 2, 3, 4, 5, 6, 7, 8]
_CORES_PER_NODE = 16

_host_queue = queue.Queue()
for host in _AVAILABLE_HOSTS:
    _host_queue.put(host)

_sirepo_lock = threading.Lock()

def _extract_foms(stat_file):
    """Extracts figures of merit from the OPAL stat file using SDDS, protecting against NaNs."""
    if not os.path.exists(stat_file):
        return None
        
    sdds_index = 0
    foms = PKDict()
    try:
        sdds.sddsdata.InitializeInput(sdds_index, stat_file)
        sdds.sddsdata.ReadPage(sdds_index)
        names = sdds.sddsdata.GetColumnNames(sdds_index)
        
        def get_val(col_names):
            col = next((c for c in col_names if c in names), None)
            if col:
                val = float(sdds.sddsdata.GetColumn(sdds_index, names.index(col))[-1])
                # Convert NaNs to None for safe JSON serialization
                return None if np.isnan(val) else val
            return None

        foms.np = get_val(['np', 'numParticles', 'N_part', 'NP'])
        foms.ex = get_val(['ex', 'epsx', 'normEmitx'])
        foms.ey = get_val(['ey', 'epsy', 'normEmity'])
    finally:
        sdds.sddsdata.Terminate(sdds_index)
        
    return foms

def _evaluate_run(foms, baseline_foms):
    """Evaluates if the run passes based on configurable particle loss and emittance ceilings."""
    eval_result = PKDict(
        passed=True,
        particles_lost=0,
        emittance_growth_x_pct=0.0,
        emittance_growth_y_pct=0.0,
        failure_reasons=[]
    )
    
    if not foms or foms.np is None:
        eval_result.passed = False
        eval_result.failure_reasons.append("Missing or invalid stat file")
        return eval_result
        
    # Check for particle loss against the configured threshold
    if foms.np < baseline_foms.np: 
        eval_result.particles_lost = int(baseline_foms.np - foms.np)
        if eval_result.particles_lost > _MAX_PARTICLE_LOSS:
            eval_result.passed = False
            eval_result.failure_reasons.append(
                f"Lost {eval_result.particles_lost} particles (Threshold: {_MAX_PARTICLE_LOSS})"
            )

    # Check for X emittance growth against the configured ceiling
    if foms.ex is not None and baseline_foms.ex is not None:
        eval_result.emittance_growth_x_pct = ((foms.ex - baseline_foms.ex) / baseline_foms.ex) * 100.0
        if eval_result.emittance_growth_x_pct > _MAX_EMITTANCE_GROWTH_PCT:
            eval_result.passed = False
            eval_result.failure_reasons.append(
                f"X emittance grew by {eval_result.emittance_growth_x_pct:.1f}% (Ceiling: {_MAX_EMITTANCE_GROWTH_PCT}%)"
            )

    # Check for Y emittance growth against the configured ceiling
    if foms.ey is not None and baseline_foms.ey is not None:
        eval_result.emittance_growth_y_pct = ((foms.ey - baseline_foms.ey) / baseline_foms.ey) * 100.0
        if eval_result.emittance_growth_y_pct > _MAX_EMITTANCE_GROWTH_PCT:
            eval_result.passed = False
            eval_result.failure_reasons.append(
                f"Y emittance grew by {eval_result.emittance_growth_y_pct:.1f}% (Ceiling: {_MAX_EMITTANCE_GROWTH_PCT}%)"
            )
            
    return eval_result

def _read_and_write_sim(run_dir, alter=False, quick=False):
    with _sirepo_lock:
        inputs = PKDict()
        d = sirepo.lib.Importer('opal').parse_file(f'{_SIM_DIR}/opal.in')
        
        if quick:
            LatticeUtil.find_first_command(d, 'track').dt = 1e-10
            LatticeUtil.find_first_command(d, 'beam').npart = 1000
            LatticeUtil.find_first_command(d, 'distribution').type = 'GAUSS'
            
        if alter:
            for e in d.models.elements:
                if e.type in _TARGET_ELEMENTS:
                    inputs[e.name] = PKDict()
                    
                    for pos_axis in ['dx', 'dy']:
                        val = random.uniform(-_ALTER_POS_MAX, _ALTER_POS_MAX)
                        e[pos_axis] = e.get(pos_axis, 0.0) + val
                        inputs[e.name][pos_axis] = val
                        
                    for rot_axis in ['dtheta', 'dphi']:
                        val = random.uniform(-_ALTER_ROT_MAX, _ALTER_ROT_MAX)
                        e[rot_axis] = e.get(rot_axis, 0.0) + val
                        inputs[e.name][rot_axis] = val
                        
        d.write_files(run_dir)
        return inputs

def _run_single_job(count, alter, quick, baseline_foms=None):
    host_id = _host_queue.get()
    
    run_dir = f'{_RUN_DIR}/run_{count:03d}' if alter else f'{_RUN_DIR}/run_baseline'
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(_OUT_DIR, exist_ok=True)
    
    try:
        for filepath in glob.glob(f'{_SIM_DIR}/*.txt'):
            filename = os.path.basename(filepath)
            dest_path = os.path.join(run_dir, filename)
            if os.path.islink(dest_path) or os.path.exists(dest_path):
                os.remove(dest_path)
            shutil.copy(filepath, dest_path)
            
        inputs = _read_and_write_sim(run_dir, alter=alter, quick=quick)
        
        dist_src = os.path.join(run_dir, 'TRACK.initial.particles.txt')
        dist_dst = os.path.join(run_dir, 'command_distribution-fname.ini_dis_opal.txt')
        if os.path.exists(dist_src):
            if os.path.islink(dist_dst) or os.path.exists(dist_dst):
                os.remove(dist_dst)
            shutil.copy(dist_src, dist_dst)

        cmd = [
            "rsmpi", 
            "-h", str(host_id), 
            "-n", str(_CORES_PER_NODE), 
            "opal", 
            "opal.in"
        ]
        
        out_file_path = os.path.join(run_dir, f"out{_CORES_PER_NODE}.txt")
        with open(out_file_path, "w") as out_f:
            subprocess.run(cmd, cwd=run_dir, stdout=out_f, stderr=subprocess.STDOUT, check=True)
            
        h5_src = os.path.join(run_dir, 'opal.h5')
        stat_src = os.path.join(run_dir, 'opal.stat')
        
        foms = _extract_foms(stat_src)
        eval_result = _evaluate_run(foms, baseline_foms) if baseline_foms and foms else PKDict(passed=True)
        
        if alter:
            if os.path.exists(h5_src):
                shutil.move(h5_src, f'{_OUT_DIR}/opal-{count:03d}.h5')
            if os.path.exists(stat_src):
                shutil.move(stat_src, f'{_OUT_DIR}/opal-{count:03d}.stat')
            
        return PKDict(count=count, inputs=inputs, foms=foms, evaluation=eval_result)
        
    except subprocess.CalledProcessError as e:
        print(f"Run {count:03d} failed on host {host_id} with exit code {e.returncode}.")
        return PKDict(count=count, error=str(e), inputs=inputs, passed=False)
        
    finally:
        _host_queue.put(host_id)

def _run_sims(total_runs, quick=False):
    print("Evaluating BASELINE lattice for FOM normalization...")
    baseline_result = _run_single_job(0, alter=False, quick=quick, baseline_foms=None)
    
    if not baseline_result.get('foms') or baseline_result.foms.get('np') is None:
        print("CRITICAL ERROR: Baseline run failed or produced invalid stat data. Check your setup.")
        return
        
    baseline_foms = baseline_result.foms
    print(f"Baseline established. Initial Particles: {baseline_foms.np}")

    # --- Append Logic ---
    existing_results = []
    start_count = 0
    if os.path.exists(_JSON_RESULTS_FILE):
        try:
            with open(_JSON_RESULTS_FILE, 'r') as f:
                existing_results = json.load(f)
            if existing_results:
                start_count = max(r.get('count', 0) for r in existing_results)
                print(f"Found {len(existing_results)} existing runs. Resuming from run {start_count + 1}.")
        except Exception as e:
            print(f"Warning: Could not read existing results: {e}. Starting fresh.")

    new_results = []
    max_workers = len(_AVAILABLE_HOSTS)
    
    print(f"Dispatching {total_runs} aligned simulations across {max_workers} hosts...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_single_job, c + 1, True, quick, baseline_foms): c
            for c in range(start_count, start_count + total_runs)
        }
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            status = "PASS" if res.get('evaluation', {}).get('passed', False) else "FAIL"
            print(f"Completed run {res.get('count', 0):03d} - {status}")
            new_results.append(res)
            
    # Combine existing and new results, ensuring it remains sorted by run count
    combined_results = existing_results + new_results
    combined_results.sort(key=lambda x: x.get('count', 0))
    pykern.pkjson.dump_pretty(combined_results, _JSON_RESULTS_FILE)
    print("All simulations complete and appended to alignment_results.json.")

if __name__ == '__main__':
    os.makedirs(_RUN_DIR, exist_ok=True)
    os.makedirs(_OUT_DIR, exist_ok=True)
    
    # You can specify any number of runs here, and they will stack continuously.
    _run_sims(1000, quick=False)