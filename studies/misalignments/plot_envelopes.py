import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sdds
from pykern.pkcollections import PKDict

_JSON_RESULTS_FILE = "alignment_results.json"
_STAT_DIR = "out"
_BASELINE_STAT = "work/run_baseline/opal.stat"

def _read_full_sdds_columns(filepath, columns):
    """Reads full arrays for requested columns from an SDDS file."""
    if not os.path.exists(filepath):
        return None
        
    sdds_idx = 0
    data = PKDict()
    try:
        sdds.sddsdata.InitializeInput(sdds_idx, filepath)
        sdds.sddsdata.ReadPage(sdds_idx)
        col_names = sdds.sddsdata.GetColumnNames(sdds_idx)
        
        for col in columns:
            if col in col_names:
                data[col] = np.array(sdds.sddsdata.GetColumn(sdds_idx, col_names.index(col)))
            else:
                data[col] = None
    finally:
        sdds.sddsdata.Terminate(sdds_idx)
        
    return data

def plot_beam_envelopes(successful_runs, failed_runs):
    """Plots the RMS beam size along the longitudinal axis 's' for X, Y, and Z(s)."""
    fig, (ax_x, ax_y, ax_s) = plt.subplots(3, 1, figsize=(8, 18))
    fig.suptitle(f"RMS Beam Envelopes Along Linac ({len(successful_runs)} Pass, {len(failed_runs)} Fail)",y=0.93, fontsize=16)
    
    # 1. Plot passing runs (Solid colored lines)
    for run_id in successful_runs:
        stat_file = f"{_STAT_DIR}/opal-{run_id:03d}.stat"
        data = _read_full_sdds_columns(stat_file, ['s', 'rms_x', 'rms_y', 'rms_s'])
        
        if data and data.s is not None:
            ax_x.plot(data.s, data.rms_x * 1e3, color='blue', alpha=0.05, linewidth=1)
            ax_y.plot(data.s, data.rms_y * 1e3, color='red', alpha=0.05, linewidth=1)
            ax_s.plot(data.s, data.rms_s * 1e3, color='green', alpha=0.05, linewidth=1)

    # 2. Plot failing runs (Dashed dark grey lines)
    for run_id in failed_runs:
        stat_file = f"{_STAT_DIR}/opal-{run_id:03d}.stat"
        data = _read_full_sdds_columns(stat_file, ['s', 'rms_x', 'rms_y', 'rms_s'])
        
        if data and data.s is not None:
            ax_x.plot(data.s, data.rms_x * 1e3, color='dimgrey', alpha=0.15, linewidth=1, linestyle='--')
            ax_y.plot(data.s, data.rms_y * 1e3, color='dimgrey', alpha=0.15, linewidth=1, linestyle='--')
            ax_s.plot(data.s, data.rms_s * 1e3, color='dimgrey', alpha=0.15, linewidth=1, linestyle='--')

    # 3. Plot Baseline (Solid bold lines on top)
    base_data = _read_full_sdds_columns(_BASELINE_STAT, ['s', 'rms_x', 'rms_y', 'rms_s'])
    if base_data and base_data.s is not None:
        ax_x.plot(base_data.s, base_data.rms_x * 1e3, color='cyan', linewidth=2)
        ax_y.plot(base_data.s, base_data.rms_y * 1e3, color='orange', linewidth=2)
        ax_s.plot(base_data.s, base_data.rms_s * 1e3, color='lime', linewidth=2)

    # 4. Format Subplots and Custom Legends
    axes = [ax_x, ax_y, ax_s]
    titles = ['Transverse X Envelope', 'Transverse Y Envelope', 'Longitudinal Z (s) Envelope']
    ylabels = ['RMS X [mm]', 'RMS Y [mm]', 'RMS Z [mm]']
    base_colors = ['cyan', 'orange', 'lime']
    pass_colors = ['blue', 'red', 'green']
    
    for ax, title, ylabel, b_color, p_color in zip(axes, titles, ylabels, base_colors, pass_colors):
        ax.set_title(title)
        ax.set_xlabel("Longitudinal Position, s [m]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        # Build custom legend
        custom_lines = [
            Line2D([0], [0], color=b_color, lw=2, label='Baseline'),
            Line2D([0], [0], color=p_color, lw=2, alpha=0.5, label='Pass'),
            Line2D([0], [0], color='dimgrey', lw=2, linestyle='--', alpha=0.5, label='Fail')
        ]
        ax.legend(handles=custom_lines)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9) 
    plt.savefig("beam_envelope_spaghetti_xyz.png", dpi=300)
    plt.show()

def main():
    if not os.path.exists(_JSON_RESULTS_FILE):
        print(f"Run the simulation sweep first to generate {_JSON_RESULTS_FILE}.")
        return

    with open(_JSON_RESULTS_FILE, 'r') as f:
        scan_data = json.load(f)

    successful_runs = []
    failed_runs = []

    for run in scan_data:
        if run.get('evaluation', {}).get('passed', False):
            successful_runs.append(run['count'])
        else:
            # Check for 'foms' to ensure we only try to plot runs that successfully wrote stat files
            if 'foms' in run:
                failed_runs.append(run['count'])

    if not successful_runs and not failed_runs:
        print("No valid runs available to plot.")
        return

    plot_beam_envelopes(successful_runs, failed_runs)

if __name__ == '__main__':
    main()