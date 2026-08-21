import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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

def plot_failed_envelopes_by_loss(failed_run_data):
    """
    Plots the RMS beam size along the longitudinal axis 's' for X, Y, and Z(s),
    color-coded by the number of particles lost, with the baseline overlaid.
    """
    if not failed_run_data:
        print("No failed run data provided to plot.")
        return

    # 1. Setup Colormap based on particle loss range
    losses = [data['particles_lost'] for data in failed_run_data]
    min_loss = min(losses)
    max_loss = max(losses)
    
    cmap = cm.get_cmap('plasma')
    if min_loss == max_loss:
        norm = mcolors.Normalize(vmin=min_loss * 0.9, vmax=max_loss * 1.1)
    else:
        norm = mcolors.Normalize(vmin=min_loss, vmax=max_loss)

    # 2. Setup Figure and Subplots
    fig, (ax_x, ax_y, ax_s) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"RMS Beam Envelopes of Failed Runs ({len(failed_run_data)} Runs) vs Baseline", fontsize=16)

    # 3. Plot Each Failed Run
    for run_info in failed_run_data:
        run_id = run_info['run_id']
        loss_val = run_info['particles_lost']
        line_color = cmap(norm(loss_val))
        
        stat_file = f"{_STAT_DIR}/opal-{run_id:03d}.stat"
        data = _read_full_sdds_columns(stat_file, ['s', 'rms_x', 'rms_y', 'rms_s'])
        
        if data and data.s is not None:
            ax_x.plot(data.s, data.rms_x * 1e3, color=line_color, alpha=0.3, linewidth=1)
            ax_y.plot(data.s, data.rms_y * 1e3, color=line_color, alpha=0.3, linewidth=1)
            ax_s.plot(data.s, data.rms_s * 1e3, color=line_color, alpha=0.3, linewidth=1)

    # 4. Plot Baseline (Solid bold lines on top)
    base_data = _read_full_sdds_columns(_BASELINE_STAT, ['s', 'rms_x', 'rms_y', 'rms_s'])
    if base_data and base_data.s is not None:
        ax_x.plot(base_data.s, base_data.rms_x * 1e3, color='cyan', linewidth=2, label='Baseline')
        ax_y.plot(base_data.s, base_data.rms_y * 1e3, color='orange', linewidth=2, label='Baseline')
        ax_s.plot(base_data.s, base_data.rms_s * 1e3, color='lime', linewidth=2, label='Baseline')

    # 5. Format Subplots
    axes = [ax_x, ax_y, ax_s]
    titles = ['Transverse X Envelope', 'Transverse Y Envelope', 'Longitudinal Z (s) Envelope']
    ylabels = ['RMS X [mm]', 'RMS Y [mm]', 'RMS Z [mm]']
    
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title)
        ax.set_xlabel("Longitudinal Position, s [m]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    # 6. Manually format spacing to avoid tight_layout conflicts
    # Reserves space on the right (up to 90% of the figure width) for the plots
    plt.subplots_adjust(left=0.05, right=0.9, top=0.85, bottom=0.15, wspace=0.25)

    # 7. Add explicitly placed colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # [left, bottom, width, height] relative to the overall figure dimensions
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) 
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Number of Particles Lost", rotation=270, labelpad=20, fontsize=12)

    plt.savefig("beam_envelope_failures_baseline_colormap.png", dpi=300)
    plt.show()

def main():
    if not os.path.exists(_JSON_RESULTS_FILE):
        print(f"Run the simulation sweep first to generate {_JSON_RESULTS_FILE}.")
        return

    with open(_JSON_RESULTS_FILE, 'r') as f:
        scan_data = json.load(f)

    failed_run_data = []

    for run in scan_data:
        eval_data = run.get('evaluation', {})
        passed = eval_data.get('passed', False)
        
        # We only want failed runs that successfully generated a stat file
        if not passed and 'foms' in run:
            particles_lost = eval_data.get('particles_lost', 0)
            failed_run_data.append({
                'run_id': run['count'],
                'particles_lost': particles_lost
            })

    if not failed_run_data:
        print("No valid failed runs available to plot.")
        return

    plot_failed_envelopes_by_loss(failed_run_data)

if __name__ == '__main__':
    main()