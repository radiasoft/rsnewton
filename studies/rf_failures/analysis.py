import json
import pandas as pd
import matplotlib.pyplot as plt

_FILE_PATH = "out/tuning_run_library.json"

def main():
    # 1. Load the run library
    with open(_FILE_PATH, "r") as f:
        data = json.load(f)
    
    # Convert to pandas DataFrame for easy manipulation
    df = pd.DataFrame(data)
    
    # Extract unique target cavities and sort them
    cavities = sorted(df["target_cavity"].unique())
    
    # ==========================================
    # PLOT 1: Phase vs. Survival (Faceted)
    # ==========================================
    fig1, axes = plt.subplots(1, len(cavities), figsize=(22, 5), sharey=True)
    fig1.suptitle("Cavity Phase vs. Particle Survival (Color: Emittance)", fontsize=16, y=1.05)
    
    # Find global min and max of emittance for consistent color mapping
    vmin, vmax = df["emit_s"].min(), df["emit_s"].max()
    
    for ax, cavity in zip(axes, cavities):
        # Filter and sort data by phase for the current cavity
        subset = df[df["target_cavity"] == cavity].sort_values("test_phase_deg")
        
        # Plot a faint connecting line to show the tuning curve shape
        ax.plot(subset["test_phase_deg"], subset["survival"], color="gray", alpha=0.4, linestyle="--")
        
        # Scatter plot colored by emittance
        # Using 'viridis_r' so lower (better) emittance is brighter/yellow
        sc = ax.scatter(
            subset["test_phase_deg"], 
            subset["survival"], 
            c=subset["emit_s"], 
            cmap="viridis_r", 
            vmin=vmin, 
            vmax=vmax,
            s=80, 
            edgecolor="k", 
            alpha=0.9
        )
        
        ax.set_title(f"Target: {cavity}")
        ax.set_xlabel("Test Phase (deg)")
        ax.set_xlim(-10, 370)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.grid(True, alpha=0.3)
        
    axes[0].set_ylabel("Particle Survival (count)")
    
    # Add a single colorbar for the entire figure
    cbar = fig1.colorbar(sc, ax=axes.ravel().tolist(), pad=0.02)
    cbar.set_label("Emittance (emit_s)", rotation=270, labelpad=20)
    plt.savefig('survival.png')
    plt.show()

    # ==========================================
    # PLOT 2: Global Pareto Front (Emittance vs. Survival)
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    # Use a distinct color for each cavity
    colors = plt.cm.tab10.colors
    
    for idx, cavity in enumerate(cavities):
        subset = df[df["target_cavity"] == cavity]
        ax2.scatter(
            subset["survival"], 
            subset["emit_s"], 
            label=cavity, 
            color=colors[idx],
            s=60,
            edgecolor="white",
            alpha=0.8
        )
        
    ax2.set_title("Pareto Visualization: Emittance vs. Survival", fontsize=14)
    ax2.set_xlabel("Particle Survival (count) -> Maximize")
    ax2.set_ylabel("Emittance (emit_s) -> Minimize")
    
    # Usually, ideal performance is bottom-right (high survival, low emittance)
    ax2.grid(True, alpha=0.3)
    ax2.legend(title="Target Cavity")
    
    plt.tight_layout()
    plt.savefig('pareto.png')
    plt.show()

def export_best_global_configuration(json_file='tuning_run_library.json', output_file='best_tuning_parameters.py'):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{json_file}'.")
        return
        
    # Filter out crashed runs
    valid_runs = [r for r in data if r.get('survival') is not None]
    
    if not valid_runs:
        print("No valid runs found in the dataset.")
        return
    
    # Sort ALL runs natively: Primary by Max Survival, Secondary by Min Emittance
    valid_runs.sort(key=lambda x: (x['survival'], -x['emit_s']), reverse=True)
    best_run = valid_runs[0]
    
    # Extract the full machine state during that specific run
    best_phases = best_run['applied_phases_rad']
    best_voltages = best_run['applied_voltages_MV_m']
    
    # Format the output as ready-to-use Python dictionaries
    output_str = f"# GLOBAL Best Configuration (Found during {best_run['target_cavity']} sweep | Run ID: {best_run['run_id']})\n"
    output_str += f"# Survival: {best_run['survival']} | Emit_s: {best_run['emit_s']:.4e}\n\n"
    
    output_str += "_BEST_PHASES_RAD = {\n"
    for cav, phase in best_phases.items():
        output_str += f'    "{cav}": {phase},\n'
    output_str = output_str.rstrip(",\n") + "\n}\n\n"
    
    output_str += "_BEST_VOLTAGES = {\n"
    for cav, volt in best_voltages.items():
        output_str += f'    "{cav}": {volt},\n'
    output_str = output_str.rstrip(",\n") + "\n}\n"
    
    with open(output_file, 'w') as f:
        f.write(output_str)
        
    print(f"Successfully wrote the global best configuration to '{output_file}'")
    print(f"Found during: {best_run['target_cavity']} sweep")
    print(f"Run ID: {best_run['run_id']} | Survival: {best_run['survival']} | Emit_s: {best_run['emit_s']:.4e}")


if __name__ == "__main__":
    main()
    export_best_global_configuration(_FILE_PATH)