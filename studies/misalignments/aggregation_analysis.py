import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

_JSON_RESULTS_FILE = "alignment_results.json"

# Define the element families based on the OPAL baseline lattice
_FAMILIES = {
    "162.5 MHz Cavities": [f"CAV{i:03d}" for i in range(1, 22)],
    "325 MHz Cavities": [f"CAV{i:03d}" for i in range(22, 50)],
    "650 MHz Cavities": [f"CAV{i:03d}" for i in range(50, 122)],
    "Solenoids": [f"SOL{i:03d}" for i in range(1, 56)]
}

def load_and_flatten_data():
    if not os.path.exists(_JSON_RESULTS_FILE):
        print(f"Cannot find {_JSON_RESULTS_FILE}.")
        return None, None

    with open(_JSON_RESULTS_FILE, 'r') as f:
        scan_data = json.load(f)

    rows = []
    losses = []

    for run in scan_data:
        eval_data = run.get('evaluation')
        if eval_data is not None and 'particles_lost' in eval_data:
            inputs = run.get('inputs', {})
            
            # Flatten the nested dictionary (e.g., 'CAV001': {'dx': 0.001} -> 'CAV001_dx': 0.001)
            flat_inputs = {}
            for elem, errors in inputs.items():
                for axis, val in errors.items():
                    # Convert m to mm and rad to mrad to keep coefficients scaled nicely
                    scale = 1e3
                    flat_inputs[f"{elem}_{axis}"] = val * scale
            
            rows.append(flat_inputs)
            losses.append(eval_data['particles_lost'])

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    target = np.array(losses)
    return df, target

def calculate_rss(inputs, element_filter=None):
    """Calculates the Root Sum Square for positional and rotational errors."""
    pos_sq = 0.0
    rot_sq = 0.0
    
    for elem, errors in inputs.items():
        if element_filter and elem not in element_filter:
            continue
            
        pos_sq += errors.get('dx', 0.0)**2 + errors.get('dy', 0.0)**2
        rot_sq += errors.get('dtheta', 0.0)**2 + errors.get('dphi', 0.0)**2
        
    return np.sqrt(pos_sq), np.sqrt(rot_sq)

def plot_global_error_scatter(scan_data):
    """Generates scatter plots for global translation vs rotation errors."""
    pos_errors = []
    rot_errors = []
    losses = []
    
    for run in scan_data:
        eval_data = run.get('evaluation')
        # Only evaluate runs that successfully recorded particle stats
        if eval_data is not None and 'particles_lost' in eval_data:
            inputs = run.get('inputs', {})
            e_pos, e_rot = calculate_rss(inputs)
            
            pos_errors.append(e_pos * 1e3) # Convert to mm
            rot_errors.append(e_rot * 1e3) # Convert to mrad for readability
            losses.append(eval_data['particles_lost'])
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Global Misalignment Magnitude vs. Particle Loss", fontsize=16)
    
    # Translation Plot
    ax1.scatter(pos_errors, losses, alpha=0.5, color='blue', edgecolor='k')
    ax1.set_title("Positional Error (Shifts)")
    ax1.set_xlabel("Global Positional RSS [mm]")
    ax1.set_ylabel("Particles Lost")
    ax1.grid(True, alpha=0.3)
    
    # Rotation Plot
    ax2.scatter(rot_errors, losses, alpha=0.5, color='red', edgecolor='k')
    ax2.set_title("Rotational Error (Tilts)")
    ax2.set_xlabel("Global Rotational RSS [mrad]")
    ax2.set_ylabel("Particles Lost")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("scatter_global_errors.png", dpi=300)
    plt.show()

def plot_family_error_scatter(scan_data):
    """Generates scatter plots isolating the positional errors of specific lattice families."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Positional Misalignment (RSS) by Component Family vs. Particle Loss", fontsize=16)
    
    axes = axes.flatten()
    colors = ['purple', 'orange', 'green', 'teal']
    
    for idx, (family_name, elements) in enumerate(_FAMILIES.items()):
        family_pos_errors = []
        losses = []
        
        for run in scan_data:
            eval_data = run.get('evaluation')
            if eval_data is not None and 'particles_lost' in eval_data:
                inputs = run.get('inputs', {})
                e_pos, _ = calculate_rss(inputs, element_filter=elements)
                
                family_pos_errors.append(e_pos * 1e3) # Convert to mm
                losses.append(eval_data['particles_lost'])
                
        ax = axes[idx]
        ax.scatter(family_pos_errors, losses, alpha=0.5, color=colors[idx], edgecolor='k')
        ax.set_title(family_name)
        ax.set_xlabel(f"{family_name} Positional RSS [mm]")
        ax.set_ylabel("Particles Lost")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("scatter_family_errors.png", dpi=300)
    plt.show()


def plot_spearman_correlation(df, target, top_n=20):
    correlations = {}
    for col in df.columns:
        coef, p_val = spearmanr(df[col], target)
        # We take the absolute value because a strong negative correlation 
        # is just as impactful as a strong positive one
        correlations[col] = abs(coef) if not np.isnan(coef) else 0.0

    # Sort and take top N
    sorted_corr = sorted(correlations.items(), key=lambda item: item[1], reverse=True)[:top_n]
    features = [x[0] for x in sorted_corr]
    scores = [x[1] for x in sorted_corr]

    plt.figure(figsize=(10, 8))
    plt.barh(features[::-1], scores[::-1], color='steelblue', edgecolor='black')
    plt.title(f"Top {top_n} Elements by Spearman Rank Correlation to Particle Loss")
    plt.xlabel("Absolute Spearman Correlation Coefficient (|ρ|)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("spearman_correlation_top20.png", dpi=300)
    plt.show()

def plot_random_forest_importance(df, target, top_n=20):
    # Initialize and train the Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(df, target)

    # Extract feature importances
    importances = rf.feature_importances_
    
    # Sort and take top N
    indices = np.argsort(importances)[::-1][:top_n]
    features = [df.columns[i] for i in indices]
    scores = [importances[i] for i in indices]

    plt.figure(figsize=(10, 8))
    plt.barh(features[::-1], scores[::-1], color='darkseagreen', edgecolor='black')
    plt.title(f"Top {top_n} Elements by Random Forest Feature Importance")
    plt.xlabel("Relative Importance (Gini Importance)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("random_forest_importance_top20.png", dpi=300)
    plt.show()
    
def main():
    if not os.path.exists(_JSON_RESULTS_FILE):
        print(f"Cannot find {_JSON_RESULTS_FILE}. Please run the simulation sweep first.")
        return

    with open(_JSON_RESULTS_FILE, 'r') as f:
        scan_data = json.load(f)

    print(f"Analyzing {len(scan_data)} runs for error correlations...")
    plot_global_error_scatter(scan_data)
    plot_family_error_scatter(scan_data)

    print("Loading data and building feature matrix...")
    df, target = load_and_flatten_data()
    
    if df is None:
        return
    print(f"Data loaded: {df.shape[0]} runs, {df.shape[1]} individual alignment features.")
    
    # Check if there is variation in target
    if np.all(target == target[0]):
        print("All runs resulted in the exact same particle loss. Cannot compute correlations.")
        return

    print("Computing Spearman Rank Correlations...")
    plot_spearman_correlation(df, target)
    
    print("Training Random Forest and extracting Feature Importances...")
    plot_random_forest_importance(df, target)

if __name__ == '__main__':
    main()