import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import logging
import numpy as np
import gc

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use a non-interactive backend suitable for scripts
plt.switch_backend('agg')

# --- Script Configuration ---
# Set this to True to only print column names from the first found simulation file and exit.
# Set to False to run the full 3D plotting analysis.
PRINT_COLUMNS_ONLY = False

def find_closest_timepoint(available_times, target_time):
    """Finds the closest available time in a list of times."""
    available_times = np.asarray(available_times)
    idx = (np.abs(available_times - target_time)).argmin()
    return available_times[idx]

def plot_3d_snapshots_grid(df, node_pair, output_path, title_prefix):
    """
    Creates and saves a 1x4 grid of 3D scatter plots for 4 timepoints,
    with cells colored categorically by the activation state of two nodes.
    """
    if df.empty:
        logging.warning(f"Cannot plot 3D snapshot grid for '{title_prefix}'. Data is empty.")
        return

    # 1. Define synergy states and colors
    node1, node2 = node_pair[0], node_pair[1]
    node1_label = node1.replace('_node', '').upper()
    node2_label = node2.replace('_node', '').upper()
    
    colors = {
        0: ('#99DDCC', f'{node1_label} OFF / {node2_label} OFF'), # Light Teal
        1: ('#0077BB', f'{node1_label} ON / {node2_label} OFF'),  # Blue
        2: ('#EE7733', f'{node1_label} OFF / {node2_label} ON'),  # Orange
        3: ('#AA3377', f'{node1_label} ON / {node2_label} ON'),   # Purple
        4: ('#000000', 'Dead')                                    # Black
    }
    
    # 2. Add synergy_state column to DataFrame
    node1_on = df[node1] > 0.5
    node2_on = df[node2] > 0.5

    df['synergy_state'] = 0  # Default: both OFF
    df.loc[node1_on & ~node2_on, 'synergy_state'] = 1
    df.loc[~node1_on & node2_on, 'synergy_state'] = 2
    df.loc[node1_on & node2_on, 'synergy_state'] = 3

    # Override with dead state if 'current_phase' column exists
    if 'current_phase' in df.columns:
        dead_cells_mask = df['current_phase'] == 'apoptotic'
        df.loc[dead_cells_mask, 'synergy_state'] = 4
    else:
        logging.warning("'current_phase' column not found, cannot color dead cells.")

    # 3. Create the plot grid
    target_timepoints = [6000]  # Only final snapshot requested
    fig, axes = plt.subplots(1, len(target_timepoints), figsize=(7*len(target_timepoints), 7), dpi=150, subplot_kw={'projection': '3d'})
    if len(target_timepoints)==1:
        axes=[axes]
    available_times = df['time'].unique()

    for i, t_target in enumerate(target_timepoints):
        ax = axes[i]
        actual_time = find_closest_timepoint(available_times, t_target)
        data_t = df[df['time'] == actual_time].copy()

        if data_t.empty:
            ax.text2D(0.5, 0.5, f'No data at t={int(actual_time)}', transform=ax.transAxes, ha='center')
            continue

        # Plotting
        ax.scatter(data_t['position_x'], data_t['position_y'], data_t['position_z'],
                             c=data_t['synergy_state'].map({k: v[0] for k, v in colors.items()}),
                             s=30, alpha=0.7) # Increased point size and transparency

        # Aspect ratio and limits (based on the full dataset for consistency)
        x_min, x_max = df['position_x'].min(), df['position_x'].max()
        y_min, y_max = df['position_y'].min(), df['position_y'].max()
        z_min, z_max = df['position_z'].min(), df['position_z'].max()
        
        mid_x, mid_y, mid_z = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
        max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0
    
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Labels and view
        ax.set_xlabel("X (μm)"); ax.set_ylabel("Y (μm)"); ax.set_zlabel("Z (μm)")
        ax.set_title(f"t = {int(actual_time)} min", fontsize=12)
        ax.view_init(elev=20, azim=-65)

    # 4. Create a single legend for the figure
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=color, markersize=10) for color, label in colors.values()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05), fontsize=12)

    fig.suptitle(title_prefix, fontsize=16, y=0.98) # Main title
    
    plt.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust layout to make space for suptitle and legend
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved 3D snapshot grid to {output_path}")

def process_snapshot_scenarios(exp_name, short_exp_name, output_base_dir):
    """Selects one run per delta_time scenario at D=600/600 and plots final 3-D snapshot."""
    summary_path = f'results/sweep_summaries/final_summary_{exp_name}.csv'
    if not os.path.exists(summary_path):
        logging.error(f"Summary file not found for {exp_name}. Skipping.")
        return
    df = pd.read_csv(summary_path)
    df['delta_time'] = df['user_parameters.drug_X_pulse_period'] - df['user_parameters.drug_Y_pulse_period']
    dose_filter = (df['user_parameters.drug_X_diffusion_coefficient']==600.0) & (df['user_parameters.drug_Y_diffusion_coefficient']==600.0)
    df = df[dose_filter]
    if df.empty:
        logging.warning(f"No D=600/600 runs for {exp_name}.")
        return
    # For each unique delta_time choose first replicate
    for dt, row in df.groupby('delta_time').head(1).iterrows():
        instance_folder = f"instance_{int(row['individual'])}_{int(row['replicate'])}"
        if 'iteration' in row and pd.notna(row['iteration']):
            instance_folder = f"instance_{int(row['iteration'])}_{instance_folder}"
        sim_file_path = os.path.join("experiments", exp_name, instance_folder, 'pcdl_total_info_sim.csv.gz')
        if not os.path.exists(sim_file_path):
            logging.warning(f"Simulation file missing: {sim_file_path}")
            continue
        try:
            sim_df = pd.read_csv(sim_file_path, compression='gzip')
        except Exception as e:
            logging.error(f"Failed reading {sim_file_path}: {e}")
            continue
        node_pair = ['pi3k_node','mek_node'] if 'pi3k' in exp_name.lower() else ['akt_node','mek_node']
        if not all(n in sim_df.columns for n in node_pair):
            logging.warning(f"Required nodes {node_pair} not in columns for {sim_file_path}")
            continue
        scenario_name=f"dt_{int(row['delta_time'])}"
        out_dir=os.path.join(output_base_dir, exp_name)
        os.makedirs(out_dir,exist_ok=True)
        output_path=os.path.join(out_dir,f"snapshot_{scenario_name}.png")
        title=f"{short_exp_name} Δt={int(row['delta_time'])}"
        plot_3d_snapshots_grid(sim_df, node_pair, output_path, title)
        del sim_df
        gc.collect()

def main():
    experiments = {
        "synergy_sweep-pi3k_mek-2606-1819-4p_3D_drugtiming_synonly_consensus_hybrid_20": "PI3Ki-MEKi",
        "synergy_sweep-akt_mek-2606-1819-4p_3D_drugtiming_synonly_consensus_hybrid_20": "AKTi-MEKi"
    }
    base_output_dir = "results/final_snapshot_3d"
    for exp, short in experiments.items():
        logging.info(f"Processing snapshots for {exp}")
        process_snapshot_scenarios(exp, short, base_output_dir)

if __name__=='__main__':
    main()