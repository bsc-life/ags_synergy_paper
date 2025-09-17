import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import matplotlib as mpl

# Set Matplotlib to use fonts that are commonly available on servers
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'dejavusans',
    'font.size': 12,
    'font.sans-serif': ['DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif', 
                        'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Helvetica', 
                        'Avant Garde', 'sans-serif']
})

def generate_variance_grid(experiment_name):
    """
    Loads data for a given experiment and generates a grid of
    box/strip plots to visualize variance.
    """
    print(f"--- Processing Variance Grid for: {experiment_name} ---")

    # --- Data Loading and Preparation ---
    try:
        timing_df = pd.read_csv(f'results/sweep_summaries/final_summary_{experiment_name}.csv')
    except FileNotFoundError:
        print(f"ERROR: Could not find the summary file for experiment '{experiment_name}'. Skipping.")
        print("----------------------------------------------------------\n")
        return

    # Calculate 'delta_time'
    timing_df['delta_time'] = timing_df['user_parameters.drug_X_pulse_period'] - timing_df['user_parameters.drug_Y_pulse_period']

    # Filter out late controls
    late_addition_threshold = 1000
    late_control_runs = (timing_df['delta_time'] == 0) & (timing_df['user_parameters.drug_X_pulse_period'] > late_addition_threshold)
    print(f"Excluding {late_control_runs.sum()} late-addition 'control' runs from analysis.")
    timing_df = timing_df[~late_control_runs]

    # Load control data for normalization
    sweep_summaries_path = "results/sweep_summaries/"
    negative_control_name = "synergy_sweep-3D-0205-1608-control_nodrug"
    try:
        final_summary_negative_control = pd.read_csv(f"{sweep_summaries_path}/final_summary_{negative_control_name}.csv")
        mean_final_number_of_alive_cells_negative_control = round(final_summary_negative_control.iloc[:, -1].mean())
    except FileNotFoundError:
        print(f"ERROR: Could not find negative control file '{negative_control_name}'. Normalization will be incorrect.")
        mean_final_number_of_alive_cells_negative_control = 1

    # Normalize the raw alive cell counts
    timing_df['percent_alive'] = (timing_df['FINAL_NUMBER_OF_ALIVE_CELLS'] / mean_final_number_of_alive_cells_negative_control) * 100
    print(f"NEGATIVE CONTROL (no drug) mean alive cells: {mean_final_number_of_alive_cells_negative_control}")

    # --- Variance Visualization Grid Plot ---
    x_diffs = sorted(timing_df['user_parameters.drug_X_diffusion_coefficient'].unique(), reverse=True)
    y_diffs = sorted(timing_df['user_parameters.drug_Y_diffusion_coefficient'].unique())

    if not x_diffs or not y_diffs:
        print("No data to plot. Skipping plot generation.")
        print("----------------------------------------------------------\n")
        return

    nrows, ncols = len(x_diffs), len(y_diffs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), sharex=True, sharey=True, dpi=300)
    fig.suptitle(f'Variance Grid for {experiment_name}', fontsize=16, fontweight='bold', y=0.98)

    for i, x_diff in enumerate(x_diffs):
        for j, y_diff in enumerate(y_diffs):
            ax = axes[i, j] if nrows > 1 and ncols > 1 else (axes[max(i, j)] if (nrows > 1 or ncols > 1) else axes)
            sub_df = timing_df[(timing_df['user_parameters.drug_X_diffusion_coefficient'] == x_diff) & (timing_df['user_parameters.drug_Y_diffusion_coefficient'] == y_diff)].sort_values('delta_time')
            
            if not sub_df.empty:
                sns.boxplot(x='delta_time', y='percent_alive', data=sub_df, ax=ax, showfliers=False, boxprops=dict(alpha=0.6), medianprops=dict(color="red", linewidth=2), whiskerprops=dict(linestyle='-'), capprops=dict(linestyle='-'))
                sns.stripplot(x='delta_time', y='percent_alive', data=sub_df, ax=ax, jitter=0.2, alpha=0.5, size=3, color='black')
                ax.set_ylim(bottom=0)
                ax.set_title(f'X: {x_diff}, Y: {y_diff}', fontsize=14, fontweight='bold')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.tick_params(axis='x', labelrotation=45, labelsize=9)
                ax.tick_params(axis='y', labelsize=12)
            else:
                ax.axis('off')

    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.15, wspace=0.18, hspace=0.4)
    fig.text(0.005, 0.55, '% Alive Cells (of Control)', va='center', ha='center', rotation='vertical', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.04, 'Delta Time (X - Y)', ha='center', va='center', fontsize=16, fontweight='bold')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = f"{script_dir}/variance_grid_plots"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_filename = f"{save_path}/variance_grid_{experiment_name}.png"
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    print(f"\nVariance grid plot saved to: {output_filename}")
    plt.close(fig)
    print("----------------------------------------------------------\n")

def main():
    """
    Main function to run the analysis for all specified experiments.
    """
    # List of experiment names to process.
    experiment_names = [
        "synergy_sweep-akt_mek-2606-1819-4p_3D_drugtiming_synonly_consensus_hybrid_20",
        "synergy_sweep-pi3k_mek-2606-1819-4p_3D_drugtiming_synonly_consensus_hybrid_20"
    ]

    for exp_name in experiment_names:
        generate_variance_grid(exp_name)

if __name__ == "__main__":
    main() 