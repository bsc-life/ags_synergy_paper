import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.lines import Line2D

def plot_top_parameter_comparison(experiment_name, top_n=10):
    """
    Creates a grouped bar chart to visualize the effectiveness of the top N
    synergistic parameter sets.

    This plot directly compares the outcome of the specific combination run
    against the mean outcomes of the control and single-drug proxies.
    It also adds markers to show the best-case outcome for the proxies.
    """
    print(f"--- Plotting Top Parameter Comparison for: {experiment_name} ---")

    # --- Configuration ---
    RESULTS_PATH = "scripts/post_emews_analysis/synergy_recovery_experiments/synergy_analysis_results"
    PLOT_SAVE_PATH = f"{RESULTS_PATH}/plots"
    INPUT_FILENAME = f"{RESULTS_PATH}/top_synergistic_params_quantified_{experiment_name}.csv"

    # --- Create Save Directory ---
    if not os.path.exists(PLOT_SAVE_PATH):
        os.makedirs(PLOT_SAVE_PATH)
        print(f"Created directory: {PLOT_SAVE_PATH}")

    # --- Data Loading ---
    try:
        df = pd.read_csv(INPUT_FILENAME)
    except FileNotFoundError:
        print(f"ERROR: Could not find results file '{INPUT_FILENAME}'. Skipping.")
        return

    if df.empty:
        print(f"WARNING: The results file '{INPUT_FILENAME}' is empty. Nothing to plot.")
        return

    # --- Data Preparation for Plotting ---
    df = df.head(top_n)
    
    # Data for the bars will be the mean values
    plot_df = df[['outcome_mean_control', 'outcome_mean_drug_A', 'outcome_mean_drug_B', 'outcome_actual_combo']].copy()
    plot_df.columns = ['Control (Mean)', 'Drug A (Mean)', 'Drug B (Mean)', 'Combo (Actual)']
    plot_df.index = [f'Top Run {i+1}' for i in range(len(plot_df))]

    # Get the best-case data for the markers
    best_case_values = df[['outcome_best_control', 'outcome_best_drug_A', 'outcome_best_drug_B']]

    # --- Plotting ---
    ax = plot_df.plot(
        kind='bar',
        figsize=(18, 10),
        width=0.8,
        color=['#c0c0c0', '#87ceeb', '#f08080', '#2e8b57'] # Silver, SkyBlue, LightCoral, SeaGreen
    )

    # --- Customization ---
    ax.set_title(f'Top {top_n} Synergistic Sets: Best Combo Run vs. Control & Single Drug Performance\n({experiment_name})', fontsize=16)
    ax.set_ylabel('Number of Alive Cells (Lower is Better)', fontsize=12)
    ax.set_xlabel('Individual Top-Performing Parameter Sets', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Add Markers for Best-Case Proxies ---
    num_groups = len(plot_df)
    num_bars_per_group = len(plot_df.columns)
    bar_width = ax.patches[0].get_width()

    # Pandas plots all bars for one category, then the next, etc.
    # So the first `num_groups` patches are for 'Control (Mean)', the next for 'Drug A (Mean)', etc.
    for i in range(num_groups):
        # Control marker
        control_bar = ax.patches[i]
        control_x = control_bar.get_x() + control_bar.get_width() / 2
        control_y_best = best_case_values.iloc[i]['outcome_best_control']
        ax.plot([control_x - bar_width*0.4, control_x + bar_width*0.4], [control_y_best, control_y_best], color='black', marker='_', mew=2, markersize=8)

        # Drug A marker
        drug_a_bar = ax.patches[i + num_groups]
        drug_a_x = drug_a_bar.get_x() + drug_a_bar.get_width() / 2
        drug_a_y_best = best_case_values.iloc[i]['outcome_best_drug_A']
        ax.plot([drug_a_x - bar_width*0.4, drug_a_x + bar_width*0.4], [drug_a_y_best, drug_a_y_best], color='black', marker='_', mew=2, markersize=8)
        
        # Drug B marker
        drug_b_bar = ax.patches[i + 2 * num_groups]
        drug_b_x = drug_b_bar.get_x() + drug_b_bar.get_width() / 2
        drug_b_y_best = best_case_values.iloc[i]['outcome_best_drug_B']
        ax.plot([drug_b_x - bar_width*0.4, drug_b_x + bar_width*0.4], [drug_b_y_best, drug_b_y_best], color='black', marker='_', mew=2, markersize=8)

    # Add value labels on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', label_type='edge', fontsize=9, padding=3)

    # --- Custom Legend ---
    handles, labels = ax.get_legend_handles_labels()
    best_case_marker = Line2D([0], [0], color='black', marker='_', mew=2, markersize=8, label='Best Single Run Outcome', linestyle='None')
    handles.append(best_case_marker)
    ax.legend(handles=handles, title='Treatment', fontsize=10)

    plt.tight_layout(pad=1.5)

    # --- Save Figure ---
    output_filename = f"{PLOT_SAVE_PATH}/top_params_comparison_with_bests_{experiment_name}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Enhanced comparison plot saved to: {output_filename}")
    print("----------------------------------------------------------\n")
    plt.close()


def main():
    """
    Main function to run the plotting for all specified experiments.
    """
    experiment_names = [
        "synergy_sweep-akt_mek-2606-1819-4p_3D_drugtiming_synonly_consensus_hybrid_20",
        "synergy_sweep-pi3k_mek-2606-1819-4p_3D_drugtiming_synonly_consensus_hybrid_20"
    ]

    for exp_name in experiment_names:
        plot_top_parameter_comparison(exp_name)

if __name__ == "__main__":
    main() 