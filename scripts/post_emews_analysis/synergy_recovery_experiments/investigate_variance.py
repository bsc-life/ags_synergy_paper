import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib as mpl
import numpy as np

# Set up publication-quality plotting fonts
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'dejavusans',
    'font.size': 12,
    'font.sans-serif': ['DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif', 
                        'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Helvetica', 
                        'Avant Garde', 'sans-serif']
})

def investigate_scenario_variance(experiments, target_x_diff, target_y_diff):
    """
    Creates a plot to investigate the variance of simulation replicates
    for a specific diffusion coefficient pair, highlighting delta_time values.
    """
    save_path = "scripts/post_emews_analysis/synergy_recovery_experiments/optimal_timings_synergy/variance_investigation"
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=300, sharey=True)
    
    all_plot_data = []

    for exp_name, title in experiments.items():
        summary_path = f"results/sweep_summaries/final_summary_{exp_name}.csv"
        if not os.path.exists(summary_path):
            print(f"Summary file not found for {exp_name}, skipping.")
            continue

        summary_df = pd.read_csv(summary_path)

        # Calculate delta_time
        summary_df['delta_time'] = summary_df['user_parameters.drug_X_pulse_period'] - summary_df['user_parameters.drug_Y_pulse_period']

        # --- FIX: Exclude "late simultaneous" runs that act as controls ---
        late_addition_threshold = 1000
        late_control_runs = (summary_df['delta_time'] == 0) & (summary_df['user_parameters.drug_X_pulse_period'] > late_addition_threshold)
        summary_df = summary_df[~late_control_runs]
        # --------------------------------------------------------------------

        # Filter for the specific scenario
        scenario_df = summary_df[
            (summary_df['user_parameters.drug_X_diffusion_coefficient'] == target_x_diff) &
            (summary_df['user_parameters.drug_Y_diffusion_coefficient'] == target_y_diff)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        if scenario_df.empty:
            print(f"No data for scenario D(X)={target_x_diff}, D(Y)={target_y_diff} in {exp_name}")
            continue

        # Add experiment info for plotting
        scenario_df['experiment'] = title
        all_plot_data.append(scenario_df)

    if not all_plot_data:
        print("No data found for any experiment. Aborting plot generation.")
        plt.close(fig)
        return
        
    combined_df = pd.concat(all_plot_data, ignore_index=True)

    # --- Create plots ---
    for i, (exp_name, title) in enumerate(experiments.items()):
        ax = axes[i]
        exp_data = combined_df[combined_df['experiment'] == title].sort_values('delta_time')

        if exp_data.empty:
            ax.text(0.5, 0.5, "Data not found", ha='center', va='center')
            ax.set_title(title, weight='bold')
            continue

        # Use seaborn for a more informative plot
        sns.boxplot(x='delta_time', y='FINAL_NUMBER_OF_ALIVE_CELLS', data=exp_data, ax=ax,
                    whis=[0, 100], width=.6, palette="vlag", showfliers=False)
        sns.stripplot(x='delta_time', y='FINAL_NUMBER_OF_ALIVE_CELLS', data=exp_data, ax=ax,
                      size=4, color=".3", linewidth=0)

        ax.set_title(title, weight='bold')
        ax.set_xlabel('Delta Time (min)', weight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Final figure adjustments
    axes[0].set_ylabel('Final Number of Alive Cells (Individual Replicates)', weight='bold')
    fig.suptitle(f'Variance of Replicates for D(X)={target_x_diff}, D(Y)={target_y_diff}', fontsize=16, weight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_save_path = os.path.join(save_path, f"variance_dist_X_{target_x_diff}_Y_{target_y_diff}.png")
    plt.savefig(plot_save_path, dpi=600)
    plt.savefig(plot_save_path.replace('.png', '.svg'), format='svg')
    print(f"Variance investigation plot saved to: {plot_save_path}")
    plt.close(fig)


if __name__ == "__main__":
    experiments_to_run = {
        "synergy_sweep-pi3k_mek-1606-0214-4p_3D_drugtiming": "PI3Ki + MEKi",
        "synergy_sweep-akt_mek-1606-0214-4p_3D_drugtiming": "AKTi + MEKi"
    }
    
    # --- Investigate variance for the high-dose symmetric scenario ---
    print("\n--- Investigating variance for the high-dose symmetric scenario ---")
    investigate_scenario_variance(experiments_to_run, target_x_diff=600.0, target_y_diff=600.0) 