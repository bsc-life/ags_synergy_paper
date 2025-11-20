import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# This script plots the high-dose comparison for a given synergy sweep experiment
# The input is the summary CSV file for the synergy sweep experiment
# The output is a side-by-side plot comparing the high-dose symmetric scenario across multiple experiments
# This script generates the Figure XYZ from the main text of the paper


# Set up publication-quality plotting fonts
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'dejavusans',
    'font.size': 12,
    'font.sans-serif': ['DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif', 
                        'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Helvetica', 
                        'Avant Garde', 'sans-serif']
})

def create_high_dose_comparison_plot(experiments):
    """
    Creates a side-by-side plot comparing the high-dose symmetric scenario
    across multiple experiments.
    """
    # Define base path for data and plots
    save_path = "scripts/post_emews_analysis/synergy_recovery_experiments/optimal_timings_synergy"
    
    # Create a 1x2 figure for side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)
    
    plotted_anything = False

    for i, (exp_name, title) in enumerate(experiments.items()):
        ax = axes[i]
        
        # Load the metrics data
        metrics_path = os.path.join(save_path, f"combined_synergy_metrics_{exp_name}.csv")
        if not os.path.exists(metrics_path):
            print(f"Metrics file not found for {exp_name}, skipping subplot.")
            ax.text(0.5, 0.5, "Data not found", ha='center', va='center')
            ax.set_title(title, weight='bold')
            continue

        metrics_df = pd.read_csv(metrics_path)

        # Filter for high-dose symmetric scenario (D=600)
        high_dose_df = metrics_df[(metrics_df['x_diff'] == 600) & (metrics_df['y_diff'] == 600)].sort_values('delta_time')

        if high_dose_df.empty:
            print(f"No high-dose symmetric data for {exp_name}, skipping subplot.")
            ax.text(0.5, 0.5, "No high-dose data", ha='center', va='center')
            ax.set_title(title, weight='bold')
            continue
            
        plotted_anything = True

        # Plot efficacy (% alive cells)
        color = 'tab:red'
        ax.plot(high_dose_df['delta_time'], high_dose_df['percent_alive'], color=color, marker='o', linestyle='-')
        
        # Invert y-axis as lower % alive is better efficacy
        ax.invert_yaxis()
        
        # Set labels and title for the subplot
        ax.set_xlabel('Delta Time (X - Y) (min)', weight='bold')
        ax.set_title(title, weight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)

    # Set shared y-label for the entire figure
    if plotted_anything:
        axes[0].set_ylabel('% Alive Cells (Efficacy)', weight='bold')
    
    # Set a super title for the whole figure
    fig.suptitle('Efficacy in High-Dose Symmetric Scenario (D=600)', fontsize=16, weight='bold')
    
    # Adjust layout to prevent titles/labels overlapping
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the combined figure if we plotted at least one subplot
    if plotted_anything:
        plot_save_path = os.path.join(save_path, "high_dose_symmetric_efficacy_comparison.png")
        plt.savefig(plot_save_path, dpi=600)
        plt.savefig(plot_save_path.replace('.png', '.svg'), format='svg')
        print(f"\nCombined efficacy comparison plot saved to {plot_save_path}")
    else:
        print("\nCould not generate comparison plot as no data was found for any experiment.")
        
    plt.close(fig)

def create_individual_scenario_plots(experiments):
    """
    Generates a separate side-by-side comparison plot for each unique
    diffusion coefficient scenario found in the data.
    """
    # Define paths
    base_path = "scripts/post_emews_analysis/synergy_recovery_experiments/optimal_timings_synergy"
    plot_dir = os.path.join(base_path, "efficacy_comparison_by_scenario") # New directory
    os.makedirs(plot_dir, exist_ok=True)

    # Use data from the first experiment to find all scenarios
    first_exp_name = list(experiments.keys())[0]
    metrics_path = os.path.join(base_path, f"combined_synergy_metrics_{first_exp_name}.csv")
    if not os.path.exists(metrics_path):
        print(f"Base metrics file not found ({metrics_path}), cannot generate individual scenario plots.")
        return
    
    metrics_df = pd.read_csv(metrics_path)
    scenarios = metrics_df[['x_diff', 'y_diff']].drop_duplicates()

    # Loop through each scenario and create a plot
    for _, scenario in scenarios.iterrows():
        x_diff, y_diff = scenario['x_diff'], scenario['y_diff']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)
        plotted_anything = False

        # Create the side-by-side plot for this specific scenario
        for i, (exp_name, title) in enumerate(experiments.items()):
            ax = axes[i]
            exp_metrics_path = os.path.join(base_path, f"combined_synergy_metrics_{exp_name}.csv")
            
            if not os.path.exists(exp_metrics_path):
                print(f"Metrics file for {exp_name} not found, skipping subplot.")
                ax.text(0.5, 0.5, "Data not found", ha='center', va='center')
                ax.set_title(title, weight='bold')
                continue
            
            exp_metrics_df = pd.read_csv(exp_metrics_path)
            scenario_df = exp_metrics_df[(exp_metrics_df['x_diff'] == x_diff) & (exp_metrics_df['y_diff'] == y_diff)].sort_values('delta_time')

            if scenario_df.empty:
                ax.text(0.5, 0.5, "No data for this scenario", ha='center', va='center')
                ax.set_title(title, weight='bold')
                continue
            
            plotted_anything = True
            color = 'tab:red'
            ax.plot(scenario_df['delta_time'], scenario_df['percent_alive'], color=color, marker='o', linestyle='-')
            ax.invert_yaxis()
            ax.set_xlabel('Delta Time (X - Y) (min)', weight='bold')
            ax.set_title(title, weight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)

        if plotted_anything:
            axes[0].set_ylabel('% Alive Cells (Efficacy)', weight='bold')
            fig.suptitle(f'Efficacy Comparison for D(X)={x_diff}, D(Y)={y_diff}', fontsize=16, weight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            
            plot_save_path = os.path.join(plot_dir, f"efficacy_comparison_X_{x_diff}_Y_{y_diff}.png")
            plt.savefig(plot_save_path, dpi=600)
            plt.savefig(plot_save_path.replace('.png', '.svg'), format='svg')
            print(f"Saved comparison plot to {plot_save_path}")
        
        plt.close(fig)

def create_individual_scenario_bar_plots(experiments):
    """
    Generates a separate bar plot with error bars for each unique
    diffusion coefficient scenario found in the data.
    """
    # Define paths
    base_path = "scripts/post_emews_analysis/synergy_recovery_experiments/optimal_timings_synergy"
    plot_dir = os.path.join(base_path, "efficacy_barplot_by_scenario") # New directory for bar plots
    os.makedirs(plot_dir, exist_ok=True)

    # --- Load Control Data for Normalization ---
    sweep_summaries_path = "results/sweep_summaries/"
    negative_control_name = "synergy_sweep-3D-0205-1608-control_nodrug"
    control_path = f"{sweep_summaries_path}/final_summary_{negative_control_name}.csv"
    if not os.path.exists(control_path):
        print(f"Negative control file not found at {control_path}. Cannot generate bar plots with error bars.")
        return
    final_summary_negative_control = pd.read_csv(control_path)
    # The final cell count is the last column
    mean_final_number_of_alive_cells_negative_control = final_summary_negative_control.iloc[:, -1].mean()

    # Use data from the first experiment to find all scenarios
    first_exp_name = list(experiments.keys())[0]
    metrics_path = os.path.join(base_path, f"combined_synergy_metrics_{first_exp_name}.csv")
    if not os.path.exists(metrics_path):
        print(f"Base metrics file not found ({metrics_path}), cannot generate individual scenario plots.")
        return
    
    metrics_df = pd.read_csv(metrics_path)
    scenarios = metrics_df[['x_diff', 'y_diff']].drop_duplicates()

    # Loop through each scenario and create a plot
    for _, scenario in scenarios.iterrows():
        x_diff, y_diff = scenario['x_diff'], scenario['y_diff']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)
        plotted_anything = False

        # Create the side-by-side plot for this specific scenario
        for i, (exp_name, title) in enumerate(experiments.items()):
            ax = axes[i]
            exp_metrics_path = os.path.join(base_path, f"combined_synergy_metrics_{exp_name}.csv")
            
            if not os.path.exists(exp_metrics_path):
                print(f"Metrics file for {exp_name} not found, skipping subplot.")
                ax.text(0.5, 0.5, "Data not found", ha='center', va='center')
                ax.set_title(title, weight='bold')
                continue
            
            exp_metrics_df = pd.read_csv(exp_metrics_path)
            
            # --- Normalize the standard deviation ---
            exp_metrics_df['norm_std_alive'] = (exp_metrics_df['std_alive'] / mean_final_number_of_alive_cells_negative_control) * 100

            scenario_df = exp_metrics_df[(exp_metrics_df['x_diff'] == x_diff) & (exp_metrics_df['y_diff'] == y_diff)].sort_values('delta_time')

            if scenario_df.empty:
                ax.text(0.5, 0.5, "No data for this scenario", ha='center', va='center')
                ax.set_title(title, weight='bold')
                continue
            
            plotted_anything = True
            
            # Use a color map for delta_time to distinguish bars
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(scenario_df['delta_time'])))
            
            # Use categorical plotting for the x-axis
            x_labels = [str(int(dt)) for dt in scenario_df['delta_time']]
            x_pos = np.arange(len(x_labels))

            ax.bar(x_pos, scenario_df['percent_alive'], 
                   yerr=scenario_df['norm_std_alive'],
                   color=colors, capsize=4) # Width is handled better automatically

            ax.invert_yaxis()
            ax.set_xlabel('Delta Time (X - Y) (min)', weight='bold')
            ax.set_title(title, weight='bold')
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Set the x-tick labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha="right")

        if plotted_anything:
            axes[0].set_ylabel('% Alive Cells (Efficacy)', weight='bold')
            fig.suptitle(f'Efficacy Comparison for D(X)={x_diff}, D(Y)={y_diff}', fontsize=16, weight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            
            plot_save_path = os.path.join(plot_dir, f"efficacy_barplot_X_{x_diff}_Y_{y_diff}.png")
            plt.savefig(plot_save_path, dpi=600)
            plt.savefig(plot_save_path.replace('.png', '.svg'), format='svg')
            print(f"Saved bar plot to {plot_save_path}")
        
        plt.close(fig)

def create_aggregated_timing_barplot(experiments, target_x_diff, target_y_diff):
    """
    Generates a bar plot that aggregates timings into three categories:
    'Drug X First', 'Simultaneous', and 'Drug Y First' for a specific scenario.
    """
    # 1. Define paths
    base_path = "scripts/post_emews_analysis/synergy_recovery_experiments/optimal_timings_synergy"
    plot_dir = os.path.join(base_path, "efficacy_aggregated_barplot")
    os.makedirs(plot_dir, exist_ok=True)

    # 2. Load Control Data for Normalization
    sweep_summaries_path = "results/sweep_summaries/"
    negative_control_name = "synergy_sweep-3D-0205-1608-control_nodrug"
    control_path = f"{sweep_summaries_path}/final_summary_{negative_control_name}.csv"
    if not os.path.exists(control_path):
        print(f"Control file not found at {control_path}. Cannot normalize.")
        return
    final_summary_negative_control = pd.read_csv(control_path)
    mean_control_alive = final_summary_negative_control.iloc[:, -1].mean()

    # 3. Create Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300, sharey=True)
    
    # 4. Loop through experiments and process data
    for i, (exp_name, title) in enumerate(experiments.items()):
        ax = axes[i]
        
        summary_path = f"results/sweep_summaries/final_summary_{exp_name}.csv"
        if not os.path.exists(summary_path):
            ax.text(0.5, 0.5, "Data not found", ha='center', va='center')
            ax.set_title(title, weight='bold')
            continue

        df = pd.read_csv(summary_path)
        df['delta_time'] = df['user_parameters.drug_X_pulse_period'] - df['user_parameters.drug_Y_pulse_period']
        
        late_addition_threshold = 1000
        late_control_runs = (df['delta_time'] == 0) & (df['user_parameters.drug_X_pulse_period'] > late_addition_threshold)
        df = df[~late_control_runs]
        
        scenario_df = df[
            (df['user_parameters.drug_X_diffusion_coefficient'] == target_x_diff) &
            (df['user_parameters.drug_Y_diffusion_coefficient'] == target_y_diff)
        ].copy()

        if scenario_df.empty:
            ax.text(0.5, 0.5, "No data for this scenario", ha='center', va='center')
            ax.set_title(title, weight='bold')
            continue
            
        # 5. Create timing categories
        drug_x_name = "PI3Ki" if "pi3k" in exp_name.lower() else "AKTi"
        
        def categorize_timing(dt):
            if dt < 0:
                return f'{drug_x_name} First'
            elif dt == 0:
                return 'Simultaneous'
            else:
                return 'MEKi First'

        scenario_df['Timing Category'] = scenario_df['delta_time'].apply(categorize_timing)
        
        # 6. Aggregate by the new category
        aggregated_data = scenario_df.groupby('Timing Category')['FINAL_NUMBER_OF_ALIVE_CELLS'].agg(['mean', 'std']).reset_index()
        
        # 7. Normalize results
        aggregated_data['percent_alive'] = (aggregated_data['mean'] / mean_control_alive) * 100
        aggregated_data['norm_std_alive'] = (aggregated_data['std'] / mean_control_alive) * 100
        
        category_order = [f'{drug_x_name} First', 'Simultaneous', 'MEKi First']
        aggregated_data['Timing Category'] = pd.Categorical(aggregated_data['Timing Category'], categories=category_order, ordered=True)
        aggregated_data = aggregated_data.sort_values('Timing Category')
        
        # 8. Plotting
        colors = ['#4c72b0', '#c44e52', '#8172b2']
        bar_positions = np.arange(len(aggregated_data['Timing Category']))
        ax.bar(bar_positions, aggregated_data['percent_alive'],
               yerr=aggregated_data['norm_std_alive'],
               capsize=5, color=colors,
               edgecolor='black')
               
        ax.set_title(title, weight='bold')
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(aggregated_data['Timing Category'], rotation=45, ha="right")

        # 9. Perform and annotate pairwise statistical tests against 'Simultaneous'
        drug_x_first_data = scenario_df[scenario_df['Timing Category'] == f'{drug_x_name} First']['FINAL_NUMBER_OF_ALIVE_CELLS']
        simultaneous_data = scenario_df[scenario_df['Timing Category'] == 'Simultaneous']['FINAL_NUMBER_OF_ALIVE_CELLS']
        meki_first_data = scenario_df[scenario_df['Timing Category'] == 'MEKi First']['FINAL_NUMBER_OF_ALIVE_CELLS']

        def p_to_stars(p):
            if p < 0.001: return '***'
            elif p < 0.01: return '**'
            elif p < 0.05: return '*'
            return 'ns'

        # Welch's t-test (does not assume equal variance)
        if not drug_x_first_data.empty and not simultaneous_data.empty:
            p_val_x_vs_sim = stats.ttest_ind(drug_x_first_data, simultaneous_data, equal_var=False).pvalue
            y_max = ax.get_ylim()[0] # Inverted axis
            y_pos = y_max * 1.1
            ax.plot([0, 1], [y_pos, y_pos], lw=1.5, c='k')
            ax.text(0.5, y_pos, p_to_stars(p_val_x_vs_sim), ha='center', va='bottom', color='k')

        if not meki_first_data.empty and not simultaneous_data.empty:
            p_val_mek_vs_sim = stats.ttest_ind(meki_first_data, simultaneous_data, equal_var=False).pvalue
            y_max = ax.get_ylim()[0]
            y_pos = y_max * 1.2
            ax.plot([1, 2], [y_pos, y_pos], lw=1.5, c='k')
            ax.text(1.5, y_pos, p_to_stars(p_val_mek_vs_sim), ha='center', va='bottom', color='k')


    # Final figure adjustments
    if 'ax' in locals() and ax.get_figure() is not None:
        axes[0].set_ylabel('% Alive Cells (Efficacy)', weight='bold')
        axes[0].invert_yaxis()
        fig.suptitle(f'Aggregated Efficacy for D(X)={target_x_diff}, D(Y)={target_y_diff}', fontsize=16, weight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        plot_save_path = os.path.join(plot_dir, f"efficacy_aggregated_barplot_X_{target_x_diff}_Y_{target_y_diff}.png")
        plt.savefig(plot_save_path, dpi=600)
        plt.savefig(plot_save_path.replace('.png', '.svg'), format='svg')
        print(f"Saved aggregated bar plot to {plot_save_path}")
    
    plt.close(fig)

def create_efficacy_grid_plot(exp_name, title):
    """
    Loads combined metrics for an experiment and creates a grid of plots,
    with each subplot showing efficacy vs. delta time for a specific
    pair of diffusion coefficients.
    """
    # 1. Define paths
    base_path = "scripts/post_emews_analysis/synergy_recovery_experiments/optimal_timings_synergy"
    plot_dir = os.path.join(base_path, "efficacy_delta_time_plots")
    os.makedirs(plot_dir, exist_ok=True)

    metrics_path = os.path.join(base_path, f"combined_synergy_metrics_{exp_name}.csv")
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found for {exp_name}, skipping.")
        return

    metrics_df = pd.read_csv(metrics_path)

    # 2. Get unique diffusion coefficients for grid dimensions
    x_diffs = sorted(metrics_df['x_diff'].unique(), reverse=True)
    y_diffs = sorted(metrics_df['y_diff'].unique())
    nrows, ncols = len(x_diffs), len(y_diffs)

    if nrows == 0 or ncols == 0:
        print(f"No data to plot for {exp_name}.")
        return

    # 3. Create grid plot, ensuring `axes` is always a 2D array
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), 
                             sharex=True, sharey=True, dpi=300, squeeze=False)

    # 4. Iterate and plot
    for i, x_diff in enumerate(x_diffs):
        for j, y_diff in enumerate(y_diffs):
            ax = axes[i, j]
            
            subdf = metrics_df[(metrics_df['x_diff'] == x_diff) & (metrics_df['y_diff'] == y_diff)].sort_values('delta_time')

            if not subdf.empty:
                ax.plot(subdf['delta_time'], subdf['percent_alive'], color='tab:red', marker='o', linestyle='-')
                ax.set_title(f'X: {x_diff}, Y: {y_diff}', fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
            else:
                ax.axis('off')

    # 5. Final figure adjustments
    fig.suptitle(f'Efficacy vs. Delta Time for {title}', fontsize=16, weight='bold')
    fig.supxlabel('Delta Time (X - Y) (min)', weight='bold', fontsize=14)
    fig.supylabel('% Alive Cells (Efficacy)', weight='bold', fontsize=14)
    
    # Invert all y-axes as lower % alive is better
    for ax in axes.flat:
        ax.invert_yaxis()

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # 6. Save figure
    plot_save_path = os.path.join(plot_dir, f"efficacy_grid_{exp_name}.png")
    plt.savefig(plot_save_path, dpi=600)
    plt.savefig(plot_save_path.replace('.png', '.svg'), format='svg')
    print(f"Efficacy grid plot saved to: {plot_save_path}")
    plt.close(fig)

if __name__ == "__main__":
    # Define experiments to run the analysis for
    experiments_to_run = {
        "synergy_sweep-pi3k_mek-1606-0214-4p_3D_drugtiming": "PI3Ki + MEKi",
        "synergy_sweep-akt_mek-1606-0214-4p_3D_drugtiming": "AKTi + MEKi"
    }
    
    # --- Generate full grid plots for supplementary materials ---
    for exp_name, title in experiments_to_run.items():
        print(f"\n--- Generating efficacy grid plot for {title} ---")
        create_efficacy_grid_plot(exp_name, title)

    # --- Generate focused side-by-side comparison plot ---
    print("\n--- Generating high-dose comparison plot ---")
    create_high_dose_comparison_plot(experiments_to_run)

    # --- Generate a comparison plot for each individual diffusion scenario ---
    print("\n--- Generating individual comparison plot for each diffusion scenario ---")
    create_individual_scenario_plots(experiments_to_run)

    # --- Generate individual bar plots with error bars for each diffusion scenario ---
    print("\n--- Generating individual bar plots with error bars for each diffusion scenario ---")
    create_individual_scenario_bar_plots(experiments_to_run)

    # --- Generate aggregated bar plot for the high-dose scenario ---
    print("\n--- Generating aggregated bar plot for high-dose scenario ---")
    create_aggregated_timing_barplot(experiments_to_run, target_x_diff=600.0, target_y_diff=600.0) 