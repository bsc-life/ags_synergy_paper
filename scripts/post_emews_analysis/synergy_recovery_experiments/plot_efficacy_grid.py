import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
import matplotlib.colors as mcolors
from scipy.stats import mannwhitneyu
from pathlib import Path  # For safer path handling

# --- Global Style Configuration for Publication Quality ---
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 16,
    'legend.fontsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

def get_base_color_scheme():
    """Returns the base color scheme for different drugs."""
    return {
        'PI3Ki': '#1E88E5',
        'MEKi': '#2E7D32',
        'AKTi': '#7B1FA2',
    }

def get_timing_colors(drug_x_name, drug_y_name):
    """
    Returns a consistent color palette for timing strategies.
    Uses shades of orange to indicate all scenarios are synergistic combinations.
    A standard orange is for simultaneous, with dimmer shades for sequential.
    """
    return {
        f'{drug_x_name} First': '#fdd0a2',  # Lighter, dimmer orange
        'Simultaneous': '#fd8d3c',         # Standard orange
        f'{drug_y_name} First': '#d94801'   # Darker, dimmer orange
    }

def p_to_stars(p):
    """Converts a p-value to a significance star string."""
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'

def get_detailed_timing_colors(delta_times, color_x, color_y):
    """Generates a color gradient for detailed timing plots."""
    delta_times_sorted = sorted(delta_times)
    colors = {}
    
    neg_deltas = sorted([d for d in delta_times_sorted if d < 0])
    pos_deltas = sorted([d for d in delta_times_sorted if d > 0])
    
    # Blue gradient for negative delta times (X first)
    if neg_deltas:
        cmap_x = mcolors.LinearSegmentedColormap.from_list("cmap_x", [color_x, "lightsteelblue"])
        min_neg, max_neg = min(neg_deltas), max(neg_deltas)
        for dt in neg_deltas:
            norm_val = (dt - min_neg) / (max_neg - min_neg) if len(neg_deltas) > 1 else 0.5
            colors[dt] = cmap_x(norm_val)

    # Red gradient for positive delta times (Y first)
    if pos_deltas:
        cmap_y = mcolors.LinearSegmentedColormap.from_list("cmap_y", ["lightcoral", color_y])
        min_pos, max_pos = min(pos_deltas), max(pos_deltas)
        for dt in pos_deltas:
            norm_val = (dt - min_pos) / (max_pos - min_pos) if len(pos_deltas) > 1 else 0.5
            colors[dt] = cmap_y(norm_val)
    
    # Grey for simultaneous
    if 0 in delta_times_sorted:
        colors[0] = '#CCCCCC'
            
    return [colors[dt] for dt in delta_times]

def generate_detailed_timing_grid(analysis_df, experiment_name, mean_control_alive, save_path):
    """Generates a grid of plots showing efficacy vs. detailed timing differences."""
    print(f"--- Generating Detailed Efficacy-vs-Timing Grid for: {experiment_name} ---")
    
    drug_x_name = "AKTi" if "akt_mek" in experiment_name else "PI3Ki"
    drug_y_name = "MEKi"
    
    detailed_grouped = analysis_df.groupby(
        ['user_parameters.drug_X_diffusion_coefficient', 'user_parameters.drug_Y_diffusion_coefficient', 'delta_time']
    ).agg(
        mean_alive=('FINAL_NUMBER_OF_ALIVE_CELLS', 'mean'),
        std_alive=('FINAL_NUMBER_OF_ALIVE_CELLS', 'std')
    ).reset_index()
    detailed_grouped['std_alive'].fillna(0, inplace=True)

    x_diffs = sorted(detailed_grouped['user_parameters.drug_X_diffusion_coefficient'].unique(), reverse=True)
    y_diffs = sorted(detailed_grouped['user_parameters.drug_Y_diffusion_coefficient'].unique())
    
    if not x_diffs or not y_diffs:
        return

    nrows, ncols = len(x_diffs), len(y_diffs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=True, sharey=True, dpi=300)
    axes = np.atleast_2d(axes)

    for i, x_diff in enumerate(x_diffs):
        for j, y_diff in enumerate(y_diffs):
            ax = axes[i, j]
            subdf = detailed_grouped[(detailed_grouped['user_parameters.drug_X_diffusion_coefficient'] == x_diff) & 
                                     (detailed_grouped['user_parameters.drug_Y_diffusion_coefficient'] == y_diff)]
            subdf = subdf.sort_values('delta_time')

            if not subdf.empty:
                efficacy_percent = (subdf['mean_alive'] / mean_control_alive) * 100
                std_percent = (subdf['std_alive'] / mean_control_alive) * 100
                
                x_pos = np.arange(len(subdf))
                bar_width = 0.8
                
                base_colors = get_base_color_scheme()
                plot_colors = get_detailed_timing_colors(subdf['delta_time'], base_colors.get(drug_x_name), base_colors.get(drug_y_name))

                ax.bar(x_pos, efficacy_percent, yerr=std_percent, capsize=3, width=bar_width, color=plot_colors)
                
                ax.set_ylim(0, 105)
                ax.grid(axis='y', linestyle='--', alpha=0.6)
                ax.set_title(f'D({drug_x_name})={x_diff}, D({drug_y_name})={y_diff}', fontweight='bold')

                ax.set_xticks(x_pos)
                ax.set_xticklabels(subdf['delta_time'], rotation=45, ha="right")

                if i == nrows - 1:
                    ax.set_xlabel('Timing Difference (min)')
                if j == 0:
                    ax.set_ylabel('% Alive Cells')
            else:
                ax.axis('off')

    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.1, wspace=0.2, hspace=0.4)
    fig.suptitle(f'Detailed Efficacy vs. Timing for {drug_x_name} + {drug_y_name}', fontsize=18, weight='bold')
    output_filename_base = f"{save_path}/efficacy_grid_detailed_{experiment_name}_publication"
    plt.savefig(f"{output_filename_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_filename_base}.svg", format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Detailed timing grid saved to {output_filename_base}.png/svg")


def generate_efficacy_grid(experiment_name):
    """
    Loads data, filters out confounding parameter sets, and generates a grid of
    efficacy plots and a summary CSV.
    """
    print(f"--- Processing Efficacy Grid for: {experiment_name} (Filtered) ---")

    # --- Data Loading and Preparation ---
    try:
        timing_df = pd.read_csv(f'results/sweep_summaries/final_summary_{experiment_name}.csv')
    except FileNotFoundError:
        print(f"ERROR: Could not find the summary file for experiment '{experiment_name}'. Skipping.")
        return None, None, None

    timing_df['delta_time'] = timing_df['user_parameters.drug_X_pulse_period'] - timing_df['user_parameters.drug_Y_pulse_period']

    # --- Normalization Factor (from full dataset, before any filtering) ---
    control_condition = (
        (timing_df['user_parameters.drug_X_diffusion_coefficient'] == 6.0) &
        (timing_df['user_parameters.drug_Y_diffusion_coefficient'] == 6.0) &
        (timing_df['user_parameters.drug_X_pulse_period'] >= 4000) &
        (timing_df['user_parameters.drug_Y_pulse_period'] >= 4000)
    )
    control_runs_df = timing_df[control_condition]

    if control_runs_df.empty:
        print("ERROR: Could not find internal control runs. Skipping analysis.")
        return None, None, None
    mean_final_number_of_alive_cells_negative_control = control_runs_df['FINAL_NUMBER_OF_ALIVE_CELLS'].mean()
    
    # --- Filter out confounding individuals based on unreasonable efficacy ---
    timing_df['percent_alive'] = (timing_df['FINAL_NUMBER_OF_ALIVE_CELLS'] / mean_final_number_of_alive_cells_negative_control) * 100

    all_diff_coeffs = pd.unique(timing_df[['user_parameters.drug_X_diffusion_coefficient', 'user_parameters.drug_Y_diffusion_coefficient']].values.ravel('K'))
    high_dose = sorted([c for c in all_diff_coeffs if c != 6.0])[-1]
    low_dose = 6.0

    # Benchmark: Median efficacy of high-dose, simultaneous synergy runs
    synergy_condition = (
        (timing_df['user_parameters.drug_X_diffusion_coefficient'] == high_dose) &
        (timing_df['user_parameters.drug_Y_diffusion_coefficient'] == high_dose) &
        (timing_df['delta_time'] == 0) &
        (timing_df['user_parameters.drug_X_pulse_period'] < 4000)
    )
    synergy_runs = timing_df[synergy_condition]
    
    if synergy_runs.empty:
        print("WARNING: No synergy runs found to define benchmark. Skipping confounding filter.")
        filtered_df = timing_df.copy()
    else:
        median_synergy_efficacy = synergy_runs['percent_alive'].median()
        
        # Confounding: Single drug runs more effective than median synergy
        drug_x_alone = (timing_df['user_parameters.drug_X_diffusion_coefficient'] == high_dose) & (timing_df['user_parameters.drug_Y_diffusion_coefficient'] == low_dose)
        drug_y_alone = (timing_df['user_parameters.drug_X_diffusion_coefficient'] == low_dose) & (timing_df['user_parameters.drug_Y_diffusion_coefficient'] == high_dose)
        single_drug_runs = timing_df[drug_x_alone | drug_y_alone]
        confounding_sd_individuals = single_drug_runs[single_drug_runs['percent_alive'] < median_synergy_efficacy]['individual'].unique()

        # Confounding: Control runs that are too effective
        confounding_ctrl_individuals = timing_df[control_condition & (timing_df['percent_alive'] < 80)]['individual'].unique()

        exclusion_list = np.union1d(confounding_sd_individuals, confounding_ctrl_individuals)
        
        if len(exclusion_list) > 0:
            print(f"Excluding {len(exclusion_list)} confounding individuals.")
            filtered_df = timing_df[~timing_df['individual'].isin(exclusion_list)].copy()
        else:
            print("No confounding individuals found. Using all data.")
            filtered_df = timing_df.copy()
    
    # --- Define Timing Categories on the Cleaned Data ---
    drug_x_name = "AKTi" if "akt_mek" in experiment_name else "PI3Ki"
    drug_y_name = "MEKi"

    def get_delta_time_category(dt):
        if dt < 0:
            return f'{drug_x_name} First'
        elif dt > 0:
            return f'{drug_y_name} First'
        else:
            return 'Simultaneous'

    filtered_df['delta_time_category'] = filtered_df['delta_time'].apply(get_delta_time_category)
    
    # Exclude late-addition 'control' runs from the main analysis group
    late_addition_threshold = 1000
    late_control_runs_mask = (filtered_df['delta_time'] == 0) & (filtered_df['user_parameters.drug_X_pulse_period'] > late_addition_threshold)
    print(f"Excluding {late_control_runs_mask.sum()} late-addition 'control' runs from timing analysis.")
    analysis_df = filtered_df[~late_control_runs_mask]

    # Group the cleaned data
    grouped = analysis_df.groupby(
        ['user_parameters.drug_X_diffusion_coefficient', 
         'user_parameters.drug_Y_diffusion_coefficient', 
         'delta_time_category']
    ).agg(
        mean_alive=('FINAL_NUMBER_OF_ALIVE_CELLS', 'mean'),
        std_alive=('FINAL_NUMBER_OF_ALIVE_CELLS', 'std')
    ).reset_index()
    # Handle cases where a group has only one member, resulting in NaN for std
    grouped['std_alive'].fillna(0, inplace=True)

    # --- Save Efficacy Data to CSV ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Each experiment writes to its own sub-directory to avoid clashes
    save_dir = _exp_save_dir(script_dir, "efficacy_grid_plots", experiment_name)
    save_path = str(save_dir)

    efficacy_df = grouped.copy()
    efficacy_df['percent_alive'] = (efficacy_df['mean_alive'] / mean_final_number_of_alive_cells_negative_control) * 100
    efficacy_df['std_alive_percent'] = (efficacy_df['std_alive'] / mean_final_number_of_alive_cells_negative_control) * 100
    efficacy_df.rename(columns={
        'user_parameters.drug_X_diffusion_coefficient': 'x_diff',
        'user_parameters.drug_Y_diffusion_coefficient': 'y_diff',
        'mean_alive': 'mean_alive_raw',
        'std_alive': 'std_alive_raw'
    }, inplace=True)
    csv_output_filename = save_dir / f"efficacy_summary_aggregated_{experiment_name}.csv"
    efficacy_df.to_csv(csv_output_filename, index=False)
    print(f"Aggregated efficacy summary data saved to: {csv_output_filename}")

    # --- Grid Plot Generation ---
    x_diffs = sorted(grouped['user_parameters.drug_X_diffusion_coefficient'].unique(), reverse=True)
    y_diffs = sorted(grouped['user_parameters.drug_Y_diffusion_coefficient'].unique())

    if not x_diffs or not y_diffs:
        print("No data to plot. Skipping plot generation.")
        print("----------------------------------------------------------\n")
        return efficacy_df, analysis_df, mean_final_number_of_alive_cells_negative_control

    # Define categories and a professional color palette
    categories = [f'{drug_x_name} First', 'Simultaneous', f'{drug_y_name} First']
    category_colors = get_timing_colors(drug_x_name, drug_y_name)
    
    nrows, ncols = len(x_diffs), len(y_diffs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows), sharex=True, sharey=True, dpi=300)

    for i, x_diff in enumerate(x_diffs):
        for j, y_diff in enumerate(y_diffs):
            ax = axes[i, j] if nrows > 1 and ncols > 1 else (axes[max(i, j)] if (nrows > 1 or ncols > 1) else axes)
            subdf = grouped[(grouped['user_parameters.drug_X_diffusion_coefficient'] == x_diff) & (grouped['user_parameters.drug_Y_diffusion_coefficient'] == y_diff)]
            
            # Ensure consistent order for plotting
            subdf['delta_time_category'] = pd.Categorical(subdf['delta_time_category'], categories=categories, ordered=True)
            subdf = subdf.sort_values('delta_time_category').reset_index(drop=True)

            if not subdf.empty:
                xvals = np.arange(len(subdf))
                efficacy_percent = (subdf['mean_alive'] / mean_final_number_of_alive_cells_negative_control) * 100
                std_percent = (subdf['std_alive'] / mean_final_number_of_alive_cells_negative_control) * 100
                std_percent.fillna(0, inplace=True) # Ensure no NaNs in error bars
                colors = [category_colors.get(cat, 'gray') for cat in subdf['delta_time_category']]
                
                bars = ax.bar(xvals, efficacy_percent, yerr=std_percent, capsize=3, color=colors, ecolor='black', width=0.8)
                ax.set_ylim(0, 100)
                ax.grid(axis='y', linestyle='--', alpha=0.6, color='#CCCCCC')
                ax.set_title(f'D(X)={x_diff}, D(Y)={y_diff}', fontweight='bold')

                if len(efficacy_percent) > 1:
                    min_idx = efficacy_percent.idxmin()
                    min_y = efficacy_percent.loc[min_idx]
                    
                    bars[min_idx].set_edgecolor('black')
                    bars[min_idx].set_linewidth(1.5)
                    bars[min_idx].set_zorder(10)
                    
                    # Add a cleaner annotation for the best result
                    ax.annotate(f'{min_y:.1f}%', (xvals[min_idx], min_y + std_percent.loc[min_idx]), 
                                textcoords="offset points", xytext=(0, 4), ha='center', color='black', 
                                fontsize=9, weight='bold')

                # --- Add statistical comparisons between timing strategies ---
                condition_df = analysis_df[(analysis_df['user_parameters.drug_X_diffusion_coefficient'] == x_diff) & 
                                         (analysis_df['user_parameters.drug_Y_diffusion_coefficient'] == y_diff)]
                
                # Get data for each category
                drug_x_first_vals = condition_df[condition_df['delta_time_category'] == categories[0]]['FINAL_NUMBER_OF_ALIVE_CELLS']
                simultaneous_vals = condition_df[condition_df['delta_time_category'] == categories[1]]['FINAL_NUMBER_OF_ALIVE_CELLS']
                drug_y_first_vals = condition_df[condition_df['delta_time_category'] == categories[2]]['FINAL_NUMBER_OF_ALIVE_CELLS']
                
                comparisons = [
                    (drug_x_first_vals, simultaneous_vals, 0, 1), # DrugX vs Sim
                    (drug_y_first_vals, simultaneous_vals, 2, 1)  # DrugY vs Sim
                ]

                # y_pos needs to be calculated based on the bars that will be annotated
                annotation_y_start = (efficacy_percent + std_percent).max()
                y_offset = annotation_y_start * 0.08 # Start annotations 8% above the highest bar

                for group1_vals, group2_vals, idx1, idx2 in comparisons:
                    if not group1_vals.empty and not group2_vals.empty and len(group1_vals) > 1 and len(group2_vals) > 1:
                        try:
                            _, p_value = mannwhitneyu(group1_vals, group2_vals, alternative='two-sided')
                            
                            stars = p_to_stars(p_value)

                            # Always draw the bracket, style text based on significance
                            y_pos = max(efficacy_percent.iloc[idx1] + std_percent.iloc[idx1], efficacy_percent.iloc[idx2] + std_percent.iloc[idx2]) + y_offset
                            
                            bar_height = annotation_y_start * 0.02
                            ax.plot([xvals[idx1], xvals[idx1], xvals[idx2], xvals[idx2]], 
                                    [y_pos, y_pos + bar_height, y_pos + bar_height, y_pos], 
                                    lw=1, c='black')
                            
                            is_significant = stars != 'n.s.'
                            text_fontweight = 'bold' if is_significant else 'normal'
                            text_fontsize = 12 if is_significant else 9
                            
                            ax.text((xvals[idx1] + xvals[idx2]) / 2, y_pos + bar_height, stars, 
                                    ha='center', va='bottom', fontsize=text_fontsize, color='black', fontweight=text_fontweight)
                            
                            # Increase offset to stack the next annotation bar
                            y_offset += annotation_y_start * 0.15

                        except ValueError:
                            pass # Happens if all values in a sample are identical

                ax.set_xticks(xvals)
                ax.set_xticklabels(subdf['delta_time_category'], rotation=45, ha="right")
                ax.tick_params(axis='x', which='major', pad=2)
                ax.tick_params(axis='y', which='major', pad=2)
            else:
                ax.axis('off')

    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.15, wspace=0.22, hspace=0.6)
    fig.text(0.01, 0.55, '% Alive Cells (of Control)', va='center', ha='center', rotation='vertical', fontweight='bold', fontsize=14)

    output_filename_base = save_dir / f"efficacy_grid_aggregated_{experiment_name}_publication"
    plt.savefig(f"{output_filename_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_filename_base}.svg", format='svg', bbox_inches='tight')
    print(f"\nAggregated efficacy grid plot saved to: {output_filename_base}.png/svg")
    plt.close(fig)
    print("----------------------------------------------------------\n")

    # --- Generate the new detailed plot ---
    generate_detailed_timing_grid(analysis_df, experiment_name, mean_final_number_of_alive_cells_negative_control, save_path)

    return efficacy_df, analysis_df, mean_final_number_of_alive_cells_negative_control

def plot_high_dose_comparison(all_efficacy_data, all_analysis_data):
    """
    Generates a 1x2 side-by-side plot comparing timing strategies for the highest
    dose condition across both experiments.
    """
    print("\n--- Generating High-Dose Comparison Plot ---")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, dpi=300)
    
    exp_names = list(all_efficacy_data.keys())
    high_dose = 6000.0

    for i, exp_name in enumerate(exp_names):
        ax = axes[i]
        efficacy_df = all_efficacy_data[exp_name]

        drug_x_name = "AKTi" if "akt_mek" in exp_name else "PI3Ki"
        drug_y_name = "MEKi"
        categories = [f'{drug_x_name} First', 'Simultaneous', f'{drug_y_name} First']
        color_map = get_timing_colors(drug_x_name, drug_y_name)

        high_dose_df = efficacy_df[(efficacy_df['x_diff'] == high_dose) & (efficacy_df['y_diff'] == high_dose)].copy()
        high_dose_df['delta_time_category'] = pd.Categorical(high_dose_df['delta_time_category'], categories=categories, ordered=True)
        high_dose_df = high_dose_df.sort_values('delta_time_category').reset_index(drop=True)

        if not high_dose_df.empty:
            xvals = np.arange(len(high_dose_df))
            colors = [color_map.get(cat, 'gray') for cat in high_dose_df['delta_time_category']]
            
            efficacy_percent = high_dose_df['percent_alive']
            std_percent = high_dose_df['std_alive_percent']
            
            ax.bar(xvals, efficacy_percent, yerr=std_percent,
                   capsize=4, color=colors, ecolor='black', width=0.8)
            
            # --- Add statistical annotations ---
            raw_analysis_df, _ = all_analysis_data[exp_name]
            condition_df = raw_analysis_df[(raw_analysis_df['user_parameters.drug_X_diffusion_coefficient'] == high_dose) & 
                                           (raw_analysis_df['user_parameters.drug_Y_diffusion_coefficient'] == high_dose)]
            
            g1_vals = condition_df[condition_df['delta_time_category'] == categories[0]]['FINAL_NUMBER_OF_ALIVE_CELLS']
            g2_vals = condition_df[condition_df['delta_time_category'] == categories[1]]['FINAL_NUMBER_OF_ALIVE_CELLS']
            g3_vals = condition_df[condition_df['delta_time_category'] == categories[2]]['FINAL_NUMBER_OF_ALIVE_CELLS']

            comparisons = [(g1_vals, g2_vals, 0, 1), (g3_vals, g2_vals, 2, 1), (g1_vals, g3_vals, 0, 2)]
            
            annotation_y_start = (efficacy_percent + std_percent).max()
            y_offset = annotation_y_start * 0.1 # Start annotations 10% above the highest bar
            
            for group1, group2, idx1, idx2 in comparisons:
                if not group1.empty and not group2.empty and len(group1) > 1 and len(group2) > 1:
                    try:
                        _, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                        stars = p_to_stars(p_val)
                        
                        y_pos = max(efficacy_percent.iloc[idx1] + std_percent.iloc[idx1], efficacy_percent.iloc[idx2] + std_percent.iloc[idx2]) + y_offset
                        bar_height = annotation_y_start * 0.02

                        ax.plot([xvals[idx1], xvals[idx1], xvals[idx2], xvals[idx2]],
                                [y_pos, y_pos + bar_height, y_pos + bar_height, y_pos], lw=1, c='black')

                        is_sig = stars != 'n.s.'
                        ax.text((xvals[idx1] + xvals[idx2]) / 2, y_pos + bar_height, stars,
                                ha='center', va='bottom', fontsize=12 if is_sig else 9,
                                color='black', fontweight='bold' if is_sig else 'normal')
                        
                        y_offset += annotation_y_start * 0.18
                    except ValueError:
                        pass

            ax.set_title(f"{drug_x_name} + {drug_y_name}", fontweight='bold')
            ax.set_xticks(xvals)
            ax.set_xticklabels(categories, rotation=45, ha="right")
        else:
            ax.text(0.5, 0.5, 'No data for D=6000', ha='center', va='center')
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    axes[0].set_ylabel('% Alive Cells (of Control)', fontweight='bold')
    axes[0].set_ylim(0, 125) # Increase limit for annotations
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dims_tag = _dims_tag(exp_names[0]) if exp_names else 'out'
    comparison_dir = Path(script_dir) / f"efficacy_grid_plots_{dims_tag}"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    output_filename_base = comparison_dir / "high_dose_comparison_publication"
    plt.savefig(f"{output_filename_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_filename_base}.svg", format='svg', bbox_inches='tight')
    print(f"High-dose comparison plot saved to: {output_filename_base}.png/svg")
    plt.close(fig)


def plot_detailed_high_dose_comparison(all_analysis_data):
    """Generates a 1x2 bar plot with detailed timing for the high-dose scenario."""
    print("\n--- Generating Detailed High-Dose Timing Comparison Plot ---")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, dpi=300)
    fig.suptitle("Efficacy vs. Detailed Timing at High Dose (D=600.0)", fontsize=18, weight='bold')
    high_dose = 600.0

    for i, exp_name in enumerate(all_analysis_data.keys()):
        ax = axes[i]
        analysis_df, mean_control_alive = all_analysis_data[exp_name]
        
        high_dose_df = analysis_df[(analysis_df['user_parameters.drug_X_diffusion_coefficient'] == high_dose) &
                                 (analysis_df['user_parameters.drug_Y_diffusion_coefficient'] == high_dose)]
        
        if high_dose_df.empty:
            drug_x_name = "AKTi" if "akt_mek" in exp_name else "PI3Ki"
            ax.set_title(f"{drug_x_name} + MEKi", fontweight='bold')
            ax.text(0.5, 0.5, 'No data for D=600', ha='center', va='center')
            continue

        detailed_grouped = high_dose_df.groupby('delta_time').agg(
            mean_alive=('FINAL_NUMBER_OF_ALIVE_CELLS', 'mean'),
            std_alive=('FINAL_NUMBER_OF_ALIVE_CELLS', 'std')
        ).reset_index()
        detailed_grouped['std_alive'].fillna(0, inplace=True)
        detailed_grouped = detailed_grouped.sort_values('delta_time')

        efficacy_percent = (detailed_grouped['mean_alive'] / mean_control_alive) * 100
        std_percent = (detailed_grouped['std_alive'] / mean_control_alive) * 100
        
        x_pos = np.arange(len(detailed_grouped))
        bar_width = 0.8
        
        drug_x_name = "AKTi" if "akt_mek" in exp_name else "PI3Ki"
        drug_y_name = "MEKi"
        base_colors = get_base_color_scheme()
        plot_colors = get_detailed_timing_colors(detailed_grouped['delta_time'], base_colors.get(drug_x_name), base_colors.get(drug_y_name))

        ax.bar(x_pos, efficacy_percent, yerr=std_percent, capsize=4, width=bar_width, color=plot_colors, ecolor='black')
        
        ax.set_title(f"{drug_x_name} + {drug_y_name}", fontweight='bold')
        ax.set_xlabel('Delta Time (X - Y) (min)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(detailed_grouped['delta_time'], rotation=45, ha="right")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    axes[0].set_ylabel('% Alive Cells (Efficacy)', fontweight='bold')
    axes[0].set_ylim(0, 105)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dims_tag = _dims_tag(next(iter(all_analysis_data))) if all_analysis_data else 'out'
    comparison_dir = Path(script_dir) / f"efficacy_grid_plots_{dims_tag}"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    output_filename_base = comparison_dir / "high_dose_comparison_detailed_publication"
    plt.savefig(f"{output_filename_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_filename_base}.svg", format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Detailed high-dose comparison plot saved to: {output_filename_base}.png/svg")


def plot_high_dose_comparison_with_jitter(all_efficacy_data, all_analysis_data):
    """
    Generates a 1x2 side-by-side plot comparing timing strategies for the highest
    dose condition with individual data points overlaid on bars.
    """
    print("\n--- Generating High-Dose Comparison Plot with Individual Data Points ---")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, dpi=300)
    
    exp_names = list(all_efficacy_data.keys())
    high_dose = 6000.0

    for i, exp_name in enumerate(exp_names):
        ax = axes[i]
        efficacy_df = all_efficacy_data[exp_name]
        raw_analysis_df, mean_control_alive = all_analysis_data[exp_name]

        drug_x_name = "AKTi" if "akt_mek" in exp_name else "PI3Ki"
        drug_y_name = "MEKi"
        categories = [f'{drug_x_name} First', 'Simultaneous', f'{drug_y_name} First']
        color_map = get_timing_colors(drug_x_name, drug_y_name)

        high_dose_df = efficacy_df[(efficacy_df['x_diff'] == high_dose) & (efficacy_df['y_diff'] == high_dose)].copy()
        high_dose_df['delta_time_category'] = pd.Categorical(high_dose_df['delta_time_category'], categories=categories, ordered=True)
        high_dose_df = high_dose_df.sort_values('delta_time_category').reset_index(drop=True)

        if not high_dose_df.empty:
            xvals = np.arange(len(high_dose_df))
            colors = [color_map.get(cat, 'gray') for cat in high_dose_df['delta_time_category']]
            
            efficacy_percent = high_dose_df['percent_alive']
            std_percent = high_dose_df['std_alive_percent']
            
            # Plot bars
            bars = ax.bar(xvals, efficacy_percent, yerr=std_percent,
                         capsize=4, color=colors, ecolor='black', width=0.8, alpha=0.7)
            
            # Add individual data points with jitter
            condition_df = raw_analysis_df[(raw_analysis_df['user_parameters.drug_X_diffusion_coefficient'] == high_dose) & 
                                           (raw_analysis_df['user_parameters.drug_Y_diffusion_coefficient'] == high_dose)]
            
            for j, category in enumerate(categories):
                category_data = condition_df[condition_df['delta_time_category'] == category]['FINAL_NUMBER_OF_ALIVE_CELLS']
                if not category_data.empty:
                    # Convert to percentage
                    category_percent = (category_data / mean_control_alive) * 100
                    
                    # Add jitter to x-position
                    jitter = np.random.normal(0, 0.1, len(category_percent))
                    x_jittered = xvals[j] + jitter
                    
                    # Plot individual points
                    ax.scatter(x_jittered, category_percent, 
                              color='black', alpha=0.3, s=8, zorder=5)
            
            # Add statistical annotations
            g1_vals = condition_df[condition_df['delta_time_category'] == categories[0]]['FINAL_NUMBER_OF_ALIVE_CELLS']
            g2_vals = condition_df[condition_df['delta_time_category'] == categories[1]]['FINAL_NUMBER_OF_ALIVE_CELLS']
            g3_vals = condition_df[condition_df['delta_time_category'] == categories[2]]['FINAL_NUMBER_OF_ALIVE_CELLS']

            comparisons = [(g1_vals, g2_vals, 0, 1), (g3_vals, g2_vals, 2, 1), (g1_vals, g3_vals, 0, 2)]
            
            annotation_y_start = (efficacy_percent + std_percent).max()
            y_offset = annotation_y_start * 0.1
            
            for group1, group2, idx1, idx2 in comparisons:
                if not group1.empty and not group2.empty and len(group1) > 1 and len(group2) > 1:
                    try:
                        _, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                        stars = p_to_stars(p_val)
                        
                        y_pos = max(efficacy_percent.iloc[idx1] + std_percent.iloc[idx1], efficacy_percent.iloc[idx2] + std_percent.iloc[idx2]) + y_offset
                        bar_height = annotation_y_start * 0.02

                        ax.plot([xvals[idx1], xvals[idx1], xvals[idx2], xvals[idx2]],
                                [y_pos, y_pos + bar_height, y_pos + bar_height, y_pos], lw=1, c='black')

                        is_sig = stars != 'n.s.'
                        ax.text((xvals[idx1] + xvals[idx2]) / 2, y_pos + bar_height, stars,
                                ha='center', va='bottom', fontsize=12 if is_sig else 9,
                                color='black', fontweight='bold' if is_sig else 'normal')
                        
                        y_offset += annotation_y_start * 0.18
                    except ValueError:
                        pass

            ax.set_title(f"{drug_x_name} + {drug_y_name}", fontweight='bold')
            ax.set_xticks(xvals)
            ax.set_xticklabels(categories, rotation=45, ha="right")
        else:
            ax.text(0.5, 0.5, 'No data for D=6000', ha='center', va='center')
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    axes[0].set_ylabel('% Alive Cells (of Control)', fontweight='bold')
    axes[0].set_ylim(0, 125)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dims_tag = _dims_tag(exp_names[0]) if exp_names else 'out'
    comparison_dir = Path(script_dir) / f"efficacy_grid_plots_{dims_tag}"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    output_filename_base = comparison_dir / "high_dose_comparison_with_jitter_publication"
    plt.savefig(f"{output_filename_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_filename_base}.svg", format='svg', bbox_inches='tight')
    print(f"High-dose comparison plot with jitter saved to: {output_filename_base}.png/svg")
    plt.close(fig)


def plot_high_dose_comparison_violin(all_analysis_data):
    """
    Generates a 1x2 side-by-side violin plot comparing timing strategies for the highest
    dose condition across both experiments.
    """
    print("\n--- Generating High-Dose Comparison Violin Plot ---")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, dpi=300)
    
    exp_names = list(all_analysis_data.keys())
    high_dose = 6000.0

    for i, exp_name in enumerate(exp_names):
        ax = axes[i]
        analysis_df, mean_control_alive = all_analysis_data[exp_name]

        drug_x_name = "AKTi" if "akt_mek" in exp_name else "PI3Ki"
        drug_y_name = "MEKi"
        categories = [f'{drug_x_name} First', 'Simultaneous', f'{drug_y_name} First']
        color_map = get_timing_colors(drug_x_name, drug_y_name)

        # Filter for high-dose condition
        high_dose_df = analysis_df[(analysis_df['user_parameters.drug_X_diffusion_coefficient'] == high_dose) & 
                                 (analysis_df['user_parameters.drug_Y_diffusion_coefficient'] == high_dose)]

        if not high_dose_df.empty:
            # Prepare data for violin plot
            violin_data = []
            violin_colors = []
            
            for category in categories:
                category_data = high_dose_df[high_dose_df['delta_time_category'] == category]['FINAL_NUMBER_OF_ALIVE_CELLS']
                if not category_data.empty:
                    # Convert to percentage
                    category_percent = (category_data / mean_control_alive) * 100
                    violin_data.append(category_percent.values)
                    violin_colors.append(color_map.get(category, 'gray'))
                else:
                    violin_data.append([])
                    violin_colors.append(color_map.get(category, 'gray'))

            # Create violin plot
            violin_parts = ax.violinplot(violin_data, positions=range(len(categories)), 
                                       showmeans=True, showmedians=True)
            
            # Customize violin plot appearance
            for pc, color in zip(violin_parts['bodies'], violin_colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            # Customize mean and median lines
            violin_parts['cmeans'].set_color('black')
            violin_parts['cmeans'].set_linewidth(2)
            violin_parts['cmedians'].set_color('white')
            violin_parts['cmedians'].set_linewidth(2)
            
            # Remove default box plot elements
            violin_parts['cbars'].set_visible(False)
            violin_parts['cmins'].set_visible(False)
            violin_parts['cmaxes'].set_visible(False)
            
            # Add statistical annotations
            g1_vals = high_dose_df[high_dose_df['delta_time_category'] == categories[0]]['FINAL_NUMBER_OF_ALIVE_CELLS']
            g2_vals = high_dose_df[high_dose_df['delta_time_category'] == categories[1]]['FINAL_NUMBER_OF_ALIVE_CELLS']
            g3_vals = high_dose_df[high_dose_df['delta_time_category'] == categories[2]]['FINAL_NUMBER_OF_ALIVE_CELLS']

            comparisons = [(g1_vals, g2_vals, 0, 1), (g3_vals, g2_vals, 2, 1), (g1_vals, g3_vals, 0, 2)]
            
            # Calculate y position for annotations
            max_vals = [max(data) if len(data) > 0 else 0 for data in violin_data]
            annotation_y_start = max(max_vals) if max_vals else 100
            y_offset = annotation_y_start * 0.1
            
            for group1, group2, idx1, idx2 in comparisons:
                if not group1.empty and not group2.empty and len(group1) > 1 and len(group2) > 1:
                    try:
                        _, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                        stars = p_to_stars(p_val)
                        
                        y_pos = max(max_vals[idx1], max_vals[idx2]) + y_offset
                        bar_height = annotation_y_start * 0.02

                        ax.plot([idx1, idx1, idx2, idx2],
                                [y_pos, y_pos + bar_height, y_pos + bar_height, y_pos], lw=1, c='black')

                        is_sig = stars != 'n.s.'
                        ax.text((idx1 + idx2) / 2, y_pos + bar_height, stars,
                                ha='center', va='bottom', fontsize=12 if is_sig else 9,
                                color='black', fontweight='bold' if is_sig else 'normal')
                        
                        y_offset += annotation_y_start * 0.18
                    except ValueError:
                        pass

            ax.set_title(f"{drug_x_name} + {drug_y_name}", fontweight='bold')
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45, ha="right")
        else:
            ax.text(0.5, 0.5, 'No data for D=6000', ha='center', va='center')
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    axes[0].set_ylabel('% Alive Cells (of Control)', fontweight='bold')
    axes[0].set_ylim(0, 125)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dims_tag = _dims_tag(exp_names[0]) if exp_names else 'out'
    comparison_dir = Path(script_dir) / f"efficacy_grid_plots_{dims_tag}"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    output_filename_base = comparison_dir / "high_dose_comparison_violin_publication"
    plt.savefig(f"{output_filename_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_filename_base}.svg", format='svg', bbox_inches='tight')
    print(f"High-dose comparison violin plot saved to: {output_filename_base}.png/svg")
    plt.close(fig)


def main():
    """
    Main function to run the analysis for all specified experiments.
    """
    experiment_names = [
        "synergy_sweep-akt_mek-2606-1819-4p_2D_drugtiming_synonly_consensus_hybrid_20",
        "synergy_sweep-pi3k_mek-2606-1819-4p_2D_drugtiming_synonly_consensus_hybrid_20"
    ]

    all_efficacy_data = {}
    all_analysis_data = {}
    for exp_name in experiment_names:
        efficacy_df, analysis_df, mean_control_alive = generate_efficacy_grid(exp_name)
        if efficacy_df is not None and analysis_df is not None:
            all_efficacy_data[exp_name] = efficacy_df
            all_analysis_data[exp_name] = (analysis_df, mean_control_alive)

    if len(all_efficacy_data) == 2:
        plot_high_dose_comparison(all_efficacy_data, all_analysis_data)
        plot_detailed_high_dose_comparison(all_analysis_data)
        plot_high_dose_comparison_with_jitter(all_efficacy_data, all_analysis_data)
        plot_high_dose_comparison_violin(all_analysis_data)
    else:
        print("\nCould not generate high-dose comparison plot due to missing data.")

# -------------------------------------------------------------------
# Helpers for output paths to keep different experiments separated
# -------------------------------------------------------------------


def _dims_tag(exp_name: str) -> str:
    """Return '2D' or '3D' depending on the experiment name."""
    return '2D' if '_2D_' in exp_name else '3D'


def _exp_save_dir(script_dir: str, base_folder: str, exp_name: str) -> Path:
    """Return Path object for `script_dir/base_folder/<exp_name>` and create it."""
    p = Path(script_dir) / base_folder / exp_name
    p.mkdir(parents=True, exist_ok=True)
    return p

if __name__ == "__main__":
    main() 