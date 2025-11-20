import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import numpy as np
from scipy import stats
import seaborn as sns

# Set up publication-quality plotting fonts and style
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'dejavusans',
    'font.size': 14,
    'font.sans-serif': ['DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif', 
    
                        'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Helvetica', 
                        'Avant Garde', 'sans-serif']
})
sns.set_style("ticks")

def format_p_value(p):
    """Converts a p-value into a significance star string."""
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'

def add_stat_annotation(ax, x1, x2, y, h, text):
    """Adds a significance bracket and text to a plot."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c='k')
    ax.text((x1 + x2) * .5, y + h, text, ha='center', va='bottom', fontsize=12, weight='bold')

def get_color_scheme():
    """Returns a color scheme for different experimental conditions consistent with the paper."""
    return {
        'Control': '#808080',
        'PI3Ki': '#1E88E5',
        'MEKi': '#2E7D32',
        'AKTi': '#7B1FA2',
        'Synergy_PI3Ki_MEKi': '#F57C00',
        'Synergy_AKTi_MEKi': '#E65100'
    }

def get_plot_palette(drug_x_label, drug_y_label):
    """Generates the color palette for the plot based on drug names."""
    scheme = get_color_scheme()
    synergy_key = f"Synergy_{drug_x_label}_{drug_y_label}"
    return {
        'Control': scheme['Control'],
        f'{drug_x_label} Alone': scheme[drug_x_label],
        f'{drug_y_label} Alone': scheme[drug_y_label],
        'Synergy': scheme[synergy_key]
    }

def _plot_validation_subplot(ax, synergy_exp_name, drug_x_label, drug_y_label):
    """
    Filters out confounding parameter sets where single-drug or control runs are 
    unrealistically effective, providing a clearer view of true synergy.
    """
    print(f"--- Plotting data for {drug_x_label} + {drug_y_label} from {synergy_exp_name} ---")

    # --- 1. Load Full Dataset ---
    raw_summary_path = f"results/sweep_summaries/final_summary_{synergy_exp_name}.csv"
    try:
        raw_df = pd.read_csv(raw_summary_path)
    except FileNotFoundError:
        ax.text(0.5, 0.5, "Data not found", ha='center', va='center', fontsize=12, color='red')
        ax.set_title(f"{drug_x_label} + {drug_y_label}", weight='bold')
        return False

    # --- 2. Define Experimental Conditions and Get Normalization Factor ---
    late_pulse_threshold = 4000
    all_diff_coeffs = pd.unique(raw_df[['user_parameters.drug_X_diffusion_coefficient', 'user_parameters.drug_Y_diffusion_coefficient']].values.ravel('K'))
    high_dose = sorted([c for c in all_diff_coeffs if c != 6.0])[-1]
    low_dose = 6.0

    control_condition_exp = (
        (raw_df['user_parameters.drug_X_diffusion_coefficient'] == low_dose) &
        (raw_df['user_parameters.drug_Y_diffusion_coefficient'] == low_dose) &
        (raw_df['user_parameters.drug_X_pulse_period'] >= late_pulse_threshold) &
        (raw_df['user_parameters.drug_Y_pulse_period'] >= late_pulse_threshold)
    )
    control_runs_df = raw_df[control_condition_exp]
    
    if control_runs_df.empty:
        ax.text(0.5, 0.5, "Control data missing", ha='center', va='center', fontsize=12, color='red')
        return False
    mean_control_alive = control_runs_df['FINAL_NUMBER_OF_ALIVE_CELLS'].mean()

    # --- 3. Normalize Entire Dataset First ---
    raw_df['percent_alive'] = (raw_df['FINAL_NUMBER_OF_ALIVE_CELLS'] / mean_control_alive) * 100

    # --- 4. Identify Confounding Parameter Sets to Exclude ---
    # Find median synergy efficacy to use as a benchmark
    synergy_condition_exp = (
        (raw_df['user_parameters.drug_X_diffusion_coefficient'] == high_dose) &
        (raw_df['user_parameters.drug_Y_diffusion_coefficient'] == high_dose) &
        (raw_df['user_parameters.drug_X_pulse_period'] == 4) &
        (raw_df['user_parameters.drug_Y_pulse_period'] == 4)
    )
    synergy_runs = raw_df[synergy_condition_exp]
    if synergy_runs.empty:
        ax.text(0.5, 0.5, "Synergy data missing", ha='center', va='center', fontsize=12, color='red')
        return False
    median_synergy_efficacy = synergy_runs['percent_alive'].median()
    
    # Identify single drug runs that are more effective than median synergy
    drug_x_condition_exp = (
        (raw_df['user_parameters.drug_X_diffusion_coefficient'] == high_dose) &
        (raw_df['user_parameters.drug_Y_diffusion_coefficient'] == low_dose) &
        (raw_df['user_parameters.drug_X_pulse_period'] == 4) &
        (raw_df['user_parameters.drug_Y_pulse_period'] >= late_pulse_threshold)
    )
    drug_y_condition_exp = (
        (raw_df['user_parameters.drug_X_diffusion_coefficient'] == low_dose) &
        (raw_df['user_parameters.drug_Y_diffusion_coefficient'] == high_dose) &
        (raw_df['user_parameters.drug_X_pulse_period'] >= late_pulse_threshold) &
        (raw_df['user_parameters.drug_Y_pulse_period'] == 4)
    )
    single_drug_runs = raw_df[drug_x_condition_exp | drug_y_condition_exp]
    confounding_sd_individuals = single_drug_runs[single_drug_runs['percent_alive'] < median_synergy_efficacy]['individual'].unique()

    # Identify control runs that are unreasonably effective
    confounding_ctrl_individuals = raw_df[control_condition_exp & (raw_df['percent_alive'] < 80)]['individual'].unique()

    # Create final exclusion list
    exclusion_list = np.union1d(confounding_sd_individuals, confounding_ctrl_individuals)
    
    if len(exclusion_list) > 0:
        print(f"Excluding {len(exclusion_list)} confounding individuals.")
        filtered_df = raw_df[~raw_df['individual'].isin(exclusion_list)].copy()
    else:
        print("No confounding individuals found. Using all data.")
        filtered_df = raw_df.copy()

    # --- 5. Select Data for Plotting from the Cleaned Dataset ---
    control_data = filtered_df[control_condition_exp]
    drug_x_data = filtered_df[drug_x_condition_exp]
    drug_y_data = filtered_df[drug_y_condition_exp]
    synergy_data = filtered_df[synergy_condition_exp]
    
    # --- 6. Prepare Data for Plotting ---
    data_map = {
        'Control': control_data,
        f'{drug_x_label} Alone': drug_x_data,
        f'{drug_y_label} Alone': drug_y_data,
        'Synergy': synergy_data
    }
    
    plot_data_frames = []
    for name, df in data_map.items():
        if df.empty: print(f"Warning: No data for '{name}' after filtering.")
        
        df_copy = df.copy()
        df_copy['Condition'] = name
        plot_data_frames.append(df_copy[['percent_alive', 'Condition']])

    plot_df = pd.concat(plot_data_frames, ignore_index=True)
    
    if plot_df.empty or plot_df['percent_alive'].isnull().all():
        ax.text(0.5, 0.5, "No data available\nfor plotting", ha='center', va='center', fontsize=12, color='red')
        return False
        
    # --- 7. Plotting ---
    order = list(data_map.keys())
    palette = get_plot_palette(drug_x_label, drug_y_label)
    
    sns.violinplot(
        ax=ax, x='Condition', y='percent_alive', data=plot_df,
        order=order, hue='Condition', palette=palette, inner='box', legend=False
    )
    
    # --- 8. Statistical Analysis & Annotations ---
    drug_x_series = data_map[f'{drug_x_label} Alone']['percent_alive']
    drug_y_series = data_map[f'{drug_y_label} Alone']['percent_alive']
    synergy_series = data_map['Synergy']['percent_alive']

    # Perform Mann-Whitney U tests and add annotations
    # Comparison 1: Drug X vs Synergy
    if not drug_x_series.empty and not synergy_series.empty:
        _, p_val_x = stats.mannwhitneyu(drug_x_series, synergy_series, alternative='two-sided')
        print(f"  {drug_x_label} vs Synergy: p={p_val_x:.4f}")
        p_text_x = format_p_value(p_val_x)
        add_stat_annotation(ax, 1, 3, 102, 2, p_text_x)

    # Comparison 2: Drug Y vs Synergy
    if not drug_y_series.empty and not synergy_series.empty:
        _, p_val_y = stats.mannwhitneyu(drug_y_series, synergy_series, alternative='two-sided')
        print(f"  {drug_y_label} vs Synergy: p={p_val_y:.4f}")
        p_text_y = format_p_value(p_val_y)
        add_stat_annotation(ax, 2, 3, 112, 2, p_text_y)
    
    # --- 9. Final Touches ---
    ax.set_title(f'{drug_x_label} + {drug_y_label}', fontsize=16, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    ax.set_xlabel(None)
    ax.tick_params(axis='x', rotation=30, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    sns.despine(ax=ax)
    return True

def main():
    # --- Create Figure for Side-by-Side Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, dpi=300)

    # --- Plot 1: PI3K + MEK ---
    success1 = _plot_validation_subplot(
        ax=axes[0],
        synergy_exp_name="synergy_sweep-pi3k_mek-2606-1819-4p_2D_drugtiming_synonly_consensus_hybrid_20",
        drug_x_label="PI3Ki",
        drug_y_label="MEKi"
    )

    # --- Plot 2: AKT + MEK ---
    success2 = _plot_validation_subplot(
        ax=axes[1],
        synergy_exp_name="synergy_sweep-akt_mek-2606-1819-4p_2D_drugtiming_synonly_consensus_hybrid_20",
        drug_x_label="AKTi",
        drug_y_label="MEKi"
    )

    # --- Final Figure Adjustments ---
    axes[0].set_ylabel('% Alive Cells (Relative to Control)', weight='bold', fontsize=14)
    axes[0].set_ylim(0, 125)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    save_dir = "scripts/post_emews_analysis/synergy_recovery_experiments/synergy_validation_plots_2D"
    os.makedirs(save_dir, exist_ok=True)


    output_filename = "internal_synergy_validation_publication.png"
    save_path = os.path.join(save_dir, output_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"\nCombined synergy validation plot saved to {save_path}")
    plt.close(fig) 

if __name__ == "__main__":
    main()