import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import seaborn as sns

# REPLACE the current font settings with more SSH-friendly options
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',  # More widely available on servers
    'mathtext.fontset': 'dejavusans',
    'font.size': 12,
    # Fallback to default fonts that are commonly available on Linux systems
    'font.sans-serif': ['DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif', 
                        'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Helvetica', 
                        'Avant Garde', 'sans-serif']
})

# Load the timing CSV
# experiment_name = "synergy_sweep-pi3k_mek-0505-1008-4p_3D_drugtiming"
# experiment_name = "synergy_sweep-akt_mek-0505-1910-4p_3D_drugtiming"

# NEW DATA
experiment_name = "synergy_sweep-pi3k_mek-2606-0158-4p_3D_drugtiming_synonly_consensus_hybrid_20"
# experiment_name = "synergy_sweep-akt_mek-2506-1757-4p_2D_drugtiming_consensus"
timing_df = pd.read_csv(f'results/sweep_summaries/final_summary_{experiment_name}.csv')

# Create delta time as the signed difference between the two pulse periods
timing_df['delta_time'] = timing_df['user_parameters.drug_X_pulse_period'] - timing_df['user_parameters.drug_Y_pulse_period']

# --- FIX: Exclude "late simultaneous" runs that act as controls ---
# These are runs where delta_time is 0 but because the pulse period is very large,
# it's effectively a no-drug scenario.
late_addition_threshold = 1000 # A reasonable threshold for a "late" addition
late_control_runs = (timing_df['delta_time'] == 0) & (timing_df['user_parameters.drug_X_pulse_period'] > late_addition_threshold)
print(f"\nExcluding {late_control_runs.sum()} late-addition 'control' runs from analysis.")
timing_df = timing_df[~late_control_runs]
# --------------------------------------------------------------------

# Group by diffusion coefficients and delta_time, compute mean alive cells
grouped = timing_df.groupby(
    ['user_parameters.drug_X_diffusion_coefficient', 
     'user_parameters.drug_Y_diffusion_coefficient', 
     'delta_time']
).agg(
    mean_alive=('FINAL_NUMBER_OF_ALIVE_CELLS', 'mean'),
    std_alive=('FINAL_NUMBER_OF_ALIVE_CELLS', 'std')
).reset_index()

# --- Load Control Data ---
# This is now at the top to ensure the negative control value is available for all plots.
sweep_summaries_path = "results/sweep_summaries/"
negative_control_name = "synergy_sweep-3D-0205-1608-control_nodrug"
final_summary_negative_control = pd.read_csv(f"{sweep_summaries_path}/final_summary_{negative_control_name}.csv")
mean_final_number_of_alive_cells_negative_control = round(final_summary_negative_control.iloc[:, -1].mean())
std_final_number_of_alive_cells_negative_control = round(final_summary_negative_control.iloc[:, -1].std(), 2)
print(f"NEGATIVE CONTROL mean: {mean_final_number_of_alive_cells_negative_control}")
print(f"NEGATIVE CONTROL std: {std_final_number_of_alive_cells_negative_control}")
print("--------------------------------\n")

# Get sorted unique values for grid arrangement
x_diffs = sorted(grouped['user_parameters.drug_X_diffusion_coefficient'].unique())
y_diffs = sorted(grouped['user_parameters.drug_Y_diffusion_coefficient'].unique())

# CHANGE: Reverse the x_diffs list to put lowest values at the bottom
x_diffs = x_diffs[::-1]

# Set up the grid of plots with shared y-axis
nrows = len(x_diffs)
ncols = len(y_diffs)
# ADAPTED AESTHETICS
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.7*nrows), sharex=True, sharey=True, dpi=300)

for i, x_diff in enumerate(x_diffs):
    for j, y_diff in enumerate(y_diffs):
        if nrows == 1 and ncols == 1:
            ax = axes
        elif nrows == 1 or ncols == 1:
            ax = axes[max(i, j)]
        else:
            ax = axes[i, j]
        grouped_subdf = grouped[
            (grouped['user_parameters.drug_X_diffusion_coefficient'] == x_diff) &
            (grouped['user_parameters.drug_Y_diffusion_coefficient'] == y_diff)
        ].sort_values('delta_time')
        if not grouped_subdf.empty:
            # ADAPTED AESTHETICS: Use categorical x-axis
            delta_times = list(grouped_subdf['delta_time'])
            xvals = np.arange(len(delta_times))
            
            # Plot line for means (in gray) - NORMALIZED
            ax.plot(xvals, (grouped_subdf['mean_alive'] / mean_final_number_of_alive_cells_negative_control) * 100, color='gray', zorder=1)
            
            # Plot each point with its own color and error bar - NORMALIZED
            for plot_idx, (idx, row) in enumerate(grouped_subdf.iterrows()):
                color = 'green' if row['delta_time'] == 0 else ('blue' if row['delta_time'] > 0 else 'orange')
                ax.errorbar(
                    xvals[plot_idx], (row['mean_alive'] / mean_final_number_of_alive_cells_negative_control) * 100,
                    yerr=(row['std_alive'] / mean_final_number_of_alive_cells_negative_control) * 100,
                    fmt='o', color=color, ecolor='black', elinewidth=2, capsize=3,
                    markerfacecolor=color, zorder=2
                )

            # ADAPTED AESTHETICS: Add vertical line at delta_time == 0
            zero_idx_list = [idx for idx, dt in enumerate(delta_times) if dt == 0]
            if zero_idx_list:
                ax.axvline(zero_idx_list[0], color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

            ax.set_ylim(bottom=0)
            ax.set_title(f'X: {x_diff}, Y: {y_diff}')
            
            # Highlight the minimum - NORMALIZED
            min_idx = grouped_subdf['mean_alive'].idxmin()
            min_pos = list(grouped_subdf.index).index(min_idx)
            min_x = xvals[min_pos]
            min_y_raw = grouped_subdf.loc[min_idx, 'mean_alive']
            min_y_percent = (min_y_raw / mean_final_number_of_alive_cells_negative_control) * 100
            ax.plot(min_x, min_y_percent, 'ro', zorder=4)
            ax.annotate(f'{min_y_percent:.1f}%', (min_x, min_y_percent), textcoords="offset points", xytext=(0, -10), ha='center', color='red', fontsize=10)
            
            # ADAPTED AESTHETICS: Set categorical x-ticks
            ax.set_xticks(xvals)
            ax.set_xticklabels([str(int(dt)) if dt == int(dt) else f"{dt:.1f}" for dt in delta_times], rotation=45, fontsize=8)

        else:
            ax.axis('off')

# ADAPTED AESTHETICS: Use subplots_adjust and consistent labels
plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12, wspace=0.18, hspace=0.25)

fig.text(0.005, 0.55, '% Alive Cells (of Control)', 
           va='center', ha='center', rotation='vertical',
           fontsize=16, fontweight='bold')

fig.text(0.5, 0.03, 'Delta Time (X - Y)', 
           ha='center', va='center',
           fontsize=16, fontweight='bold')

for ax in axes.flat:
    if not ax.get_visible():
        continue
    current_title = ax.get_title()
    ax.set_title(current_title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)

# Save the figure with higher DPI
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = f"{script_dir}/optimal_timings_synergy"

if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(f"{save_path}/optimal_timings_synergy_{experiment_name}.png", dpi=600, bbox_inches='tight')
plt.show()

#################################################################################################
# Reading the singledrug control experiments
#################################################################################################
if "pi3k_mek" in experiment_name.lower():
    drug1_single_drug_name = "synergy_sweep-pi3k_mek-3D-0505-0218-logscale_singledrug_pi3k"
    drug2_single_drug_name = "synergy_sweep-pi3k_mek-3D-0505-0218-logscale_singledrug_mek"
    drug1_label, drug2_label = 'Drug X (PI3Ki)', 'Drug Y (MEKi)'
elif "akt_mek" in experiment_name.lower():
    drug1_single_drug_name = "synergy_sweep-akt_mek-3D-0505-1910-logscale_singledrug_akt"
    drug2_single_drug_name = "synergy_sweep-akt_mek-3D-0505-1910-logscale_singledrug_mek"
    drug1_label, drug2_label = 'Drug X (AKTi)', 'Drug Y (MEKi)'
else:
    raise ValueError("Could not determine drugs from experiment name")

drug1_single_drug = pd.read_csv(f"{sweep_summaries_path}/final_summary_{drug1_single_drug_name}.csv")
drug2_single_drug = pd.read_csv(f"{sweep_summaries_path}/final_summary_{drug2_single_drug_name}.csv")

# we need to obtain a dictionary of the diffusion coefficient and the average final number of alive cells
drug1_grouped = drug1_single_drug.groupby(drug1_single_drug.columns[0]).agg({drug1_single_drug.columns[-1]: 'mean'})
drug1_dict = drug1_grouped[drug1_single_drug.columns[-1]].to_dict()

drug2_grouped = drug2_single_drug.groupby(drug2_single_drug.columns[0]).agg({drug2_single_drug.columns[-1]: 'mean'})
drug2_dict = drug2_grouped[drug2_single_drug.columns[-1]].to_dict()

print("--------------------------------")
print(f"Drug 1 ({drug1_label}) single drug dict: {drug1_dict}")
print(f"Drug 2 ({drug2_label}) single drug dict: {drug2_dict}")
print("--------------------------------\n")

# --- NEW: Fit and Plot Single-Drug Dose-Response Curves using a 4PL Model ---

def four_param_logistic(x, bottom, top, ic50, hill_slope):
    """Four-parameter logistic model function."""
    return bottom + (top - bottom) / (1 + (x / ic50)**hill_slope)

def fit_and_plot_dr_curve(ax, drug_dict, drug_name, control_mean, marker, color):
    """Fits a 4PL model to dose-response data and plots it."""
    doses = np.array(sorted(drug_dict.keys()))
    responses_raw = np.array([drug_dict[d] for d in doses])
    
    # Use bounds to guide the fit. IC50 must be positive.
    # Top should be around the control mean, bottom should be around the min response.
    p0 = [np.min(responses_raw), control_mean, np.median(doses), -1]
    bounds = (
        [0, 0, 0, -np.inf],
        [control_mean*1.2, control_mean*1.2, np.inf, np.inf]
    )

    try:
        params, _ = curve_fit(four_param_logistic, doses, responses_raw, p0=p0, bounds=bounds, maxfev=5000)
        
        # Plot raw data
        ax.plot(doses, (responses_raw / control_mean) * 100, marker=marker, linestyle='none', label=f'{drug_name} (data)', color=color)
        
        # Plot fitted curve
        fine_doses = np.logspace(np.log10(min(doses)), np.log10(max(doses)), 100)
        fitted_responses = four_param_logistic(fine_doses, *params)
        ax.plot(fine_doses, (fitted_responses / control_mean) * 100, linestyle='-', label=f'{drug_name} (4PL Fit)', color=color)
        
        print(f"4PL Fit Parameters for {drug_name}: {params}")
        return params
    except RuntimeError:
        print(f"Could not fit 4PL curve for {drug_name}. Plotting raw data only.")
        # Fallback to plotting raw data if fit fails
        ax.plot(doses, (responses_raw / control_mean) * 100, marker=marker, linestyle='--', label=f'{drug_name} (raw data)', color=color)
        return None

# Create the plot
fig_dr, ax_dr = plt.subplots(figsize=(8, 6), dpi=300)

# Fit and plot for each drug
drug1_params = fit_and_plot_dr_curve(ax_dr, drug1_dict, drug1_label, mean_final_number_of_alive_cells_negative_control, 'o', 'blue')
drug2_params = fit_and_plot_dr_curve(ax_dr, drug2_dict, drug2_label, mean_final_number_of_alive_cells_negative_control, 's', 'green')

# Add the no-drug control line for reference
ax_dr.axhline(100, color='gray', linestyle='--', label=f'No-Drug Control')

# Formatting
ax_dr.set_xscale('log')
ax_dr.set_xlabel('Diffusion Coefficient (Dose)', fontsize=12, fontweight='bold')
ax_dr.set_ylabel('% Alive Cells (of Control)', fontsize=12, fontweight='bold')
ax_dr.set_title('Single-Drug Dose-Response Curves (4PL Fit)', fontsize=14, fontweight='bold')
ax_dr.legend()
ax_dr.grid(True, which="both", ls="--", linewidth=0.5)
ax_dr.set_ylim(bottom=0)

# Improve layout and save
plt.tight_layout()
dose_response_save_path = f"{save_path}/single_drug_dose_response_{experiment_name}.png"
plt.savefig(dose_response_save_path, dpi=600)
print(f"\nSingle-drug dose-response curve plot saved to: {dose_response_save_path}")
plt.show()

def get_experimental_dose_interpolators():
    """
    Loads experimental dose-response curves from CSV files and creates
    interpolation functions to map a given drug effect to the required dose.
    """
    from scipy import interpolate
    import pandas as pd
    
    # Define paths to the experimental data
    base_path = "/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/data/AGS_data/AGS_growth_data/drug_response_curves"
    paths = {
        "akt": f"{base_path}/AKTi/points_from_pdf/AKTi_DR_averaged_linear_uM.csv",
        "mek": f"{base_path}/PD0/PD0_DR_averaged_linear_uM.csv",
        "pi3k": f"{base_path}/PI103/PI103_DR_averaged_linear_uM.csv"
    }
    
    # Load data
    curves = {drug: pd.read_csv(path) for drug, path in paths.items()}
    
    # Create interpolation functions (effect -> dose)
    # Assumes 'drug_effect' is the target metric.
    interpolators = {
        "akt": interpolate.interp1d(curves["akt"]['drug_effect'], curves["akt"]['drug_concentration'], bounds_error=False, fill_value="extrapolate"),
        "mek": interpolate.interp1d(curves["mek"]['drug_effect'], curves["mek"]['drug_concentration'], bounds_error=False, fill_value="extrapolate"),
        "pi3k": interpolate.interp1d(curves["pi3k"]['drug_effect'], curves["pi3k"]['drug_concentration'], bounds_error=False, fill_value="extrapolate")
    }
    
    return interpolators['akt'], interpolators['mek'], interpolators['pi3k']

def create_diffusion_to_concentration_mappers(drug_dict, control_mean, get_dose_exp):
    """
    Creates a mapping function from simulation diffusion coefficient to an
    equivalent experimental concentration based on the observed effect.
    """
    # Get the effect for each simulated dose (diffusion coeff)
    sim_doses = np.array(sorted(drug_dict.keys()))
    sim_effects = np.array([drug_dict[d] for d in sim_doses]) / control_mean
    
    # Find the experimental concentration that produces the same effect
    exp_equiv_concs = get_dose_exp(sim_effects)
    
    # Create the mapping function: diffusion_coeff -> concentration (µM)
    # Sort by diffusion coefficient to ensure the interpolator is monotonic
    sorted_pairs = sorted(zip(sim_doses, exp_equiv_concs))
    mapper = interp1d(
        [p[0] for p in sorted_pairs], 
        [p[1] for p in sorted_pairs], 
        bounds_error=False, 
        fill_value="extrapolate"
    )
    return mapper

# --- Build Comprehensive Synergy Dataframe ---
print("\nBuilding comprehensive dataframe with all synergy metrics...")

all_metrics_rows = []

# Define hard-coded concentrations for d1 and d2 based on GI50 values from the user's table.
# d = (GI50_mM * 1000 uM/mM) / 2
concentrations_uM = {
    'pi3k': (7e-4 * 1000) / 2,   # 0.35 uM
    'mek': (3.5e-5 * 1000) / 2, # 0.0175 uM
    'akt': (1e-2 * 1000) / 2     # 5.0 uM
}


# Get Loewe interpolators and drug names
get_dose_akt_exp, get_dose_mek_exp, get_dose_pi3k_exp = get_experimental_dose_interpolators()

# --- Determine drug combination based on experiment name ---
if "pi3k_mek" in experiment_name.lower():
    get_dose_drug1_exp, get_dose_drug2_exp = get_dose_pi3k_exp, get_dose_mek_exp
    d1_conc = concentrations_uM['pi3k']
    d2_conc = concentrations_uM['mek']
elif "akt_mek" in experiment_name.lower():
    get_dose_drug1_exp, get_dose_drug2_exp = get_dose_akt_exp, get_dose_mek_exp
    d1_conc = concentrations_uM['akt']
    d2_conc = concentrations_uM['mek']
else:
    d1_conc, d2_conc = np.nan, np.nan


# Main loop over all experimental conditions
for _, row in grouped.iterrows():
    # Common variables
    x_diff = row['user_parameters.drug_X_diffusion_coefficient']
    y_diff = row['user_parameters.drug_Y_diffusion_coefficient']
    delta_time = row['delta_time']
    observed_alive = row['mean_alive']
    std_alive = row['std_alive']
    
    # 1. % Alive Cells
    percent_alive = (observed_alive / mean_final_number_of_alive_cells_negative_control) * 100
    
    # 2. Bliss Score
    bliss_score = np.nan
    drug1_single_effect = drug1_dict.get(x_diff)
    drug2_single_effect = drug2_dict.get(y_diff)
    if drug1_single_effect is not None and drug2_single_effect is not None:
        E_A = drug1_single_effect / mean_final_number_of_alive_cells_negative_control
        E_B = drug2_single_effect / mean_final_number_of_alive_cells_negative_control
        E_AB = observed_alive / mean_final_number_of_alive_cells_negative_control
        bliss_expected = E_A * E_B
        # Invert to match paper's convention (negative score = synergy)
        bliss_score = E_AB - bliss_expected

    # 3. Loewe Score (Combination Index)
    loewe_CI = np.nan
    observed_effect_loewe = observed_alive / mean_final_number_of_alive_cells_negative_control
    try:
        # Get the doses of single drugs that would produce the same effect as the combination
        D1_exp = get_dose_drug1_exp(observed_effect_loewe)
        D2_exp = get_dose_drug2_exp(observed_effect_loewe)
        
        # d1 and d2 are now the fixed concentrations based on half GI50
        if D1_exp > 0 and D2_exp > 0 and not (np.isnan(d1_conc) or np.isnan(d2_conc)):
            loewe_CI = (d1_conc / D1_exp) + (d2_conc / D2_exp)
    except (KeyError, ValueError):
        pass

    all_metrics_rows.append({
        'x_diff': x_diff,
        'y_diff': y_diff,
        'delta_time': delta_time,
        'percent_alive': percent_alive,
        'bliss_score': bliss_score,
        'loewe_CI': loewe_CI,
        'std_alive': std_alive
    })

combined_metrics_df = pd.DataFrame(all_metrics_rows)
combined_metrics_path = f"{save_path}/combined_synergy_metrics_{experiment_name}.csv"
combined_metrics_df.to_csv(combined_metrics_path, index=False)
print(f"Combined synergy metrics saved to: {combined_metrics_path}")

# --- NEW: Plot to Compare Synergy Metrics ---
def plot_metric_comparison(df, save_path, experiment_name):
    """
    Creates a 3-panel scatter plot to compare efficacy, Bliss, and Loewe scores.
    """
    # Create the figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    fig.suptitle(f'Comparison of Synergy Metrics for {experiment_name}', fontsize=16, weight='bold', y=1.02)

    # Use a diverging colormap for delta_time
    cmap = plt.get_cmap('coolwarm')
    norm = plt.Normalize(df['delta_time'].min(), df['delta_time'].max())
    
    # --- Plot 1: Efficacy vs. Bliss ---
    scatter1 = ax1.scatter(df['bliss_score'], df['percent_alive'], c=df['delta_time'], cmap=cmap, norm=norm, alpha=0.7)
    ax1.set_xlabel('Bliss Score (Synergy > 0)', weight='bold')
    ax1.set_ylabel('% Alive Cells (Efficacy)', weight='bold')
    ax1.set_title('Efficacy vs. Bliss Synergy', weight='bold')
    ax1.axvline(0, color='gray', linestyle='--')
    ax1.axhline(100, color='red', linestyle='--', label='No-Drug Control')
    ax1.invert_yaxis() # Lower % alive is better efficacy
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Efficacy vs. Loewe ---
    scatter2 = ax2.scatter(df['loewe_CI'], df['percent_alive'], c=df['delta_time'], cmap=cmap, norm=norm, alpha=0.7)
    ax2.set_xlabel('Loewe CI (Synergy < 1)', weight='bold')
    ax2.set_ylabel('% Alive Cells (Efficacy)', weight='bold')
    ax2.set_title('Efficacy vs. Loewe Synergy', weight='bold')
    ax2.axvline(1, color='gray', linestyle='--')
    ax2.axhline(100, color='red', linestyle='--')
    ax2.invert_yaxis()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 3: Bliss vs. Loewe ---
    scatter3 = ax3.scatter(df['bliss_score'], df['loewe_CI'], c=df['delta_time'], cmap=cmap, norm=norm, alpha=0.7)
    ax3.set_xlabel('Bliss Score', weight='bold')
    ax3.set_ylabel('Loewe CI', weight='bold')
    ax3.set_title('Bliss vs. Loewe Models', weight='bold')
    ax3.axvline(0, color='gray', linestyle='--')
    ax3.axhline(1, color='gray', linestyle='--')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Add a shared colorbar
    cbar = fig.colorbar(scatter1, ax=[ax1, ax2, ax3], orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Delta Time (X - Y)', weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    comparison_save_path = f"{save_path}/metrics_comparison_scatter_{experiment_name}.png"
    plt.savefig(comparison_save_path, dpi=600)
    print(f"\nMetrics comparison plot saved to: {comparison_save_path}")
    plt.show()

# Call the new plotting function
plot_metric_comparison(combined_metrics_df, save_path, experiment_name)

def plot_synergy_radar_chart(df, save_path, experiment_name):
    """
    Creates a radar chart to compare normalized Efficacy, Bliss, and Loewe scores for specific cases.
    This version aggregates data for each scenario instead of picking one optimal point.
    """
    # 1. Data Normalization
    metrics_df = df.copy()

    # Handle NaNs before normalization
    metrics_df['bliss_score'].fillna(0, inplace=True)
    # For Loewe, filling with a high number represents strong antagonism before normalization
    metrics_df['loewe_CI'].fillna(10, inplace=True) 

    # Normalize Efficacy (lower % alive is better)
    min_alive = metrics_df['percent_alive'].min()
    max_alive = metrics_df['percent_alive'].max()
    if max_alive > min_alive:
        metrics_df['norm_efficacy'] = (max_alive - metrics_df['percent_alive']) / (max_alive - min_alive)
    else:
        metrics_df['norm_efficacy'] = 0.5

    # Normalize Bliss Score (higher is better, assuming synergy > 0)
    min_bliss = metrics_df['bliss_score'].min()
    max_bliss = metrics_df['bliss_score'].max()
    if (max_bliss - min_bliss) > 0:
        # Invert scale: lower (more negative) score -> higher normalized value
        metrics_df['norm_bliss'] = (max_bliss - metrics_df['bliss_score']) / (max_bliss - min_bliss)
    else:
        metrics_df['norm_bliss'] = 0.5

    # Normalize Loewe Score (lower is better, clip at a reasonable upper bound like 2)
    metrics_df['loewe_CI_clipped'] = metrics_df['loewe_CI'].clip(lower=0, upper=2)
    metrics_df['norm_loewe'] = 1 - (metrics_df['loewe_CI_clipped'] / 2) # 0->1, 1->0.5, 2->0

    # 2. Define and apply scenario categories
    def categorize_scenario(row):
        is_symmetric = row['x_diff'] == row['y_diff']
        is_low_dose = row['x_diff'] <= 60 and row['y_diff'] <= 60
        is_high_dose = row['x_diff'] >= 600 and row['y_diff'] >= 600
        
        if is_symmetric:
            if is_low_dose:
                return 'Symmetric Low Dose'
            if is_high_dose:
                return 'Symmetric High Dose'
            # Any other symmetric cases can be ignored for this specific plot
            return 'Other'

        # Asymmetric cases
        x_is_fast = row['x_diff'] > row['y_diff']
        # delta_time = T_x - T_y. Negative means X is first.
        x_is_first = row['delta_time'] < 0

        if x_is_fast:
            return 'Asymmetric: Fast Drug First' if x_is_first else 'Asymmetric: Slow Drug First'
        else: # y_is_fast
            return 'Asymmetric: Slow Drug First' if x_is_first else 'Asymmetric: Fast Drug First'

    metrics_df['scenario'] = metrics_df.apply(categorize_scenario, axis=1)

    # 3. Aggregate data by scenario
    scenario_summary = metrics_df[metrics_df['scenario'] != 'Other'].groupby('scenario').agg(
        Efficacy=('norm_efficacy', 'mean'),
        Bliss=('norm_bliss', 'mean'),
        Loewe=('norm_loewe', 'mean')
    ).reset_index()

    # 4. Create Radar Plot
    labels = ['Efficacy', 'Bliss Synergy', 'Loewe Synergy']
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Complete the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)

    for i, row in scenario_summary.iterrows():
        values = row[['Efficacy', 'Bliss', 'Loewe']].tolist()
        values += values[:1] # Complete the loop
        ax.plot(angles, values, label=row['scenario'], linewidth=2, linestyle='solid', marker='o')
        ax.fill(angles, values, alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    ax.set_title('Aggregated Synergy & Efficacy Comparison', size=16, weight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
    ax.set_ylim(0, 1)

    plt.tight_layout()
    radar_save_path = f"{save_path}/synergy_radar_plot_aggregated_{experiment_name}.png"
    plt.savefig(radar_save_path, dpi=600, bbox_inches='tight')
    print(f"\nAggregated synergy radar plot saved to: {radar_save_path}")
    plt.show()

# Call the radar plot function
plot_synergy_radar_chart(combined_metrics_df, save_path, experiment_name)

def plot_metric_trends_by_timing(df, x_diffs, y_diffs, save_path, experiment_name, control_mean):
    """
    Creates a 2x2 grid of line plots to show how Efficacy, Bliss, and Loewe scores
    trend across different delta times for four key scenarios.
    """
    metrics_df = df.copy()

    # --- Normalize metrics (0-1 scale, higher is better) ---
    metrics_df['norm_efficacy'] = (control_mean - metrics_df['percent_alive'] * control_mean / 100) / control_mean
    min_bliss, max_bliss = metrics_df['bliss_score'].min(), metrics_df['bliss_score'].max()
    if (max_bliss - min_bliss) > 0:
        # Correctly normalize Bliss: more negative is better
        metrics_df['norm_bliss'] = (max_bliss - metrics_df['bliss_score']) / (max_bliss - min_bliss)
    else:
        metrics_df['norm_bliss'] = 0.5
    
    metrics_df['loewe_CI_clipped'] = metrics_df['loewe_CI'].clip(lower=0, upper=2)
    metrics_df['norm_loewe'] = 1 - (metrics_df['loewe_CI_clipped'] / 2)
    
    # --- Define and apply scenario categories from radar plot ---
    def categorize_scenario(row):
        is_symmetric = row['x_diff'] == row['y_diff']
        is_low_dose = row['x_diff'] <= 60 and row['y_diff'] <= 60
        is_high_dose = row['x_diff'] >= 600 and row['y_diff'] >= 600
        
        if is_symmetric:
            if is_low_dose: return 'Symmetric Low Dose'
            if is_high_dose: return 'Symmetric High Dose'
            return 'Other'

        x_is_fast = row['x_diff'] > row['y_diff']
        x_is_first = row['delta_time'] < 0
        if x_is_fast:
            return 'Asymmetric: Fast Drug First' if x_is_first else 'Asymmetric: Slow Drug First'
        else: # y_is_fast
            return 'Asymmetric: Slow Drug First' if x_is_first else 'Asymmetric: Fast Drug First'

    metrics_df['scenario'] = metrics_df.apply(categorize_scenario, axis=1)

    # Filter out 'Other' and aggregate data for asymmetric cases
    plot_df = metrics_df[metrics_df['scenario'] != 'Other'].copy()
    
    # For asymmetric, we need to average across different diffusion pairs for each delta_time
    numeric_cols = ['norm_efficacy', 'norm_bliss', 'norm_loewe']
    asym_fast_df = plot_df[plot_df['scenario'] == 'Asymmetric: Fast Drug First'].groupby('delta_time')[numeric_cols].mean().reset_index()
    asym_slow_df = plot_df[plot_df['scenario'] == 'Asymmetric: Slow Drug First'].groupby('delta_time')[numeric_cols].mean().reset_index()
    
    # Get the specific symmetric data and aggregate it by timing as well
    sym_low_df = plot_df[plot_df['scenario'] == 'Symmetric Low Dose'].groupby('delta_time')[numeric_cols].mean().reset_index()
    sym_high_df = plot_df[plot_df['scenario'] == 'Symmetric High Dose'].groupby('delta_time')[numeric_cols].mean().reset_index()
    
    scenarios_data = {
        'Symmetric Low Dose': sym_low_df,
        'Symmetric High Dose': sym_high_df,
        'Asymmetric: Fast Drug First': asym_fast_df,
        'Asymmetric: Slow Drug First': asym_slow_df
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True, dpi=300)
    axes = axes.flatten()
    fig.suptitle(f'Aggregated Metric Trends vs. Drug Timing for {experiment_name}', fontsize=16, weight='bold', y=1.0)

    for i, (title, data) in enumerate(scenarios_data.items()):
        ax = axes[i]
        if not data.empty:
            ax.plot(data['delta_time'], data['norm_efficacy'], marker='o', linestyle='-', label='Efficacy', color='red')
            ax.plot(data['delta_time'], data['norm_bliss'], marker='s', linestyle='-', label='Bliss Synergy', color='blue')
            ax.plot(data['delta_time'], data['norm_loewe'], marker='^', linestyle='-', label='Loewe Synergy', color='green')
            
            ax.set_title(title, fontsize=12, weight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.axvline(0, color='gray', linestyle=':', linewidth=1.5)
            ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        else:
            ax.text(0.5, 0.5, 'No Data for this Scenario', ha='center', va='center')
            ax.set_title(title, fontsize=12, weight='bold')

    # Shared labels and legend
    fig.text(0.5, 0.01, 'Delta Time (Negative = Fast Drug First)', ha='center', va='center', fontsize=14, weight='bold')
    fig.text(0.01, 0.5, 'Normalized Score (0 to 1, Higher is Better)', ha='center', va='center', rotation='vertical', fontsize=14, weight='bold')
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=12)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.93])
    trends_save_path = f"{save_path}/metric_trends_by_scenario_{experiment_name}.png"
    plt.savefig(trends_save_path, dpi=600)
    print(f"\nMetric trends plot by scenario saved to: {trends_save_path}")
    plt.show()

# Call the new function at the end
plot_metric_trends_by_timing(combined_metrics_df, x_diffs, y_diffs, save_path, experiment_name, mean_final_number_of_alive_cells_negative_control)

summary = []

for i, x_diff in enumerate(x_diffs):
    for j, y_diff in enumerate(y_diffs):
        grouped_subdf = grouped[
            (grouped['user_parameters.drug_X_diffusion_coefficient'] == x_diff) &
            (grouped['user_parameters.drug_Y_diffusion_coefficient'] == y_diff)
        ].sort_values('delta_time')
        if not grouped_subdf.empty:
            min_idx = grouped_subdf['mean_alive'].idxmin()
            min_row = grouped_subdf.loc[min_idx]
            delta_time = min_row['delta_time']
            mean_alive = min_row['mean_alive']
            # Classify timing
            if delta_time > 0:
                timing_type = "X after Y"
            elif delta_time < 0:
                timing_type = "X before Y"
            else:
                timing_type = "Simultaneous"
            summary.append({
                'X_diff': x_diff,
                'Y_diff': y_diff,
                'Optimal_delta_time': delta_time,
                'Mean_alive_cells_percent': (mean_alive / mean_final_number_of_alive_cells_negative_control) * 100,
                'Timing_type': timing_type
            })

summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values(['X_diff', 'Y_diff'])

# Print the summary
# print("\nQuantitative summary of optimal timings for each diffusion pair:")
# print(summary_df)

# Optionally, save to CSV
summary_csv_path = f"{save_path}/optimal_timing_summary_{experiment_name}.csv"
summary_df.to_csv(summary_csv_path, index=False)
print(f"\nSummary table saved to: {summary_csv_path}")

# --- Bliss Independence Score Calculation and Plotting ---

# The calculation is now done above and stored in combined_metrics_df.
# We will use this dataframe for plotting.
bliss_df = combined_metrics_df.copy()

# When creating your figure, keep the larger figure size
fig_bliss, axes_bliss = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.7*nrows), sharex=True, sharey=True, dpi=300)

for i, x_diff in enumerate(x_diffs):
    for j, y_diff in enumerate(y_diffs):
        if nrows == 1 and ncols == 1:
            ax = axes_bliss
        elif nrows == 1 or ncols == 1:
            ax = axes_bliss[max(i, j)]
        else:
            ax = axes_bliss[i, j]
        subdf = bliss_df[
            (bliss_df['x_diff'] == x_diff) &
            (bliss_df['y_diff'] == y_diff)
        ].sort_values('delta_time')
        if not subdf.empty:
            # Use delta_time as categories
            delta_times = list(subdf['delta_time'])
            xvals = np.arange(len(delta_times))
            # Plot the line connecting points
            ax.plot(xvals, subdf['bliss_score'], color='gray', zorder=1)
            # Plot each point with its own color and error bar
            for plot_idx, (idx, row) in enumerate(subdf.iterrows()):
                color = 'green' if row['delta_time'] == 0 else ('blue' if row['delta_time'] > 0 else 'orange')
                ax.errorbar(
                    xvals[plot_idx], row['bliss_score'],
                    yerr=row['std_alive'] / mean_final_number_of_alive_cells_negative_control,
                    fmt='o', color=color, markerfacecolor=color, ecolor='black', elinewidth=2, capsize=3, zorder=2
                )
            # Highlight the minimum (most negative = strongest synergy)
            min_idx = subdf['bliss_score'].idxmin()
            min_pos = list(subdf.index).index(min_idx)
            min_x = xvals[min_pos]
            min_y = subdf.loc[min_idx, 'bliss_score']
            ax.plot(min_x, min_y, 'ro', zorder=4)
            ax.annotate(f'{min_y:.2f}', (min_x, min_y), textcoords="offset points", xytext=(0, -10), ha='center', color='red', fontsize=8)
            # Set categorical x-ticks
            ax.set_xticks(xvals)
            ax.set_xticklabels([str(int(dt)) if dt == int(dt) else f"{dt:.1f}" for dt in delta_times], rotation=45, fontsize=8)
            ax.set_xlabel('Delta Time (X - Y)')
            ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.set_title(f'X: {x_diff}, Y: {y_diff}')
        else:
            ax.axis('off')

# REPLACE the current subplot adjustment with the same tight margin
plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12, wspace=0.18, hspace=0.25)

# REPLACE the y-axis label with separate text elements - one bold, one regular
# Main Bliss Score label in bold
fig_bliss.text(0.005, 0.55, 'Bliss Score', 
               va='center', ha='center', rotation='vertical',
               fontsize=16, fontweight='bold')

# Explanation text in regular weight
# fig_bliss.text(0.005, 0.45, '(Negative = Synergy\nPositive = Antagonism)', 
#                va='center', ha='left', rotation='vertical',
#                fontsize=12, fontweight='normal')

# Keep the global x-axis label with larger font
fig_bliss.text(0.5, 0.03, 'Delta Time (X - Y)', 
               ha='center', va='center',
               fontsize=16, fontweight='bold')

# Improve subplot titles with larger font
for ax in axes_bliss.flat:
    # Skip empty subplots
    if not ax.get_visible():
        continue
        
    # Get current title and format it better
    current_title = ax.get_title()
    ax.set_title(current_title, fontsize=14, fontweight='bold')
    
    # Make tick labels larger
    ax.tick_params(axis='both', labelsize=12)
    
    # Strengthen the zero line for better visual reference
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Increase annotation font size
    for child in ax.get_children():
        if isinstance(child, plt.Text) and child.get_text().startswith('-0.') and child.get_color() == 'red':
            child.set_fontsize(10)  # Increase size of minimum value annotations

# Save with higher DPI and tight bbox
plt.savefig(f"{save_path}/optimal_timings_bliss_{experiment_name}.png", 
           dpi=600, bbox_inches='tight')
plt.show()

# --- Loewe Additivity Score Calculation and Plotting ---

# The calculation is now done above and stored in combined_metrics_df.
# We will use this dataframe for plotting.
loewe_df = combined_metrics_df.copy()

# Plotting Loewe scores vs. timing
plt.figure(figsize=(10, 6))
sns.boxplot(x='x_diff', y='loewe_CI', data=loewe_df)
plt.axhline(1.0, color='red', linestyle='--', label='Additivity (CI=1)')
plt.title(f'Loewe Synergy vs. Drug Timing for {experiment_name}')
plt.xlabel('Timing of Second Drug (minutes)')
plt.ylabel('Loewe Combination Index (CI)')
plt.legend()
    
# Create the output directory if it doesn't exist
output_dir = 'results/synergy_analysis'
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f'{output_dir}/loewe_synergy_{experiment_name}.png')
plt.show()

# --- NEW: Generate Summary Heatmaps ---

# 1. Find the optimal delta_time and max synergy for each diffusion pair
# We use idxmin() to get the index of the row with the minimum bliss_score for each group
optimal_df = combined_metrics_df.loc[combined_metrics_df.groupby(['x_diff', 'y_diff'])['bliss_score'].idxmin()].reset_index(drop=True)

# 2. Pivot the data to create 2D matrices for the heatmaps
# The 'index' in pivot becomes the rows (y-axis of heatmap), and 'columns' becomes the columns (x-axis of heatmap).
# We want x_diff on the y-axis and y_diff on the x-axis, matching the grid plots.
synergy_pivot = optimal_df.pivot(index='x_diff', columns='y_diff', values='bliss_score')
timing_pivot = optimal_df.pivot(index='x_diff', columns='y_diff', values='delta_time')

# Ensure the pivot tables have the same order as the original grid plots
synergy_pivot = synergy_pivot.reindex(index=x_diffs, columns=y_diffs)
timing_pivot = timing_pivot.reindex(index=x_diffs, columns=y_diffs)

# 3. Create the heatmap plots
fig_heat, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
plt.suptitle(f'Synergy Analysis Summary for {os.path.basename(experiment_name)}', fontsize=16, y=1.02)

# --- Maximum Synergy Heatmap ---
# Use a reversed sequential colormap so more negative (better synergy) is more prominent
cmap_synergy = plt.get_cmap('plasma_r') 
im1 = ax1.imshow(synergy_pivot, cmap=cmap_synergy, interpolation='nearest')

# Add colorbar
cbar1 = fig_heat.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Maximum Synergy (Min Bliss Score)', weight='bold')

# Add annotations
for i in range(len(x_diffs)):
    for j in range(len(y_diffs)):
        ax1.text(j, i, f"{synergy_pivot.iloc[i, j]:.2f}",
                 ha="center", va="center", color="white", fontsize=10)

# Set labels and title
ax1.set_title('Maximum Synergy', fontsize=14, weight='bold')
ax1.set_xlabel('Drug Y Diffusion Coeff.', weight='bold')
ax1.set_ylabel('Drug X Diffusion Coeff.', weight='bold')
ax1.set_xticks(np.arange(len(y_diffs)))
ax1.set_yticks(np.arange(len(x_diffs)))
ax1.set_xticklabels(y_diffs)
ax1.set_yticklabels(x_diffs)

# --- Optimal Timing Heatmap ---
# Use a diverging colormap to show positive vs negative delta time
cmap_timing = plt.get_cmap('coolwarm')
# Find the maximum absolute value for symmetric color scaling
vmax = np.abs(timing_pivot.values).max()
im2 = ax2.imshow(timing_pivot, cmap=cmap_timing, interpolation='nearest', vmin=-vmax, vmax=vmax)

# Add colorbar
cbar2 = fig_heat.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Optimal Timing (X-Y) for Max Synergy', weight='bold')

# Add annotations
for i in range(len(x_diffs)):
    for j in range(len(y_diffs)):
        ax2.text(j, i, f"{timing_pivot.iloc[i, j]:.0f}",
                 ha="center", va="center", color="black", fontsize=10)

# Set labels and title
ax2.set_title('Optimal Timing', fontsize=14, weight='bold')
ax2.set_xlabel('Drug Y Diffusion Coeff.', weight='bold')
ax2.set_ylabel('Drug X Diffusion Coeff.', weight='bold')
ax2.set_xticks(np.arange(len(y_diffs)))
ax2.set_yticks(np.arange(len(x_diffs)))
ax2.set_xticklabels(y_diffs)
ax2.set_yticklabels(x_diffs)

# Final adjustments and save
plt.tight_layout(rect=[0, 0, 1, 0.96])
heatmap_save_path = f"{save_path}/heatmap_summary_{experiment_name}.png"
plt.savefig(heatmap_save_path, dpi=600, bbox_inches='tight')
print(f"\nHeatmap summary saved to: {heatmap_save_path}")

plt.show()

# --- NEW: Generate Loewe Summary Heatmaps ---

# 1. Find the optimal delta_time and max synergy (min Loewe CI) for each diffusion pair
optimal_loewe_df = combined_metrics_df.loc[combined_metrics_df.groupby(['x_diff', 'y_diff'])['loewe_CI'].idxmin()].reset_index(drop=True)

# 2. Pivot the data to create 2D matrices for the heatmaps
loewe_synergy_pivot = optimal_loewe_df.pivot(index='x_diff', columns='y_diff', values='loewe_CI')
loewe_timing_pivot = optimal_loewe_df.pivot(index='x_diff', columns='y_diff', values='delta_time')

# Ensure the pivot tables have the same order as the original grid plots
loewe_synergy_pivot = loewe_synergy_pivot.reindex(index=x_diffs, columns=y_diffs)
loewe_timing_pivot = loewe_timing_pivot.reindex(index=x_diffs, columns=y_diffs)

# 3. Create the heatmap plots
fig_loewe_heat, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
plt.suptitle(f'Loewe Synergy Analysis Summary for {os.path.basename(experiment_name)}', fontsize=16, y=1.02)

# --- Maximum Synergy Heatmap (Loewe) ---
# Use a reversed sequential colormap so lower CI (better synergy) is more prominent
cmap_synergy_loewe = plt.get_cmap('plasma_r') 
im1 = ax1.imshow(loewe_synergy_pivot, cmap=cmap_synergy_loewe, interpolation='nearest')

cbar1 = fig_loewe_heat.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Maximum Synergy (Min Loewe CI)', weight='bold')

for i in range(len(x_diffs)):
    for j in range(len(y_diffs)):
        ax1.text(j, i, f"{loewe_synergy_pivot.iloc[i, j]:.2f}",
                 ha="center", va="center", color="white", fontsize=10)

ax1.set_title('Maximum Synergy (Loewe)', fontsize=14, weight='bold')
ax1.set_xlabel('Drug Y Diffusion Coeff.', weight='bold')
ax1.set_ylabel('Drug X Diffusion Coeff.', weight='bold')
ax1.set_xticks(np.arange(len(y_diffs)))
ax1.set_yticks(np.arange(len(x_diffs)))
ax1.set_xticklabels(y_diffs)
ax1.set_yticklabels(x_diffs)

# --- Optimal Timing Heatmap (Loewe) ---
# This part is identical to the Bliss timing heatmap
cmap_timing = plt.get_cmap('coolwarm')
vmax = np.abs(loewe_timing_pivot.values).max()
im2 = ax2.imshow(loewe_timing_pivot, cmap=cmap_timing, interpolation='nearest', vmin=-vmax, vmax=vmax)

cbar2 = fig_loewe_heat.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Optimal Timing (X-Y) for Max Synergy', weight='bold')

for i in range(len(x_diffs)):
    for j in range(len(y_diffs)):
        ax2.text(j, i, f"{loewe_timing_pivot.iloc[i, j]:.0f}",
                 ha="center", va="center", color="black", fontsize=10)

ax2.set_title('Optimal Timing (Loewe)', fontsize=14, weight='bold')
ax2.set_xlabel('Drug Y Diffusion Coeff.', weight='bold')
ax2.set_ylabel('Drug X Diffusion Coeff.', weight='bold')
ax2.set_xticks(np.arange(len(y_diffs)))
ax2.set_yticks(np.arange(len(x_diffs)))
ax2.set_xticklabels(y_diffs)
ax2.set_yticklabels(x_diffs)

plt.tight_layout(rect=[0, 0, 1, 0.96])
heatmap_save_path_loewe = f"{save_path}/heatmap_summary_loewe_{experiment_name}.png"
plt.savefig(heatmap_save_path_loewe, dpi=600, bbox_inches='tight')
print(f"\nLoewe heatmap summary saved to: {heatmap_save_path_loewe}")
plt.show()

# --- NEW: Generate 2x2 plot for specific cases ---

# Define the four cases of interest based on the results paragraph
cases = [
    {'x_diff': 6.0, 'y_diff': 6.0, 'title': 'Symmetric: Low D (X) vs. Low D (Y)'},
    {'x_diff': 6000.0, 'y_diff': 6000.0, 'title': 'Symmetric: High D (X) vs. High D (Y)'},
    {'x_diff': 6000.0, 'y_diff': 6.0, 'title': 'Asymmetric: Max Efficacy\n(Fast Drug First)', 'regime': 'neg_delta', 'optimize_for': 'efficacy'},
    {'x_diff': 6000.0, 'y_diff': 6.0, 'title': 'Asymmetric: Max Synergy\n(Slow Drug First)', 'regime': 'pos_delta', 'optimize_for': 'bliss'}
]


fig_cases, axes_cases = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
axes_cases = axes_cases.flatten()

for i, case in enumerate(cases):
    ax = axes_cases[i]
    subdf = combined_metrics_df[
        (combined_metrics_df['x_diff'] == case['x_diff']) &
        (combined_metrics_df['y_diff'] == case['y_diff'])
    ].sort_values('delta_time')

    if subdf.empty:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        ax.set_title(case['title'], fontsize=12, weight='bold')
        ax.axis('off')
        continue

    # For the mixed diffusion cases, we highlight the optimal point within the specified timing regime.
    # The user's rationale states that max efficacy occurs when X is first (delta<0)
    # and max synergy occurs when Y is first (delta>0).
    if "X is administered first" in case['title']:
        # Find minimum bliss score for delta_time <= 0
        regime_df = subdf[subdf['delta_time'] <= 0]
        if not regime_df.empty:
            min_idx = regime_df['bliss_score'].idxmin()
        else: # handle case where there are no points
            min_idx = subdf['bliss_score'].idxmin() 
    elif "Y is administered first" in case['title']:
        # Find minimum bliss score for delta_time >= 0
        regime_df = subdf[subdf['delta_time'] >= 0]
        if not regime_df.empty:
            min_idx = regime_df['bliss_score'].idxmin()
        else: # handle case where there are no points
            min_idx = subdf['bliss_score'].idxmin()
    else:
        # For symmetric cases, find the overall minimum synergy point
        min_idx = subdf['bliss_score'].idxmin()

    delta_times = list(subdf['delta_time'])
    xvals = np.arange(len(delta_times))
    
    ax.plot(xvals, subdf['bliss_score'], color='gray', zorder=1, marker='o', markerfacecolor='gray', markersize=4)
    
    for plot_idx, (idx, row) in enumerate(subdf.iterrows()):
        color = 'green' if row['delta_time'] == 0 else ('blue' if row['delta_time'] > 0 else 'orange')
        ax.errorbar(
            xvals[plot_idx], row['bliss_score'],
            yerr=row['std_alive'] / mean_final_number_of_alive_cells_negative_control,
            fmt='o', color=color, markerfacecolor=color, ecolor='black', elinewidth=1.5, capsize=3, zorder=2, markersize=8
        )
    
    # Highlight the specific optimal point for the case/regime
    min_pos = list(subdf.index).index(min_idx)
    min_x = xvals[min_pos]
    min_y = subdf.loc[min_idx, 'bliss_score']
    ax.plot(min_x, min_y, 'ro', zorder=4, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    # ADJUSTED ANNOTATION: Only show the number
    ax.annotate(f'{min_y:.2f}', (min_x, min_y), textcoords="offset points", xytext=(0, -20),
                ha='center', color='red', fontsize=9, weight='bold')

    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(xvals)
    # ADJUSTED LABELS: Tilt the x-axis labels
    ax.set_xticklabels([str(int(dt)) for dt in delta_times], rotation=45, ha='right')
    ax.set_xlabel('Delta Time (X - Y)', fontsize=10, weight='bold')
    ax.set_ylabel('Bliss Score', fontsize=10, weight='bold')
    ax.set_title(case['title'], fontsize=12, weight='bold')
    
    # Add a vertical line for simultaneous treatment for reference
    zero_idx_list = [idx for idx, dt in enumerate(delta_times) if dt == 0]
    if zero_idx_list:
        ax.axvline(zero_idx_list[0], color='black', linestyle=':', linewidth=1.5)


fig_cases.suptitle('Analysis of Optimal Synergy Under Specific Diffusion and Timing Scenarios', fontsize=16, y=1.03)
plt.tight_layout(rect=[0, 0, 1, 0.98])
cases_save_path = f"{save_path}/cases_summary_{experiment_name}.png"
plt.savefig(cases_save_path, dpi=600, bbox_inches='tight')
print(f"\nSpecific cases summary saved to: {cases_save_path}")
plt.show()

# --- NEW: Generate 2x2 plot for specific cases (Loewe) ---

fig_cases_loewe, axes_cases_loewe = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
axes_cases_loewe = axes_cases_loewe.flatten()

for i, case in enumerate(cases):
    ax = axes_cases_loewe[i]
    subdf = combined_metrics_df[
        (combined_metrics_df['x_diff'] == case['x_diff']) &
        (combined_metrics_df['y_diff'] == case['y_diff'])
    ].sort_values('delta_time')

    if subdf.empty:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        ax.set_title(case['title'], fontsize=12, weight='bold')
        ax.axis('off')
        continue

    if "X is administered first" in case['title']:
        regime_df = subdf[subdf['delta_time'] <= 0]
        if not regime_df.empty:
            min_idx = regime_df['delta_time'].idxmin()
        else:
            min_idx = subdf['delta_time'].idxmin() 
    elif "Y is administered first" in case['title']:
        regime_df = subdf[subdf['delta_time'] >= 0]
        if not regime_df.empty:
            min_idx = regime_df['delta_time'].idxmin()
        else:
            min_idx = subdf['delta_time'].idxmin()
    else:
        min_idx = subdf['delta_time'].idxmin()

    delta_times = list(subdf['delta_time'])
    xvals = np.arange(len(delta_times))
    
    ax.plot(xvals, subdf['delta_time'], color='gray', zorder=1, marker='o', markerfacecolor='gray', markersize=4)
    
    for plot_idx, (idx, row) in enumerate(subdf.iterrows()):
        color = 'green' if row['delta_time'] == 0 else ('blue' if row['delta_time'] > 0 else 'orange')
        ax.plot(xvals[plot_idx], row['delta_time'], 'o', color=color, markerfacecolor=color, zorder=2, markersize=8)
    
    min_pos = list(subdf.index).index(min_idx)
    min_x = xvals[min_pos]
    min_y = subdf.loc[min_idx, 'delta_time']
    ax.plot(min_x, min_y, 'ro', zorder=4, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    ax.annotate(f'{min_y:.2f}', (min_x, min_y), textcoords="offset points", xytext=(0, -20),
                ha='center', color='red', fontsize=9, weight='bold')

    ax.axhline(0, color='black', linestyle='--', linewidth=1) # Additivity line at CI=1
    ax.set_xticks(xvals)
    ax.set_xticklabels([str(int(dt)) for dt in delta_times], rotation=45, ha='right')
    ax.set_xlabel('Delta Time (X - Y)', fontsize=10, weight='bold')
    ax.set_ylabel('Delta Time (X - Y)', fontsize=10, weight='bold')
    ax.set_title(case['title'], fontsize=12, weight='bold')
    
    zero_idx_list = [idx for idx, dt in enumerate(delta_times) if dt == 0]
    if zero_idx_list:
        ax.axvline(zero_idx_list[0], color='black', linestyle=':', linewidth=1.5)

fig_cases_loewe.suptitle('Loewe Analysis of Optimal Synergy Under Specific Diffusion and Timing Scenarios', fontsize=16, y=1.03)
plt.tight_layout(rect=[0, 0, 1, 0.98])
cases_save_path_loewe = f"{save_path}/cases_summary_loewe_{experiment_name}.png"
plt.savefig(cases_save_path_loewe, dpi=600, bbox_inches='tight')
print(f"\nSpecific cases summary for Loewe saved to: {cases_save_path_loewe}")
plt.show()

# --- NEW: Generate Combined Synergy vs. Efficacy Scatter Plot ---

# The data for the current experiment is already in `combined_metrics_df`.
# We will use that directly instead of reprocessing hardcoded experiments.
combined_df = combined_metrics_df.copy()
combined_df['exp_name'] = experiment_name

# NORMALIZE ALIVE CELLS to percentage of control
# This is already done in the 'percent_alive' column.
combined_df['alive_cells_percent'] = combined_df['percent_alive']

# 3. Categorize the data
def get_category(row):
    if row['x_diff'] > row['y_diff']:
        diff_cat = "Fast X, Slow Y"
    elif row['x_diff'] < row['y_diff']:
        diff_cat = "Slow X, Fast Y"
    else:
        diff_cat = "Symmetric D"
        
    if row['delta_time'] > 0:
        time_cat = "Y First"
    elif row['delta_time'] < 0:
        time_cat = "X First"
    else:
        time_cat = "Simultaneous"
    return f"{diff_cat} | {time_cat}"

combined_df['category'] = combined_df.apply(get_category, axis=1)

# 4. Create the scatter plot for Bliss Score
plt.figure(figsize=(14, 10), dpi=300)

categories = combined_df['category'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
markers = ['o', 's', '^', 'v', '>', '<', 'p', '*', 'X', 'D']

for i, category in enumerate(sorted(categories)):
    subset = combined_df[combined_df['category'] == category]
    plt.scatter(subset['bliss_score'], subset['alive_cells_percent'],
                label=category,
                alpha=0.7,
                s=50,
                marker=markers[i % len(markers)])

plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.axhline(y=100, color='red', linestyle='--', linewidth=2, label=f'No-Drug Control (100%)')
plt.xlabel('Bliss Score (Negative = Synergy)', fontsize=14, weight='bold')
plt.ylabel('% Alive Cells (of Control)', fontsize=14, weight='bold')
plt.title(f'Synergy vs. Efficacy for {experiment_name}', fontsize=18, weight='bold')
plt.legend(title='Condition Category', bbox_to_anchor=(1.04, 1), loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout(rect=[0, 0, 0.85, 1])

scatter_save_path = f"{save_path}/synergy_vs_efficacy_scatter_{experiment_name}_bliss.png"
plt.savefig(scatter_save_path, dpi=600)
print(f"\nCombined Bliss scatter plot saved to: {scatter_save_path}")
plt.show()

# 5. Create the scatter plot for Loewe Score
plt.figure(figsize=(14, 10), dpi=300)

for i, category in enumerate(sorted(categories)):
    subset = combined_df[combined_df['category'] == category]
    plt.scatter(subset['loewe_CI'], subset['alive_cells_percent'],
                label=category,
                alpha=0.7,
                s=50,
                marker=markers[i % len(markers)])

plt.axvline(1, color='gray', linestyle='--', linewidth=1) # Additivity line at CI=1
plt.axhline(y=100, color='red', linestyle='--', linewidth=2, label=f'No-Drug Control (100%)')
plt.xlabel('Loewe Score (CI, < 1 is Synergy)', fontsize=14, weight='bold')
plt.ylabel('% Alive Cells (of Control)', fontsize=14, weight='bold')
plt.title(f'Loewe Synergy vs. Efficacy for {experiment_name}', fontsize=18, weight='bold')
plt.legend(title='Condition Category', bbox_to_anchor=(1.04, 1), loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout(rect=[0, 0, 0.85, 1])

scatter_save_path_loewe = f"{save_path}/synergy_vs_efficacy_scatter_{experiment_name}_loewe.png"
plt.savefig(scatter_save_path_loewe, dpi=600)
print(f"\nCombined Loewe scatter plot saved to: {scatter_save_path_loewe}")
plt.show()