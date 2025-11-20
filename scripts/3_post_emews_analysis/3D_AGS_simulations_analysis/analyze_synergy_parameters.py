import pandas as pd
import numpy as np
import os
import itertools

def find_top_synergistic_parameters(experiment_name, top_n=10):
    """
    Analyzes a synergy experiment to find the best-performing parameter sets.

    This version uses "proxy" conditions within the same experiment file for
    control and single-drug scenarios to ensure a consistent comparison.

    1. Defines proxies for Control, Drug A, and Drug B from the main data.
    2. Identifies "good" combination conditions where combo < single drugs < control.
    3. Pools all parameter sets from these "good" conditions.
    4. For each parameter set, it finds its specific outcome under combo, single-drug,
       and control proxy conditions.
    5. Ranks the fully quantified sets and reports the top N performers.
    """
    print(f"--- Analyzing Synergistic Efficacy for: {experiment_name} (using internal proxies) ---")

    # --- Configuration for Proxy Conditions ---
    SWEEP_SUMMARIES_PATH = "results/sweep_summaries/"
    HIGH_DOSE = 600.0
    LOW_DOSE = 6.0
    ABSENT_DOSE_EQUIVALENT = 6.0
    LATE_PULSE_THRESHOLD = 3000
    LATE_ADDITION_TIME = 4000

    # --- Data Loading ---
    try:
        df = pd.read_csv(f'{SWEEP_SUMMARIES_PATH}/final_summary_{experiment_name}.csv')
    except FileNotFoundError:
        print(f"ERROR: Could not find summary file for '{experiment_name}'. Skipping.")
        return

    # --- Data Preparation ---
    df['delta_time'] = df['user_parameters.drug_X_pulse_period'] - df['user_parameters.drug_Y_pulse_period']
    # Rename columns to match proxy logic from validation script
    df = df.rename(columns={
        'user_parameters.drug_X_diffusion_coefficient': 'x_diff',
        'user_parameters.drug_Y_diffusion_coefficient': 'y_diff'
    })

    # --- Define and Extract Proxy Data ---
    print("\nExtracting data for proxy conditions from within the experiment file...")
    # Control is defined as low diffusion (6.0) and very late drug addition (>=4000 min)
    control_proxy_condition = (
        (df['x_diff'] == LOW_DOSE) &
        (df['y_diff'] == LOW_DOSE) &
        (df['user_parameters.drug_X_pulse_period'] >= LATE_ADDITION_TIME) &
        (df['user_parameters.drug_Y_pulse_period'] >= LATE_ADDITION_TIME)
    )
    control_proxy_df = df[control_proxy_condition].copy()

    drug_a_proxy_condition = (
        (df['x_diff'] == ABSENT_DOSE_EQUIVALENT) & (df['y_diff'] == LOW_DOSE) &
        (df['delta_time'] < -LATE_PULSE_THRESHOLD)
    )
    drug_a_proxy_df = df[drug_a_proxy_condition].copy()

    drug_b_proxy_condition = (
        (df['x_diff'] == LOW_DOSE) & (df['y_diff'] == ABSENT_DOSE_EQUIVALENT) &
        (df['delta_time'] > LATE_PULSE_THRESHOLD)
    )
    drug_b_proxy_df = df[drug_b_proxy_condition].copy()

    if control_proxy_df.empty or drug_a_proxy_df.empty or drug_b_proxy_df.empty:
        print("ERROR: Could not find data for one or more proxy conditions. Cannot proceed.")
        return
    
    mean_control_alive = control_proxy_df['FINAL_NUMBER_OF_ALIVE_CELLS'].mean()
    mean_drug_a_alive = drug_a_proxy_df['FINAL_NUMBER_OF_ALIVE_CELLS'].mean()
    mean_drug_b_alive = drug_b_proxy_df['FINAL_NUMBER_OF_ALIVE_CELLS'].mean()
    print(f"Proxy Means Found -> Control: {mean_control_alive:.2f}, Drug A: {mean_drug_a_alive:.2f}, Drug B: {mean_drug_b_alive:.2f}")

    # Also find the best-case (minimum) for each proxy to allow for best-vs-best comparison
    min_control_alive = control_proxy_df['FINAL_NUMBER_OF_ALIVE_CELLS'].min()
    min_drug_a_alive = drug_a_proxy_df['FINAL_NUMBER_OF_ALIVE_CELLS'].min()
    min_drug_b_alive = drug_b_proxy_df['FINAL_NUMBER_OF_ALIVE_CELLS'].min()
    print(f"Proxy Bests Found -> Control: {min_control_alive:.2f}, Drug A: {min_drug_a_alive:.2f}, Drug B: {min_drug_b_alive:.2f}")

    # --- Filter for Synergistically Effective Combo Conditions (More Rigorous) ---
    print("\nFiltering for combination conditions that are truly synergistic...")
    
    # 1. Determine the benchmark for an effective single drug, which MUST be better than the control.
    best_single_drug_mean = min(mean_drug_a_alive, mean_drug_b_alive)
    if best_single_drug_mean >= mean_control_alive:
        print(f"VALIDATION FAILED: The best single drug treatment ({best_single_drug_mean:.2f}) was not more effective than the control ({mean_control_alive:.2f}).")
        print("No meaningful synergy can be determined. Skipping this experiment.")
        return

    print(f"Effective Single-Drug Benchmark Found: {best_single_drug_mean:.2f} (better than control's {mean_control_alive:.2f})")
    
    # 2. Find combo conditions that are better than this effective benchmark by aggregating delta_times.
    print("\nAggregating by positive/negative delta_time to find synergistic conditions...")
    good_parameter_sets = []
    combo_coeffs = [d for d in df['x_diff'].unique() if d not in [LOW_DOSE, ABSENT_DOSE_EQUIVALENT]]

    # Aggregate by positive/negative delta_time
    df['delta_time_sign'] = 'zero' # Default for delta_time == 0
    df.loc[df['delta_time'] > 0, 'delta_time_sign'] = 'positive'
    df.loc[df['delta_time'] < 0, 'delta_time_sign'] = 'negative'

    # We only care about aggregated positive/negative delta_times, not individual or zero ones
    signed_df = df[df['delta_time_sign'] != 'zero'].copy()

    grouped = signed_df.groupby(['x_diff', 'y_diff', 'delta_time_sign'])

    for (dx, dy, sign), group_df in grouped:
        # We only care about actual combination drug coefficients
        if dx not in combo_coeffs or dy not in combo_coeffs:
            continue

        mean_ab = group_df['FINAL_NUMBER_OF_ALIVE_CELLS'].mean()

        # Check if the aggregated condition is better than the single drug benchmark
        if mean_ab < best_single_drug_mean:
            print(f"  Synergistic condition found: x_diff={dx}, y_diff={dy}, delta_time_sign='{sign}' (Mean alive: {mean_ab:.2f})")
            good_parameter_sets.append(group_df)

    if not good_parameter_sets:
        print("ERROR: No combination conditions matched the synergistic efficacy criteria.")
        return

    all_good_params_df = pd.concat(good_parameter_sets, ignore_index=True)
    print(f"Found {len(all_good_params_df)} individual parameter sets from {len(good_parameter_sets)} effective combo conditions.")

    # --- Consolidate and Quantify Outcomes ---
    # Instead of merging, we will now add the mean proxy outcomes as new columns.
    # This provides a robust comparison even if exact parameter matches don't exist
    # for control/single-drug runs.
    print("\nQuantifying efficacy by comparing against mean proxy outcomes...")

    # Start with all the good parameter sets found
    results_df = all_good_params_df.copy()

    # Rename the combo outcome column for clarity
    results_df = results_df.rename(columns={'FINAL_NUMBER_OF_ALIVE_CELLS': 'outcome_actual_combo'})

    # Add the mean values from the proxy conditions as new columns
    results_df['outcome_mean_control'] = mean_control_alive
    results_df['outcome_mean_drug_A'] = mean_drug_a_alive
    results_df['outcome_mean_drug_B'] = mean_drug_b_alive

    # Add the best-case (min) values as well for a more fair comparison in plots
    results_df['outcome_best_control'] = min_control_alive
    results_df['outcome_best_drug_A'] = min_drug_a_alive
    results_df['outcome_best_drug_B'] = min_drug_b_alive
        
    # --- Identify and Report Top Performing Parameter Sets ---
    top_params_df = results_df.sort_values(by='outcome_actual_combo', ascending=True).head(top_n)

    outcome_cols = [
        'outcome_mean_control', 'outcome_best_control',
        'outcome_mean_drug_A', 'outcome_best_drug_A',
        'outcome_mean_drug_B', 'outcome_best_drug_B',
        'outcome_actual_combo'
    ]
    param_cols = [c for c in top_params_df.columns if c.startswith('user_parameters.')]
    display_cols = outcome_cols + param_cols
    top_params_df = top_params_df[display_cols]

    print(f"\n--- Top {top_n} Overall Performing Synergistic Parameter Sets ---")
    print(top_params_df.to_string(float_format="%.2f"))
    
    save_path = "scripts/post_emews_analysis/synergy_recovery_experiments/synergy_analysis_results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    output_filename = f"{save_path}/top_synergistic_params_quantified_{experiment_name}.csv"
    top_params_df.to_csv(output_filename, index=False, float_format="%.2f")
    print(f"\nTop parameter sets saved to: {output_filename}")
    print("----------------------------------------------------------\n")

def main():
    """
    Main function to run the analysis for all specified experiments.
    """
    experiment_names = [
        "synergy_sweep-akt_mek-2606-1819-4p_3D_drugtiming_synonly_consensus_hybrid_20",
        "synergy_sweep-pi3k_mek-2606-1819-4p_3D_drugtiming_synonly_consensus_hybrid_20"
    ]

    for exp_name in experiment_names:
        find_top_synergistic_parameters(exp_name)

if __name__ == "__main__":
    main() 