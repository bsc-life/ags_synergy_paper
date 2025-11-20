import pandas as pd
import os
import numpy as np
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def categorize_scenario(row):
    """
    Categorizes a simulation run into one of four scenarios based on dosage and pulse periods.
    This version is adapted for experiments that vary pulse periods instead of delta_time.
    """
    # Use a tolerance for floating-point comparisons
    is_symmetric_diff = abs(row['x_diff'] - row['y_diff']) < 1e-9
    is_symmetric_pulse = abs(row['x_pulse'] - row['y_pulse']) < 1e-9
    
    is_low_dose = row['x_diff'] <= 60 and row['y_diff'] <= 60
    is_high_dose = row['x_diff'] >= 600 and row['y_diff'] >= 600

    if is_symmetric_diff and is_symmetric_pulse:
        if is_low_dose:
            return 'Symmetric Low Dose'
        if is_high_dose:
            return 'Symmetric High Dose'
        return 'Other'

    # Asymmetric cases are now defined by differences in pulse period
    is_fast_drug_X = row['x_pulse'] < row['y_pulse']  # Shorter pulse period is "faster"

    if is_fast_drug_X:
        return 'Asymmetric: Fast Drug Dominant'
    else:
        return 'Asymmetric: Slow Drug Dominant'

def select_representative_simulations(experiment_name):
    """
    Selects four representative simulations from a large dataset based on
    predefined scenarios and performance metrics.
    """
    logging.info(f"Starting selection for experiment: {experiment_name}")

    # --- 1. Define File Paths ---
    # Path to the metrics file we generated previously
    metrics_path = f"scripts/post_emews_analysis/synergy_recovery_experiments/optimal_timings_synergy/combined_synergy_metrics_{experiment_name}.csv"
    
    # Path to the original EMEWS summary file which contains instance IDs
    strategy = "sweep" # Based on the experiment name convention
    summary_path = f"results/{strategy}_summaries/final_summary_{experiment_name}.csv"

    # --- 2. Load and Prepare Data ---
    if not os.path.exists(metrics_path):
        logging.error(f"Metrics file not found: {metrics_path}")
        return
    if not os.path.exists(summary_path):
        logging.error(f"Summary file not found: {summary_path}")
        return
        
    logging.info("Loading and merging data files...")
    metrics_df = pd.read_csv(metrics_path)
    summary_df = pd.read_csv(summary_path)

    # Rename summary columns to match metrics_df for merging
    # This experiment uses pulse periods, not delta_time
    summary_df.rename(columns={
        'user_parameters.drug_X_diffusion_coefficient': 'x_diff',
        'user_parameters.drug_Y_diffusion_coefficient': 'y_diff',
        'user_parameters.drug_X_pulse_period': 'x_pulse',
        'user_parameters.drug_Y_pulse_period': 'y_pulse'
    }, inplace=True)

    # Merge dataframes to link metrics with instance identifiers
    # We need to handle potential floating point precision issues in the merge keys
    # Note: The merge keys are updated to reflect the available columns
    merged_df = pd.merge(
        summary_df,
        metrics_df,
        on=['x_diff', 'y_diff'], # Merging only on diffusion coeffs
        how='inner'
    )
    
    # --- 3. Categorize Scenarios ---
    logging.info("Categorizing scenarios for each simulation...")
    merged_df['scenario'] = merged_df.apply(categorize_scenario, axis=1)

    # --- 4. Select Best Representative for Each Scenario ---
    logging.info("Selecting the best representative for each scenario...")
    representatives = []
    
    # Scenario 1: Symmetric Low Dose (Goal: Best Synergy -> min Bliss score)
    sym_low_df = merged_df[merged_df['scenario'] == 'Symmetric Low Dose']
    if not sym_low_df.empty:
        best_sym_low = sym_low_df.loc[sym_low_df['bliss_score'].idxmin()]
        representatives.append(best_sym_low)
        logging.info(f"Selected 'Symmetric Low Dose': index {best_sym_low.name}")

    # Scenario 2: Symmetric High Dose (Goal: Best Efficacy -> min % alive)
    sym_high_df = merged_df[merged_df['scenario'] == 'Symmetric High Dose']
    if not sym_high_df.empty:
        best_sym_high = sym_high_df.loc[sym_high_df['percent_alive'].idxmin()]
        representatives.append(best_sym_high)
        logging.info(f"Selected 'Symmetric High Dose': index {best_sym_high.name}")

    # Scenario 3: Asymmetric, Fast Drug First (Goal: Best Efficacy -> min % alive)
    asym_fast_df = merged_df[merged_df['scenario'] == 'Asymmetric: Fast Drug Dominant']
    if not asym_fast_df.empty:
        best_asym_fast = asym_fast_df.loc[asym_fast_df['percent_alive'].idxmin()]
        representatives.append(best_asym_fast)
        logging.info(f"Selected 'Asymmetric: Fast Drug Dominant': index {best_asym_fast.name}")

    # Scenario 4: Asymmetric, Slow Drug First (Goal: Best Synergy -> min Bliss score)
    asym_slow_df = merged_df[merged_df['scenario'] == 'Asymmetric: Slow Drug Dominant']
    if not asym_slow_df.empty:
        best_asym_slow = asym_slow_df.loc[asym_slow_df['bliss_score'].idxmin()]
        representatives.append(best_asym_slow)
        logging.info(f"Selected 'Asymmetric: Slow Drug Dominant': index {best_asym_slow.name}")

    if not representatives:
        logging.warning("Could not select any representative simulations.")
        return

    # --- 5. Save the Subset ---
    final_df = pd.DataFrame(representatives)
    
    output_dir = "scripts/post_emews_analysis/synergy_recovery_experiments/optimal_timings_synergy/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"representative_simulations_{experiment_name}.csv")
    
    final_df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved 4 representative simulations to {output_path}")


if __name__ == '__main__':
    # The name of the experiment we are analyzing
    experiment_name_pi3k_mek = "synergy_sweep-pi3k_mek-0505-1008-4p_3D_drugtiming"
    
    # Run the selection process
    select_representative_simulations(experiment_name_pi3k_mek)
