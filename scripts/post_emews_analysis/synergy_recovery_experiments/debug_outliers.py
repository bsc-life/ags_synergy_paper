import pandas as pd
import numpy as np
import os

def find_and_print_outliers(experiment_name, target_x_diff, target_y_diff, target_delta_time=0, outlier_threshold=2000):
    """
    Identifies and prints the full parameters for outlier and non-outlier runs
    for a specific experimental scenario.
    """
    print(f"\n--- Investigating: {experiment_name} ---")
    print(f"Scenario: D(X)={target_x_diff}, D(Y)={target_y_diff}, Delta Time={target_delta_time}")
    
    summary_path = f"results/sweep_summaries/final_summary_{experiment_name}.csv"
    if not os.path.exists(summary_path):
        print(f"ERROR: Summary file not found at {summary_path}")
        return

    df = pd.read_csv(summary_path)

    # Calculate delta_time
    df['delta_time'] = df['user_parameters.drug_X_pulse_period'] - df['user_parameters.drug_Y_pulse_period']

    # --- FIX: Exclude "late simultaneous" runs that act as controls ---
    late_addition_threshold = 1000
    late_control_runs = (df['delta_time'] == 0) & (df['user_parameters.drug_X_pulse_period'] > late_addition_threshold)
    df = df[~late_control_runs]
    # --------------------------------------------------------------------

    # Filter for the specific scenario
    scenario_df = df[
        (df['user_parameters.drug_X_diffusion_coefficient'] == target_x_diff) &
        (df['user_parameters.drug_Y_diffusion_coefficient'] == target_y_diff) &
        (df['delta_time'] == target_delta_time)
    ].copy()

    if scenario_df.empty:
        print("No data found for this specific scenario.")
        return

    # Identify outliers based on a simple threshold
    outliers = scenario_df[scenario_df['FINAL_NUMBER_OF_ALIVE_CELLS'] > outlier_threshold]
    non_outliers = scenario_df[scenario_df['FINAL_NUMBER_OF_ALIVE_CELLS'] <= outlier_threshold]

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)

    print("\n" + "="*50)
    print("         >>> OUTLIER RUNS <<<")
    print("="*50)
    if not outliers.empty:
        print(outliers)
    else:
        print("No outliers found.")

    print("\n" + "="*50)
    print("       >>> NON-OUTLIER RUNS <<<")
    print("="*50)
    if not non_outliers.empty:
        print(non_outliers)
    else:
        print("No non-outliers found.")

if __name__ == "__main__":
    experiments_to_run = {
        "synergy_sweep-pi3k_mek-1606-0214-4p_3D_drugtiming": "PI3Ki + MEKi",
        "synergy_sweep-akt_mek-1606-0214-4p_3D_drugtiming": "AKTi + MEKi"
    }

    # Set the scenario to investigate
    TARGET_X_DIFF = 600.0
    TARGET_Y_DIFF = 600.0

    for exp_name in experiments_to_run.keys():
        find_and_print_outliers(exp_name, TARGET_X_DIFF, TARGET_Y_DIFF) 