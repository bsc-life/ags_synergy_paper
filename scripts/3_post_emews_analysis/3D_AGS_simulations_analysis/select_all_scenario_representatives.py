import pandas as pd
import os
import logging
import numpy as np

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_control_and_single_drug_data(experiment_name, sweep_summaries_path):
    """Loads negative control and single-drug data needed for metric calculations."""
    
    # Load negative control data (no drug)
    neg_control_name = "synergy_sweep-3D-0205-1608-control_nodrug"
    neg_control_df = pd.read_csv(os.path.join(sweep_summaries_path, f"final_summary_{neg_control_name}.csv"))
    control_mean = neg_control_df.iloc[:, -1].mean()

    # Determine single drug experiment names
    if "pi3k_mek" in experiment_name.lower():
        drug1_name = "synergy_sweep-pi3k_mek-3D-0505-0218-logscale_singledrug_pi3k"
        drug2_name = "synergy_sweep-pi3k_mek-3D-0505-0218-logscale_singledrug_mek"
    elif "akt_mek" in experiment_name.lower():
        drug1_name = "synergy_sweep-akt_mek-3D-0505-1910-logscale_singledrug_akt"
        drug2_name = "synergy_sweep-akt_mek-3D-0505-1910-logscale_singledrug_mek"
    else:
        raise ValueError(f"Cannot determine single drug experiments for {experiment_name}")

    # Load single drug data and create effect dictionaries
    drug1_df = pd.read_csv(os.path.join(sweep_summaries_path, f"final_summary_{drug1_name}.csv"))
    drug1_dict = drug1_df.groupby(drug1_df.columns[0]).agg({drug1_df.columns[-1]: 'mean'})[drug1_df.columns[-1]].to_dict()

    drug2_df = pd.read_csv(os.path.join(sweep_summaries_path, f"final_summary_{drug2_name}.csv"))
    drug2_dict = drug2_df.groupby(drug2_df.columns[0]).agg({drug2_df.columns[-1]: 'mean'})[drug2_df.columns[-1]].to_dict()

    return control_mean, drug1_dict, drug2_dict

def select_representatives(experiment_name):
    """
    For a given experiment, iterates through every diffusion coefficient scenario,
    selects a set of representative points for each, and saves them to a CSV file.
    """
    logging.info(f"Processing experiment: {experiment_name}")

    # --- 1. Define Paths and Load Data ---
    sweep_summaries_path = "results/sweep_summaries"
    output_dir = f"scripts/post_emews_analysis/synergy_recovery_experiments/representative_simulations/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    summary_path = os.path.join(sweep_summaries_path, f"final_summary_{experiment_name}.csv")
    if not os.path.exists(summary_path):
        logging.error(f"Summary file not found: {summary_path}")
        return

    main_df = pd.read_csv(summary_path)

    # --- 2. Get Control Data and Calculate Metrics ---
    try:
        control_mean, drug1_dict, drug2_dict = get_control_and_single_drug_data(experiment_name, sweep_summaries_path)
    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Could not load required control/single-drug data for {experiment_name}. Error: {e}")
        return

    # Rename columns for consistency
    main_df = main_df.rename(columns={
        'user_parameters.drug_X_diffusion_coefficient': 'x_diff',
        'user_parameters.drug_Y_diffusion_coefficient': 'y_diff',
        'user_parameters.drug_X_pulse_period': 'x_pulse',
        'user_parameters.drug_Y_pulse_period': 'y_pulse',
        'FINAL_NUMBER_OF_ALIVE_CELLS': 'alive_cells'
    })

    # Calculate metrics for each run
    main_df['delta_time'] = main_df['x_pulse'] - main_df['y_pulse']
    main_df['percent_alive'] = (main_df['alive_cells'] / control_mean) * 100
    
    # Calculate Bliss score
    e_a = main_df['x_diff'].map(drug1_dict) / control_mean
    e_b = main_df['y_diff'].map(drug2_dict) / control_mean
    e_ab = main_df['alive_cells'] / control_mean
    main_df['bliss_score'] = e_ab - (e_a * e_b)

    # --- 3. Loop Through Scenarios and Select Representatives ---
    scenarios = main_df[['x_diff', 'y_diff']].drop_duplicates()
    logging.info(f"Found {len(scenarios)} unique diffusion scenarios to process.")

    for _, row in scenarios.iterrows():
        x_diff, y_diff = row['x_diff'], row['y_diff']
        scenario_df = main_df[(main_df['x_diff'] == x_diff) & (main_df['y_diff'] == y_diff)].copy()

        if scenario_df.empty:
            continue

        # --- NEW LOGIC: Select representatives based on MEAN performance per timing ---
        # 1. Aggregate data to find the mean performance for each unique delta_time
        agg_df = scenario_df.groupby('delta_time').agg(
            mean_percent_alive=('percent_alive', 'mean'),
            mean_bliss_score=('bliss_score', 'mean')
        ).reset_index()

        representatives = {}

        # 2. Select Max Efficacy based on the minimum of the *mean* percent_alive
        best_efficacy_timing = agg_df.loc[agg_df['mean_percent_alive'].idxmin()]
        # Select the first run from the original df that matches this best timing
        representatives['Max Efficacy'] = scenario_df[scenario_df['delta_time'] == best_efficacy_timing['delta_time']].iloc[0]

        # 3. Select Min Efficacy, excluding t=0, based on the maximum of the *mean* percent_alive
        non_simultaneous_agg_df = agg_df[agg_df['delta_time'] != 0]
        if not non_simultaneous_agg_df.empty:
            worst_efficacy_timing = non_simultaneous_agg_df.loc[non_simultaneous_agg_df['mean_percent_alive'].idxmax()]
            representatives['Min Efficacy'] = scenario_df[scenario_df['delta_time'] == worst_efficacy_timing['delta_time']].iloc[0]
            logging.info(f"For D(X)={x_diff}, D(Y)={y_diff}, Min Efficacy (avg, excl t=0) chosen from delta_time={worst_efficacy_timing['delta_time']}.")
        else: # Fallback if only t=0 runs exist
            worst_efficacy_timing = agg_df.loc[agg_df['mean_percent_alive'].idxmax()]
            representatives['Min Efficacy'] = scenario_df[scenario_df['delta_time'] == worst_efficacy_timing['delta_time']].iloc[0]
            logging.warning(f"For D(X)={x_diff}, D(Y)={y_diff}, only t=0 runs found. Min Efficacy chosen from this group.")

        # 4. Select Best and Worst Synergy based on the *mean* bliss_score
        best_synergy_timing = agg_df.loc[agg_df['mean_bliss_score'].idxmin()]
        representatives['Best Synergy (Bliss)'] = scenario_df[scenario_df['delta_time'] == best_synergy_timing['delta_time']].iloc[0]

        worst_synergy_timing = agg_df.loc[agg_df['mean_bliss_score'].idxmax()]
        representatives['Worst Synergy (Bliss)'] = scenario_df[scenario_df['delta_time'] == worst_synergy_timing['delta_time']].iloc[0]

        # 5. Select Simultaneous case (if it exists) based on the single best-performing run
        simultaneous_df = scenario_df[scenario_df['delta_time'] == 0]
        if not simultaneous_df.empty:
            # If multiple simultaneous runs, pick the one with best efficacy
            representatives['Simultaneous'] = simultaneous_df.loc[simultaneous_df['percent_alive'].idxmin()]
        # --- END NEW LOGIC ---

        # --- 4. Assemble and Save Results ---
        result_df = pd.DataFrame(representatives.values(), index=representatives.keys())
        result_df = result_df.reset_index().rename(columns={'index': 'selection_reason'})
        
        # Drop duplicates in case one run fits multiple criteria
        result_df = result_df.drop_duplicates(subset=[col for col in result_df.columns if col != 'selection_reason'])

        output_path = os.path.join(output_dir, f"representatives_X_{x_diff}_Y_{y_diff}.csv")
        result_df.to_csv(output_path, index=False)
        logging.info(f"Saved representative simulations for D(X)={x_diff}, D(Y)={y_diff} to {output_path}")

if __name__ == '__main__':
    # List of experiments to process
    experiments_to_run = [
        "synergy_sweep-pi3k_mek-1606-0214-4p_3D_drugtiming",
        "synergy_sweep-akt_mek-1606-0214-4p_3D_drugtiming"
    ]
    
    for exp_name in experiments_to_run:
        select_representatives(exp_name) 