import pandas as pd
import json
import os

def generate_sweep_json(synergy_exp_name, output_json_path, top_n="1p"):
    """
    Generates a JSON file for a uniform sweep based on the min/max values
    from the top parameter sets of a synergy experiment.

    Args:
        synergy_exp_name (str): Name of the synergy experiment.
        output_json_path (str): Path to save the generated JSON file.
        top_n (str): The number/percentage of top parameter sets to use (e.g., "1p", "100").
    """
    print(f"--- Generating sweep JSON for: {synergy_exp_name} using top {top_n} ---")

    # --- 1. Load Data ---
    try:
        source_csv = f'results/sweep_summaries/final_summary_{synergy_exp_name}/top_{top_n}.csv'
        top_df = pd.read_csv(source_csv)
    except FileNotFoundError as e:
        print(f"Error: Could not find required summary file: {source_csv}. {e}")
        return

    # Exclude the last column which is usually the score/fitness
    params_df = top_df.iloc[:, :-1]

    # Also exclude identifier columns that are not model parameters
    identifier_cols_to_drop = ['individual', 'replicate']
    existing_identifiers = [col for col in identifier_cols_to_drop if col in params_df.columns]
    if existing_identifiers:
        params_df = params_df.drop(columns=existing_identifiers)
        print(f"Info: Excluding identifier columns found in source data: {existing_identifiers}")

    # --- 2. Calculate Min/Max/Mean/Std for each parameter ---
    min_values = params_df.min()
    max_values = params_df.max()
    mean_values = params_df.mean()
    std_values = params_df.std()

    sweep_parameters = {}
    for param in params_df.columns:
        # Correct the parameter name to match the simulation's expected format
        if param.startswith("user_parameters."):
            corrected_param_name = param
        else:
            corrected_param_name = f"cell_definitions.cell_definition.custom_data.{param}"

        sweep_parameters[corrected_param_name] = {
            "min": float(min_values[param]),
            "max": float(max_values[param]),
            "loc": float(mean_values[param]),
            "scale": float(std_values[param])
        }

    # --- 3. Add the fixed sweep parameters ---
    # These are taken from the example sweep_4p_3D_drugaddition_drugtiming.json
    fixed_sweep_params = {
        "user_parameters.drug_X_diffusion_coefficient": {
            "min": 6.0,
            "max": 6000.0,
            "loc": 1.0,
            "scale": 50.0
        },
        "user_parameters.drug_Y_diffusion_coefficient": {
            "min": 6.0,
            "max": 6000.0,
            "loc": 1.0,
            "scale": 50.0
        },
        "user_parameters.drug_X_pulse_period": {
            "min": 4.0,
            "max": 4000.0,
            "loc": 1280.0,
            "scale": 1280.0
        },
        "user_parameters.drug_Y_pulse_period": {
            "min": 4.0,
            "max": 4000.0,
            "loc": 1280.0,
            "scale": 1280.0
        }
    }
    sweep_parameters.update(fixed_sweep_params)

    # --- 4. Write New JSON File ---
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(sweep_parameters, f, indent=4)
    
    print(f"Successfully wrote sweep JSON to: {output_json_path}\n")


if __name__ == "__main__":
    top_n_setting = "1p"
    # --- Configuration for AKT + MEK ---
    aktmek_synergy_name = "synergy_sweep-akt_mek-1204-1639-18p_transient_delayed_uniform_postdrug_RMSE_5k"
    aktmek_output_json = f"data/JSON/sweep/sweep_consensus_akt_mek_top{top_n_setting}.json"
    
    generate_sweep_json(
        synergy_exp_name=aktmek_synergy_name,
        output_json_path=aktmek_output_json,
        top_n=top_n_setting
    )

    # --- Configuration for PI3K + MEK ---
    pi3kmek_synergy_name = "synergy_sweep-pi3k_mek-1104-2212-18p_transient_delayed_uniform_5k_10p"
    pi3kmek_output_json = f"data/JSON/sweep/sweep_consensus_pi3k_mek_top{top_n_setting}.json"

    generate_sweep_json(
        synergy_exp_name=pi3kmek_synergy_name,
        output_json_path=pi3kmek_output_json,
        top_n=top_n_setting
    ) 