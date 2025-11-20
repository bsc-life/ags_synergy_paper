# this script will take the top 1% of parameters of the synergy experiments, and 
# generate a consensus parameter set that can be used for the synergy recovery experiments

import pandas as pd
import xml.etree.ElementTree as ET
import os

def generate_consensus_xml(synergy_exp_name, drug_x_exp_name, drug_y_exp_name, template_xml_path, output_xml_path, drug_x_identifiers, drug_y_identifiers, top_n="1p"):
    """
    Generates a consensus XML configuration by combining parameters from synergy and single-drug experiments.

    Args:
        synergy_exp_name (str): Name of the synergy experiment.
        drug_x_exp_name (str): Name of the single-drug experiment for Drug X.
        drug_y_exp_name (str): Name of the single-drug experiment for Drug Y.
        template_xml_path (str): Path to the template settings XML file.
        output_xml_path (str): Path to save the generated XML file.
        drug_x_identifiers (list): List of strings to identify Drug X parameters (e.g., ['drug_x', 'akt']).
        drug_y_identifiers (list): List of strings to identify Drug Y parameters (e.g., ['drug_y', 'mek']).
        top_n (str): The number/percentage of top parameter sets to average (e.g., "1p", "100").
    """
    print(f"--- Generating consensus for: {synergy_exp_name} using top {top_n} ---")

    # --- 1. Load and Average Data ---
    try:
        synergy_df = pd.read_csv(f'results/sweep_summaries/final_summary_{synergy_exp_name}/top_{top_n}.csv')
        drug_x_df = pd.read_csv(f'results/sweep_summaries/final_summary_{drug_x_exp_name}/top_{top_n}.csv')
        drug_y_df = pd.read_csv(f'results/sweep_summaries/final_summary_{drug_y_exp_name}/top_{top_n}.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find required summary file. {e}")
        return

    synergy_avg = synergy_df.iloc[:, :-1].mean()
    drug_x_avg = drug_x_df.iloc[:, :-1].mean()
    drug_y_avg = drug_y_df.iloc[:, :-1].mean()

    # --- 2. Build Consensus Parameter Set ---
    consensus_parameters = {}
    for param in synergy_avg.index:
        # Check if it's a Drug X parameter
        if any(identifier in param.lower() for identifier in drug_x_identifiers):
            if param in drug_x_avg:
                consensus_parameters[param] = drug_x_avg[param]
            else:
                consensus_parameters[param] = synergy_avg[param] # Fallback
        # Check if it's a Drug Y parameter
        elif any(identifier in param.lower() for identifier in drug_y_identifiers):
            # In the single-drug run, Drug Y was configured as Drug X
            param_x_equivalent = param.replace("drug_Y", "drug_X")
            if param_x_equivalent in drug_y_avg:
                consensus_parameters[param] = drug_y_avg[param_x_equivalent]
            else:
                consensus_parameters[param] = synergy_avg[param] # Fallback
        # It's a general parameter
        else:
            consensus_parameters[param] = synergy_avg[param]

    # --- 3. Update XML Template ---
    try:
        tree = ET.parse(template_xml_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: Template XML not found at {template_xml_path}")
        return

    user_params = root.find('.//user_parameters')
    custom_data = root.find('.//cell_definition[@name="default"]/custom_data')

    for param, value in consensus_parameters.items():
        updated = False
        # Try updating in <user_parameters>
        if user_params is not None and user_params.find(param) is not None:
            user_params.find(param).text = str(float(value))
            updated = True
        # If not there, try <custom_data>
        elif custom_data is not None and custom_data.find(param) is not None:
            custom_data.find(param).text = str(float(value))
            updated = True

    # --- 4. Write New XML File ---
    os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
    tree.write(output_xml_path)
    print(f"Successfully wrote consensus XML to: {output_xml_path}\n")


def generate_synergy_only_xml(synergy_exp_name, template_xml_path, output_xml_path, top_n="1p"):
    """
    Generates a consensus XML by averaging the top parameter sets from a single synergy experiment.
    Args:
        synergy_exp_name (str): Name of the synergy experiment.
        template_xml_path (str): Path to the template settings XML file.
        output_xml_path (str): Path to save the generated XML file.
        top_n (str): The number/percentage of top parameter sets to average (e.g., "1p", "100").
    """
    print(f"--- Generating synergy-only consensus for: {synergy_exp_name} using top {top_n} ---")

    # --- 1. Load and Average Data ---
    try:
        synergy_df = pd.read_csv(f'results/sweep_summaries/final_summary_{synergy_exp_name}/top_{top_n}.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find required summary file. {e}")
        return

    synergy_only_avg = synergy_df.iloc[:, :-1].mean()

    # --- 2. Update XML Template ---
    try:
        tree = ET.parse(template_xml_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: Template XML not found at {template_xml_path}")
        return

    user_params = root.find('.//user_parameters')
    custom_data = root.find('.//cell_definition[@name="default"]/custom_data')

    for param, value in synergy_only_avg.items():
        updated = False
        # Try updating in <user_parameters>
        if user_params is not None and user_params.find(param) is not None:
            user_params.find(param).text = str(float(value))
            updated = True
        # If not there, try <custom_data>
        elif custom_data is not None and custom_data.find(param) is not None:
            custom_data.find(param).text = str(float(value))
            updated = True

    # --- 3. Write New XML File ---
    os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
    tree.write(output_xml_path)
    print(f"Successfully wrote synergy-only consensus XML to: {output_xml_path}\n")


if __name__ == "__main__":
    top_n_setting = "1p"
    # --- Configuration for AKT + MEK ---
    aktmek_synergy_name = "synergy_sweep-akt_mek-1204-1639-18p_transient_delayed_uniform_postdrug_RMSE_5k"
    aktmek_akt_name = "synergy_sweep-akt_mek-1104-2212-18p_AKT_transient_delayed_uniform_5k_singledrug"
    aktmek_mek_name = "synergy_sweep-akt_mek-1104-2212-18p_MEK_transient_delayed_uniform_5k_singledrug"
    aktmek_template_xml = "data/physiboss_config/3D_above_drugtreatment/settings_AGSv2_3D_SYN_AKT_MEK_drugfromabove_top1p_average.xml"
    aktmek_output_xml = f"data/physiboss_config/3D_above_drugtreatment/settings_AGSv2_3D_SYN_AKT_MEK_consensus_top{top_n_setting}.xml"
    aktmek_synergy_only_output_xml = f"data/physiboss_config/3D_above_drugtreatment/settings_AGSv2_3D_SYN_AKT_MEK_synergy_only_consensus_top{top_n_setting}.xml"

    generate_consensus_xml(
        synergy_exp_name=aktmek_synergy_name,
        drug_x_exp_name=aktmek_akt_name,
        drug_y_exp_name=aktmek_mek_name,
        template_xml_path=aktmek_template_xml,
        output_xml_path=aktmek_output_xml,
        drug_x_identifiers=['drug_x', 'akt'],
        drug_y_identifiers=['drug_y', 'mek'],
        top_n=top_n_setting
    )

    generate_synergy_only_xml(
        synergy_exp_name=aktmek_synergy_name,
        template_xml_path=aktmek_template_xml,
        output_xml_path=aktmek_synergy_only_output_xml,
        top_n=top_n_setting
    )

    # --- Configuration for PI3K + MEK ---
    pi3kmek_synergy_name = "synergy_sweep-pi3k_mek-1104-2212-18p_transient_delayed_uniform_5k_10p"
    pi3kmek_pi3k_name = "synergy_sweep-pi3k_mek-1104-2212-18p_PI3K_transient_delayed_uniform_5k_10p"
    pi3kmek_mek_name = "synergy_sweep-pi3k_mek-1104-2212-18p_MEK_transient_delayed_uniform_5k_10p"
    pi3kmek_template_xml = "data/physiboss_config/3D_above_drugtreatment/settings_AGSv2_3D_SYN_PI3K_MEK_drugfromabove_top1p_average.xml" # Assumed name
    pi3kmek_output_xml = f"data/physiboss_config/3D_above_drugtreatment/settings_AGSv2_3D_SYN_PI3K_MEK_consensus_top{top_n_setting}.xml"
    pi3kmek_synergy_only_output_xml = f"data/physiboss_config/3D_above_drugtreatment/settings_AGSv2_3D_SYN_PI3K_MEK_synergy_only_consensus_top{top_n_setting}.xml"

    generate_consensus_xml(
        synergy_exp_name=pi3kmek_synergy_name,
        drug_x_exp_name=pi3kmek_pi3k_name,
        drug_y_exp_name=pi3kmek_mek_name,
        template_xml_path=pi3kmek_template_xml,
        output_xml_path=pi3kmek_output_xml,
        drug_x_identifiers=['drug_x', 'pi3k'],
        drug_y_identifiers=['drug_y', 'mek'],
        top_n=top_n_setting
    )

    generate_synergy_only_xml(
        synergy_exp_name=pi3kmek_synergy_name,
        template_xml_path=pi3kmek_template_xml,
        output_xml_path=pi3kmek_synergy_only_output_xml,
        top_n=top_n_setting
    )



