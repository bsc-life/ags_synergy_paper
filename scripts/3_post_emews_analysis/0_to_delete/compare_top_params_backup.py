# This script takes two JSON files with top N parameter distributions from a CMA / GA run and runs different tests on them.
# The idea is to employ this as a previous step to find overlapping parameter distributions between the single-drug experiments.
# From here, we can define a JSON file for running the synergy experiment.

import json
import numpy as np
from scipy import stats
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os

def detect_evolutionary_algorithm(experiment_name):
    if "CMA" in experiment_name:
        return "CMA"
    elif "GA" in experiment_name:
        return "GA"
    elif "sweep" in experiment_name:
        return "sweep"
    else:
        raise ValueError("Unknown evolutionary algorithm")

# Function to load data from JSON files
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_experiment_json(experiment_name, top_n):
    algorithm = detect_evolutionary_algorithm(experiment_name)
    top_params_path = f"results/{algorithm}_summaries/final_summary_{experiment_name}/final_summary_{experiment_name}_param_distribution_top_{top_n}.json"
    return load_json_data(top_params_path)


# Function to perform tests and check significance
def run_tests(param1, param2, param_name):

    # Create arrays of the actual data points
    actual_data1 = np.linspace(param1['min'], param1['max'], param1['count'])
    actual_data2 = np.linspace(param2['min'], param2['max'], param2['count'])
    
    # Mann-Whitney U test for two samples
    mw_statistic, mw_pvalue = stats.mannwhitneyu(actual_data1, actual_data2, alternative='two-sided')
    
    # print(f"Parameter: {param_name}")
    # print(f"Mann-Whitney U test - statistic: {mw_statistic:.4f}, p-value: {mw_pvalue:.4f}")
    # print("Statistically significant:" if mw_pvalue < 0.05 else "Not statistically significant")
    # print()

    significant = mw_pvalue < 0.05
    
    return param_name, mw_statistic, mw_pvalue, significant

# def run_tests_for_all_params(data_pi3k, data_mek, data_akt):
#     # Create dataframes to store the results for PI3K-MEK and AKT-MEK comparisons
#     pi3k_mek_results_df = pd.DataFrame(columns=['Parameter', 'M-W test statistic', 'M-W test p-value', 'Statistically significant'])
#     akt_mek_results_df = pd.DataFrame(columns=['Parameter', 'M-W test statistic', 'M-W test p-value', 'Statistically significant'])

#     # Iterate through parameters and run tests for PI3K-MEK
#     for param in data_pi3k.keys():
#         if param in data_mek and param != "drug_X_permeability":
#             parameter, mw_statistic, mw_pvalue, significant = run_tests(data_pi3k[param], data_mek[param], param)
#             pi3k_mek_results_df.loc[len(pi3k_mek_results_df)] = [parameter, mw_statistic, mw_pvalue, significant]

#     # Iterate through parameters and run tests for AKT-MEK
#     for param in data_akt.keys():
#         if param in data_mek and param != "drug_X_permeability":
#             parameter, mw_statistic, mw_pvalue, significant = run_tests(data_akt[param], data_mek[param], param)
#             akt_mek_results_df.loc[len(akt_mek_results_df)] = [parameter, mw_statistic, mw_pvalue, significant]

#     # Save results to separate CSV files
#     pi3k_mek_results_df.to_csv(f"results/comparing_top_distributions/final_summary_{pi3k_experiment_name}_and_{mek_experiment_name}_top_comparison.csv", index=False, header=True)
#     akt_mek_results_df.to_csv(f"results/comparing_top_distributions/final_summary_{akt_experiment_name}_and_{mek_experiment_name}_top_comparison.csv", index=False, header=True)

#     return pi3k_mek_results_df, akt_mek_results_df


def run_tests_for_all_params(data_pi3k, data_mek, data_akt, top_n):
    """
    Run statistical tests comparing parameter distributions between experiments,
    excluding all drug-specific parameters (those starting with 'drug_X').
    """
    # Create dataframes to store the results for PI3K-MEK and AKT-MEK comparisons
    pi3k_mek_results_df = pd.DataFrame(columns=['Parameter', 'M-W test statistic', 'M-W test p-value', 'Statistically significant'])
    akt_mek_results_df = pd.DataFrame(columns=['Parameter', 'M-W test statistic', 'M-W test p-value', 'Statistically significant'])

    # Iterate through parameters and run tests for PI3K-MEK
    for param in data_pi3k.keys():
        # Skip drug-specific parameters
        if not param.startswith('drug_X'):
            if param in data_mek:
                parameter, mw_statistic, mw_pvalue, significant = run_tests(data_pi3k[param], data_mek[param], param)
                pi3k_mek_results_df.loc[len(pi3k_mek_results_df)] = [parameter, mw_statistic, mw_pvalue, significant]

    # Iterate through parameters and run tests for AKT-MEK
    for param in data_akt.keys():
        # Skip drug-specific parameters
        if not param.startswith('drug_X'):
            if param in data_mek:
                parameter, mw_statistic, mw_pvalue, significant = run_tests(data_akt[param], data_mek[param], param)
                akt_mek_results_df.loc[len(akt_mek_results_df)] = [parameter, mw_statistic, mw_pvalue, significant]

    # Save results to separate CSV files
    pi3k_mek_results_df.to_csv(f"results/comparing_top_distributions/final_summary_{pi3k_experiment_name}_and_{mek_experiment_name}_top_{top_n}_comparison.csv", index=False, header=True)
    akt_mek_results_df.to_csv(f"results/comparing_top_distributions/final_summary_{akt_experiment_name}_and_{mek_experiment_name}_top_{top_n}_comparison.csv", index=False, header=True)

    return pi3k_mek_results_df, akt_mek_results_df


# added the same function, but generalizes to any parameter with "drug_X" in the name
# this handles the drug-specific parameters
def generate_combined_distribution(data_pi3k, data_mek, param_comparison_df, original_params):
    """
    Generate a consensus distribution of the top N parameters from two experiments.
    Returns a list of dictionaries in the same format as the deap JSON files.
    Handles all drug-specific parameters (those starting with 'drug_X') by keeping them separate.
    """
    combined_distribution = []

    for param in original_params:
        base_name = param['name'].split('.')[-1]
        
        # Check if this is a drug-specific parameter
        is_drug_specific = base_name.startswith('drug_X')
        
        if not is_drug_specific and base_name in data_pi3k and base_name in data_mek:
            # Handle shared parameters as before
            param1 = data_pi3k[base_name]
            param2 = data_mek[base_name]
            
            # Compute min and max using mean and std
            min1 = param1['mean'] - 3 * param1['std']
            max1 = param1['mean'] + 3 * param1['std']
            min2 = param2['mean'] - 3 * param2['std']
            max2 = param2['mean'] + 3 * param2['std']
            
            # Check if the parameter distributions are statistically different
            is_significant = param_comparison_df.loc[param_comparison_df['Parameter'] == base_name, 'Statistically significant'].iloc[0]
            
            if is_significant:
                # If distributions are significantly different, use the union
                combined_min = min(min1, min2)
                combined_max = max(max1, max2)
                
                # If there's no overlap, use the midpoint between the two ranges
                if combined_min > combined_max:
                    combined_min = (max1 + min2) / 2
                    combined_max = combined_min
            else:
                # If distributions are not significantly different, use the intersection
                combined_min = max(min1, min2)
                combined_max = min(max1, max2)

            # Ensure the combined range is within the original bounds
            combined_min = max(combined_min, param['lower'])
            combined_max = min(combined_max, param['upper'])

            combined_distribution.append({
                "name": param['name'],
                "type": param['type'],
                "lower": combined_min,
                "upper": combined_max,
                "sigma": 5.0
            })
        else:
            # For drug-specific parameters or parameters not in both datasets,
            # use the original values
            combined_distribution.append(param)

    return combined_distribution



def generate_combined_sweep(data_experiment1, data_experiment2, param_comparison_df, exp1_name, exp2_name, top_n):
    """
    Generate a combined sweep configuration based on two input distributions.
    All drug-specific parameters (starting with 'drug_X') from the first input are kept as is.
    All drug-specific parameters from the second input are renamed from 'drug_X' to 'drug_Y' in the combined sweep.
    Saves the combined sweep configurations to JSON files.
    """
    combined_sweep = {}
    exp1_combined_sweep = {}
    exp2_combined_sweep = {}

    # Handle all drug-specific parameters from experiment1
    for param_name in data_experiment1.keys():
        if param_name.startswith('drug_X'):
            param1 = data_experiment1[param_name]
            # Keep original name for combined and exp1 sweeps
            param_path = f"cell_definitions.cell_definition.custom_data.{param_name}"
            param_dict = {
                "min": param1['min'],
                "max": param1['max'],
                "loc": param1['mean'],
                "scale": param1['std']
            }
            combined_sweep[param_path] = param_dict
            exp1_combined_sweep[param_path] = param_dict

    # Handle all drug-specific parameters from experiment2
    for param_name in data_experiment2.keys():
        if param_name.startswith('drug_X'):
            param2 = data_experiment2[param_name]
            # Rename to drug_Y for combined sweep
            renamed_param = param_name.replace('drug_X', 'drug_Y')
            combined_sweep[f"cell_definitions.cell_definition.custom_data.{renamed_param}"] = {
                "min": param2['min'],
                "max": param2['max'],
                "loc": param2['mean'],
                "scale": param2['std']
            }
            # Keep original name for exp2 sweep
            exp2_combined_sweep[f"cell_definitions.cell_definition.custom_data.{param_name}"] = {
                "min": param2['min'],
                "max": param2['max'],
                "loc": param2['mean'],
                "scale": param2['std']
            }

    # Handle all non-drug-specific parameters
    for param in data_experiment1.keys():
        if param in data_experiment2 and not param.startswith('drug_X'):
            param1 = data_experiment1[param]
            param2 = data_experiment2[param]
            
            # Check if the parameter distributions are statistically different
            is_significant = param_comparison_df.loc[param_comparison_df['Parameter'] == param, 'Statistically significant'].iloc[0]
            
            if is_significant:  # Distros are different
                # If distributions are significantly different, use the union to capture all data
                combined_min = min(param1['min'], param2['min'])
                combined_max = max(param1['max'], param2['max'])
                combined_mean = (param1['mean'] + param2['mean']) / 2
                combined_std = max(param1['std'], param2['std'])
            else:
                # If distributions are not significantly different, use the intersection
                combined_min = max(param1['min'], param2['min'])
                combined_max = min(param1['max'], param2['max'])
                
                # But only include the parameter if there's an overlap
                if combined_min <= combined_max:
                    combined_mean = (param1['mean'] + param2['mean']) / 2
                    combined_std = max(param1['std'], param2['std'])
                else:
                    # Handle the case where there is no overlap with also the union
                    combined_min = min(param1['min'], param2['min'])
                    combined_max = max(param1['max'], param2['max'])
                    combined_mean = (param1['mean'] + param2['mean']) / 2
                    combined_std = max(param1['std'], param2['std'])

            param_dict = {
                "min": combined_min,
                "max": combined_max,
                "loc": combined_mean,
                "scale": combined_std
            }

            if "user_parameters" in param:
                param_path = f"{param}"
            else:
                param_path = f"cell_definitions.cell_definition.custom_data.{param}"
            
            combined_sweep[param_path] = param_dict
            exp1_combined_sweep[param_path] = param_dict
            exp2_combined_sweep[param_path] = param_dict

    # Save the combined sweep configurations to JSON files
    output_file_combined = f"results/comparing_top_distributions/sweep_combined_{exp1_name}_and_{exp2_name}_top_{top_n}.json"
    with open(output_file_combined, 'w') as f:
        json.dump(combined_sweep, f, indent=4)

    output_file_exp1 = f"results/comparing_top_distributions/{exp1_name}_single_drug_sweep_combined_{exp1_name}_{exp2_name}_top_{top_n}.json"
    with open(output_file_exp1, 'w') as f:
        json.dump(exp1_combined_sweep, f, indent=4)

    output_file_exp2 = f"results/comparing_top_distributions/{exp2_name}_single_drug_sweep_combined_{exp1_name}_{exp2_name}_top_{top_n}.json"
    with open(output_file_exp2, 'w') as f:
        json.dump(exp2_combined_sweep, f, indent=4)

    print(f"Combined sweep configuration saved to {output_file_combined}")
    print(f"Experiment 1 combined sweep configuration saved to {output_file_exp1}")
    print(f"Experiment 2 combined sweep configuration saved to {output_file_exp2}")

    return combined_sweep, exp1_combined_sweep, exp2_combined_sweep                 


def plot_parameter_distributions(data_dict1, data_dict2, exp_name1, exp_name2, output_dir):
    # Prepare the data
    plot_data = []
    params = []
    for param in data_dict1.keys():
        if param in data_dict2:
            params.append(param)
            values1 = np.linspace(data_dict1[param]['min'], data_dict1[param]['max'], data_dict1[param]['count'])
            values2 = np.linspace(data_dict2[param]['min'], data_dict2[param]['max'], data_dict2[param]['count'])
            plot_data.extend([(param, exp_name1, val) for val in values1])
            plot_data.extend([(param, exp_name2, val) for val in values2])

    df = pd.DataFrame(plot_data, columns=['Parameter', 'Experiment', 'Value'])

    # Set up the plot
    n_params = len(params)
    fig_width = 12.0
    fig_height = 7.0
    fig, axes = plt.subplots(1, n_params, figsize=(fig_width, fig_height), sharey=False)
    if n_params == 1:
        axes = [axes]  # Ensure axes is always a list
    plt.subplots_adjust(top=0.85, bottom=0.3, left=0.03, right=0.95, wspace=1.0)
    sns.set_style("whitegrid")

    # Color palette - using ColorBrewer Set2 green and orange
    colors = sns.color_palette("Set2")
    
    # Function to get significance stars
    def get_stars(p_value):
        if p_value <= 0.001:
            return "***"
        elif p_value <= 0.01:
            return "**"
        elif p_value <= 0.05:
            return "*"
        else:
            return "ns"

    # Create box plots for each parameter
    for ax, param in zip(axes, params):
        sns.boxplot(data=df[df['Parameter'] == param], x='Parameter', y='Value', 
                    hue='Experiment', ax=ax, width=0.6, palette=colors)  # Adjust the linewidth of the box
        
        # Remove the box around the plot
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Customize the subplot
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')  # Remove Y-axis label
        
        # Adjust tick label size and rotation based on figsize
        # tick_label_size = 12 * (fig_width / 8.0)  # Proportional to figure width
        ax.tick_params(axis='both', which='major')
        ax.set_xticklabels([param], rotation=90, ha='right')
        
        # Remove the legend from individual subplots
        ax.get_legend().remove()
        
        # Calculate and add significance stars
        values1 = np.linspace(data_dict1[param]['min'], data_dict1[param]['max'], data_dict1[param]['count'])
        values2 = np.linspace(data_dict2[param]['min'], data_dict2[param]['max'], data_dict2[param]['count'])
        _, p_value = stats.mannwhitneyu(values1, values2)
        stars = get_stars(p_value)
        
        ax.text(0.5, 1.05, stars, 
                ha='center', va='bottom', fontsize=9.6 * (fig_width / 8.0), transform=ax.transAxes)

    # Add a simple legend to the top of the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='upper center', bbox_to_anchor=(0.5, 1.05), 
               ncol=2, fontsize=8.4 * (fig_width / 8.0), frameon=False)

    # Save the plot
    output_filename = f"combined_dist_{exp_name1}_{exp_name2}_A4_third_boxplot.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved as {output_path}")


########################################################################################
# Defining the data for single-drug and synergy experiments
########################################################################################

# Read the single-drug top params JSON file
# pi3k_experiment_name = "PI3Ki_CMA-1410-1014-12p_rmse_final_50gen"
# mek_experiment_name = "MEKi_CMA-1410-1026-12p_rmse_final_50gen"
# akt_experiment_name = "AKTi_CMA-1710-0934-12p_rmse_final_50gen"

# Testing with linear mapping
# pi3k_experiment_name = "PI3Ki_CMA-1002-0147-8p_linear_mapping"
# mek_experiment_name = "MEKi_CMA-1002-0147-8p_linear_mapping"
# akt_experiment_name = "AKTi_CMA-1002-0147-8p_linear_mapping"

# Testing with the 20p final experiments
pi3k_experiment_name = "PI3Ki_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen"
mek_experiment_name = "MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen"
akt_experiment_name = "AKTi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen"

# Load data from both JSON files
top_n = "10p"
data_pi3k_singledrug = get_experiment_json(pi3k_experiment_name, top_n)
data_mek_singledrug = get_experiment_json(mek_experiment_name, top_n)
data_akt_singledrug = get_experiment_json(akt_experiment_name, top_n)

# Read the PI3K-MEK synergy top params JSON file
pi3kmek_syn_experiment_name = "synergy_sweep-pi3k_mek-1610-1455-13p_uniform_5k"
pi3kmek_syn_PI3K_experiment_name = "synergy_sweep-pi3k_mek-1510-1508-12p_PI3K_singledrug"
pi3kmek_syn_MEK_experiment_name = "synergy_sweep-pi3k_mek-1510-1508-12p_MEK_singledrug"

# Read the AKT-MEK synergy top params JSON file
aktmek_syn_experiment_name = "synergy_sweep-akt_mek-1810-1101-13p_uniform_5k"
aktmek_syn_AKT_experiment_name = "synergy_sweep-akt_mek-1810-1101-12p_AKT_singledrug"
aktmek_syn_MEK_experiment_name = "synergy_sweep-akt_mek-1810-1502-12p_MEK_singledrug"

# Load data from the synergy top params JSON files
data_pi3kmek_syn = get_experiment_json(pi3kmek_syn_experiment_name, top_n)
data_pi3kmek_syn_PI3K = get_experiment_json(pi3kmek_syn_PI3K_experiment_name, top_n)
data_pi3kmek_syn_MEK = get_experiment_json(pi3kmek_syn_MEK_experiment_name, top_n)

# # Load data from the AKT-MEK synergy top params JSON files
data_aktmek_syn = get_experiment_json(aktmek_syn_experiment_name, top_n)
data_aktmek_syn_AKT = get_experiment_json(aktmek_syn_AKT_experiment_name, top_n)
data_aktmek_syn_MEK = get_experiment_json(aktmek_syn_MEK_experiment_name, top_n)



#######################################################################################################
# Selecting the top N parameters for the single-drug and synergy experiments
#######################################################################################################

top_n = top_n


#######################################################################################################
# Running the statistical tests for all parameters in single-drug experiments and synergy experiments
#######################################################################################################

pi3k_mek_single_drug_results_df, akt_mek_single_drug_results_df = run_tests_for_all_params(data_pi3k_singledrug, data_mek_singledrug, data_akt_singledrug, top_n)


########################################################################################
# Generating the combined JSON distributions for single-drug experiments
########################################################################################


# Load the original parameter structure
with open('data/JSON/deap/deap_18p_single_drug_exp_v2.json', 'r') as f:
    original_params = json.load(f)

# Generate the combined distribution for PI3K and MEK
combined_dist_pi3k_mek = generate_combined_distribution(data_pi3k_singledrug, data_mek_singledrug, pi3k_mek_single_drug_results_df, original_params)
# Save the combined distribution to a JSON file
output_file_pi3k_mek = f"results/comparing_top_distributions/combined_distribution_{pi3k_experiment_name}_and_{mek_experiment_name}_top{top_n}.json"
with open(output_file_pi3k_mek, 'w') as f:
    json.dump(combined_dist_pi3k_mek, f, indent=4)

print(f"Combined distribution (PI3K and MEK) saved to {output_file_pi3k_mek}")

# Generate the combined distribution for MEK and AKT
combined_dist_mek_akt = generate_combined_distribution(data_mek_singledrug, data_akt_singledrug, akt_mek_single_drug_results_df, original_params)

# Save the combined distribution to a JSON file
output_file_mek_akt = f"results/comparing_top_distributions/combined_distribution_{mek_experiment_name}_and_{akt_experiment_name}_top{top_n}.json"
with open(output_file_mek_akt, 'w') as f:
    json.dump(combined_dist_mek_akt, f, indent=4)

print(f"Combined distribution (MEK and AKT) saved to {output_file_mek_akt}")



########################################################################################
# Generating the combined sweep configurations for single-drug experiments
########################################################################################

# Generate the combined sweep configuration for PI3K and MEK 
combined_sweep_pi3k_mek, exp1_combined_sweep_pi3k_mek, exp2_combined_sweep_pi3k_mek = generate_combined_sweep(data_pi3k_singledrug, data_mek_singledrug, pi3k_mek_single_drug_results_df, pi3k_experiment_name, mek_experiment_name, top_n)

# Generate the combined sweep configuration for MEK and AKT
combined_sweep_akt_mek, exp1_combined_sweep_akt_mek, exp2_combined_sweep_akt_mek = generate_combined_sweep(data_akt_singledrug, data_mek_singledrug, akt_mek_single_drug_results_df, akt_experiment_name, mek_experiment_name, top_n)



########################################################################################
# Plotting the single-drug and synergy parameter distributions
########################################################################################

# Define the output directory
output_dir_single_drug = "results/comparing_top_distributions/violin_plots_single_drug"
os.makedirs(output_dir_single_drug, exist_ok=True)
# Generate the plots
plot_parameter_distributions(data_pi3k_singledrug, data_mek_singledrug, f"PI3K_{pi3k_experiment_name}", f"MEK_{mek_experiment_name}", output_dir_single_drug)
plot_parameter_distributions(data_akt_singledrug, data_mek_singledrug, f"AKT_{akt_experiment_name}", f"MEK_{mek_experiment_name}", output_dir_single_drug)


# synergy experiments
output_dir_synergy_pi3kmek = "results/comparing_top_distributions/violin_plots_synergy_pi3kmek"
os.makedirs(output_dir_synergy_pi3kmek, exist_ok=True) 
output_dir_synergy_aktmek = "results/comparing_top_distributions/violin_plots_synergy_aktmek"
os.makedirs(output_dir_synergy_aktmek, exist_ok=True) 
output_dir_synergy = "results/comparing_top_distributions/violin_plots_both_synergies"
os.makedirs(output_dir_synergy, exist_ok=True) 


# Comparing top 10 parameters from both synergies
plot_parameter_distributions(data_pi3kmek_syn, data_aktmek_syn, "PI3K+MEK", "AKT+MEK", output_dir_synergy)

# Focusing on the PI3K+MEK synergy
# Best synergy parameters and PI3K synergy sweep parameters
plot_parameter_distributions(data_pi3kmek_syn, data_pi3kmek_syn_PI3K, "PI3K+MEK", "PI3K_synergy_sweep", output_dir_synergy_pi3kmek)
plot_parameter_distributions(data_pi3kmek_syn, data_pi3kmek_syn_MEK, "PI3K+MEK", "MEK_synergy_sweep", output_dir_synergy_pi3kmek)

# Focusing on the AKT+MEK synergy
# Best synergy parameters and MEK synergy sweep parameters
plot_parameter_distributions(data_aktmek_syn, data_aktmek_syn_AKT, "AKT+MEK", "AKT_synergy_sweep", output_dir_synergy_aktmek)
plot_parameter_distributions(data_aktmek_syn, data_aktmek_syn_MEK, "AKT+MEK", "MEK_synergy_sweep", output_dir_synergy_aktmek)

output_dir_synergy_single_drug_pi3k = "results/comparing_top_distributions/violin_plots_synergy_single_drug_pi3k"
os.makedirs(output_dir_synergy_single_drug_pi3k, exist_ok=True)
output_dir_synergy_single_drug_mek = "results/comparing_top_distributions/violin_plots_synergy_single_drug_mek"
os.makedirs(output_dir_synergy_single_drug_mek, exist_ok=True)
output_dir_synergy_single_drug_akt = "results/comparing_top_distributions/violin_plots_synergy_single_drug_akt"
os.makedirs(output_dir_synergy_single_drug_akt, exist_ok=True)

# Comparing for each drug, the top single-drug and synergy parameters 
plot_parameter_distributions(data_pi3k_singledrug, data_pi3kmek_syn_PI3K, "PI3K_single_drug", "PI3K_synergy_sweep", output_dir_synergy_single_drug_pi3k)
plot_parameter_distributions(data_mek_singledrug, data_pi3kmek_syn_MEK, "MEK_single_drug", "MEK_synergy_sweep", output_dir_synergy_single_drug_mek)
plot_parameter_distributions(data_mek_singledrug, data_aktmek_syn_MEK, "MEK_single_drug", "MEK_synergy_sweep", output_dir_synergy_single_drug_mek)
plot_parameter_distributions(data_akt_singledrug, data_aktmek_syn_AKT, "AKT_single_drug", "AKT_synergy_sweep", output_dir_synergy_single_drug_akt)


########################################################################################
# 
#                                         OLD CODE 
# 
# 
########################################################################################


# def generate_combined_sweep(data_experiment1, data_experiment2, param_comparison_df, exp1_name, exp2_name):
#     """
#     Generate a combined sweep configuration based on two input distributions.
#     The drug_X_permeability of the first input (data_experiment1) is kept as is and added to the dictionary.
#     The drug_X_permeability of the second input (data_experiment2) is renamed to drug_Y_permeability and added to the dictionary without any operations.
#     This also has to be don
#     Saves the combined sweep configurations to JSON files.
#     """
#     combined_sweep = {}
#     exp1_combined_sweep = {}
#     exp2_combined_sweep = {}

#     # Keep drug_X_permeability from data_experiment1
#     if "drug_X_permeability" in data_experiment1:
#         param1 = data_experiment1["drug_X_permeability"]
#         combined_sweep["cell_definitions.cell_definition.custom_data.drug_X_permeability"] = {
#             "min": param1['min'],
#             "max": param1['max'],
#             "loc": param1['mean'],
#             "scale": param1['std']
#         }
#         exp1_combined_sweep["cell_definitions.cell_definition.custom_data.drug_X_permeability"] = {
#             "min": param1['min'],
#             "max": param1['max'],
#             "loc": param1['mean'],
#             "scale": param1['std']
#         }

#     # Rename drug_X_permeability from data_experiment2 to drug_Y_permeability
#     if "drug_X_permeability" in data_experiment2:
#         param2 = data_experiment2["drug_X_permeability"]
#         combined_sweep["cell_definitions.cell_definition.custom_data.drug_Y_permeability"] = {
#             "min": param2['min'],
#             "max": param2['max'],
#             "loc": param2['mean'],
#             "scale": param2['std']
#         }
    
#     # Keep drug_X_permeability from data_experiment2 in exp2_combined_sweep
#     if "drug_X_permeability" in data_experiment2:
#         exp2_combined_sweep["cell_definitions.cell_definition.custom_data.drug_X_permeability"] = {
#             "min": param2['min'],
#             "max": param2['max'],
#             "loc": param2['mean'],
#             "scale": param2['std']
#         }

#     for param in data_experiment1.keys():
#         if param in data_experiment2 and param != "drug_X_permeability":
#             param1 = data_experiment1[param]
#             param2 = data_experiment2[param]
            
#             # Check if the parameter distributions are statistically different
#             is_significant = param_comparison_df.loc[param_comparison_df['Parameter'] == param, 'Statistically significant'].iloc[0]
            
#             if is_significant:  # Distros are different
#                 # If distributions are significantly different, use the union to capture all data
#                 combined_min = min(param1['min'], param2['min'])
#                 combined_max = max(param1['max'], param2['max'])
#                 combined_mean = (param1['mean'] + param2['mean']) / 2
#                 combined_std = max(param1['std'], param2['std'])
#             else:
#                 # If distributions are not significantly different, use the intersection
#                 combined_min = max(param1['min'], param2['min'])
#                 combined_max = min(param1['max'], param2['max'])
                
#                 # But only include the parameter if there's an overlap
#                 if combined_min <= combined_max:
#                     combined_mean = (param1['mean'] + param2['mean']) / 2
#                     combined_std = max(param1['std'], param2['std'])
#                 else:
#                     # Handle the case where there is no overlap with also the union
#                     combined_min = min(param1['min'], param2['min'])
#                     combined_max = max(param1['max'], param2['max'])
#                     combined_mean = (param1['mean'] + param2['mean']) / 2
#                     combined_std = max(param1['std'], param2['std'])

#             combined_sweep[f"cell_definitions.cell_definition.custom_data.{param}"] = {
#                 "min": combined_min,
#                 "max": combined_max,
#                 "loc": combined_mean,
#                 "scale": combined_std
#             }

#             # Add to exp1_combined_sweep (excluding drug_Y_permeability)
#             exp1_combined_sweep[f"cell_definitions.cell_definition.custom_data.{param}"] = {
#                 "min": combined_min,
#                 "max": combined_max,
#                 "loc": combined_mean,
#                 "scale": combined_std
#             }

#             # Add to exp2_combined_sweep (excluding drug_X_permeability)
#             exp2_combined_sweep[f"cell_definitions.cell_definition.custom_data.{param}"] = {
#                 "min": combined_min,
#                 "max": combined_max,
#                 "loc": combined_mean,
#                 "scale": combined_std
#             }

#     # Save the combined sweep configurations to JSON files
#     output_file_combined = f"results/comparing_top_distributions/sweep_combined_{exp1_name}_and_{exp2_name}.json"
#     with open(output_file_combined, 'w') as f:
#         json.dump(combined_sweep, f, indent=4)

#     output_file_exp1 = f"results/comparing_top_distributions/{exp1_name}_single_drug_sweep_combined_{exp1_name}_{exp2_name}.json"
#     with open(output_file_exp1, 'w') as f:
#         json.dump(exp1_combined_sweep, f, indent=4)

#     output_file_exp2 = f"results/comparing_top_distributions/{exp2_name}_single_drug_sweep_combined_{exp1_name}_{exp2_name}.json"
#     with open(output_file_exp2, 'w') as f:
#         json.dump(exp2_combined_sweep, f, indent=4)

#     print(f"Combined sweep configuration saved to {output_file_combined}")
#     print(f"Experiment 1 combined sweep configuration saved to {output_file_exp1}")
#     print(f"Experiment 2 combined sweep configuration saved to {output_file_exp2}")

#     return combined_sweep, exp1_combined_sweep, exp2_combined_sweep


# def generate_combined_distribution(data_pi3k, data_mek, param_comparison_df, original_params):
#     """
#     Generate a consensus distribution of the top N parameters from two experiments.
#     Returns a list of dictionaries in the same format as the deap JSON files.
#     """
#     combined_distribution = []

#     for param in original_params:
#         base_name = param['name'].split('.')[-1]
#         if base_name in data_pi3k and base_name in data_mek and base_name != "drug_X_permeability":
#             param1 = data_pi3k[base_name]
#             param2 = data_mek[base_name]
            
#             # Compute min and max using mean and std
#             min1 = param1['mean'] - 3 * param1['std']
#             max1 = param1['mean'] + 3 * param1['std']
#             min2 = param2['mean'] - 3 * param2['std']
#             max2 = param2['mean'] + 3 * param2['std']
            
#             # Check if the parameter distributions are statistically different
#             is_significant = param_comparison_df.loc[param_comparison_df['Parameter'] == base_name, 'Statistically significant'].iloc[0]
            
#             if is_significant:
#                 # If distributions are significantly different, use the union
#                 combined_min = min(min1, min2)
#                 combined_max = max(max1, max2)
                
#                 # If there's no overlap, use the midpoint between the two ranges
#                 if combined_min > combined_max:
#                     combined_min = (max1 + min2) / 2
#                     combined_max = combined_min
#             else:
#                 # If distributions are not significantly different, use the intersection
#                 combined_min = max(min1, min2)
#                 combined_max = min(max1, max2)

#             # Ensure the combined range is within the original bounds
#             combined_min = max(combined_min, param['lower'])
#             combined_max = min(combined_max, param['upper'])

#             combined_distribution.append({
#                 "name": param['name'],
#                 "type": param['type'],
#                 "lower": combined_min,
#                 "upper": combined_max,
#                 "sigma": 5.0
#             })
#         else:
#             # If the parameter is not in all datasets or is drug_X_permeability, use the original values
#             combined_distribution.append(param)

#     return combined_distribution

