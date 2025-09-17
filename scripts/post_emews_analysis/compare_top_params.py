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
import xml.etree.ElementTree as ET
import re

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

def run_tests_for_all_params(data_pi3k, data_mek, data_akt, top_n, pi3k_experiment_name, mek_experiment_name, akt_experiment_name, output_dir):
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
    pi3k_mek_results_df.to_csv(os.path.join(output_dir, f"final_summary_{pi3k_experiment_name}_and_{mek_experiment_name}_top_{top_n}_comparison.csv"), index=False, header=True)
    akt_mek_results_df.to_csv(os.path.join(output_dir, f"final_summary_{akt_experiment_name}_and_{mek_experiment_name}_top_{top_n}_comparison.csv"), index=False, header=True)

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



def generate_combined_sweep(data_experiment1, data_experiment2, param_comparison_df, exp1_name, exp2_name, top_n, output_dir):
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
    output_file_combined = os.path.join(output_dir, f"sweep_combined_{exp1_name}_and_{exp2_name}_top_{top_n}.json")
    with open(output_file_combined, 'w') as f:
        json.dump(combined_sweep, f, indent=4)

    output_file_exp1 = os.path.join(output_dir, f"{exp1_name}_single_drug_sweep_combined_{exp1_name}_{exp2_name}_top_{top_n}.json")
    with open(output_file_exp1, 'w') as f:
        json.dump(exp1_combined_sweep, f, indent=4)

    output_file_exp2 = os.path.join(output_dir, f"{exp2_name}_single_drug_sweep_combined_{exp1_name}_{exp2_name}_top_{top_n}.json")
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


def plot_parameter_distributions_extended(data_dicts, exp_names, output_dir, title_suffix=""):
    """
    Create box plots comparing parameter distributions across multiple experimental contexts.
    
    Parameters:
    -----------
    data_dicts : list of dict
        List of dictionaries containing parameter distribution data
    exp_names : list of str
        List of experiment names corresponding to data_dicts
    output_dir : str
        Directory to save the output plots
    title_suffix : str
        Optional suffix for the plot title
    """
    # Ensure we have matching data dictionaries and experiment names
    assert len(data_dicts) == len(exp_names), "Number of data dictionaries must match number of experiment names"
    
    # Find common parameters across all data dictionaries
    common_params = set(data_dicts[0].keys())
    for data_dict in data_dicts[1:]:
        common_params = common_params.intersection(set(data_dict.keys()))
    
    # Filter out any parameters that start with 'drug_X' to focus on non-drug specific parameters
    # This is optional and depends on your analysis needs
    common_params = [param for param in common_params if not param.startswith('drug_X')]
    
    # Convert to sorted list for consistent plotting
    params = sorted(list(common_params))
    
    if not params:
        print("No common parameters found across distributions.")
        return
    
    # Prepare the data
    plot_data = []
    for param in params:
        for i, (data_dict, exp_name) in enumerate(zip(data_dicts, exp_names)):
            # Generate values using np.linspace based on min, max, and count
            # This is the key fix - using the same approach as in plot_parameter_distributions
            values = np.linspace(data_dict[param]['min'], data_dict[param]['max'], data_dict[param]['count'])
            plot_data.extend([(param, exp_name, val) for val in values])
    
    df = pd.DataFrame(plot_data, columns=['Parameter', 'Experiment', 'Value'])
    
    # Set up the plot
    n_params = len(params)
    fig_width = min(max(14, n_params * 1.5), 36)  # Adjust width based on number of parameters
    fig_height = 7.0
    fig, axes = plt.subplots(1, n_params, figsize=(fig_width, fig_height), sharey=False)
    if n_params == 1:
        axes = [axes]  # Ensure axes is always a list
    plt.subplots_adjust(top=0.85, bottom=0.3, left=0.03, right=0.95, wspace=1.0)
    sns.set_style("whitegrid")
    
    # Color palette for 3+ distributions - using ColorBrewer palette
    if len(exp_names) <= 3:
        colors = sns.color_palette("Set2", len(exp_names))  # Set2 works well for 2-3 categories
    else:
        colors = sns.color_palette("husl", len(exp_names))  # husl scales better for many categories
    
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
                    hue='Experiment', ax=ax, width=0.6, palette=colors)
        
        # Remove the box around the plot
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Customize the subplot
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='major')
        ax.set_xticklabels([param], rotation=90, ha='right')
        
        # Remove the legend from individual subplots
        ax.get_legend().remove()
        
        # Calculate pairwise significance and add annotations
        sig_text = []
        for i in range(len(data_dicts)):
            for j in range(i+1, len(data_dicts)):
                # Generate values for statistical tests
                values_i = np.linspace(data_dicts[i][param]['min'], data_dicts[i][param]['max'], data_dicts[i][param]['count'])
                values_j = np.linspace(data_dicts[j][param]['min'], data_dicts[j][param]['max'], data_dicts[j][param]['count'])
                _, p_value = stats.mannwhitneyu(values_i, values_j)
                stars = get_stars(p_value)
                sig_text.append(f"{exp_names[i]} vs {exp_names[j]}: {stars}")
        
        # Add significance annotations at top of plot
        y_pos = 1.05
        for text in sig_text:
            ax.text(0.5, y_pos, text, ha='center', va='bottom', 
                    fontsize=8, transform=ax.transAxes)
            y_pos += 0.05
    
    # Add a legend to the top of the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
               ncol=len(exp_names), fontsize=10, frameon=False)
    
    # Add overall title
    exp_names_str = "_".join(exp_names)
    plt.suptitle(f"Parameter Distributions Comparison {title_suffix}", fontsize=14, y=1.05)
    
    # Save the plot
    output_filename = f"param_dist_{'_vs_'.join(exp_names)}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as {output_path}")
    
    # Also perform statistical tests and save results
    test_results = []
    for param in params:
        for i in range(len(data_dicts)):
            for j in range(i+1, len(data_dicts)):
                # Generate values for statistical tests
                values_i = np.linspace(data_dicts[i][param]['min'], data_dicts[i][param]['max'], data_dicts[i][param]['count'])
                values_j = np.linspace(data_dicts[j][param]['min'], data_dicts[j][param]['max'], data_dicts[j][param]['count'])
                stat, p_value = stats.mannwhitneyu(values_i, values_j)
                test_results.append({
                    'Parameter': param,
                    'Distribution1': exp_names[i],
                    'Distribution2': exp_names[j],
                    'Mann-Whitney U': stat,
                    'p-value': p_value,
                    'Significant': p_value < 0.05
                })
    
    # Save statistical test results
    results_df = pd.DataFrame(test_results)
    results_path = os.path.join(output_dir, f"statistical_tests_{'_vs_'.join(exp_names)}.csv")
    results_df.to_csv(results_path, index=False)
    
    return results_df


def generate_parameter_statistics_table(data_dicts, exp_names, output_dir):
    """
    Generate two CSV tables with parameter statistics (mean and std) for each condition:
    1. One for drug-specific parameters (starting with 'drug_X' or 'drug_Y')
    2. One for all other parameters
    
    Parameters:
    -----------
    data_dicts : list of dict
        List of dictionaries containing parameter distribution data
    exp_names : list of str
        List of experiment names corresponding to data_dicts
    output_dir : str
        Directory to save the output CSV files
    """
    # Get all unique parameters across all conditions
    all_params = set()
    for data_dict in data_dicts:
        all_params.update(data_dict.keys())
    
    # Separate drug-specific and non-drug-specific parameters
    drug_params = [param for param in all_params if param.startswith('drug_')]
    non_drug_params = [param for param in all_params if not param.startswith('drug_')]
    
    # Sort parameters alphabetically
    drug_params.sort()
    non_drug_params.sort()
    
    # Function to generate statistics table for a set of parameters
    def generate_table(params_list, is_drug_specific=False):
        stats_data = []
        
        for param in params_list:
            row_data = {'Parameter': param}
            
            # Add statistics for each condition
            for data_dict, exp_name in zip(data_dicts, exp_names):
                if param in data_dict:
                    param_data = data_dict[param]
                    row_data[f'{exp_name}_mean'] = param_data['mean']
                    row_data[f'{exp_name}_std'] = param_data['std']
                else:
                    row_data[f'{exp_name}_mean'] = None
                    row_data[f'{exp_name}_std'] = None
            
            stats_data.append(row_data)
        
        # Create DataFrame
        stats_df = pd.DataFrame(stats_data)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV with appropriate filename
        suffix = 'drug_specific' if is_drug_specific else 'non_drug_specific'
        output_file = os.path.join(output_dir, f'parameter_statistics_table_{suffix}.csv')
        stats_df.to_csv(output_file, index=False)
        print(f"Parameter statistics table ({suffix}) saved to {output_file}")
        
        return stats_df
    
    # Generate both tables
    drug_stats_df = generate_table(drug_params, is_drug_specific=True)
    non_drug_stats_df = generate_table(non_drug_params, is_drug_specific=False)
    
    return drug_stats_df, non_drug_stats_df


def generate_synergy_comparison_table(data_dicts, exp_names, output_dir):
    
    """
    Generate a CSV table comparing only the two main synergy conditions:
    1. PI3K+MEK synergy
    2. AKT+MEK synergy
    
    Computes effect sizes and distribution overlap between the conditions.
    
    Parameters:
    -----------
    data_dicts : list of dict
        List of dictionaries containing parameter distribution data
    exp_names : list of str
        List of experiment names corresponding to data_dicts
    output_dir : str
        Directory to save the output CSV file
    """
    # Get all unique parameters across all conditions
    all_params = set()
    for data_dict in data_dicts:
        all_params.update(data_dict.keys())
    
    # Sort parameters alphabetically
    all_params = sorted(list(all_params))
    
    # Create a list to store the statistics
    stats_data = []
    
    for param in all_params:
        row_data = {'Parameter': param}
        
        # Add statistics for each condition
        for data_dict, exp_name in zip(data_dicts, exp_names):
            if param in data_dict:
                param_data = data_dict[param]
                row_data[f'{exp_name}_mean'] = param_data['mean']
                row_data[f'{exp_name}_std'] = param_data['std']
                # Calculate min and max using mean and std
                row_data[f'{exp_name}_min'] = param_data['mean'] - 3 * param_data['std']
                row_data[f'{exp_name}_max'] = param_data['mean'] + 3 * param_data['std']
            else:
                row_data[f'{exp_name}_mean'] = None
                row_data[f'{exp_name}_std'] = None
                row_data[f'{exp_name}_min'] = None
                row_data[f'{exp_name}_max'] = None
        
        # Calculate effect size and overlap if parameter exists in both conditions
        if param in data_dicts[0] and param in data_dicts[1]:
            # Calculate effect size (Cohen's d)
            d = (data_dicts[0][param]['mean'] - data_dicts[1][param]['mean']) / np.sqrt(
                (data_dicts[0][param]['std']**2 + data_dicts[1][param]['std']**2) / 2
            )
            row_data['Effect_Size'] = abs(d)
            
            # Add interpretation of effect size
            if abs(d) < 0.2:
                row_data['Effect_Interpretation'] = 'Negligible'
            elif abs(d) < 0.5:
                row_data['Effect_Interpretation'] = 'Small'
            elif abs(d) < 0.8:
                row_data['Effect_Interpretation'] = 'Medium'
            else:
                row_data['Effect_Interpretation'] = 'Large'
            
            # Calculate distribution overlap
            min1 = data_dicts[0][param]['mean'] - 3 * data_dicts[0][param]['std']
            max1 = data_dicts[0][param]['mean'] + 3 * data_dicts[0][param]['std']
            min2 = data_dicts[1][param]['mean'] - 3 * data_dicts[1][param]['std']
            max2 = data_dicts[1][param]['mean'] + 3 * data_dicts[1][param]['std']
            
            # Calculate union and intersection
            union_min = min(min1, min2)
            union_max = max(max1, max2)
            intersection_min = max(min1, min2)
            intersection_max = min(max1, max2)
            
            # Calculate overlap percentage
            if intersection_max > intersection_min:
                overlap_range = intersection_max - intersection_min
                total_range = union_max - union_min
                overlap_percentage = (overlap_range / total_range) * 100
            else:
                overlap_percentage = 0
            
            row_data['Overlap_Percentage'] = overlap_percentage
            
            # Add interpretation of overlap
            if overlap_percentage > 65:
                row_data['Overlap_Interpretation'] = 'High'
            elif overlap_percentage > 35:
                row_data['Overlap_Interpretation'] = 'Medium'
            else:
                row_data['Overlap_Interpretation'] = 'Low'
        else:
            # If parameter doesn't exist in both conditions, add None values
            row_data['Effect_Size'] = None
            row_data['Effect_Interpretation'] = None
            row_data['Overlap_Percentage'] = None
            row_data['Overlap_Interpretation'] = None
        
        stats_data.append(row_data)
    
    # Create DataFrame and save to CSV
    stats_df = pd.DataFrame(stats_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'synergy_comparison_table.csv')
    stats_df.to_csv(output_file, index=False)
    print(f"Synergy comparison table saved to {output_file}")
    
    # Print summary of parameters with high effect sizes or low overlap
    high_effect_params = stats_df[
        (stats_df['Effect_Size'] >= 0.5) | 
        (stats_df['Overlap_Percentage'] < 35)
    ]
    
    if len(high_effect_params) > 0:
        print("\nParameters with notable differences between PI3K+MEK and AKT+MEK:")
        for _, row in high_effect_params.iterrows():
            print(f"\n{row['Parameter']}:")
            print(f"  Effect size: {row['Effect_Size']:.3f} ({row['Effect_Interpretation']})")
            print(f"  Overlap: {row['Overlap_Percentage']:.1f}% ({row['Overlap_Interpretation']})")
            print(f"  PI3K+MEK: {row['PI3K+MEK_synergy_mean']:.3f} ± {row['PI3K+MEK_synergy_std']:.3f}")
            print(f"  AKT+MEK: {row['AKT+MEK_synergy_mean']:.3f} ± {row['AKT+MEK_synergy_std']:.3f}")
    else:
        print("\nNo parameters showed notable differences between PI3K+MEK and AKT+MEK")
    
    return stats_df


def generate_comprehensive_comparison_table(data_dicts, exp_names, output_dir):
    """
    Generate a comprehensive CSV table comparing all conditions for each parameter simultaneously.
    Uses Kruskal-Wallis H-test for overall comparison and Dunn's test for post-hoc analysis.
    
    Parameters:
    -----------
    data_dicts : list of dict
        List of dictionaries containing parameter distribution data
    exp_names : list of str
        List of experiment names corresponding to data_dicts
    output_dir : str
        Directory to save the output CSV file
    """
    # Get all unique parameters across all conditions
    all_params = set()
    for data_dict in data_dicts:
        all_params.update(data_dict.keys())
    
    # Sort parameters alphabetically
    all_params = sorted(list(all_params))
    
    # Create a list to store all comparison results
    comparison_results = []
    
    for param in all_params:
        # Prepare data for all conditions
        values_list = []  # List to store all values
        groups_list = []  # List to store group labels
        means = []        # List to store means
        stds = []         # List to store standard deviations
        
        for data_dict, exp_name in zip(data_dicts, exp_names):
            if param in data_dict:
                # Generate values for this condition
                values = np.linspace(data_dict[param]['min'], data_dict[param]['max'], data_dict[param]['count'])
                values_list.extend(values)
                groups_list.extend([exp_name] * len(values))
                means.append(data_dict[param]['mean'])
                stds.append(data_dict[param]['std'])
        
        if len(set(groups_list)) > 1:  # Only perform tests if we have at least 2 conditions
            # Perform Kruskal-Wallis H-test for overall difference
            h_stat, h_pvalue = stats.kruskal(*[np.array(values_list)[np.array(groups_list) == name] 
                                             for name in set(groups_list)])
            
            # Calculate effect size (eta-squared)
            n = len(values_list)
            eta_squared = (h_stat - len(set(groups_list)) + 1) / (n - len(set(groups_list)))
            
            # Determine effect size interpretation
            if eta_squared < 0.01:
                effect_interpretation = 'Negligible'
            elif eta_squared < 0.06:
                effect_interpretation = 'Small'
            elif eta_squared < 0.14:
                effect_interpretation = 'Medium'
            else:
                effect_interpretation = 'Large'
            
            # Add overall comparison result
            comparison_results.append({
                'Parameter': param,
                'Test': 'Kruskal-Wallis',
                'Statistic': h_stat,
                'p_value': h_pvalue,
                'Significant': h_pvalue < 0.05,
                'Effect_Size': eta_squared,
                'Effect_Interpretation': effect_interpretation,
                'Number_of_Conditions': len(set(groups_list))
            })
            
            # Add summary statistics for each condition
            for exp_name, mean, std in zip(set(groups_list), means, stds):
                comparison_results.append({
                    'Parameter': param,
                    'Test': 'Condition',
                    'Condition': exp_name,
                    'Mean': mean,
                    'Std': std,
                    'Significant': h_pvalue < 0.05,  # Same significance as overall test
                    'Effect_Size': eta_squared,      # Same effect size as overall test
                    'Effect_Interpretation': effect_interpretation
                })
    
    # Create DataFrame and save to CSV
    comparison_df = pd.DataFrame(comparison_results)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'comprehensive_comparison_table.csv')
    comparison_df.to_csv(output_file, index=False)
    print(f"Comprehensive comparison table saved to {output_file}")
    
    # Print summary of significant differences
    significant_params = comparison_df[
        (comparison_df['Test'] == 'Kruskal-Wallis') & 
        (comparison_df['Significant'] == True)
    ]
    
    if len(significant_params) > 0:
        print("\nParameters with significant differences across conditions:")
        for _, row in significant_params.iterrows():
            param = row['Parameter']
            print(f"\n{param}:")
            print(f"  Kruskal-Wallis p-value: {row['p_value']:.3e}")
            print(f"  Effect size (η²): {row['Effect_Size']:.3f} ({row['Effect_Interpretation']})")
            
            # Print means for each condition
            condition_stats = comparison_df[
                (comparison_df['Parameter'] == param) & 
                (comparison_df['Test'] == 'Condition')
            ]
            print("  Condition means:")
            for _, cond_row in condition_stats.iterrows():
                print(f"    {cond_row['Condition']}: {cond_row['Mean']:.3f} ± {cond_row['Std']:.3f}")
    else:
        print("\nNo parameters showed significant differences across conditions")
    
    return comparison_df


def generate_synergy_parameter_analysis(data_dicts, exp_names, output_dir):
    """
    Generate a comprehensive analysis of synergy-related parameters, comparing:
    1. PI3K+MEK synergy vs PI3K and MEK alone in PI3K+MEK space
    2. AKT+MEK synergy vs AKT and MEK alone in AKT+MEK space
    
    Parameters:
    -----------
    data_dicts : list of dict
        List of dictionaries containing parameter distribution data
    exp_names : list of str
        List of experiment names corresponding to data_dicts
    output_dir : str
        Directory to save the output CSV file
    """
    # Create a list to store all comparison results
    comparison_results = []
    
    # Define the groups for each synergy analysis
    pi3kmek_groups = {
        'synergy': 'PI3K+MEK_synergy',
        'pi3k': 'PI3K_in_PI3K+MEK_space',
        'mek': 'MEK_in_PI3K+MEK_space'
    }
    
    aktmek_groups = {
        'synergy': 'AKT+MEK_synergy',
        'akt': 'AKT_in_AKT+MEK_space',
        'mek': 'MEK_in_AKT+MEK_space'
    }
    
    # Get all unique parameters across all conditions
    all_params = set()
    for data_dict in data_dicts:
        all_params.update(data_dict.keys())
    
    # Sort parameters alphabetically
    all_params = sorted(list(all_params))
    
    for param in all_params:
        # Analyze PI3K+MEK synergy group
        pi3kmek_values = []
        pi3kmek_labels = []
        pi3kmek_means = []
        pi3kmek_stds = []
        
        for group_key, group_name in pi3kmek_groups.items():
            idx = exp_names.index(group_name)
            if param in data_dicts[idx]:
                values = np.linspace(data_dicts[idx][param]['min'], 
                                   data_dicts[idx][param]['max'], 
                                   data_dicts[idx][param]['count'])
                pi3kmek_values.extend(values)
                pi3kmek_labels.extend([group_name] * len(values))
                pi3kmek_means.append(data_dicts[idx][param]['mean'])
                pi3kmek_stds.append(data_dicts[idx][param]['std'])
        
        if len(set(pi3kmek_labels)) > 1:
            # Perform Kruskal-Wallis test for PI3K+MEK group
            h_stat, h_pvalue = stats.kruskal(*[np.array(pi3kmek_values)[np.array(pi3kmek_labels) == name] 
                                             for name in set(pi3kmek_labels)])
            
            # Calculate effect size (eta-squared)
            n = len(pi3kmek_values)
            eta_squared = (h_stat - len(set(pi3kmek_labels)) + 1) / (n - len(set(pi3kmek_labels)))
            
            # Determine effect size interpretation
            if eta_squared < 0.01:
                effect_interpretation = 'Negligible'
            elif eta_squared < 0.06:
                effect_interpretation = 'Small'
            elif eta_squared < 0.14:
                effect_interpretation = 'Medium'
            else:
                effect_interpretation = 'Large'
            
            # Add PI3K+MEK group results
            comparison_results.append({
                'Parameter': param,
                'Synergy_Group': 'PI3K+MEK',
                'Test': 'Kruskal-Wallis',
                'Statistic': h_stat,
                'p_value': h_pvalue,
                'Significant': h_pvalue < 0.05,
                'Effect_Size': eta_squared,
                'Effect_Interpretation': effect_interpretation
            })
            
            # Add condition statistics for PI3K+MEK group
            for group_name, mean, std in zip(set(pi3kmek_labels), pi3kmek_means, pi3kmek_stds):
                comparison_results.append({
                    'Parameter': param,
                    'Synergy_Group': 'PI3K+MEK',
                    'Test': 'Condition',
                    'Condition': group_name,
                    'Mean': mean,
                    'Std': std,
                    'Significant': h_pvalue < 0.05,
                    'Effect_Size': eta_squared,
                    'Effect_Interpretation': effect_interpretation
                })
        
        # Analyze AKT+MEK synergy group
        aktmek_values = []
        aktmek_labels = []
        aktmek_means = []
        aktmek_stds = []
        
        for group_key, group_name in aktmek_groups.items():
            idx = exp_names.index(group_name)
            if param in data_dicts[idx]:
                values = np.linspace(data_dicts[idx][param]['min'], 
                                   data_dicts[idx][param]['max'], 
                                   data_dicts[idx][param]['count'])
                aktmek_values.extend(values)
                aktmek_labels.extend([group_name] * len(values))
                aktmek_means.append(data_dicts[idx][param]['mean'])
                aktmek_stds.append(data_dicts[idx][param]['std'])
        
        if len(set(aktmek_labels)) > 1:
            # Perform Kruskal-Wallis test for AKT+MEK group
            h_stat, h_pvalue = stats.kruskal(*[np.array(aktmek_values)[np.array(aktmek_labels) == name] 
                                             for name in set(aktmek_labels)])
            
            # Calculate effect size (eta-squared)
            n = len(aktmek_values)
            eta_squared = (h_stat - len(set(aktmek_labels)) + 1) / (n - len(set(aktmek_labels)))
            
            # Determine effect size interpretation
            if eta_squared < 0.01:
                effect_interpretation = 'Negligible'
            elif eta_squared < 0.06:
                effect_interpretation = 'Small'
            elif eta_squared < 0.14:
                effect_interpretation = 'Medium'
            else:
                effect_interpretation = 'Large'
            
            # Add AKT+MEK group results
            comparison_results.append({
                'Parameter': param,
                'Synergy_Group': 'AKT+MEK',
                'Test': 'Kruskal-Wallis',
                'Statistic': h_stat,
                'p_value': h_pvalue,
                'Significant': h_pvalue < 0.05,
                'Effect_Size': eta_squared,
                'Effect_Interpretation': effect_interpretation
            })
            
            # Add condition statistics for AKT+MEK group
            for group_name, mean, std in zip(set(aktmek_labels), aktmek_means, aktmek_stds):
                comparison_results.append({
                    'Parameter': param,
                    'Synergy_Group': 'AKT+MEK',
                    'Test': 'Condition',
                    'Condition': group_name,
                    'Mean': mean,
                    'Std': std,
                    'Significant': h_pvalue < 0.05,
                    'Effect_Size': eta_squared,
                    'Effect_Interpretation': effect_interpretation
                })
    
    # Create DataFrame and save to CSV
    comparison_df = pd.DataFrame(comparison_results)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'synergy_parameter_analysis.csv')
    comparison_df.to_csv(output_file, index=False)
    print(f"Synergy parameter analysis table saved to {output_file}")
    
    # Print summary of significant differences
    significant_params = comparison_df[
        (comparison_df['Test'] == 'Kruskal-Wallis') & 
        (comparison_df['Significant'] == True)
    ]
    
    if len(significant_params) > 0:
        print("\nParameters with significant differences in synergy groups:")
        for _, row in significant_params.iterrows():
            param = row['Parameter']
            synergy_group = row['Synergy_Group']
            print(f"\n{param} ({synergy_group}):")
            print(f"  Kruskal-Wallis p-value: {row['p_value']:.3e}")
            print(f"  Effect size (η²): {row['Effect_Size']:.3f} ({row['Effect_Interpretation']})")
            
            # Print means for each condition in this synergy group
            condition_stats = comparison_df[
                (comparison_df['Parameter'] == param) & 
                (comparison_df['Synergy_Group'] == synergy_group) &
                (comparison_df['Test'] == 'Condition')
            ]
            print("  Condition means:")
            for _, cond_row in condition_stats.iterrows():
                print(f"    {cond_row['Condition']}: {cond_row['Mean']:.3f} ± {cond_row['Std']:.3f}")
    else:
        print("\nNo parameters showed significant differences in synergy groups")
    
    return comparison_df


def generate_parameter_count_table(data_dicts, exp_names, output_dir):
    """
    Generates a table summarizing the number of total, common, and drug-specific
    parameters for each experimental condition.

    Parameters:
    -----------
    data_dicts : list of dict
        List of dictionaries containing parameter distribution data.
    exp_names : list of str
        List of experiment names corresponding to data_dicts.
    output_dir : str
        Directory to save the output CSV file.
    """
    counts_data = []

    for data_dict, exp_name in zip(data_dicts, exp_names):
        param_names = list(data_dict.keys())
        total_params = len(param_names)
        drug_specific_params = sum(1 for p in param_names if p.startswith('drug_'))
        common_params = total_params - drug_specific_params

        counts_data.append({
            'Condition': exp_name,
            'Total Parameters': total_params,
            'Common Parameters': common_params,
            'Drug-Specific Parameters': drug_specific_params
        })

    # Create DataFrame
    counts_df = pd.DataFrame(counts_data)

    # Save to CSV
    output_file = os.path.join(output_dir, 'parameter_counts_per_condition.csv')
    counts_df.to_csv(output_file, index=False)
    print(f"Parameter count table saved to {output_file}")

    return counts_df


def load_top_instances(experiment_name, top_n):
    """Loads top parameter instances from a CSV file."""
    algorithm = detect_evolutionary_algorithm(experiment_name)
    summary_dir = f"results/{algorithm}_summaries/final_summary_{experiment_name}"
    
    file_path = os.path.join(summary_dir, f"top_{top_n}.csv")

    if not os.path.exists(file_path):
        print(f"Error: Could not find the top instances file at {file_path}")
        return None

    try:
        # Load the CSV without assuming an index column
        df = pd.read_csv(file_path)
        # Convert dataframe to list of dictionaries
        return df.to_dict('records')
    except Exception as e:
        print(f"Error loading or parsing {file_path}: {e}")
        return None


def generate_pairplot_from_top_instances(pi3k_exp, mek_exp, akt_exp, top_n, output_dir):
    """
    Generates a pairplot from the top parameter instances of the three single-drug
    experiments, colored by drug.

    This involves:
    1. Loading the top instances from CSV for each experiment.
    2. Combining them into a single DataFrame.
    3. Adding an 'Experiment' column to distinguish the drugs.
    4. Plotting the common, non-drug-specific parameters.
    5. Saving the plot in both PNG and PDF formats at 300 dpi.
    """
    # Load top instances for each experiment
    pi3k_instances = load_top_instances(pi3k_exp, top_n)
    mek_instances = load_top_instances(mek_exp, top_n)
    akt_instances = load_top_instances(akt_exp, top_n)

    if pi3k_instances is None or mek_instances is None or akt_instances is None:
        print("Could not load instances for all single-drug experiments. Skipping pairplot from instances.")
        return

    # Convert to DataFrames
    df_pi3k = pd.DataFrame(pi3k_instances)
    df_mek = pd.DataFrame(mek_instances)
    df_akt = pd.DataFrame(akt_instances)

    # Add experiment labels
    df_pi3k['Experiment'] = 'PI3K'
    df_mek['Experiment'] = 'MEK'
    df_akt['Experiment'] = 'AKT'

    # Concatenate dataframes
    combined_df = pd.concat([df_pi3k, df_mek, df_akt], ignore_index=True)
    
    # Find common parameters, excluding drug-specific ones and unwanted columns
    common_params = set(df_pi3k.columns) & set(df_mek.columns) & set(df_akt.columns)
    
    # Exclude unwanted columns
    excluded_columns = {'replicate', 'RMSE_SK_POSTDRUG', 'Experiment'}
    consensus_params = sorted([p for p in common_params 
                             if not p.startswith('drug_') 
                             and p not in excluded_columns])

    if not consensus_params:
        print("No common consensus parameters found to generate a pairplot from instances.")
        return

    # Select only consensus parameters and the Experiment column
    plot_df = combined_df[consensus_params + ['Experiment']]

    if plot_df.empty:
        print("Plotting dataframe is empty. Cannot generate plot.")
        return

    # Generate the pairplot
    print("Generating pairplot from top parameter instances...")
    sns.set_style("whitegrid")
    pairplot = sns.pairplot(plot_df, hue='Experiment', corner=True, diag_kind='kde',
                            palette=sns.color_palette("Set2", 3))
    
    pairplot.fig.suptitle("Pairplot of Top Parameter Instances from Single-Drug Calibrations", y=1.02, fontsize=16)

    # Save the plot in PNG format
    output_path_png = os.path.join(output_dir, 'top_instances_pairplot.png')
    pairplot.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"Pairplot saved as PNG to {output_path_png}")

    # Save the plot in PDF format
    output_path_pdf = os.path.join(output_dir, 'top_instances_pairplot.pdf')
    pairplot.savefig(output_path_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Pairplot saved as PDF to {output_path_pdf}")

    plt.close()


def generate_single_drug_parameter_table(data_dicts, exp_names, output_dir):
    """
    Generate a single CSV table with parameter distributions (mean and std) for the three single-drug conditions.
    Values are rounded to 2 decimal places.
    
    Parameters:
    -----------
    data_dicts : list of dict
        List of dictionaries containing parameter distribution data for PI3K, MEK, and AKT single-drug conditions
    exp_names : list of str
        List of experiment names corresponding to data_dicts
    output_dir : str
        Directory to save the output CSV file
    """
    # Get all unique parameters across all conditions
    all_params = set()
    for data_dict in data_dicts:
        all_params.update(data_dict.keys())
    
    # Sort parameters alphabetically
    all_params = sorted(list(all_params))
    
    # Create a list to store the statistics
    stats_data = []
    
    for param in all_params:
        row_data = {'Parameter': param}
        
        # Add statistics for each condition
        for data_dict, exp_name in zip(data_dicts, exp_names):
            if param in data_dict:
                param_data = data_dict[param]
                row_data[f'{exp_name}_mean'] = round(param_data['mean'], 2)
                row_data[f'{exp_name}_std'] = round(param_data['std'], 2)
            else:
                row_data[f'{exp_name}_mean'] = None
                row_data[f'{exp_name}_std'] = None
        
        stats_data.append(row_data)
    
    # Create DataFrame
    stats_df = pd.DataFrame(stats_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'single_drug_parameter_distributions.csv')
    stats_df.to_csv(output_file, index=False, float_format='%.2f')
    print(f"Single drug parameter distributions table saved to {output_file}")
    
    return stats_df


def generate_union_distribution(data_pi3k, data_mek, original_params):
    """
    Generate a distribution using only the union of parameter ranges from two experiments.
    This creates a more inclusive parameter space by taking the widest range for each parameter.
    """
    union_distribution = []

    for param in original_params:
        base_name = param['name'].split('.')[-1]
        
        # Check if this is a drug-specific parameter
        is_drug_specific = base_name.startswith('drug_X')
        
        if not is_drug_specific and base_name in data_pi3k and base_name in data_mek:
            # Get parameters from both experiments
            param1 = data_pi3k[base_name]
            param2 = data_mek[base_name]
            
            # Compute min and max using mean and std
            min1 = param1['mean'] - 3 * param1['std']
            max1 = param1['mean'] + 3 * param1['std']
            min2 = param2['mean'] - 3 * param2['std']
            max2 = param2['mean'] + 3 * param2['std']
            
            # Always use the union (widest range)
            union_min = min(min1, min2)
            union_max = max(max1, max2)

            # Ensure the union range is within the original bounds
            union_min = max(union_min, param['lower'])
            union_max = min(union_max, param['upper'])

            union_distribution.append({
                "name": param['name'],
                "type": param['type'],
                "lower": union_min,
                "upper": union_max,
                "sigma": 5.0
            })
        else:
            # For drug-specific parameters or parameters not in both datasets,
            # use the original values
            union_distribution.append(param)

    return union_distribution


def generate_six_way_comparison_table(data_dicts, exp_names, output_dir):
    """
    Generate CSV tables comparing all six conditions simultaneously:
    1. PI3K+MEK synergy
    2. PI3K in PI3K+MEK space
    3. MEK in PI3K+MEK space
    4. AKT+MEK synergy
    5. AKT in AKT+MEK space
    6. MEK in AKT+MEK space
    
    Generates two files:
    1. A comprehensive table with all comparisons
    2. A simplified table with just the overall overlap information
    """
    # Get all unique parameters across all conditions
    all_params = set()
    for data_dict in data_dicts:
        all_params.update(data_dict.keys())
    
    # Sort parameters alphabetically
    all_params = sorted(list(all_params))
    
    # Create a list to store the statistics
    stats_data = []
    
    for param in all_params:
        row_data = {'Parameter': param}
        
        # Add statistics for each condition
        param_exists_in_all = True
        all_mins = []
        all_maxs = []
        
        for data_dict, exp_name in zip(data_dicts, exp_names):
            if param in data_dict:
                param_data = data_dict[param]
                row_data[f'{exp_name}_mean'] = param_data['mean']
                row_data[f'{exp_name}_std'] = param_data['std']
                # Calculate min and max using mean and std
                param_min = param_data['mean'] - 3 * param_data['std']
                param_max = param_data['mean'] + 3 * param_data['std']
                row_data[f'{exp_name}_min'] = param_min
                row_data[f'{exp_name}_max'] = param_max
                all_mins.append(param_min)
                all_maxs.append(param_max)
            else:
                param_exists_in_all = False
                row_data[f'{exp_name}_mean'] = None
                row_data[f'{exp_name}_std'] = None
                row_data[f'{exp_name}_min'] = None
                row_data[f'{exp_name}_max'] = None
        
        # Calculate overall overlap between all conditions if parameter exists in all
        if param_exists_in_all:
            # Calculate union and intersection across all conditions
            union_min = min(all_mins)
            union_max = max(all_maxs)
            intersection_min = max(all_mins)
            intersection_max = min(all_maxs)
            
            # Calculate overall overlap percentage
            if intersection_max > intersection_min:
                overlap_range = intersection_max - intersection_min
                total_range = union_max - union_min
                overall_overlap_percentage = (overlap_range / total_range) * 100
            else:
                overall_overlap_percentage = 0
            
            row_data['Overall_Overlap_Percentage'] = overall_overlap_percentage
            
            # Add interpretation of overall overlap
            if overall_overlap_percentage > 65:
                overlap_interpretation = 'High'
            elif overall_overlap_percentage > 35:
                overlap_interpretation = 'Medium'
            else:
                overlap_interpretation = 'Low'
            row_data['Overall_Overlap_Interpretation'] = overlap_interpretation
            
            # Calculate pairwise comparisons
            for i in range(len(data_dicts)):
                for j in range(i+1, len(data_dicts)):
                    # Calculate effect size (Cohen's d)
                    d = (data_dicts[i][param]['mean'] - data_dicts[j][param]['mean']) / np.sqrt(
                        (data_dicts[i][param]['std']**2 + data_dicts[j][param]['std']**2) / 2
                    )
                    row_data[f'Effect_Size_{exp_names[i]}_vs_{exp_names[j]}'] = abs(d)
                    
                    # Add interpretation of effect size
                    if abs(d) < 0.2:
                        effect_interpretation = 'Negligible'
                    elif abs(d) < 0.5:
                        effect_interpretation = 'Small'
                    elif abs(d) < 0.8:
                        effect_interpretation = 'Medium'
                    else:
                        effect_interpretation = 'Large'
                    row_data[f'Effect_Interpretation_{exp_names[i]}_vs_{exp_names[j]}'] = effect_interpretation
                    
                    # Calculate pairwise overlap
                    min1 = data_dicts[i][param]['mean'] - 3 * data_dicts[i][param]['std']
                    max1 = data_dicts[i][param]['mean'] + 3 * data_dicts[i][param]['std']
                    min2 = data_dicts[j][param]['mean'] - 3 * data_dicts[j][param]['std']
                    max2 = data_dicts[j][param]['mean'] + 3 * data_dicts[j][param]['std']
                    
                    union_min = min(min1, min2)
                    union_max = max(max1, max2)
                    intersection_min = max(min1, min2)
                    intersection_max = min(max1, max2)
                    
                    if intersection_max > intersection_min:
                        overlap_range = intersection_max - intersection_min
                        total_range = union_max - union_min
                        overlap_percentage = (overlap_range / total_range) * 100
                    else:
                        overlap_percentage = 0
                    
                    row_data[f'Overlap_Percentage_{exp_names[i]}_vs_{exp_names[j]}'] = overlap_percentage
        else:
            row_data['Overall_Overlap_Percentage'] = None
            row_data['Overall_Overlap_Interpretation'] = None
        
        stats_data.append(row_data)
    
    # Create DataFrame and save comprehensive CSV
    stats_df = pd.DataFrame(stats_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comprehensive CSV
    output_file = os.path.join(output_dir, 'six_way_comparison_table.csv')
    stats_df.to_csv(output_file, index=False)
    print(f"Six-way comparison table saved to {output_file}")
    
    # Create and save simplified version with only overall overlap information
    simplified_df = stats_df[['Parameter', 'Overall_Overlap_Percentage', 'Overall_Overlap_Interpretation']].copy()
    simplified_df = simplified_df.dropna()  # Remove parameters that don't exist in all conditions
    simplified_df = simplified_df.sort_values('Overall_Overlap_Percentage', ascending=False)  # Sort by overlap percentage
    
    # Round the overlap percentage to 2 decimal places
    simplified_df['Overall_Overlap_Percentage'] = simplified_df['Overall_Overlap_Percentage'].round(2)
    
    # Save simplified CSV
    output_file_simplified = os.path.join(output_dir, 'six_way_comparison_simplified.csv')
    simplified_df.to_csv(output_file_simplified, index=False)
    print(f"Simplified six-way comparison table saved to {output_file_simplified}")
    
    # Print summary of parameters with notable overall overlap
    print("\nParameters sorted by overall overlap percentage:")
    for _, row in simplified_df.iterrows():
        print(f"{row['Parameter']}:")
        print(f"  Overall overlap: {row['Overall_Overlap_Percentage']:.1f}% ({row['Overall_Overlap_Interpretation']})")
    
    return stats_df, simplified_df


def update_param(name, val, root):
    """Updates a single parameter in the XML tree."""
    xpath = name.replace(".", "/")
    elements = root.findall(f"./{xpath}")

    if len(elements) == 0:
        return
    elif len(elements) > 1:
        print(f"Warning: Found {len(elements)} matches for '{name}' in the XML. Updating the first one.")

    el = elements[0]
    
    val = float(val)
    value = str(val) 
    if ':' not in value:
        el.text = value
    else:
        t, u, v = value.split(':')
        el.set('type', t)
        el.set('units', u)
        el.text = v

def params_to_xml(params, template_xml, xml_out):
    """Writes a dictionary of parameters to an XML file based on a template."""
    tree = ET.parse(template_xml)
    root = tree.getroot()

    for p, val in params.items():
        update_param(p, val, root)

    tree.write(xml_out)

def generate_xml_from_average_params(data_dict, original_params_structure, template_xml_path, output_xml_path, top_n, exp_name):
    """
    Generate XML files from averaged parameters, using the template XML structure.
    """
    import xml.etree.ElementTree as ET
    import re
    
    print(f"[INFO] Using template: {template_xml_path}")
    
    # Parse the template XML
    tree = ET.parse(template_xml_path)
    root = tree.getroot()
    
    # Improved drug name recognition
    template_name = os.path.basename(template_xml_path)
    template_name_lower = template_name.lower()
    if "pi3k" in template_name_lower:
        drug_key = "pi3k"
    elif "mek" in template_name_lower:
        drug_key = "mek"
    elif "akt" in template_name_lower:
        drug_key = "akt"
    else:
        raise ValueError(f"Could not recognize drug name in template filename: {template_name}")
    drug_data = data_dict.get(drug_key, {})
    
    # Find the user_parameters section
    user_params = root.find('user_parameters')
    if user_params is None:
        raise ValueError("Could not find user_parameters section in template XML")
    
    # Add 3D setup parameters
    params_3d = {
        'use_3D_setup': {'type': 'bool', 'value': 'true'},
        'z_height': {'type': 'double', 'value': '40'},
        'sphere_distribution': {'type': 'bool', 'value': 'true'}
    }
    
    for param_name, param_info in params_3d.items():
        # Check if parameter already exists
        existing_param = user_params.find(f".//{param_name}")
        if existing_param is None:
            # Create new parameter element
            param = ET.SubElement(user_params, param_name)
            param.set('type', param_info['type'])
            param.set('units', 'dimensionless')
            param.set('description', '')
            param.text = param_info['value']
    
    # Update parameters from the data dictionary
    for base_name, stats in drug_data.items():
        # Handle intracellular and scaling parameters robustly
        if 'intracellular' in base_name or 'scaling' in base_name:
            # Always update under <user_parameters>
            param_tag = base_name.replace('user_parameters.', '')
            param_elem = user_params.find(param_tag)
            if param_elem is None:
                # Create the element if it doesn't exist
                param_elem = ET.SubElement(user_params, param_tag)
                param_elem.set('type', 'double')
                param_elem.set('units', 'min' if 'dt' in param_tag else 'dimensionless')
                param_elem.set('description', '')
            param_elem.text = str(stats['mean'])
            continue
        # Update other parameters as before
        update_param(base_name, stats['mean'], root)
    
    # Get the template directory and create output filename
    template_dir = os.path.dirname(template_xml_path)
    output_name = f"{os.path.splitext(template_name)[0]}_top{top_n}_averaged.xml"
    output_path = os.path.join(template_dir, output_name)
    
    print(f"[INFO] Writing output XML to: {output_path}")
    # Write the modified XML
    tree.write(output_path)
    print(f"[INFO] Generated averaged XML for {drug_key} at: {output_path}")


def main():
    ########################################################################################
    # Configuration
    ########################################################################################
    
    # Define experiment names
    pi3k_experiment_name = "PI3Ki_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen"
    mek_experiment_name = "MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen"
    akt_experiment_name = "AKTi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen"
    
    pi3kmek_syn_experiment_name = "synergy_sweep-pi3k_mek-1104-2212-18p_transient_delayed_uniform_5k_10p"
    pi3kmek_syn_PI3K_experiment_name = "synergy_sweep-pi3k_mek-1104-2212-18p_PI3K_transient_delayed_uniform_5k_10p"
    pi3kmek_syn_MEK_experiment_name = "synergy_sweep-pi3k_mek-1104-2212-18p_MEK_transient_delayed_uniform_5k_10p"
    
    aktmek_syn_experiment_name = "synergy_sweep-akt_mek-1204-1639-18p_transient_delayed_uniform_postdrug_RMSE_5k"
    aktmek_syn_AKT_experiment_name = "synergy_sweep-akt_mek-1104-2212-18p_AKT_transient_delayed_uniform_5k_singledrug"
    aktmek_syn_MEK_experiment_name = "synergy_sweep-akt_mek-1104-2212-18p_MEK_transient_delayed_uniform_5k_singledrug"
    
    top_n = "10p"
    
    # Define output directories
    output_dir_root = "results/comparing_top_distributions"
    output_dir_single_drug_plots = os.path.join(output_dir_root, "violin_plots_single_drug")
    output_dir_synergy_pi3kmek_plots = os.path.join(output_dir_root, "violin_plots_synergy_pi3kmek")
    output_dir_synergy_aktmek_plots = os.path.join(output_dir_root, "violin_plots_synergy_aktmek")
    output_dir_both_synergies_plots = os.path.join(output_dir_root, "violin_plots_both_synergies")
    output_dir_synergy_single_drug_pi3k_plots = os.path.join(output_dir_root, "violin_plots_synergy_single_drug_pi3k")
    output_dir_synergy_single_drug_mek_plots = os.path.join(output_dir_root, "violin_plots_synergy_single_drug_mek")
    output_dir_synergy_single_drug_akt_plots = os.path.join(output_dir_root, "violin_plots_synergy_single_drug_akt")
    output_dir_three_way_plots = os.path.join(output_dir_root, "three_way_comparisons")
    output_dir_tables = os.path.join(output_dir_root, "top_params_table")
    output_dir_pairplots = os.path.join(output_dir_root, "pairplots")
    output_dir_distribution_approaches = os.path.join(output_dir_root, "distribution_approaches")

    # Create all directories
    for d in [output_dir_root, output_dir_single_drug_plots, output_dir_synergy_pi3kmek_plots,
              output_dir_synergy_aktmek_plots, output_dir_both_synergies_plots,
              output_dir_synergy_single_drug_pi3k_plots, output_dir_synergy_single_drug_mek_plots,
              output_dir_synergy_single_drug_akt_plots, output_dir_three_way_plots, output_dir_tables,
              output_dir_pairplots, output_dir_distribution_approaches]:
        os.makedirs(d, exist_ok=True)

    ########################################################################################
    # Data Loading
    ########################################################################################
    
    # Load single-drug data
    data_pi3k_singledrug = get_experiment_json(pi3k_experiment_name, top_n)
    data_mek_singledrug = get_experiment_json(mek_experiment_name, top_n)
    data_akt_singledrug = get_experiment_json(akt_experiment_name, top_n)
    
    # Load PI3K-MEK synergy data
    data_pi3kmek_syn = get_experiment_json(pi3kmek_syn_experiment_name, top_n)
    data_pi3kmek_syn_PI3K = get_experiment_json(pi3kmek_syn_PI3K_experiment_name, top_n)
    data_pi3kmek_syn_MEK = get_experiment_json(pi3kmek_syn_MEK_experiment_name, top_n)
    
    # Load AKT-MEK synergy data
    data_aktmek_syn = get_experiment_json(aktmek_syn_experiment_name, top_n)
    data_aktmek_syn_AKT = get_experiment_json(aktmek_syn_AKT_experiment_name, top_n)
    data_aktmek_syn_MEK = get_experiment_json(aktmek_syn_MEK_experiment_name, top_n)

    # Load original parameter structure for combined distributions
    with open('data/JSON/deap/deap_18p_single_drug_exp_v2.json', 'r') as f:
        original_params = json.load(f)

    #######################################################################################################
    # Generate XML from average top 10% parameters for PI3K single-drug experiment
    #######################################################################################################
    
    print("Generating XML files from averaged parameters for PI3K single-drug experiment")


    # Generate XML files from averaged parameters
    pi3k_template_xml = 'data/physiboss_config/dose_curves_experiments/PI3Ki_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_top_10p_averaged.xml'
    generate_xml_from_average_params(
        data_dict={'pi3k': data_pi3k_singledrug, 'mek': data_mek_singledrug, 'akt': data_akt_singledrug},
        original_params_structure=original_params,
        template_xml_path=pi3k_template_xml,
        output_xml_path=None,
        top_n=top_n,
        exp_name=pi3k_experiment_name
    )

    mek_template_xml = 'data/physiboss_config/dose_curves_experiments/MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_top_10p_averaged.xml'
    generate_xml_from_average_params(
        data_dict={'pi3k': data_pi3k_singledrug, 'mek': data_mek_singledrug, 'akt': data_akt_singledrug},
        original_params_structure=original_params,
        template_xml_path=mek_template_xml,
        output_xml_path=None,
        top_n=top_n,
        exp_name=mek_experiment_name
    )

    akt_template_xml = 'data/physiboss_config/dose_curves_experiments/AKTi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_top_10p_averaged.xml'
    generate_xml_from_average_params(
        data_dict={'pi3k': data_pi3k_singledrug, 'mek': data_mek_singledrug, 'akt': data_akt_singledrug},
        original_params_structure=original_params,
        template_xml_path=akt_template_xml,
        output_xml_path=None,
        top_n=top_n,
        exp_name=akt_experiment_name
    )

if __name__ == "__main__":
    main()