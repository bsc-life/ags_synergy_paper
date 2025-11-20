import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from functools import partial
import gc
import logging

# Set the maximum number of open figures before a warning is issued
plt.rcParams['figure.max_open_warning'] = 0  # Suppress the warning
# Use a faster backend for matplotlib
plt.switch_backend('agg')

# Global cache for processed data
DATA_CACHE = {}

# NOTE: All plots in this script have been updated to use compact proportions (1.67 x 2.5 inches) for journal-style figures.
# This ensures consistency with other scripts and is suitable for multi-panel layouts in publications.
#
# For two-panel plots, the width should be doubled (3.34 x 2.5 inches).
#
# Font sizes, line widths, and marker sizes have also been adjusted for compactness and clarity.

# Colorblind-friendly and Nature-approved palette (see: https://www.nceas.ucsb.edu/sites/default/files/2022-06/Colorblind%20Safe%20Color%20Schemes.pdf)
CB_PALETTE = {
    'blue':   '#0072B2',
    'orange': '#E69F00',
    'green':  '#009E73',
    'red':    '#D55E00',
    'purple': '#CC79A7',
    'yellow': '#F0E442',
    'black':  '#000000',
    'grey':   '#999999',
}

def get_experiment_dynamics(experiment_name, strategy, top_n=10):
    """
    Loads and processes the top N simulation runs for a given experiment
    and returns aggregated dynamics data.
    """
    logging.info(f"--- Loading data for: {experiment_name} ---")

    # 1. Find the top_n.csv file
    summaries_folder = f"results/{strategy}_summaries"
    exp_summary_dir = os.path.join(summaries_folder, f"final_summary_{experiment_name}")
    top_n_path = None

    if os.path.isdir(exp_summary_dir):
        for f in os.listdir(exp_summary_dir):
            if f.endswith(f"top_{top_n}.csv"):
                top_n_path = os.path.join(exp_summary_dir, f)
                logging.info(f"Found data file: {top_n_path}")
                break
    
    if not top_n_path:
        logging.warning(f"Could not find top_{top_n}.csv for {experiment_name}. Skipping.")
        return None

    top_n_df = pd.read_csv(top_n_path)
    
    # 2. Process each run to get dynamics data
    dataframes = {'apoptotic_live': [], 'cell_rates': [], 'cell_signals': [], 'node_states': []}
    experiment_folder = f"experiments/{experiment_name}"

    for _, row in top_n_df.iterrows():
        # Build path to instance data
        instance_folder_parts = []
        if "iteration" in row.index:
            instance_folder_parts.append(str(int(row['iteration'])))
        instance_folder_parts.append(str(int(row['individual'])))
        instance_folder_parts.append(str(int(row['replicate'])))
        instance_folder = f"instance_{'_'.join(instance_folder_parts)}"
        full_path = os.path.join(experiment_folder, instance_folder, 'pcdl_total_info_sim.csv.gz')

        if os.path.exists(full_path):
            try:
                # Use caching if possible
                cache_key = f"{full_path}_{os.path.getmtime(full_path)}"
                if cache_key in DATA_CACHE:
                    cached_data = DATA_CACHE[cache_key]
                else:
                    raw_df = pd.read_csv(full_path, compression='gzip')
                    cached_data = {
                        'apoptotic_live': get_alive_apoptotic(raw_df),
                        'cell_rates': get_cell_rates(raw_df),
                        'cell_signals': get_cell_signals(raw_df),
                        'node_states': get_node_states(raw_df)
                    }
                    DATA_CACHE[cache_key] = cached_data
                    del raw_df
                    gc.collect()

                dataframes['apoptotic_live'].append(cached_data['apoptotic_live'])
                dataframes['cell_rates'].append(cached_data['cell_rates'])
                dataframes['cell_signals'].append(cached_data['cell_signals'])
                if 'node_states' in cached_data and not cached_data['node_states'].empty:
                    dataframes['node_states'].append(cached_data['node_states'])
            except Exception as e:
                logging.error(f"Error processing {full_path}: {e}")
        else:
            logging.warning(f"File not found: {full_path}")
            
    if not dataframes['apoptotic_live']:
        logging.warning(f"No valid simulation data found for {experiment_name}.")
        return None

    # 3. Aggregate the data across all runs
    combined_counts = pd.concat(dataframes['apoptotic_live'], ignore_index=True)
    stats_counts = combined_counts.groupby('time').agg({
        'alive': ['mean', 'std'], 'apoptotic': ['mean', 'std']
    }).reset_index()
    stats_counts.columns = ['time', 'alive_mean', 'alive_std', 'apoptotic_mean', 'apoptotic_std']

    combined_rates = pd.concat(dataframes['cell_rates'], ignore_index=True)
    stats_rates = combined_rates.groupby(['time', 'rate_type']).agg(
        value_mean=('value', 'mean'), value_std=('value', 'std')
    ).reset_index()

    combined_signals = pd.concat(dataframes['cell_signals'], ignore_index=True)
    stats_signals = combined_signals.groupby(['time', 'signal_type']).agg(
        value_mean=('value', 'mean'), value_std=('value', 'std')
    ).reset_index()

    if not dataframes['node_states']:
        stats_nodes = None
    else:
        combined_nodes = pd.concat(dataframes['node_states'], ignore_index=True)
        stats_nodes = combined_nodes.groupby(['time', 'node_type']).agg(
            value_mean=('value', 'mean'), value_std=('value', 'std')
        ).reset_index()

    return {
        'alive_apoptotic': stats_counts,
        'cell_rates': stats_rates,
        'cell_signals': stats_signals,
        'node_states': stats_nodes
    }

def plot_single_drug_grid(pi3k_data, akt_data, mek_data):
    """
    Creates a 2x2 grid comparing single-drug calibration dynamics for
    PI3Ki, AKTi, and MEKi.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), dpi=300, sharex=True, sharey=True)
    
    datasets = {
        'PI3K inhibitor': pi3k_data,
        'MEK inhibitor': mek_data,
        'AKT inhibitor': akt_data
    }
    
    ax_map = {
        'PI3K inhibitor': axes[0, 0],
        'MEK inhibitor': axes[0, 1],
        'AKT inhibitor': axes[1, 0]
    }

    lines = []
    labels = []

    # Note: This function currently plots only simulated data as experimental data is not available here.
    # The legend has been formatted to reflect this.
    for title, data in datasets.items():
        if data is None:
            ax_map[title].set_title(f"{title}\n(Data not found)", fontsize=11)
            ax_map[title].axis('off')
            continue
        
        ax = ax_map[title]
        data_df = data['alive_apoptotic']
        
        alive_line, = ax.plot(data_df['time'], data_df['alive_mean'], color=CB_PALETTE['green'], label='Simulated Alive', linewidth=2.0)
        ax.fill_between(data_df['time'], data_df['alive_mean'] - data_df['alive_std'], data_df['alive_mean'] + data_df['alive_std'], color=CB_PALETTE['green'], alpha=0.2)
        
        apop_line, = ax.plot(data_df['time'], data_df['apoptotic_mean'], color=CB_PALETTE['red'], label='Simulated Apoptotic', linewidth=2.0)
        ax.fill_between(data_df['time'], data_df['apoptotic_mean'] - data_df['apoptotic_std'], data_df['apoptotic_mean'] + data_df['apoptotic_std'], color=CB_PALETTE['red'], alpha=0.2)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axvspan(1280, 1292, color=CB_PALETTE['grey'], alpha=1.0, zorder=0)
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
        ax.set_xlim(right=4200)

    # Use the lines from the last valid plot for the legend
    if alive_line:
        lines.extend([alive_line, apop_line])
        labels.extend(['Simulated Alive', 'Simulated Apoptotic'])

    # Common X and Y labels
    fig.text(0.5, 0.02, 'Time (min)', ha='center', va='center', fontsize=12, fontweight="bold")
    fig.text(0.02, 0.5, 'Cell Count', ha='center', va='center', rotation='vertical', fontsize=12, fontweight="bold")

    # Turn off the empty subplot
    axes[1, 1].axis('off')
    
    # Add a unified legend below the plots, anchored to the empty subplot space
    fig.legend(lines, labels, loc='lower right', bbox_to_anchor=(0.95, 0.1), ncol=1, frameon=True, fontsize=10)

    plt.tight_layout(rect=[0.04, 0.04, 1, 1])
    
    save_dir = "results/comparison_grids"
    os.makedirs(save_dir, exist_ok=True)
    base_save_path = os.path.join(save_dir, "single_drug_calibration_grid")
    
    plt.savefig(f"{base_save_path}.png", dpi=300)
    plt.savefig(f"{base_save_path}.svg", format='svg')
    
    plt.close(fig)
    logging.info(f"Saved single-drug calibration grid to {base_save_path}.png/.svg")

def plot_combination_comparison(pi3k_mek_data, akt_mek_data):
    """
    Creates a side-by-side comparison of the combination therapies.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300, sharey=True)

    datasets = {
        'PI3Ki + MEKi': pi3k_mek_data,
        'AKTi + MEKi': akt_mek_data
    }

    for i, (title, data) in enumerate(datasets.items()):
        ax = axes[i]
        if data is None:
            ax.set_title(f"{title}\n(Data not found)", fontsize=11)
            ax.axis('off')
            continue
        
        data_df = data['alive_apoptotic']

        ax.plot(data_df['time'], data_df['alive_mean'], color=CB_PALETTE['green'], label='Alive', linewidth=2.0)
        ax.fill_between(data_df['time'], data_df['alive_mean'] - data_df['alive_std'], data_df['alive_mean'] + data_df['alive_std'], color=CB_PALETTE['green'], alpha=0.2)
        
        ax.plot(data_df['time'], data_df['apoptotic_mean'], color=CB_PALETTE['red'], label='Apoptotic', linewidth=2.0)
        ax.fill_between(data_df['time'], data_df['apoptotic_mean'] - data_df['apoptotic_std'], data_df['apoptotic_mean'] + data_df['apoptotic_std'], color=CB_PALETTE['red'], alpha=0.2)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axvspan(1280, 1292, color=CB_PALETTE['grey'], alpha=1.0, zorder=0)
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
        ax.set_xlim(right=4200)
        ax.set_xlabel('Time (min)', fontsize=10, fontweight="bold")
        ax.legend()

    axes[0].set_ylabel('Cell Count', fontsize=10, fontweight="bold")

    plt.tight_layout()
    
    save_dir = "results/comparison_grids"
    os.makedirs(save_dir, exist_ok=True)
    base_save_path = os.path.join(save_dir, "combination_therapy_comparison")
    
    plt.savefig(f"{base_save_path}.png", dpi=300)
    plt.savefig(f"{base_save_path}.svg", format='svg')
    
    plt.close(fig)
    logging.info(f"Saved combination therapy comparison to {base_save_path}.png/.svg")

def plot_top_n_comparison_grid(all_data, output_path):
    """
    Creates a 3x5 summary grid plot for the top N simulation dynamics across all experiments.
    Rows are dynamics types, columns are different experiments.
    Legends are only shown on the rightmost plot of each row.
    """
    row_keys = ['cell_signals', 'cell_rates', 'alive_apoptotic']
    
    # Define the order and titles for the columns
    col_order = ['pi3k', 'mek', 'akt', 'pi3k_mek', 'akt_mek']
    col_titles = ['PI3Ki', 'MEKi', 'AKTi', 'PI3Ki + MEKi', 'AKTi + MEKi']

    # Set figsize to create square plots (width/cols == height/rows)
    fig, axes = plt.subplots(3, 5, figsize=(12.5, 7.5), dpi=300, sharex=True, sharey='row')

    def _plot_on_ax(ax, data, plot_info, plot_key):
        """Helper to plot data on a given axes object."""
        lines = []
        
        # Filter time=0 data for dynamics plots (rates, signals)
        if plot_key in ['cell_rates', 'cell_signals']:
            data = data[data['time'] != 0].copy()
            
        for col, props in plot_info['series'].items():
            if props['mean_col'] in data.columns:
                line, = ax.plot(data['time'], data[props['mean_col']], label=props.get('label', col), color=props['color'], linewidth=2.0)
                ax.fill_between(data['time'], 
                                data[props['mean_col']] - data[props['std_col']],
                                data[props['mean_col']] + data[props['std_col']],
                                color=props['color'], alpha=0.2, linewidth=1.5)
                lines.append(line)
                
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
        ax.axvspan(1280, 1292, color=CB_PALETTE['grey'], alpha=1.0, zorder=0) # Drug addition visualization
        
        if plot_info.get('ylim'):
            ax.set_ylim(plot_info['ylim'])
            
        return lines

    # Define configurations for each row of plots
    plot_configs = {
        'alive_apoptotic': {
            'ylabel': "Cell Count",
            'series': {
                'live': {'label': 'Alive', 'color': CB_PALETTE['green'], 'mean_col': 'alive_mean', 'std_col': 'alive_std'},
                'apoptotic': {'label': 'Apoptotic', 'color': CB_PALETTE['red'], 'mean_col': 'apoptotic_mean', 'std_col': 'apoptotic_std'}
            }
        },
        'cell_rates': {
            'ylabel': "Rate (1/min)", 'ylim': (0, 0.0012),
            'series': {
                'growth': {'label': 'Growth', 'color': CB_PALETTE['green'], 'mean_col': 'value_mean', 'std_col': 'value_std'},
                'apoptosis': {'label': 'Apoptosis', 'color': CB_PALETTE['red'], 'mean_col': 'value_mean', 'std_col': 'value_std'}
            }
        },
        'cell_signals': {
            'ylabel': "Signal Value",
            'series': {
                'pro': {'label': 'Pro-survival', 'color': CB_PALETTE['green'], 'mean_col': 'value_mean', 'std_col': 'value_std'},
                'anti': {'label': 'Anti-survival', 'color': CB_PALETTE['red'], 'mean_col': 'value_mean', 'std_col': 'value_std'}
            }
        }
    }
    
    # Map our internal data keys to the plot configurations
    data_map = {
        'cell_signals': 'cell_signals',
        'cell_rates': 'cell_rates', 
        'alive_apoptotic': 'alive_apoptotic'
    }

    # Iterate through the grid and plot
    for i, key in enumerate(row_keys):
        plot_info = plot_configs[key]
        axes[i, 0].set_ylabel(plot_info['ylabel'], fontweight='bold', fontsize=10)
        
        for j, case_key in enumerate(col_order):
            ax = axes[i, j]
            
            # Set column titles on the first row
            if i == 0:
                ax.set_title(col_titles[j], fontweight='bold', fontsize=11)

            # Check if data for this experiment exists
            if case_key not in all_data or all_data[case_key] is None:
                ax.text(0.5, 0.5, "Data Not\nFound", ha='center', va='center', fontsize=9, alpha=0.5)
                ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
                continue
            
            data_container = all_data[case_key]
            data_df = data_container[data_map[key]]

            # Special handling for cell_rates and cell_signals dataframes
            if key in ['cell_rates', 'cell_signals']:
                rate_type_map = {
                    'cell_rates': {'growth': 'growth_rate_mean', 'apoptosis': 'apoptosis_rate_mean'},
                    'cell_signals': {'pro': 'S_pro_real_mean', 'anti': 'S_anti_real_mean'}
                }

                if key == 'cell_rates':
                    series1_key, series2_key = 'growth', 'apoptosis'
                    col_name = 'rate_type'
                else:  # key == 'cell_signals'
                    series1_key, series2_key = 'pro', 'anti'
                    col_name = 'signal_type'
                
                series1_df = data_df[data_df[col_name] == rate_type_map[key][series1_key]]
                series2_df = data_df[data_df[col_name] == rate_type_map[key][series2_key]]
                
                lines1 = _plot_on_ax(ax, series1_df, {'series': {series1_key: plot_info['series'][series1_key]}}, key)
                lines2 = _plot_on_ax(ax, series2_df, {'series': {series2_key: plot_info['series'][series2_key]}}, key)
                lines = lines1 + lines2
            else:
                lines = _plot_on_ax(ax, data_df, plot_info, key)
            
            # Only show legend on the right-most plot
            if j == len(col_order) - 1 and lines:
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, fontsize=8, loc='best')
            
            # Set common x-label on the last row
            if i == len(row_keys) - 1:
                ax.set_xlabel("Time (min)", fontsize=10, fontweight='bold')
            
            ax.tick_params(axis='both', labelsize=9)
            ax.set_xlim(right=4200)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved top N comparison grid plot to {output_path}")

def plot_survival_nodes_grid(all_data, output_path):
    """
    Creates a 2x5 summary grid plot for pro- and anti-survival node dynamics.
    """
    col_order = ['pi3k', 'mek', 'akt', 'pi3k_mek', 'akt_mek']
    col_titles = ['PI3Ki', 'MEKi', 'AKTi', 'PI3Ki + MEKi', 'AKTi + MEKi']

    fig, axes = plt.subplots(2, 5, figsize=(12.5, 5), dpi=300, sharex=True, sharey=True)

    pro_survival_nodes = {
        "node_cMYC_mean": {'label': 'cMYC', 'color': CB_PALETTE['blue']},
        "node_RSK_mean": {'label': 'RSK', 'color': CB_PALETTE['green']},
        "node_TCF_mean": {'label': 'TCF', 'color': CB_PALETTE['orange']}
    }
    
    anti_survival_nodes = {
        "node_FOXO_mean": {'label': 'FOXO', 'color': CB_PALETTE['red']},
        "node_Caspase8_mean": {'label': 'Caspase8', 'color': CB_PALETTE['purple']},
        "node_Caspase9_mean": {'label': 'Caspase9', 'color': CB_PALETTE['grey']}
    }

    node_sets = [pro_survival_nodes, anti_survival_nodes]
    y_labels = ["Pro-survival Signal", "Anti-survival Signal"]

    for i, node_set in enumerate(node_sets):
        axes[i, 0].set_ylabel(y_labels[i], fontweight='bold', fontsize=10)
        
        for j, case_key in enumerate(col_order):
            ax = axes[i, j]
            
            if i == 0:
                ax.set_title(col_titles[j], fontweight='bold', fontsize=11)
            
            if case_key not in all_data or all_data[case_key] is None or all_data[case_key].get('node_states') is None:
                ax.text(0.5, 0.5, "Data Not\nFound", ha='center', va='center', fontsize=9, alpha=0.5)
                ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
                continue
            
            data_df = all_data[case_key]['node_states']
            data_df = data_df[data_df['time'] != 0].copy()

            lines = []
            for node_name, props in node_set.items():
                node_data = data_df[data_df['node_type'] == node_name]
                if not node_data.empty:
                    line, = ax.plot(node_data['time'], node_data['value_mean'], label=props['label'], color=props['color'], linewidth=2.0)
                    ax.fill_between(node_data['time'],
                                    node_data['value_mean'] - node_data['value_std'],
                                    node_data['value_mean'] + node_data['value_std'],
                                    color=props['color'], alpha=0.2, linewidth=1.5)
                    lines.append(line)

            ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
            ax.axvspan(1280, 1292, color=CB_PALETTE['grey'], alpha=1.0, zorder=0)
            ax.set_ylim(0, 1)

            if j == len(col_order) - 1 and lines:
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, fontsize=8, loc='best')

            if i == len(node_sets) - 1:
                ax.set_xlabel("Time (min)", fontsize=10, fontweight='bold')
            
            ax.tick_params(axis='both', labelsize=9)
            ax.set_xlim(right=4200)

    plt.tight_layout(rect=[0.04, 0.04, 1, 0.95])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved survival nodes grid plot to {output_path}")

def process_top_10(top_10_df, experiment_name):
    """
    Optimized version that checks for existing files and caches data
    """
    print(f"\nProcessing experiment: {experiment_name}")
    print(f"DataFrame columns: {top_10_df.columns.tolist()}")
    print(f"DataFrame shape: {top_10_df.shape}")
    
    # Define output directories to check if all files exist
    output_dirs = {
        'alive_apoptotic': os.path.join('results', 'alive_apoptotic_plots', f'alive_apoptotic_plot_{experiment_name}.png'),
        'node_states_full': os.path.join('results', 'node_states_plots', f'node_states_plot_full_{experiment_name}.png'),
        'anti_node_states': os.path.join('results', 'node_states_plots', f'anti_node_states_plot_{experiment_name}.png'),
        'cell_rates': os.path.join('results', 'cell_rates_plots', f'cell_rates_plot_{experiment_name}.png'),
        'cell_signals': os.path.join('results', 'cell_signals_plots', f'cell_signals_plot_{experiment_name}.png')
    }
    
    # Check which plots need to be generated
    need_apoptotic_live = not os.path.exists(output_dirs['alive_apoptotic'])
    need_node_states = not os.path.exists(output_dirs['node_states_full'])
    need_anti_node_states = not os.path.exists(output_dirs['anti_node_states'])
    need_cell_rates = not os.path.exists(output_dirs['cell_rates'])
    need_cell_signals = not os.path.exists(output_dirs['cell_signals'])
    
    # Create output directories if they don't exist
    for path in [os.path.dirname(f) for f in output_dirs.values()]:
        os.makedirs(path, exist_ok=True)
    
    # Get the experiment folder from the csv path
    experiment_folder = os.path.join("experiments", experiment_name)
    
    # Initialize data containers
    dataframes = {
        'apoptotic_live': [],
        'node_states': [],
        'cell_rates': [],
        'cell_signals': []
    }
    
    # Process data for each instance
    instance_data = {
        'node_states': [],
        'cell_rates': [],
        'cell_signals': []
    }
    
    # Check if this is a drug treatment experiment by looking for drug-related columns
    is_drug_experiment = any(col.startswith('drug_') for col in top_10_df.columns)
    
    # Process regular plots if needed and if this is a drug experiment
    if is_drug_experiment and any([need_apoptotic_live, need_node_states, need_anti_node_states, need_cell_rates, need_cell_signals]):
        # Process each row in the top 10 CSV
        for _, row in top_10_df.iterrows():
            if "iteration" in row.index:
                instance_folder = f"instance_{int(row['iteration'])}_{int(row['individual'])}_{int(row['replicate'])}"
            else:
                instance_folder = f"instance_{int(row['individual'])}_{int(row['replicate'])}"
                
            full_path = os.path.join(experiment_folder, instance_folder, 'pcdl_total_info_sim.csv.gz')
            
            # Check if file exists and process it
            if os.path.exists(full_path):
                # Use cache key based on file path and modification time
                cache_key = f"{full_path}_{os.path.getmtime(full_path)}"
                
                # Check if data is in cache
                if cache_key in DATA_CACHE:
                    cached_data = DATA_CACHE[cache_key]
                    
                    if need_apoptotic_live and 'apoptotic_live' in cached_data:
                        dataframes['apoptotic_live'].append(cached_data['apoptotic_live'])
                    if (need_node_states or need_anti_node_states) and 'node_states' in cached_data:
                        instance_data['node_states'].append(cached_data['node_states'])
                    if need_cell_rates and 'cell_rates' in cached_data:
                        instance_data['cell_rates'].append(cached_data['cell_rates'])
                    if need_cell_signals and 'cell_signals' in cached_data:
                        instance_data['cell_signals'].append(cached_data['cell_signals'])
                else:
                    # Read file with optimized settings
                    # - Use only needed columns
                    # - Use dtype specifications for efficiency
                    # - Use optimized chunk processing for large files
                    
                    # List of columns we need for different analyses
                    needed_columns = []
                    if need_apoptotic_live:
                        needed_columns.extend(['current_phase', 'time'])
                    if need_node_states or need_anti_node_states:
                        needed_columns.extend(['current_phase', 'time'])
                        # Add node columns (can't specify exact names as they vary)
                    if need_cell_rates:
                        needed_columns.extend(['time', 'apoptosis_rate', 'growth_rate'])
                    if need_cell_signals:
                        needed_columns.extend(['time', 'S_pro_real', 'S_anti_real'])
                    
                    # Read file (allowing for all columns if we need node states)
                    if need_node_states or need_anti_node_states:
                        full_path_df = pd.read_csv(full_path, compression='gzip')
                    else:
                        full_path_df = pd.read_csv(full_path, 
                                                 usecols=lambda col: col in needed_columns,
                                                 compression='gzip')

                    # Process data as needed and store in cache
                    cached_data = {}

                    if need_apoptotic_live:
                        instance_apoptotic_live_df = get_alive_apoptotic(full_path_df)
                        dataframes['apoptotic_live'].append(instance_apoptotic_live_df)
                        cached_data['apoptotic_live'] = instance_apoptotic_live_df
                    
                    if need_node_states or need_anti_node_states:
                        instance_node_states_df = get_node_states(full_path_df)
                        instance_data['node_states'].append(instance_node_states_df)
                        cached_data['node_states'] = instance_node_states_df
                    
                    if need_cell_rates:
                        instance_cell_rates_df = get_cell_rates(full_path_df)
                        instance_data['cell_rates'].append(instance_cell_rates_df)
                        cached_data['cell_rates'] = instance_cell_rates_df
                
                    if need_cell_signals:
                        instance_cell_signals_df = get_cell_signals(full_path_df)
                        instance_data['cell_signals'].append(instance_cell_signals_df)
                        cached_data['cell_signals'] = instance_cell_signals_df
                    
                    # Store in cache
                    DATA_CACHE[cache_key] = cached_data
                    
                    # Free memory
                    del full_path_df
                    gc.collect()
    
    # Generate plots in parallel if possible
    plot_tasks = []
    
    if need_apoptotic_live and dataframes['apoptotic_live']:
        combined_df = pd.concat(dataframes['apoptotic_live'], ignore_index=True)
        plot_top_10_alive_apoptotic(combined_df, experiment_name)
    
    if (need_node_states or need_anti_node_states) and instance_data['node_states']:
        # Combine instances
        all_nodes_df = pd.concat(instance_data['node_states'])
        # Now aggregate across instances by time and node_type - FIXED
        aggregated_nodes = all_nodes_df.groupby(['time', 'node_type']).agg({
            'value': ['mean', 'std']  # Calculate mean and std directly
        }).reset_index()
        
        # Flatten column names
        aggregated_nodes.columns = ['time', 'node_type', 'value', 'std']
        
        # Generate both node state plots if needed
        if need_node_states:
            plot_top_10_node_states(aggregated_nodes, experiment_name)
        
        if need_anti_node_states:
            plot_top_10_anti_node_states(aggregated_nodes, experiment_name)
    
    if need_cell_rates and instance_data['cell_rates']:
        # Combine instances
        all_rates_df = pd.concat(instance_data['cell_rates'])
        # Now aggregate across instances by time and rate_type - FIXED
        aggregated_rates = all_rates_df.groupby(['time', 'rate_type']).agg({
            'value': ['mean', 'std']  # Calculate mean and std directly
        }).reset_index()
        
        # Flatten column names
        aggregated_rates.columns = ['time', 'rate_type', 'value', 'std']
        plot_top_10_cell_rates(aggregated_rates, experiment_name)
    
    
    if need_cell_signals and instance_data['cell_signals']:
        # Combine instances
        all_signals_df = pd.concat(instance_data['cell_signals'])
        # Now aggregate across instances by time and signal_type - FIXED
        aggregated_signals = all_signals_df.groupby(['time', 'signal_type']).agg({
            'value': ['mean', 'std']  # Calculate mean and std directly
        }).reset_index()
        
        # Flatten column names
        aggregated_signals.columns = ['time', 'signal_type', 'value', 'std']
        plot_top_10_cell_signals(aggregated_signals, experiment_name)
    
    # Clear data frames to free memory
    for key in dataframes:
        dataframes[key].clear()
    
    # --- Process node weights plots independently ---
    # Check if we have the required columns for weight plots
    weight_cols = ["w_pro_cMYC", "w_pro_RSK", "w_pro_TCF", "w_anti_Caspase8", "w_anti_Caspase9", "w_anti_FOXO"]
    has_weight_cols = all(col in top_10_df.columns for col in weight_cols)
    
    print(f"\nChecking weight columns for {experiment_name}:")
    print(f"Required columns: {weight_cols}")
    print(f"Has all required columns: {has_weight_cols}")
    if not has_weight_cols:
        missing = [col for col in weight_cols if col not in top_10_df.columns]
        print(f"Missing columns: {missing}")
    
    if has_weight_cols:
        # Check if weight plots already exist
        pro_weights_file = os.path.join('results', 'node_weights_plots', f'prosurvival_weights_boxplot_{experiment_name}.png')
        anti_weights_file = os.path.join('results', 'node_weights_plots', f'antisurvival_weights_boxplot_{experiment_name}.png')
        
        need_weight_plots = not (os.path.exists(pro_weights_file) and os.path.exists(anti_weights_file))
        
        print(f"Need weight plots: {need_weight_plots}")
        print(f"Pro weights file exists: {os.path.exists(pro_weights_file)}")
        print(f"Anti weights file exists: {os.path.exists(anti_weights_file)}")
        
        if need_weight_plots:
            print(f"Generating weight plots for {experiment_name}")
            os.makedirs(os.path.join('results', 'node_weights_plots'), exist_ok=True)
            plot_prosurvival_weights_boxplot(top_10_df, experiment_name)
            plot_antisurvival_weights_boxplot(top_10_df, experiment_name)
        else:
            print(f"Weight plots for {experiment_name} already exist, skipping...")
    else:
        print(f"Skipping weight plots for {experiment_name} - missing required columns")
    
    gc.collect()
    return "OK"

# Optimize data extraction functions

def get_alive_apoptotic(simulation_df):
    """Optimized version that uses more efficient pandas operations"""
    # Extract only the columns we need
    alive_apoptotic_df = simulation_df[['current_phase', 'time']]
    
    # Use value_counts for better performance than groupby
    time_values = alive_apoptotic_df['time'].unique()
    alive_counts = pd.Series(0, index=time_values, name='alive')
    apoptotic_counts = pd.Series(0, index=time_values, name='apoptotic')
    
    # Count phases more efficiently
    phases_by_time = alive_apoptotic_df.groupby(['time', 'current_phase']).size().unstack(fill_value=0)
    
    # Derive counts
    if 'apoptotic' in phases_by_time.columns:
        apoptotic_counts = phases_by_time['apoptotic']
    else:
        apoptotic_counts = pd.Series(0, index=time_values, name='apoptotic')

    # Live cells = total cells minus apoptotic (or sum of all non-apoptotic phases).
    non_apoptotic_cols = [c for c in phases_by_time.columns if c != 'apoptotic']
    if non_apoptotic_cols:
        alive_counts = phases_by_time[non_apoptotic_cols].sum(axis=1)
    else:
        # Fallback to zeros if no columns present (should not happen)
        alive_counts = pd.Series(0, index=time_values, name='alive')
    
    # Create dataframe
    combined_df = pd.DataFrame({
        'time': time_values,
        'alive': alive_counts.values,
        'apoptotic': apoptotic_counts.values
    })
    
    return combined_df

def get_node_states(simulation_df):
    """Properly aggregates node states across cells within one instance"""
    # Filter for relevant rows only
    df = simulation_df[simulation_df['current_phase'] != 'alive']
    
    # Select node columns
    node_cols = [col for col in df.columns if 'node' in col and 'anti' not in col]
    
    # Group by time and calculate statistics for each node
    result_rows = []
    
    for t in df['time'].unique():
        time_slice = df[df['time'] == t]
        
        for col in node_cols:
            mean_val = time_slice[col].mean()
            
            result_rows.append({
                'time': t,
                'node_type': col + '_mean',
                'value': mean_val
            })
    
    return pd.DataFrame(result_rows)

def get_cell_rates(simulation_df):
    """Properly aggregates cell rates within one instance"""
    rate_cols = ['apoptosis_rate', 'growth_rate']
    result_rows = []
    
    # Group by time and calculate statistics for each rate
    for t in simulation_df['time'].unique():
        time_slice = simulation_df[simulation_df['time'] == t]
        
        for col in rate_cols:
            mean_val = time_slice[col].mean()
            
            result_rows.append({
                'time': t,
                'rate_type': col + '_mean', 
                'value': mean_val
            })
    
    return pd.DataFrame(result_rows)

def get_cell_signals(simulation_df):
    """Properly aggregates cell signals within one instance"""
    signal_cols = ['S_pro_real', 'S_anti_real']
    result_rows = []
    
    # Group by time and calculate statistics for each signal
    for t in simulation_df['time'].unique():
        time_slice = simulation_df[simulation_df['time'] == t]
        
        for col in signal_cols:
            mean_val = time_slice[col].mean()
            
            result_rows.append({
                'time': t,
                'signal_type': col + '_mean',
                'value': mean_val
            })
    
    return pd.DataFrame(result_rows)

# Modified plotting functions with file existence checking

def plot_top_10_alive_apoptotic(top_10_combined_df, experiment_name):
    """
    Plot the mean and standard deviation of alive and apoptotic cells over time for the top 10 parameter sets.
    Figure size: (2.5, 2.5) inches for square, journal-style layout.
    Uses colorblind-friendly and Nature-approved palette.
    """
    output_file = os.path.join('results', 'alive_apoptotic_plots', f'alive_apoptotic_plot_{experiment_name}.png')
    output_file_svg = os.path.join('results', 'alive_apoptotic_plots', f'alive_apoptotic_plot_{experiment_name}.svg')
    legend_file_svg = os.path.join('results', 'alive_apoptotic_plots', f'alive_apoptotic_legend_{experiment_name}.svg')
    if os.path.exists(output_file) and os.path.exists(output_file_svg) and os.path.exists(legend_file_svg):
        print(f"Alive/apoptotic plot for {experiment_name} already exists, skipping...")
        return

    print(f"Plotting alive/apoptotic cells for {experiment_name}")
    
    # Calculate statistics across instances for each timepoint
    stats_df = top_10_combined_df.groupby('time').agg({
        'alive': ['mean', 'std'],
        'apoptotic': ['mean', 'std']
    }).reset_index()
    stats_df.columns = ['time', 'alive_mean', 'alive_std', 'apoptotic_mean', 'apoptotic_std']

    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
    alive_line = ax.plot(stats_df['time'], stats_df['alive_mean'], 
            color=CB_PALETTE['green'], label='Alive', linewidth=2.0)[0]
    ax.fill_between(stats_df['time'], 
                    stats_df['alive_mean'] - stats_df['alive_std'],
                    stats_df['alive_mean'] + stats_df['alive_std'],
                    color=CB_PALETTE['green'], alpha=0.2, linewidth=1.5)
    apoptotic_line = ax.plot(stats_df['time'], stats_df['apoptotic_mean'], 
            color=CB_PALETTE['red'], label='Apoptotic', linewidth=2.0)[0]
    ax.fill_between(stats_df['time'], 
                    stats_df['apoptotic_mean'] - stats_df['apoptotic_std'],
                    stats_df['apoptotic_mean'] + stats_df['apoptotic_std'],
                    color=CB_PALETTE['red'], alpha=0.2, linewidth=1.5)
    ax.axvspan(1280, 1292, color=CB_PALETTE['grey'], alpha=1.0, zorder=0)
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    ax.set_xlabel('Time (min)', fontsize=10, fontweight="bold")
    ax.set_ylabel('Cell count', fontsize=10, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', width=0.8, length=2, labelsize=9, colors='black')
    plt.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)
    fig.savefig(output_file_svg, bbox_inches='tight', dpi=300, transparent=True)

    # Now create and save the legend
    legend_fig, legend_ax = plt.subplots(figsize=(2.5, 0.5))
    legend = legend_ax.legend([alive_line, apoptotic_line], ['Alive', 'Apoptotic'], loc='center', ncol=2, frameon=False)
    legend_ax.axis('off')
    legend_fig.savefig(legend_file_svg, bbox_inches='tight', dpi=300, transparent=True)

    plt.close(fig)
    plt.close(legend_fig)

def plot_top_10_node_states(top_10_combined_df, experiment_name):
    """
    Plot the mean and standard deviation of node states over time for the top 10 parameter sets.
    Figure size: (2.5, 2.5) inches for square, journal-style layout.
    Uses colorblind-friendly and Nature-approved palette.
    """
    output_file = os.path.join('results', 'node_states_plots', f'node_states_plot_full_{experiment_name}.png')
    if os.path.exists(output_file):
        print(f"Node states plot for {experiment_name} already exists, skipping...")
        return
    print(f"Plotting node states for {experiment_name}")
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
    filtered_nodes = ["node_cMYC_mean", "node_TCF_mean", "node_RSK_mean",
                     "node_FOXO_mean", "node_Caspase8_mean", "node_Caspase9_mean"]
    node_colors = {
        "node_cMYC_mean": CB_PALETTE['orange'],
        "node_TCF_mean": CB_PALETTE['green'],
        "node_RSK_mean": CB_PALETTE['blue'],
        "node_FOXO_mean": CB_PALETTE['red'],
        "node_Caspase8_mean": CB_PALETTE['purple'],
        "node_Caspase9_mean": CB_PALETTE['grey'],
    }
    filtered_df = top_10_combined_df[top_10_combined_df["node_type"].isin(filtered_nodes)]
    for node in filtered_nodes:
        node_df = filtered_df[filtered_df["node_type"] == node]
        if not node_df.empty:
            color = node_colors.get(node, CB_PALETTE['black'])
            label = node.replace("node_", "").replace("_mean", "")
            ax.plot(node_df['time'], node_df['value'], color=color, label=label, linewidth=2.0)
            ax.fill_between(node_df['time'], 
                          node_df['value'] - node_df['std'],
                          node_df['value'] + node_df['std'],
                          color=color, alpha=0.2, linewidth=1.5)
    ax.axvspan(1280, 1292, color=CB_PALETTE['grey'], alpha=1.0, zorder=0)
    ax.set_xlabel('Time (min)', fontsize=10, fontweight="bold")
    ax.set_ylabel('Node Value', fontsize=10, fontweight="bold")
    ax.tick_params(axis='both', width=0.8, length=2, labelsize=9, colors='black')
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    plt.close('all')

def plot_top_10_anti_node_states(top_10_combined_df, experiment_name):
    """
    Plot the mean and standard deviation of anti-drug node states over time for the top 10 parameter sets.
    Figure size: (2.5, 2.5) inches for square, journal-style layout.
    Uses colorblind-friendly and Nature-approved palette.
    """
    output_file = os.path.join('results', 'node_states_plots', f'anti_node_states_plot_{experiment_name}.png')
    if os.path.exists(output_file):
        print(f"Anti-node states plot for {experiment_name} already exists, skipping...")
        return

    print(f"Plotting anti-drug node states for {experiment_name}")

    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
    anti_drug_nodes = ["PI3K_node", "MEK_node", "AKT_node"]
    node_colors = {
        "PI3K_node": CB_PALETTE['blue'],
        "MEK_node": CB_PALETTE['orange'],
        "AKT_node": CB_PALETTE['green'],
    }
    filtered_df = top_10_combined_df[top_10_combined_df["node_type"].isin(anti_drug_nodes)]
    for node in anti_drug_nodes:
        node_df = filtered_df[filtered_df["node_type"] == node]
        if not node_df.empty:
            color = node_colors.get(node, CB_PALETTE['black'])
            label = node.replace("anti_", "").replace("_node", "").upper()
            ax.plot(node_df['time'], node_df['value'], color=color, label=label, linewidth=2.0)
            ax.fill_between(node_df['time'], 
                         node_df['value'] - node_df['std'],
                         node_df['value'] + node_df['std'],
                         color=color, alpha=0.2, linewidth=1.5)
    ax.axvspan(1280, 1292, color=CB_PALETTE['grey'], alpha=1.0, zorder=0)
    ax.set_xlabel('Time (min)', fontsize=10, fontweight="bold")
    ax.set_ylabel('Node Value', fontsize=10, fontweight="bold")
    ax.tick_params(axis='both', width=0.8, length=2, labelsize=9, colors='black')
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    plt.close('all')
    
def plot_top_10_cell_rates(top_10_combined_df, experiment_name):
    """
    Plot the mean and standard deviation of cell rates over time for the top 10 parameter sets.
    Figure size: (2.5, 2.5) inches for square, journal-style layout.
    Uses colorblind-friendly and Nature-approved palette.
    """
    output_file = os.path.join('results', 'cell_rates_plots', f'cell_rates_plot_{experiment_name}.png')
    output_file_svg = os.path.join('results', 'cell_rates_plots', f'cell_rates_plot_{experiment_name}.svg')
    legend_file_svg = os.path.join('results', 'cell_rates_plots', f'cell_rates_legend_{experiment_name}.svg')
    if os.path.exists(output_file) and os.path.exists(output_file_svg) and os.path.exists(legend_file_svg):
        print(f"Cell rates plot for {experiment_name} already exists, skipping...")
        return

    print(f"Plotting cell rates for {experiment_name}")
    
    top_10_combined_df = top_10_combined_df[top_10_combined_df['time'] != 0].copy()
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
    rate_colors = {
        'growth_rate_mean': CB_PALETTE['green'],  # Green for growth rate
        'apoptosis_rate_mean': CB_PALETTE['red']  # Red for apoptosis rate
    }
    lines = []
    labels = []
    for rate_type, color in rate_colors.items():
        rate_df = top_10_combined_df[top_10_combined_df["rate_type"] == rate_type]
        if not rate_df.empty:
            line = ax.plot(rate_df['time'], rate_df['value'], color=color, 
                   label=rate_type.replace('_mean', ''), linewidth=2.0)[0]
            ax.fill_between(rate_df['time'], 
                          rate_df['value'] - rate_df['std'],
                          rate_df['value'] + rate_df['std'],
                          color=color, alpha=0.2, linewidth=1.5)
            lines.append(line)
            labels.append(rate_type.replace('_mean', ''))
    ax.axvspan(1280, 1292, color=CB_PALETTE['grey'], alpha=1.0, zorder=0)
    ax.set_xlabel('Time (min)', fontsize=10, fontweight="bold")
    ax.set_ylabel('Rate Value', fontsize=10, fontweight="bold")
    ax.set_ylim(bottom=0, top=0.0012)
    ax.tick_params(axis='both', width=0.8, length=2, labelsize=9, colors='black')
    legend_fig, legend_ax = plt.subplots(figsize=(2.5, 0.5))
    legend = legend_ax.legend(lines, labels, loc='center', ncol=2, frameon=False)
    legend_ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)
    plt.savefig(output_file_svg, bbox_inches='tight', dpi=300, transparent=True)
    legend_fig.savefig(legend_file_svg, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    plt.close(legend_fig)
    plt.close('all')

def plot_top_10_cell_signals(top_10_combined_df, experiment_name):
    """
    Plot the mean and standard deviation of cell signals over time for the top 10 parameter sets.
    Figure size: (2.5, 2.5) inches for square, journal-style layout.
    Uses colorblind-friendly and Nature-approved palette.
    """
    output_file = os.path.join('results', 'cell_signals_plots', f'cell_signals_plot_{experiment_name}.png')
    output_file_svg = os.path.join('results', 'cell_signals_plots', f'cell_signals_plot_{experiment_name}.svg')
    legend_file_svg = os.path.join('results', 'cell_signals_plots', f'cell_signals_legend_{experiment_name}.svg')
    if os.path.exists(output_file) and os.path.exists(output_file_svg) and os.path.exists(legend_file_svg):
        print(f"Cell signals plot for {experiment_name} already exists, skipping...")
        return

    print(f"Plotting cell signals for {experiment_name}")

    top_10_combined_df = top_10_combined_df[top_10_combined_df['time'] != 0].copy()

    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
    signal_colors = {
        'S_pro_real_mean': CB_PALETTE['green'],  # Green for pro-survival
        'S_anti_real_mean': CB_PALETTE['red']    # Red for anti-survival
    }
    lines = []
    labels = []
    for signal_type, color in signal_colors.items():
        signal_df = top_10_combined_df[top_10_combined_df["signal_type"] == signal_type]
        if not signal_df.empty:
            line = ax.plot(signal_df['time'], signal_df['value'], color=color, 
                   label=signal_type.replace('_real_mean', ''), linewidth=2.0)[0]
            ax.fill_between(signal_df['time'], 
                          signal_df['value'] - signal_df['std'],
                          signal_df['value'] + signal_df['std'],
                          color=color, alpha=0.2, linewidth=1.5)
            lines.append(line)
            labels.append(signal_type.replace('_real_mean', ''))
    ax.axvspan(1280, 1292, color=CB_PALETTE['grey'], alpha=1.0, zorder=0)
    ax.set_xlabel('Time (min)', fontsize=10, fontweight="bold")
    ax.set_ylabel('Signal Value', fontsize=10, fontweight="bold")
    ax.tick_params(axis='both', width=0.8, length=2, labelsize=9, colors='black')
    legend_fig, legend_ax = plt.subplots(figsize=(2.5, 0.5))
    legend = legend_ax.legend(lines, labels, loc='center', ncol=2, frameon=False)
    legend_ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)
    plt.savefig(output_file_svg, bbox_inches='tight', dpi=300, transparent=True)
    legend_fig.savefig(legend_file_svg, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    plt.close(legend_fig)
    plt.close('all')

def plot_prosurvival_weights_boxplot(top_n_df, experiment_name):
    """
    Plots boxplots for pro-survival node weights across the top N parameter sets.
    Figure size: (2.5, 2.5) inches for square, journal-style layout.
    Uses colorblind-friendly and Nature-approved palette.
    """
    print(f"\nAttempting to plot pro-survival weights for {experiment_name}")
    print(f"Available columns: {top_n_df.columns.tolist()}")
    print(f"DataFrame shape: {top_n_df.shape}")
    
    output_file = os.path.join('results', 'node_weights_plots', f'prosurvival_weights_boxplot_{experiment_name}.png')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        print(f"Pro-survival weights boxplot for {experiment_name} already exists, skipping...")
        return

    print(f"Plotting pro-survival node weights for {experiment_name}")

    pro_cols = ["w_pro_cMYC", "w_pro_RSK", "w_pro_TCF"]
    missing_cols = [col for col in pro_cols if col not in top_n_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for pro-survival weights: {missing_cols}")
        return
        
    data = top_n_df[pro_cols]
    print(f"Data shape: {data.shape}")
    print(f"Data head:\n{data.head()}")

    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
    box = ax.boxplot([data[col] for col in pro_cols], patch_artist=True, labels=[col.replace("w_pro_", "") for col in pro_cols])
    colors = [CB_PALETTE['orange'], CB_PALETTE['green'], CB_PALETTE['blue']]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Weight", fontsize=10, fontweight="bold")
    ax.set_xlabel("Pro-survival Node", fontsize=10, fontweight="bold")
    ax.set_title("Pro-survival Node Weights", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    ax.tick_params(axis='both', width=0.8, length=2, labelsize=9, colors='black')
    plt.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)

    # Now create and save the legend
    legend_fig, legend_ax = plt.subplots(figsize=(2.5, 0.5))
    legend = legend_ax.legend([box['boxes'][i] for i in range(len(pro_cols))], [col.replace("w_pro_", "") for col in pro_cols], loc='center', ncol=2, frameon=False)
    legend_ax.axis('off')
    legend_fig.savefig(os.path.join('results', 'node_weights_plots', f'prosurvival_weights_legend_{experiment_name}.svg'), bbox_inches='tight', dpi=300, transparent=True)

    plt.close(fig)
    plt.close(legend_fig)
    print(f"Saved pro-survival weights plot to {output_file}")

def plot_antisurvival_weights_boxplot(top_n_df, experiment_name):
    """
    Plots boxplots for anti-survival node weights across the top N parameter sets.
    Figure size: (2.5, 2.5) inches for square, journal-style layout.
    Uses colorblind-friendly and Nature-approved palette.
    """
    print(f"\nAttempting to plot anti-survival weights for {experiment_name}")
    print(f"Available columns: {top_n_df.columns.tolist()}")
    print(f"DataFrame shape: {top_n_df.shape}")
    
    output_file = os.path.join('results', 'node_weights_plots', f'antisurvival_weights_boxplot_{experiment_name}.png')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        print(f"Anti-survival weights boxplot for {experiment_name} already exists, skipping...")
        return

    print(f"Plotting anti-survival node weights for {experiment_name}")

    anti_cols = ["w_anti_Caspase8", "w_anti_Caspase9", "w_anti_FOXO"]
    missing_cols = [col for col in anti_cols if col not in top_n_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for anti-survival weights: {missing_cols}")
        return
        
    data = top_n_df[anti_cols]
    print(f"Data shape: {data.shape}")
    print(f"Data head:\n{data.head()}")

    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
    box = ax.boxplot([data[col] for col in anti_cols], patch_artist=True, labels=[col.replace("w_anti_", "") for col in anti_cols])
    colors = [CB_PALETTE['purple'], CB_PALETTE['red'], CB_PALETTE['grey']]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Weight", fontsize=10, fontweight="bold")
    ax.set_xlabel("Anti-survival Node", fontsize=10, fontweight="bold")
    ax.set_title("Anti-survival Node Weights", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    ax.tick_params(axis='both', width=0.8, length=2, labelsize=9, colors='black')
    plt.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)

    # Now create and save the legend
    legend_fig, legend_ax = plt.subplots(figsize=(2.5, 0.5))
    legend = legend_ax.legend([box['boxes'][i] for i in range(len(anti_cols))], [col.replace("w_anti_", "") for col in anti_cols], loc='center', ncol=2, frameon=False)
    legend_ax.axis('off')
    legend_fig.savefig(os.path.join('results', 'node_weights_plots', f'antisurvival_weights_legend_{experiment_name}.svg'), bbox_inches='tight', dpi=300, transparent=True)

    plt.close(fig)
    plt.close(legend_fig)
    print(f"Saved anti-survival weights plot to {output_file}")

# Add parallel processing for experiment batches
def parallel_process_experiment(strategy, top_n, experiment_name, directory_name):
    summaries_folder = f"results/{strategy}_summaries"
    final_experiment_name = "final_summary_" + experiment_name
    df = pd.read_csv(os.path.join(summaries_folder, final_experiment_name, directory_name))
    return process_top_10(df, experiment_name)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # --- Define Experiments ---
    single_drug_exps = {
        'pi3k': {'name': "PI3Ki_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen", 'strategy': 'CMA'},
        'mek':  {'name': "MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen", 'strategy': 'CMA'},
        'akt':  {'name': "AKTi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen", 'strategy': 'CMA'},
    }
    
    combo_drug_exps = {
        'pi3k_mek': {'name': "synergy_sweep-pi3k_mek-1104-2212-18p_transient_delayed_uniform_5k_10p", 'strategy': 'sweep'},
        'akt_mek':  {'name': "synergy_sweep-akt_mek-1204-1639-18p_transient_delayed_uniform_postdrug_RMSE_5k", 'strategy': 'sweep'}
    }

    # --- Process and Plot Single-Drug Calibrations ---
    logging.info("--- Processing Single-Drug Experiments ---")
    all_exp_data = {}
    for key, info in single_drug_exps.items():
        all_exp_data[key] = get_experiment_dynamics(info['name'], info['strategy'])
    
    if any(s is not None for s in all_exp_data.values()):
        plot_single_drug_grid(
            pi3k_data=all_exp_data.get('pi3k'),
            akt_data=all_exp_data.get('akt'),
            mek_data=all_exp_data.get('mek')
        )
    else:
        logging.warning("Could not generate single-drug grid: no data was processed for any experiment.")

    # --- Process and Plot Combination Therapies ---
    logging.info("--- Processing Combination Therapy Experiments ---")
    for key, info in combo_drug_exps.items():
        all_exp_data[key] = get_experiment_dynamics(info['name'], info['strategy'])
    
    if all(k in all_exp_data and all_exp_data[k] is not None for k in ['pi3k_mek', 'akt_mek']):
        plot_combination_comparison(
            pi3k_mek_data=all_exp_data.get('pi3k_mek'),
            akt_mek_data=all_exp_data.get('akt_mek')
        )
    else:
        logging.warning("Could not generate combination therapy comparison: missing data for one or both experiments.")

    # --- Generate Final 3x5 Comparison Grid ---
    logging.info("--- Generating Final Comparison Grid ---")
    if any(d is not None for d in all_exp_data.values()):
        save_dir = "results/comparison_grids"
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, "topN_comparison_grid.png")
        plot_top_n_comparison_grid(all_exp_data, output_path)
    else:
        logging.warning("Could not generate topN comparison grid: no data was processed for any experiment.")

    # --- Generate Survival Nodes Grid ---
    logging.info("--- Generating Survival Nodes Grid ---")
    if any(d is not None for d in all_exp_data.values()):
        save_dir = "results/comparison_grids"
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, "survival_nodes_grid.png")
        plot_survival_nodes_grid(all_exp_data, output_path)
    else:
        logging.warning("Could not generate survival nodes grid: no data was processed for any experiment.")