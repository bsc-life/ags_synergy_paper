import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from functools import partial
import gc
import glob
from matplotlib.lines import Line2D

# Set the maximum number of open figures before a warning is issued
plt.rcParams['figure.max_open_warning'] = 0  # Suppress the warning
# Use a faster backend for matplotlib
plt.switch_backend('agg')

# Global cache for processed data
DATA_CACHE = {}

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
    
    # Extract counts
    if 'live' in phases_by_time.columns:
        alive_counts = phases_by_time['live']
    if 'apoptotic' in phases_by_time.columns:
        apoptotic_counts = phases_by_time['apoptotic']
    
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
    output_file = os.path.join('results', 'alive_apoptotic_plots', f'alive_apoptotic_plot_{experiment_name}.png')
    if os.path.exists(output_file):
        print(f"Alive/apoptotic plot for {experiment_name} already exists, skipping...")
        return

    print(f"Plotting alive/apoptotic cells for {experiment_name}")
    
    # Use existing plotting code with optimized settings
    # ... (rest of this function unchanged)
    # Calculate statistics across instances for each timepoint
    stats_df = top_10_combined_df.groupby('time').agg({
        'alive': ['mean', 'std'],
        'apoptotic': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    stats_df.columns = ['time', 'alive_mean', 'alive_std', 'apoptotic_mean', 'apoptotic_std']

    # Plotting with optimized settings
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    
    # Plot lines with confidence intervals
    ax.plot(stats_df['time'], stats_df['alive_mean'], 
            color='#2271B2', label='Alive', linewidth=1.5)
    ax.fill_between(stats_df['time'], 
                    stats_df['alive_mean'] - stats_df['alive_std'],
                    stats_df['alive_mean'] + stats_df['alive_std'],
                    color='#4CAF50', alpha=0.2)
    
    ax.plot(stats_df['time'], stats_df['apoptotic_mean'], 
            color='#D55E00', label='Apoptotic', linewidth=1.5)
    ax.fill_between(stats_df['time'], 
                    stats_df['apoptotic_mean'] - stats_df['apoptotic_std'],
                    stats_df['apoptotic_mean'] + stats_df['apoptotic_std'],
                    color='#FF6B6B', alpha=0.2)
    
    # Add treatment indicator
    ax.axvspan(1280, 1292, color='#FF6B6B', alpha=1.0, zorder=0)
    
    # Basic styling
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    ax.set_xlabel('Simulation Time (min)', fontsize=12, fontweight="bold")
    ax.set_ylabel('Cell count', fontsize=12, fontweight="bold")
    # ax.set_xlim(0, 4200)
    ax.set_ylim(bottom=0)
        
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    plt.close('all')

# Similar optimizations for the other plotting functions...
def plot_top_10_node_states(top_10_combined_df, experiment_name):
    output_file = os.path.join('results', 'node_states_plots', f'node_states_plot_full_{experiment_name}.png')
    if os.path.exists(output_file):
        print(f"Node states plot for {experiment_name} already exists, skipping...")
        return

    print(f"Plotting node states for {experiment_name}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    # Filter the dataframe for the node types we want (excluding those with "node" in name)
    filtered_nodes = ["node_cMYC_mean", "node_TCF_mean", "node_RSK_mean",
                     "node_FOXO_mean", "node_Caspase8_mean", "node_Caspase9_mean"]
    
    # Define pro-survival and anti-survival nodes
    pro_survival_nodes = ["node_cMYC_mean", "node_TCF_mean", "node_RSK_mean"]
    anti_survival_nodes = ["node_FOXO_mean", "node_Caspase8_mean", "node_Caspase9_mean"]
    
    # Define consistent colors for pro and anti-survival
    pro_survival_color = '#2E8B57'  # Nice green
    anti_survival_color = '#D55E00'  # Nice red
    
    # Create color dictionary
    node_colors = {}
    for node in pro_survival_nodes:
        node_colors[node] = pro_survival_color
    for node in anti_survival_nodes:
        node_colors[node] = anti_survival_color

    # Filter for the nodes we want
    filtered_df = top_10_combined_df[top_10_combined_df["node_type"].isin(filtered_nodes)]

    # Plot each node type with its proper color
    for node in filtered_nodes:
        node_df = filtered_df[filtered_df["node_type"] == node]
        if not node_df.empty:
            color = node_colors.get(node, "gray")
            # Remove "node_" and "_mean" from label
            label = node.replace("node_", "").replace("_mean", "")
            ax.plot(node_df['time'], node_df['value'], color=color, label=label, linewidth=1.5)
            ax.fill_between(node_df['time'], 
                          node_df['value'] - node_df['std'],
                          node_df['value'] + node_df['std'],
                          color=color, alpha=0.2)

    # Add treatment indicator
    ax.axvspan(1280, 1292, color='#FF6B6B', alpha=1.0, zorder=0)
    
    # Basic styling
    ax.set_xlabel('Time (min)', fontsize=12, fontweight="bold")
    ax.set_ylabel('Node Value', fontsize=12, fontweight="bold")
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    # ax.set_xlim(0, 4200)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    plt.close('all')

def plot_top_10_anti_node_states(top_10_combined_df, experiment_name):
    output_file = os.path.join('results', 'node_states_plots', f'anti_node_states_plot_{experiment_name}.png')
    if os.path.exists(output_file):
        print(f"Anti-node states plot for {experiment_name} already exists, skipping...")
        return

    print(f"Plotting anti-drug node states for {experiment_name}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    # Define the anti-drug nodes we want to plot
    anti_drug_nodes = ["PI3K_node", "MEK_node", "AKT_node"]
    
    # Define specific colors for each node
    node_colors = {
        "PI3K_node": '#E41A1C',  # Red
        "MEK_node": '#377EB8',   # Blue
        "AKT_node": '#4DAF4A'    # Green
    }

    # Filter for the nodes we want
    filtered_df = top_10_combined_df[top_10_combined_df["node_type"].isin(anti_drug_nodes)]

    # Plot each node type with its proper color
    for node in anti_drug_nodes:
        node_df = filtered_df[filtered_df["node_type"] == node]
        if not node_df.empty:
            color = node_colors.get(node, "gray")
            # Clean up label: remove "anti_" and "_node" from label
            label = node.replace("anti_", "").replace("_node", "").upper()
            ax.plot(node_df['time'], node_df['value'], color=color, label=label, linewidth=1.5)
            ax.fill_between(node_df['time'], 
                         node_df['value'] - node_df['std'],
                         node_df['value'] + node_df['std'],
                         color=color, alpha=0.2)

    # Add treatment indicator
    ax.axvspan(1280, 1292, color='#FF6B6B', alpha=1.0, zorder=0)
    
    # Basic styling
    ax.set_xlabel('Time (min)', fontsize=12, fontweight="bold")
    ax.set_ylabel('Anti-Drug Node Value', fontsize=12, fontweight="bold")
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    # ax.set_xlim(0, 4200)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    
    # Add title explaining what these nodes represent
    ax.set_title('Drug Resistance Mechanisms', fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    plt.close('all')
    
def plot_top_10_cell_rates(top_10_combined_df, experiment_name):
    output_file = os.path.join('results', 'cell_rates_plots', f'cell_rates_plot_{experiment_name}.png')
    if os.path.exists(output_file):
        print(f"Cell rates plot for {experiment_name} already exists, skipping...")
        return

    print(f"Plotting cell rates for {experiment_name}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    # Pro-survival in green, anti-survival in red
    rate_colors = {
        'growth_rate_mean': '#2E8B57',  # Green for growth rate
        'apoptosis_rate_mean': '#D55E00'  # Red for apoptosis rate
    }

    # Plot the aggregated data
    for rate_type, color in rate_colors.items():
        rate_df = top_10_combined_df[top_10_combined_df["rate_type"] == rate_type]
        if not rate_df.empty:
            # Plot the mean with error bands
            ax.plot(rate_df['time'], rate_df['value'], color=color, 
                   label=rate_type.replace('_mean', ''), linewidth=1.5)
            ax.fill_between(rate_df['time'], 
                          rate_df['value'] - rate_df['std'],
                          rate_df['value'] + rate_df['std'],
                          color=color, alpha=0.2)

    # Add treatment indicator
    ax.axvspan(1280, 1292, color='#FF6B6B', alpha=1.0, zorder=0)
    
    # Basic styling
    ax.set_xlabel('Time (min)', fontsize=12, fontweight="bold")
    ax.set_ylabel('Rate', fontsize=12, fontweight="bold")
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    # ax.set_xlim(0, 4200)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    plt.close('all')

def plot_top_10_cell_signals(top_10_combined_df, experiment_name):
    output_file = os.path.join('results', 'cell_signals_plots', f'cell_signals_plot_{experiment_name}.png')
    if os.path.exists(output_file):
        print(f"Cell signals plot for {experiment_name} already exists, skipping...")
        return

    print(f"Plotting cell signals for {experiment_name}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    # Pro-survival in green, anti-survival in red
    signal_colors = {
        'S_pro_real_mean': '#2E8B57',  # Green for pro-survival
        'S_anti_real_mean': '#D55E00'  # Red for anti-survival
    }

    # Plot each signal type with its proper color
    for signal_type, color in signal_colors.items():
        signal_df = top_10_combined_df[top_10_combined_df["signal_type"] == signal_type]
        if not signal_df.empty:
            ax.plot(signal_df['time'], signal_df['value'], color=color, 
                   label=signal_type.replace('_real_mean', ''), linewidth=1.5)
            ax.fill_between(signal_df['time'], 
                          signal_df['value'] - signal_df['std'],
                          signal_df['value'] + signal_df['std'],
                          color=color, alpha=0.2)

    # Add treatment indicator
    ax.axvspan(1280, 1292, color='#FF6B6B', alpha=1.0, zorder=0)
    
    # Basic styling
    ax.set_xlabel('Time (min)', fontsize=12, fontweight="bold")
    ax.set_ylabel('Signal Value', fontsize=12, fontweight="bold")
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    # ax.set_xlim(0, 4200)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    plt.close('all')

def plot_prosurvival_weights_boxplot(top_n_df, experiment_name):
    """
    Plots boxplots for pro-survival node weights across the top N parameter sets.
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
    # Check if all required columns exist
    missing_cols = [col for col in pro_cols if col not in top_n_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for pro-survival weights: {missing_cols}")
        return
        
    data = top_n_df[pro_cols]
    print(f"Data shape: {data.shape}")
    print(f"Data head:\n{data.head()}")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    box = ax.boxplot([data[col] for col in pro_cols], patch_artist=True, labels=[col.replace("w_pro_", "") for col in pro_cols])

    # Set colors to match the style (green for pro-survival)
    colors = ['#2E8B57'] * len(pro_cols)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel("Weight", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pro-survival Node", fontsize=12, fontweight="bold")
    ax.set_title("Pro-survival Node Weights", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    plt.close('all')
    print(f"Saved pro-survival weights plot to {output_file}")

def plot_antisurvival_weights_boxplot(top_n_df, experiment_name):
    """
    Plots boxplots for anti-survival node weights across the top N parameter sets.
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
    # Check if all required columns exist
    missing_cols = [col for col in anti_cols if col not in top_n_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for anti-survival weights: {missing_cols}")
        return
        
    data = top_n_df[anti_cols]
    print(f"Data shape: {data.shape}")
    print(f"Data head:\n{data.head()}")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    box = ax.boxplot([data[col] for col in anti_cols], patch_artist=True, labels=[col.replace("w_anti_", "") for col in anti_cols])

    # Set colors to match the style (red for anti-survival)
    colors = ['#D55E00'] * len(anti_cols)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel("Weight", fontsize=12, fontweight="bold")
    ax.set_xlabel("Anti-survival Node", fontsize=12, fontweight="bold")
    ax.set_title("Anti-survival Node Weights", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    plt.close('all')
    print(f"Saved anti-survival weights plot to {output_file}")

# Add parallel processing for experiment batches
def parallel_process_experiment(strategy, top_n, experiment_name, directory_name):
    summaries_folder = f"results/{strategy}_summaries"
    final_experiment_name = "final_summary_" + experiment_name
    df = pd.read_csv(os.path.join(summaries_folder, final_experiment_name, directory_name))
    return process_top_10(df, experiment_name)

# Create output directory
results_dir = "results/4p_diffusion_timing_analysis"
os.makedirs(results_dir, exist_ok=True)

# Function to find all relevant summary files
def find_4p_summary_files():
    sweep_dir = "results/sweep_summaries/"
    return glob.glob(f"{sweep_dir}/final_summary_*4p*_drugtiming.csv")

# Process each summary file
def process_summary_file(file_path):
    print(f"Processing {file_path}")
    
    # Extract experiment name from filename
    experiment_name = os.path.basename(file_path).replace("final_summary_", "").replace(".csv", "")
    
    # Read the CSV file
    timing_df = pd.read_csv(file_path)
    
    # Create delta time as the signed difference between the two pulse periods
    timing_df['delta_time'] = timing_df['user_parameters.drug_X_pulse_period'] - timing_df['user_parameters.drug_Y_pulse_period']
    
    # Group by diffusion coefficients and delta_time, compute mean alive cells and std
    grouped = timing_df.groupby(
        ['user_parameters.drug_X_diffusion_coefficient', 
         'user_parameters.drug_Y_diffusion_coefficient', 
         'delta_time']
    ).agg(
        mean_alive=('FINAL_NUMBER_OF_ALIVE_CELLS', 'mean'),
        std_alive=('FINAL_NUMBER_OF_ALIVE_CELLS', 'std')
    ).reset_index()
    
    # Get sorted unique values for diffusion coefficients
    x_diffs = sorted(grouped['user_parameters.drug_X_diffusion_coefficient'].unique())
    y_diffs = sorted(grouped['user_parameters.drug_Y_diffusion_coefficient'].unique())
    
    # Plot alive cells grid
    plot_alive_cells_grid(grouped, x_diffs, y_diffs, experiment_name, 
                         f"{results_dir}/alive_cells_vs_deltatime_{experiment_name}.png")
    
    # Calculate and plot Bliss scores
    calculate_and_plot_bliss(timing_df, grouped, x_diffs, y_diffs, experiment_name)

# Function to calculate and plot Bliss scores
def calculate_and_plot_bliss(timing_df, grouped, x_diffs, y_diffs, experiment_name):
    # Access the single drug data and control exactly as in study_optimal_timings_synergy.py
    sweep_summaries_path = "results/sweep_summaries/"
    negative_control_name = "synergy_sweep-3D-0205-1608-control_nodrug"
    
    # Determine which drug combination we're dealing with
    if "pi3k_mek" in experiment_name:
        pi3k_single_drug_name = "synergy_sweep-pi3k_mek-3D-0505-0218-logscale_singledrug_pi3k"
        mek_single_drug_name = "synergy_sweep-pi3k_mek-3D-0505-0218-logscale_singledrug_mek"
    elif "akt_mek" in experiment_name:
        pi3k_single_drug_name = "synergy_sweep-akt_mek-3D-0505-1910-logscale_singledrug_akt"  
        mek_single_drug_name = "synergy_sweep-akt_mek-3D-0505-1910-logscale_singledrug_mek"
    else:
        print(f"Unknown drug combination in {experiment_name}, skipping Bliss analysis")
        return
    
    # Read the negative control data
    final_summary_negative_control = pd.read_csv(f"{sweep_summaries_path}/final_summary_{negative_control_name}.csv")
    mean_final_number_of_alive_cells_negative_control = round(final_summary_negative_control.iloc[:, -1].mean())
    print(f"Negative control mean: {mean_final_number_of_alive_cells_negative_control}")
    
    # Read the single drug data
    pi3k_single_drug = pd.read_csv(f"{sweep_summaries_path}/final_summary_{pi3k_single_drug_name}.csv")
    mek_single_drug = pd.read_csv(f"{sweep_summaries_path}/final_summary_{mek_single_drug_name}.csv")
    
    # Create dictionaries of diffusion coefficient to mean effect
    pi3k_grouped = pi3k_single_drug.groupby(pi3k_single_drug.columns[0]).agg({pi3k_single_drug.columns[-1]: 'mean'})
    pi3k_single_drug_dict = pi3k_grouped[pi3k_single_drug.columns[-1]].to_dict()
    
    mek_grouped = mek_single_drug.groupby(mek_single_drug.columns[0]).agg({mek_single_drug.columns[-1]: 'mean'})
    mek_single_drug_dict = mek_grouped[mek_single_drug.columns[-1]].to_dict()
    
    print(f"Single drug dictionaries loaded. Keys: {list(pi3k_single_drug_dict.keys())}, {list(mek_single_drug_dict.keys())}")
    
    # Calculate Bliss scores
    bliss_rows = []
    
    for i, x_diff in enumerate(x_diffs):
        for j, y_diff in enumerate(y_diffs):
            grouped_subdf = grouped[
                (grouped['user_parameters.drug_X_diffusion_coefficient'] == x_diff) &
                (grouped['user_parameters.drug_Y_diffusion_coefficient'] == y_diff)
            ].sort_values('delta_time')
            
            if not grouped_subdf.empty:
                for idx, row in grouped_subdf.iterrows():
                    delta_time = row['delta_time']
                    observed = row['mean_alive']
                    pi3k_single = pi3k_single_drug_dict.get(x_diff, None)
                    mek_single = mek_single_drug_dict.get(y_diff, None)
                    
                    if pi3k_single is None or mek_single is None:
                        continue
                        
                    # Normalize to negative control
                    E_A = pi3k_single / mean_final_number_of_alive_cells_negative_control
                    E_B = mek_single / mean_final_number_of_alive_cells_negative_control
                    E_AB = observed / mean_final_number_of_alive_cells_negative_control
                    
                    # Calculate Bliss score
                    bliss_expected = E_A * E_B
                    bliss_score = bliss_expected - E_AB
                    
                    bliss_rows.append({
                        'x_diff': x_diff,
                        'y_diff': y_diff,
                        'delta_time': delta_time,
                        'bliss_score': bliss_score,
                        'std_alive': row['std_alive'] / mean_final_number_of_alive_cells_negative_control
                    })
    
    bliss_df = pd.DataFrame(bliss_rows)
    if bliss_df.empty:
        print("No Bliss scores could be calculated!")
        return
    
    # Plot Bliss scores grid
    plot_bliss_grid(bliss_df, x_diffs, y_diffs, experiment_name,
                   f"{results_dir}/bliss_score_vs_deltatime_{experiment_name}.png")

# Function to plot alive cells grid
def plot_alive_cells_grid(grouped, x_diffs, y_diffs, experiment_name, save_path):
    # Create a figure with subplots
    nrows = len(x_diffs)
    ncols = len(y_diffs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), sharex=True, sharey=True)
    
    for i, x_diff in enumerate(x_diffs):
        for j, y_diff in enumerate(y_diffs):
            # Get the correct axis
            if nrows == 1 and ncols == 1:
                ax = axes
            elif nrows == 1:
                ax = axes[j]
            elif ncols == 1:
                ax = axes[i]
            else:
                ax = axes[i, j]
            
            # Filter data for this diffusion coefficient combination
            subdf = grouped[
                (grouped['user_parameters.drug_X_diffusion_coefficient'] == x_diff) &
                (grouped['user_parameters.drug_Y_diffusion_coefficient'] == y_diff)
            ].sort_values('delta_time')
            
            if not subdf.empty:
                # Use delta_time as categories
                delta_times = list(subdf['delta_time'])
                xvals = np.arange(len(delta_times))
                
                # Plot the line connecting points
                ax.plot(xvals, subdf['mean_alive'], color='gray', zorder=1)
                
                # Plot each point with its own color and error bar
                for plot_idx, (idx, row) in enumerate(subdf.iterrows()):
                    if row['delta_time'] == 0:
                        # Special marker for delta_time == 0
                        ax.errorbar(
                            xvals[plot_idx], row['mean_alive'],
                            yerr=row['std_alive'],
                            fmt='o', color='green', ecolor='black', elinewidth=2, capsize=3,
                            markerfacecolor='green', markersize=10, zorder=3
                        )
                    else:
                        color = 'blue' if row['delta_time'] > 0 else 'orange'
                        ax.errorbar(
                            xvals[plot_idx], row['mean_alive'],
                            yerr=row['std_alive'],
                            fmt='o', color=color, ecolor='black', elinewidth=2, capsize=3,
                            markerfacecolor=color, zorder=2
                        )
                
                # Set custom x-ticks
                ax.set_xticks(xvals)
                ax.set_xticklabels([str(int(dt)) if dt == int(dt) else f"{dt:.1f}" for dt in delta_times], 
                                   rotation=45, fontsize=8)
                
                # Highlight the minimum
                min_idx = subdf['mean_alive'].idxmin()
                min_pos = list(subdf.index).index(min_idx)
                min_x = xvals[min_pos]
                min_y = subdf.loc[min_idx, 'mean_alive']
                
                ax.plot(min_x, min_y, 'ro', zorder=4)
                ax.annotate(f'{int(min_y)}', (min_x, min_y), textcoords="offset points", 
                           xytext=(0, -10), ha='center', color='red', fontsize=8)
                
                ax.set_title(f'X: {x_diff}, Y: {y_diff}')
                ax.set_xlabel('Delta Time (X - Y)')
                ax.set_ylim(bottom=0)  # Start y-axis at 0
            else:
                ax.axis('off')
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='X > Y', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='X < Y', markerfacecolor='orange', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='X = Y', markerfacecolor='green', markersize=8),
    ]
    fig.legend(handles=legend_elements, loc='upper right', title='Pulse Period Comparison')
    
    # Shared y-axis label
    fig.text(0.04, 0.5, 'Alive Cells', va='center', rotation='vertical', fontsize=15)
    
    plt.tight_layout(rect=[0.05, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved alive cells plot to {save_path}")

# Function to plot Bliss scores grid
def plot_bliss_grid(bliss_df, x_diffs, y_diffs, experiment_name, save_path):
    # Create a figure with subplots
    nrows = len(x_diffs)
    ncols = len(y_diffs)
    fig_bliss, axes_bliss = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.7*nrows), sharex=True, sharey=True, dpi=300)
    
    for i, x_diff in enumerate(x_diffs):
        for j, y_diff in enumerate(y_diffs):
            # Get the correct axis
            if nrows == 1 and ncols == 1:
                ax = axes_bliss
            elif nrows == 1:
                ax = axes_bliss[j]
            elif ncols == 1:
                ax = axes_bliss[i]
            else:
                ax = axes_bliss[i, j]
            
            # Filter data for this diffusion coefficient combination
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
                        yerr=row['std_alive'],
                        fmt='o', color=color, markerfacecolor=color, ecolor='black', 
                        elinewidth=2, capsize=3, zorder=2
                    )
                
                # Highlight the minimum (most negative = strongest synergy)
                min_idx = subdf['bliss_score'].idxmin()
                min_pos = list(subdf.index).index(min_idx)
                min_x = xvals[min_pos]
                min_y = subdf.loc[min_idx, 'bliss_score']
                
                ax.plot(min_x, min_y, 'ro', zorder=4)
                ax.annotate(f'{min_y:.2f}', (min_x, min_y), textcoords="offset points", 
                           xytext=(0, -10), ha='center', color='red', fontsize=8)
                
                # Set custom x-ticks
                ax.set_xticks(xvals)
                ax.set_xticklabels([str(int(dt)) if dt == int(dt) else f"{dt:.1f}" for dt in delta_times], 
                                  rotation=45, fontsize=8)
                
                ax.set_xlabel('Delta Time (X - Y)')
                ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.set_title(f'X: {x_diff}, Y: {y_diff}')
            else:
                ax.axis('off')
    
    # Layout and label adjustments
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12, wspace=0.18, hspace=0.25)
    
    # Main Bliss Score label in bold
    fig_bliss.text(0.005, 0.55, 'Bliss Score', 
                  va='center', ha='center', rotation='vertical',
                  fontsize=16, fontweight='bold')
    
    # Global x-axis label 
    fig_bliss.text(0.5, 0.03, 'Delta Time (X - Y)', 
                  ha='center', va='center',
                  fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='X > Y', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='X < Y', markerfacecolor='orange', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='X = Y', markerfacecolor='green', markersize=8),
    ]
    fig_bliss.legend(handles=legend_elements, loc='upper right', title='Pulse Period Comparison')
    
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved Bliss score plot to {save_path}")

# Main execution
def main():
    summary_files = find_4p_summary_files()
    
    if not summary_files:
        print("No 4p summary files found!")
        return
        
    print(f"Found {len(summary_files)} summary files with '4p' in the name.")
    
    for file_path in summary_files:
        process_summary_file(file_path)
        
    print("Analysis complete!")

if __name__ == "__main__":
    main()