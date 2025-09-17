import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import gc
from scipy.interpolate import interp1d
import numpy as np

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use a non-interactive backend suitable for scripts
plt.switch_backend('agg')

# Define a consistent, colorblind-friendly palette
CB_PALETTE = {
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
    'red': '#D55E00', 'purple': '#CC79A7', 'grey': '#999999',
}

def get_control_bounds_from_data(data_collector):
    """
    Gets control bounds from the case with lowest diffusion coefficient and highest time of addition (t=4000).
    This serves as an internal control instead of using external no-drug data.
    """
    control_min, control_max = None, None
    
    # Find the case with lowest diffusion and highest time
    control_case = None
    lowest_diff = float('inf')
    for case, (sim_data, row) in data_collector.items():
        # Check if this is a case with t=4000
        if ('drug_X_pulse_period' in row and row['drug_X_pulse_period'] >= 4000 and
            'drug_Y_pulse_period' in row and row['drug_Y_pulse_period'] >= 4000):
            # Get the total diffusion coefficient (sum of X and Y)
            total_diff = row.get('x_diff', 0) + row.get('y_diff', 0)
            if total_diff < lowest_diff:
                lowest_diff = total_diff
                control_case = case

    if control_case and control_case in data_collector:
        control_data, _ = data_collector[control_case]
        control_df = control_data['alive_apoptotic']
        control_min = control_df['live'].min()
        control_max = control_df['live'].max()
        logging.info(f"Using {control_case} as control. Bounds: Min={control_min}, Max={control_max}")
    else:
        logging.warning("No suitable control case found (high time, low diffusion). Using default normalization.")
        
    return control_min, control_max

def get_simulation_data(simulation_df):
    """
    Processes a raw simulation dataframe to extract and structure data for plotting.
    Note: Normalization is NOT performed here. It is handled globally.
    """
    # Extract alive/apoptotic counts
    alive_apoptotic_df = simulation_df.groupby('time')['current_phase'].value_counts().unstack(fill_value=0).reset_index()
    if 'live' not in alive_apoptotic_df.columns: alive_apoptotic_df['live'] = 0
    if 'apoptotic' not in alive_apoptotic_df.columns: alive_apoptotic_df['apoptotic'] = 0

    # Extract cell rates (mean across all cells at each time point)
    cell_rates_df = simulation_df.groupby('time')[['apoptosis_rate', 'growth_rate']].mean().reset_index()

    # Extract node states (mean across all cells at each time point)
    node_cols = [col for col in simulation_df.columns if col.startswith('node_')]
    node_states_df = simulation_df.groupby('time')[node_cols].mean().reset_index()

    # Extract cell signals (mean across all cells at each time point)
    cell_signals_df = simulation_df.groupby('time')[['S_pro_real', 'S_anti_real']].mean().reset_index()

    # Calculate fraction of activated cells for key target nodes
    target_nodes = [
        'pi3k_node', 'akt_node', 'mek_node', 'node_cMYC', 'node_RSK', 'node_TCF',
        'node_FOXO', 'node_Caspase8', 'node_Caspase9'
    ]
    available_nodes = [node for node in target_nodes if node in simulation_df.columns]
    
    if available_nodes:
        # Assuming activation threshold is > 0.5
        for node in available_nodes:
            simulation_df[f'{node}_activated'] = simulation_df[node] > 0.5
        
        # Group by time and calculate the mean of the boolean (activated) column
        frac_df = simulation_df.groupby('time')[[f'{node}_activated' for node in available_nodes]].mean().reset_index()
        
        # Convert to percentage and rename columns
        for node in available_nodes:
            frac_df[f'{node}_activated'] *= 100
            frac_df.rename(columns={f'{node}_activated': f'{node}_frac'}, inplace=True)
    else:
        frac_df = pd.DataFrame({'time': simulation_df['time'].unique()})

    return {
        "alive_apoptotic": alive_apoptotic_df,
        "cell_rates": cell_rates_df,
        "node_states": node_states_df,
        "cell_signals": cell_signals_df,
        "activated_fractions": frac_df
    }

def plot_dynamics(data, plot_info):
    """
    Generic plotting function to create and save a time-series plot.
    """
    # Compressed plot size and wider lines
    fig, ax = plt.subplots(figsize=(2.8, 2.2), dpi=300)
    
    for col, props in plot_info['series'].items():
        if col in data.columns:
            ax.plot(data['time'], data[col], label=props['label'], color=props['color'], linewidth=2.0)

    ax.set_xlabel("Time (min)", fontweight='bold', fontsize=9)
    ax.set_ylabel(plot_info['ylabel'], fontweight='bold', fontsize=9)
    ax.set_title(plot_info['title'], fontweight='bold', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=8)
    
    if plot_info.get('ylim'):
        ax.set_ylim(plot_info['ylim'])
        
    ax.legend(fontsize=8)
    
    os.makedirs(os.path.dirname(plot_info['save_path']), exist_ok=True)
    plt.savefig(plot_info['save_path'], bbox_inches='tight', transparent=True)
    plt.close(fig)

def create_summary_grid_plot(grid_data, output_path, exp_name):
    """
    Creates a 4x3 summary grid plot for the most important simulation cases.
    Rows are dynamics types, columns are simulation cases for easier comparison.
    """
    row_keys = ['target_activation', 'cell_signals', 'cell_rates', 'alive_apoptotic']
    col_order = ['Max Efficacy', 'Min Efficacy', 'Simultaneous']

    # Swapped dimensions and figsize for a vertical layout, with shared axes
    fig, axes = plt.subplots(4, 3, figsize=(10, 13), dpi=300, sharex=True, sharey='row')

    def _plot_on_ax(ax, data, plot_info):
        for col, props in plot_info['series'].items():
            if col in data.columns:
                ax.plot(data['time'], data[col], label=props.get('label', col), color=props['color'], linewidth=2.0)
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
        if plot_info.get('ylim'):
            ax.set_ylim(plot_info['ylim'])
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)

    # Base plot configurations with y-labels for all
    plot_configs = {
        'alive_apoptotic': {
            'title': "Cell Counts", 'ylabel': "Normalized Cell Count", 'ylim': (-5, 105),
            'series': {
                'normalized_live': {'label': 'Live', 'color': CB_PALETTE['green']},
                'normalized_apoptotic': {'label': 'Apoptotic', 'color': CB_PALETTE['red']}
            }
        },
        'cell_rates': {
            'title': "Cell Rates", 'ylabel': "Rate (1/min)", 'ylim': (0, 0.0012),
            'series': {'growth_rate': {'label': 'Growth', 'color': CB_PALETTE['green']}, 'apoptosis_rate': {'label': 'Apoptosis', 'color': CB_PALETTE['red']}}
        },
        'cell_signals': {
            'title': "Cell Signals", 'ylabel': "Signal Value",
            'series': {'S_pro_real': {'label': 'Pro-survival', 'color': CB_PALETTE['green']}, 'S_anti_real': {'label': 'Anti-survival', 'color': CB_PALETTE['red']}}
        }
    }
    
    target_activation_config = {
        'title': "Target Activation", 'ylabel': "% Activated Cells", 'ylim': (0, 101), 'series': {}
    }
    if 'pi3k' in exp_name.lower():
        target_activation_config['series']['pi3k_node_frac'] = {'label': 'PI3K Active', 'color': CB_PALETTE['blue']}
        target_activation_config['series']['mek_node_frac'] = {'label': 'MEK Active', 'color': CB_PALETTE['orange']}
    elif 'akt' in exp_name.lower():
        target_activation_config['series']['akt_node_frac'] = {'label': 'AKT Active', 'color': CB_PALETTE['green']}
        target_activation_config['series']['mek_node_frac'] = {'label': 'MEK Active', 'color': CB_PALETTE['orange']}
    plot_configs['target_activation'] = target_activation_config
    
    data_map = {
        'target_activation': 'activated_fractions', 'cell_signals': 'cell_signals',
        'cell_rates': 'cell_rates', 'alive_apoptotic': 'alive_apoptotic'
    }

    for i, key in enumerate(row_keys):
        plot_info = plot_configs[key]
        # Set y-label for the first plot in each row, and title for the plot type.
        axes[i, 0].set_ylabel(plot_info['ylabel'], fontweight='bold', fontsize=10)
        
        for j, case in enumerate(col_order):
            ax = axes[i, j]
            
            # Set title for the first row, indicating the case
            if i == 0:
                title_text = case
                if case in grid_data and case in ['Max Efficacy', 'Min Efficacy']:
                    _, row_data = grid_data[case]
                    # Check for 'delta_time' and that it's not NaN
                    if 'delta_time' in row_data and pd.notna(row_data['delta_time']):
                        title_text = f"{case}\\n(Δt={int(row_data['delta_time'])} min)"
                ax.set_title(title_text, fontweight='bold', fontsize=11)

            if case not in grid_data:
                ax.text(0.5, 0.5, "Data Not\\nFound", ha='center', va='center', fontsize=9, alpha=0.5)
                ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
                continue
            
            sim_data, _ = grid_data[case]
            data_df = sim_data[data_map[key]]
            
            _plot_on_ax(ax, data_df, plot_info)
            
            # Set x-label only for the last row
            if i == len(row_keys) - 1:
                ax.set_xlabel("Time (min)", fontsize=10, fontweight='bold')
            
            ax.tick_params(axis='both', labelsize=9, width=0.8, length=2)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved summary grid plot to {output_path}")

def create_cross_experiment_comparison_grid(grid_data, output_path):
    """
    Creates a 4x4 summary grid comparing Max/Min Efficacy for PI3Ki-MEKi vs AKTi-MEKi.
    """
    row_keys = ['target_activation', 'cell_signals', 'cell_rates', 'alive_apoptotic']
    col_keys = [
        'PI3Ki-MEKi_Max Efficacy', 'PI3Ki-MEKi_Min Efficacy',
        'AKTi-MEKi_Max Efficacy', 'AKTi-MEKi_Min Efficacy'
    ]

    # Adjust figsize for better proportions and compactness
    fig, axes = plt.subplots(4, 4, figsize=(11, 10), dpi=300, sharex=True, sharey='row')

    def _plot_on_ax(ax, data, plot_info):
        for col, props in plot_info['series'].items():
            if col in data.columns:
                ax.plot(data['time'], data[col], label=props.get('label', col), color=props['color'], linewidth=2.0)
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
        if plot_info.get('ylim'):
            ax.set_ylim(plot_info['ylim'])
        # Legend will be handled outside this helper

    # Base plot configurations
    base_plot_configs = {
        'alive_apoptotic': {
            'ylabel': "Normalized Cell Count", 'ylim': (-5, 105),
            'series': {
                'normalized_live': {'label': 'Live', 'color': CB_PALETTE['green']},
                'normalized_apoptotic': {'label': 'Apoptotic', 'color': CB_PALETTE['red']}
            }
        },
        'cell_rates': {
            'ylabel': "Rate (1/min)", 'ylim': (0, 0.0012),
            'series': {'growth_rate': {'label': 'Growth', 'color': CB_PALETTE['green']}, 'apoptosis_rate': {'label': 'Apoptosis', 'color': CB_PALETTE['red']}}
        },
        'cell_signals': {
            'ylabel': "Signal Value",
            'series': {'S_pro_real': {'label': 'Pro-survival', 'color': CB_PALETTE['green']}, 'S_anti_real': {'label': 'Anti-survival', 'color': CB_PALETTE['red']}}
        }
    }

    def get_target_activation_config(col_key):
        config = {'ylabel': "% Activated Cells", 'ylim': (0, 101), 'series': {}}
        if 'pi3k' in col_key.lower():
            config['series']['pi3k_node_frac'] = {'label': 'PI3K Active', 'color': CB_PALETTE['blue']}
            config['series']['mek_node_frac'] = {'label': 'MEK Active', 'color': CB_PALETTE['orange']}
        elif 'akt' in col_key.lower():
            config['series']['akt_node_frac'] = {'label': 'AKT Active', 'color': CB_PALETTE['green']}
            config['series']['mek_node_frac'] = {'label': 'MEK Active', 'color': CB_PALETTE['orange']}
        return config

    data_map = {
        'target_activation': 'activated_fractions', 'cell_signals': 'cell_signals',
        'cell_rates': 'cell_rates', 'alive_apoptotic': 'alive_apoptotic'
    }

    for i, row_key in enumerate(row_keys):
        for j, col_key in enumerate(col_keys):
            ax = axes[i, j]

            plot_info = get_target_activation_config(col_key) if row_key == 'target_activation' else base_plot_configs[row_key]

            if j == 0:
                ax.set_ylabel(plot_info['ylabel'], fontweight='bold', fontsize=10)

            if i == 0:
                title_parts = col_key.split('_')
                exp_title = title_parts[0]
                case_title = title_parts[1]
                # No newline in main title
                title_text = f"{exp_title}\\n{case_title}"
                if col_key in grid_data:
                    _, row_data = grid_data[col_key]
                    if 'delta_time' in row_data and pd.notna(row_data['delta_time']):
                        # Add time on a new line
                        title_text += f"\\n(Δt={int(row_data['delta_time'])} min)"
                ax.set_title(title_text, fontweight='bold', fontsize=10)
            
            if col_key not in grid_data:
                ax.text(0.5, 0.5, "Data Not\\nFound", ha='center', va='center', fontsize=9, alpha=0.5)
                ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
                continue

            sim_data, _ = grid_data[col_key]
            data_df = sim_data[data_map[row_key]]
            
            _plot_on_ax(ax, data_df, plot_info)
            
            # Add a legend for each inhibitor combination's column
            if j in [1, 3] and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)

            if i == len(row_keys) - 1:
                ax.set_xlabel("Time (min)", fontsize=11, fontweight='bold')
            
            ax.tick_params(axis='both', labelsize=9)
    
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved cross-experiment comparison grid to {output_path}")

def generate_individual_plots(sim_data, row, output_dir, exp_name):
    """
    Generates all individual plots for a single, processed simulation run.
    """
    selection_reason = row['selection_reason']
    
    # Filter time=0 data for dynamics plots (rates, signals, nodes)
    sim_data['cell_rates'] = sim_data['cell_rates'][sim_data['cell_rates']['time'] != 0].copy()
    sim_data['cell_signals'] = sim_data['cell_signals'][sim_data['cell_signals']['time'] != 0].copy()
    sim_data['node_states'] = sim_data['node_states'][sim_data['node_states']['time'] != 0].copy()
    sim_data['activated_fractions'] = sim_data['activated_fractions'][sim_data['activated_fractions']['time'] != 0].copy()
    
    # Add time information to title for specific cases
    title_prefix = selection_reason
    if selection_reason in ['Max Efficacy', 'Min Efficacy'] and 'delta_time' in row and pd.notna(row['delta_time']):
        title_prefix = f"{selection_reason} (Δt={int(row['delta_time'])} min)"

    scenario_name = f"X_{row['x_diff']}_Y_{row['y_diff']}"
    plots_output_dir = os.path.join(output_dir, scenario_name)

    # Define all plot configurations
    plots_to_generate = {
        'alive_apoptotic': {
            'title': f"{title_prefix}: Cell Counts",
            'ylabel': "Normalized Cell Count", 'ylim': (-5, 105),
            'save_path': os.path.join(plots_output_dir, f"{selection_reason}_cell_counts_normalized.png"),
            'series': {
                'normalized_live': {'label': 'Live', 'color': CB_PALETTE['green']},
                'normalized_apoptotic': {'label': 'Apoptotic', 'color': CB_PALETTE['red']}
            }
        },
        'cell_rates': {
            'title': f"{title_prefix}: Cell Rates",
            'ylabel': "Rate (1/min)", 'ylim': (0, 0.0012),
            'save_path': os.path.join(plots_output_dir, f"{selection_reason}_cell_rates.png"),
            'series': {
                'growth_rate': {'label': 'Growth', 'color': CB_PALETTE['green']},
                'apoptosis_rate': {'label': 'Apoptosis', 'color': CB_PALETTE['red']}
            }
        },
        'cell_signals': {
            'title': f"{title_prefix}: Cell Signals",
            'ylabel': "Signal Value",
            'save_path': os.path.join(plots_output_dir, f"{selection_reason}_cell_signals.png"),
            'series': {
                'S_pro_real': {'label': 'Pro-survival', 'color': CB_PALETTE['green']},
                'S_anti_real': {'label': 'Anti-survival', 'color': CB_PALETTE['red']}
            }
        },
        'pro_survival_nodes': {
            'title': f"{title_prefix}: Pro-Survival Nodes",
            'ylabel': "Node Activity",
            'save_path': os.path.join(plots_output_dir, f"{selection_reason}_pro_survival_nodes.png"),
            'series': {
                'node_cMYC': {'label': 'cMYC', 'color': CB_PALETTE['orange']},
                'node_RSK': {'label': 'RSK', 'color': CB_PALETTE['blue']},
                'node_TCF': {'label': 'TCF', 'color': CB_PALETTE['green']}
            }
        },
        'anti_survival_nodes': {
            'title': f"{title_prefix}: Anti-Survival Nodes",
            'ylabel': "Node Activity",
            'save_path': os.path.join(plots_output_dir, f"{selection_reason}_anti_survival_nodes.png"),
            'series': {
                'node_FOXO': {'label': 'FOXO', 'color': CB_PALETTE['red']},
                'node_Caspase8': {'label': 'Caspase8', 'color': CB_PALETTE['purple']},
                'node_Caspase9': {'label': 'Caspase9', 'color': CB_PALETTE['grey']}
            }
        },
        'pro_survival_nodes_boolean': {
            'title': f"{title_prefix}: Pro-Survival Nodes (Boolean)",
            'ylabel': "% Activated Cells", 'ylim': (0, 101),
            'save_path': os.path.join(plots_output_dir, f"{selection_reason}_pro_survival_nodes_boolean.png"),
            'series': {
                'node_cMYC_frac': {'label': 'cMYC', 'color': CB_PALETTE['orange']},
                'node_RSK_frac': {'label': 'RSK', 'color': CB_PALETTE['blue']},
                'node_TCF_frac': {'label': 'TCF', 'color': CB_PALETTE['green']}
            }
        },
        'anti_survival_nodes_boolean': {
            'title': f"{title_prefix}: Anti-Survival Nodes (Boolean)",
            'ylabel': "% Activated Cells", 'ylim': (0, 101),
            'save_path': os.path.join(plots_output_dir, f"{selection_reason}_anti_survival_nodes_boolean.png"),
            'series': {
                'node_FOXO_frac': {'label': 'FOXO', 'color': CB_PALETTE['red']},
                'node_Caspase8_frac': {'label': 'Caspase8', 'color': CB_PALETTE['purple']},
                'node_Caspase9_frac': {'label': 'Caspase9', 'color': CB_PALETTE['grey']}
            }
        }
    }

    target_activation_plot = {
        'title': f"{title_prefix}: Target Node Activation",
        'ylabel': "% Activated Cells", 'ylim': (0, 101),
        'save_path': os.path.join(plots_output_dir, f"{selection_reason}_target_activation.png"),
        'series': {}
    }
    if 'pi3k' in exp_name:
        target_activation_plot['series']['pi3k_node_frac'] = {'label': 'PI3K Active', 'color': CB_PALETTE['blue']}
        target_activation_plot['series']['mek_node_frac'] = {'label': 'MEK Active', 'color': CB_PALETTE['orange']}
    elif 'akt' in exp_name:
        target_activation_plot['series']['akt_node_frac'] = {'label': 'AKT Active', 'color': CB_PALETTE['green']}
        target_activation_plot['series']['mek_node_frac'] = {'label': 'MEK Active', 'color': CB_PALETTE['orange']}
    
    if target_activation_plot['series']:
        plots_to_generate['target_activation'] = target_activation_plot

    # Generate all plots for this run
    plot_dynamics(sim_data['alive_apoptotic'], plots_to_generate['alive_apoptotic'])
    plot_dynamics(sim_data['cell_rates'], plots_to_generate['cell_rates'])
    plot_dynamics(sim_data['node_states'], plots_to_generate['pro_survival_nodes'])
    plot_dynamics(sim_data['node_states'], plots_to_generate['anti_survival_nodes'])
    plot_dynamics(sim_data['cell_signals'], plots_to_generate['cell_signals'])
    if 'target_activation' in plots_to_generate:
        plot_dynamics(sim_data['activated_fractions'], plots_to_generate['target_activation'])
    if 'pro_survival_nodes_boolean' in plots_to_generate:
        plot_dynamics(sim_data['activated_fractions'], plots_to_generate['pro_survival_nodes_boolean'])
    if 'anti_survival_nodes_boolean' in plots_to_generate:
        plot_dynamics(sim_data['activated_fractions'], plots_to_generate['anti_survival_nodes_boolean'])

def load_representative_data(row, exp_name):
    """
    Loads and processes the data for a single representative simulation run.
    """
    selection_reason = row['selection_reason']
    
    instance_folder = f"instance_{int(row['individual'])}_{int(row['replicate'])}"
    if 'iteration' in row and pd.notna(row['iteration']):
        instance_folder = f"instance_{int(row['iteration'])}_{instance_folder}"

    sim_file_path = os.path.join("experiments", exp_name, instance_folder, 'pcdl_total_info_sim.csv.gz')

    if not os.path.exists(sim_file_path):
        logging.warning(f"Could not find simulation file: {sim_file_path} for case {selection_reason}. Skipping.")
        return None

    logging.info(f"Loading data for {selection_reason} from {os.path.basename(exp_name)}...")
    
    try:
        raw_df = pd.read_csv(sim_file_path, compression='gzip')
        sim_data = get_simulation_data(raw_df)
        del raw_df
        gc.collect()
        return sim_data
    except Exception as e:
        logging.error(f"Failed to read or process {sim_file_path}. Error: {e}")
        return None

if __name__ == '__main__':
    experiments = {
        "synergy_sweep-pi3k_mek-1606-0214-4p_3D_drugtiming": "PI3Ki-MEKi",
        "synergy_sweep-akt_mek-1606-0214-4p_3D_drugtiming": "AKTi-MEKi"
    }

    base_representatives_dir = "scripts/post_emews_analysis/synergy_recovery_experiments/representative_simulations"
    base_output_dir = "results/representative_dynamics"

    # --- New logic to iterate by scenario instead of by experiment ---
    
    pi3k_mek_exp_name = "synergy_sweep-pi3k_mek-1606-0214-4p_3D_drugtiming"
    akt_mek_exp_name = "synergy_sweep-akt_mek-1606-0214-4p_3D_drugtiming"
    
    pi3k_mek_reps_dir = os.path.join(base_representatives_dir, pi3k_mek_exp_name)
    akt_mek_reps_dir = os.path.join(base_representatives_dir, akt_mek_exp_name)

    # Make sure both representative directories exist before starting
    if not os.path.isdir(pi3k_mek_reps_dir) or not os.path.isdir(akt_mek_reps_dir):
        logging.error("One or both required representative simulation directories not found. Aborting analysis.")
        exit()

    # Use the scenarios from the PI3K-MEK experiment as the reference
    for scenario_file in os.listdir(pi3k_mek_reps_dir):
        if not (scenario_file.startswith('representatives_') and scenario_file.endswith('.csv')):
            continue
        
        logging.info(f"\n--- Processing Scenario: {scenario_file} ---")

        # Define paths for both experiments for the current scenario
        pi3k_mek_reps_path = os.path.join(pi3k_mek_reps_dir, scenario_file)
        akt_mek_reps_path = os.path.join(akt_mek_reps_dir, scenario_file)

        if not os.path.exists(akt_mek_reps_path):
            logging.warning(f"Scenario file {scenario_file} not found for AKTi-MEKi experiment. Skipping cross-comparison for this scenario.")
            continue
            
        # --- Data collection for the new 4x4 grid ---
        cross_exp_data_collector = {}
        
        # Process PI3Ki-MEKi cases
        reps_df_pi3k_mek = pd.read_csv(pi3k_mek_reps_path)
        for case in ['Max Efficacy', 'Min Efficacy']:
            case_df = reps_df_pi3k_mek[reps_df_pi3k_mek['selection_reason'] == case]
            if not case_df.empty:
                row = case_df.iloc[0]
                sim_data = load_representative_data(row, pi3k_mek_exp_name)
                if sim_data:
                    cross_exp_data_collector[f"PI3Ki-MEKi_{case}"] = (sim_data, row)
        
        # Process AKTi-MEKi cases
        reps_df_akt_mek = pd.read_csv(akt_mek_reps_path)
        for case in ['Max Efficacy', 'Min Efficacy']:
            case_df = reps_df_akt_mek[reps_df_akt_mek['selection_reason'] == case]
            if not case_df.empty:
                row = case_df.iloc[0]
                sim_data = load_representative_data(row, akt_mek_exp_name)
                if sim_data:
                    cross_exp_data_collector[f"AKTi-MEKi_{case}"] = (sim_data, row)

        # Get control bounds from the data itself
        control_min, control_max = get_control_bounds_from_data(cross_exp_data_collector)

        # Normalize all collected data for the grid
        if control_min is not None and control_max is not None and (control_max - control_min) > 0:
            norm_range = control_max - control_min
            for sim_data, _ in cross_exp_data_collector.values():
                df = sim_data['alive_apoptotic']
                df['normalized_live'] = ((df['live'] - control_min) / norm_range) * 100
                df['normalized_apoptotic'] = (df['apoptotic'] / norm_range) * 100
        
        # Generate the new cross-experiment comparison grid if all data is available
        if len(cross_exp_data_collector) == 4:
            scenario_name_for_path = scenario_file.replace('representatives_', '').replace('.csv', '')
            grid_plot_path = os.path.join(base_output_dir, "cross_experiment_comparison", f"{scenario_name_for_path}_comparison_grid.png")
            create_cross_experiment_comparison_grid(cross_exp_data_collector, grid_plot_path)
        else:
            logging.warning(f"Missing data for cross-experiment grid for scenario {scenario_file}. Skipping grid plot.")

    # --- Previous logic for individual experiment summaries (run separately) ---
    for exp_name, exp_short_name in experiments.items():
        logging.info(f"\n--- Processing Individual Summaries for Experiment: {exp_name} ---")
        
        representatives_dir = os.path.join(base_representatives_dir, exp_name)
        if not os.path.isdir(representatives_dir):
            logging.warning(f"Representatives directory not found for {exp_name}. Skipping.")
            continue

        exp_plot_dir = os.path.join(base_output_dir, exp_name)

        for f in os.listdir(representatives_dir):
            if f.startswith('representatives_') and f.endswith('.csv'):
                reps_df_path = os.path.join(representatives_dir, f)
                logging.info(f"\n--- Analyzing Scenario: {os.path.basename(reps_df_path)} for {exp_short_name} ---")
                reps_df = pd.read_csv(reps_df_path)

                # --- Step 1: Collect data for key cases for global normalization ---
                data_collector = {}
                key_cases = ['Max Efficacy', 'Min Efficacy', 'Simultaneous']
                
                for case in key_cases:
                    case_df = reps_df[reps_df['selection_reason'] == case]
                    if not case_df.empty:
                        row = case_df.iloc[0]
                        sim_data = load_representative_data(row, exp_name)
                        if sim_data:
                            data_collector[case] = (sim_data, row)
                    else:
                        logging.warning(f"Case '{case}' not found in {os.path.basename(reps_df_path)}. It will be skipped for the summary grid.")

                # If we don't have all cases for the main summary, there's no point proceeding with this file.
                if len(data_collector) < len(key_cases):
                    logging.warning(f"Could not perform global normalization for {f} - missing one or more key cases. Skipping summary plot for this scenario.")
                    continue
                
                # --- Step 2: Get control bounds from the data itself ---
                control_min, control_max = get_control_bounds_from_data(data_collector)

                # --- Step 3: Perform Global Min-Max Normalization using Control Bounds ---
                if control_min is not None and control_max is not None:
                    norm_range = control_max - control_min
                    if norm_range > 0:
                        for case_data, _ in data_collector.values():
                            df = case_data['alive_apoptotic']
                            df['normalized_live'] = ((df['live'] - control_min) / norm_range) * 100
                            df['normalized_apoptotic'] = (df['apoptotic'] / norm_range) * 100
                    else:
                        logging.warning("Control normalization range is zero. Defaulting to 50.")
                        for case_data, _ in data_collector.values():
                            df = case_data['alive_apoptotic']
                            df['normalized_live'] = 50.0
                            df['normalized_apoptotic'] = 0.0
                else:
                    logging.warning("Control bounds not available. Skipping global normalization for summary plot.")

                # --- Step 4: Generate Summary Grid Plot ---
                scenario_name = f"X_{reps_df.iloc[0]['x_diff']}_Y_{reps_df.iloc[0]['y_diff']}"
                grid_plot_path = os.path.join(exp_plot_dir, scenario_name, "summary_grid.png")
                create_summary_grid_plot(data_collector, grid_plot_path, experiments[exp_name])
                
                # --- Step 5: Generate all individual plots ---
                for _, row in reps_df.iterrows():
                    reason = row['selection_reason']
                    
                    # This logic now needs to handle the tuple format of data_collector
                    if reason in data_collector:
                        sim_data_to_plot, _ = data_collector[reason]
                    else:
                        # Load data for other cases and normalize locally if control is unavailable
                        sim_data_to_plot = load_representative_data(row, exp_name)
                        if sim_data_to_plot:
                            df = sim_data_to_plot['alive_apoptotic']
                            if control_min is not None and control_max is not None:
                                norm_range = control_max - control_min
                                if norm_range > 0:
                                    df['normalized_live'] = ((df['live'] - control_min) / norm_range) * 100
                                    df['normalized_apoptotic'] = (df['apoptotic'] / norm_range) * 100
                                else:
                                    df['normalized_live'] = 50.0
                                    df['normalized_apoptotic'] = 0.0
                            else: # Fallback to local normalization
                                min_live, max_live = df['live'].min(), df['live'].max()
                                if (max_live - min_live) > 0:
                                    norm_range = max_live - min_live
                                    df['normalized_live'] = ((df['live'] - min_live) / norm_range) * 100
                                    df['normalized_apoptotic'] = (df['apoptotic'] / norm_range) * 100
                                else:
                                    df['normalized_live'] = 50.0
                                    df['normalized_apoptotic'] = 0.0
                    
                    if sim_data_to_plot:
                        generate_individual_plots(sim_data_to_plot, row, exp_plot_dir, exp_name)

    logging.info("\n--- Analysis Complete ---") 