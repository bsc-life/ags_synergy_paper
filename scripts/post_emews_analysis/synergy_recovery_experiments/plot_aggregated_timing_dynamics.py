import pandas as pd
import matplotlib
matplotlib.use('agg') # Must be called before pyplot is imported
import matplotlib.pyplot as plt
# --- Nature-style plotting guidelines ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'Helvetica']
# Increase font sizes for better readability in Google Docs
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['figure.titlesize'] = 20
# ------------------------------------------
import os
import logging
import gc
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pathlib import Path  # For cache file paths
import argparse
import multiprocessing as mp

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use a non-interactive backend suitable for scripts - DEPRECATED, moved to top
# plt.switch_backend('agg')

# Define a consistent, colorblind-friendly palette
CB_PALETTE = {
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
    'red': '#D55E00', 'purple': '#CC79A7', 'grey': '#999999',
}

# -----------------------------------------------------------------------------
# Feather caching helpers
# -----------------------------------------------------------------------------

def _cache_path(exp_name: str, label: str, key: str, dose: float):
    """Return a Path to the cached feather file for a given group.

    The directory structure is::

        results/aggregated_timing_dynamics/<exp_name>/cache_D<dose>/<label>_<key>.feather
    """
    base_dir = Path(f"results/aggregated_timing_dynamics/{exp_name}/cache_D{int(dose)}")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{label}_{key}.feather"


def load_cached_group(exp_name: str, label: str, dose: float):
    """Load cached aggregation for a timing *label* (neg/zero/pos).

    Returns a dict identical to what ``aggregate_dynamics_for_group`` produces or
    ``None`` if any of the expected files is missing.
    """
    keys = [
        "alive_apoptotic",
        "cell_signals",
        "cell_rates",
        "activated_fractions",
    ]

    # Fast check: if the first expected file is missing we bail out early
    if not _cache_path(exp_name, label, keys[0], dose).exists():
        return None

    cached_data = {}
    try:
        for k in keys:
            path = _cache_path(exp_name, label, k, dose)
            if not path.exists():
                return None  # Incomplete cache, ignore
            df = pd.read_feather(path)
            # Backward-compatibility: older caches used 'live' instead of 'alive'
            if k == "alive_apoptotic":
                rename_map = {
                    "live_mean": "alive_mean",
                    "live_std": "alive_std",
                    "live": "alive"  # In case non-aggregated form is cached
                }
                overlap = [col for col in rename_map if col in df.columns]
                if overlap:
                    df = df.rename(columns={col: rename_map[col] for col in overlap})
            cached_data[k] = df
        return cached_data
    except Exception as exc:
        logging.warning(f"Failed to load cache for {exp_name}:{label} – ignoring. Error: {exc}")
        return None


def dump_cached_group(exp_name: str, label: str, dose: float, data_dict: dict):
    """Persist aggregated DataFrames to feather files so future runs can reuse them."""
    for k, df in data_dict.items():
        try:
            if df is not None and not df.empty:
                # --- FIX: Ensure legacy 'live' columns are renamed before saving to cache ---
                if k == "alive_apoptotic" and 'live_mean' in df.columns:
                    df = df.rename(columns={
                        'live_mean': 'alive_mean',
                        'live_std': 'alive_std'
                    })
                df.reset_index(drop=True).to_feather(_cache_path(exp_name, label, k, dose))
        except Exception as exc:
            logging.warning(f"Failed to write cache for {exp_name}:{label}:{k}. Error: {exc}")

# -----------------------------------------------------------------------------
# Fast helpers for control-bound computation
# -----------------------------------------------------------------------------

_GLOBAL_CONTROL_CACHE = {}

def _control_bounds_cache_file(exp_name: str, dose: float):
    """Return a Path where min/max control bounds are cached (simple txt)."""
    base_dir = Path(f"results/aggregated_timing_dynamics/{exp_name}/cache_D{int(dose)}")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "control_bounds.txt"

def _extract_live_counts(sim_file_path):
    """Read compressed csv and return a Series indexed by time with live counts."""
    try:
        df = pd.read_csv(sim_file_path, compression='gzip', usecols=['time', 'current_phase'])
        counts_df = df.groupby('time')['current_phase'].value_counts().unstack(fill_value=0)

        # Derive live counts as the sum of all phases except 'apoptotic'.
        if 'apoptotic' not in counts_df.columns:
            counts_df['apoptotic'] = 0
        phase_cols = [c for c in counts_df.columns if c != 'apoptotic']
        counts_df['alive'] = counts_df[phase_cols].sum(axis=1)

        return counts_df['alive']
    except Exception as exc:
        logging.warning(f"Control extraction failed for {sim_file_path}: {exc}")
        return None

def get_control_bounds_from_data(raw_df):
    """
    Gets control bounds from cases with lowest diffusion coefficient and highest time of addition (t=4000).
    This serves as an internal control instead of using external no-drug data.
    """
    late_pulse_threshold = 4000
    low_dose = 6.0  # Lowest diffusion coefficient
    
    # First: quick cache lookup (shared across runs of script)
    cache_key_global = "ctrl_minmax"
    if cache_key_global in _GLOBAL_CONTROL_CACHE:
        return _GLOBAL_CONTROL_CACHE[cache_key_global]
    
    # Find control cases (low diffusion, late addition for both drugs)
    control_condition = (
        (raw_df['user_parameters.drug_X_diffusion_coefficient'] == low_dose) &
        (raw_df['user_parameters.drug_Y_diffusion_coefficient'] == low_dose) &
        (raw_df['user_parameters.drug_X_pulse_period'] >= late_pulse_threshold) &
        (raw_df['user_parameters.drug_Y_pulse_period'] >= late_pulse_threshold)
    )
    control_runs_df = raw_df[control_condition]
    
    if control_runs_df.empty:
        logging.warning("No control cases found (high time, low diffusion). Cannot normalize.")
        return None, None

    logging.info(f"Found {len(control_runs_df)} control cases with low diffusion and late addition.")
    
    # Build list of file paths
    file_paths = []
    for _, row in control_runs_df.iterrows():
        parts = []
        if "iteration" in row.index and pd.notna(row['iteration']):
            parts.append(str(int(row['iteration'])))
        parts.append(str(int(row['individual'])))
        parts.append(str(int(row['replicate'])))
        instance_folder = f"instance_{'_'.join(parts)}"
        sim_file_path = os.path.join("experiments", row['experiment_name'], instance_folder, 'pcdl_total_info_sim.csv.gz')
        if os.path.exists(sim_file_path):
            file_paths.append(sim_file_path)

    if not file_paths:
        logging.warning("No control growth curves could be processed. Cannot normalize.")
        return None, None

    # Try cache file first
    cache_file = _control_bounds_cache_file(control_runs_df['experiment_name'].iloc[0], low_dose)
    if cache_file.exists():
        try:
            with open(cache_file) as fh:
                vals = fh.read().strip().split(',')
                control_min, control_max = float(vals[0]), float(vals[1])
                _GLOBAL_CONTROL_CACHE[cache_key_global] = (control_min, control_max)
                logging.info(f"Loaded cached control bounds Min={control_min}, Max={control_max}")
                return control_min, control_max
        except Exception:
            pass  # fallback to recompute

    # Parallel extraction of live counts
    max_workers = min(8, mp.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        series_list = list(pool.map(_extract_live_counts, file_paths))

    valid_series = [s for s in series_list if s is not None]
    if not valid_series:
        logging.warning("No valid control curves after processing.")
        return None, None

    all_curves_df = pd.concat(valid_series, axis=1)
    mean_growth = all_curves_df.mean(axis=1)
    control_min, control_max = float(mean_growth.min()), float(mean_growth.max())

    # Persist to cache file
    try:
        with open(cache_file, 'w') as fh:
            fh.write(f"{control_min},{control_max}")
    except Exception as exc:
        logging.warning(f"Failed to write control bounds cache: {exc}")

    logging.info(f"Control bounds determined: Min={control_min}, Max={control_max}")
    _GLOBAL_CONTROL_CACHE[cache_key_global] = (control_min, control_max)
    return control_min, control_max

def get_simulation_data(file_path):
    """
    Loads a single simulation file and extracts key dynamics data.
    Optimized to only read necessary columns.
    """
    try:
        # Base columns always required
        base_cols = ['time', 'current_phase', 'apoptosis_rate', 'growth_rate', 'S_pro_real', 'S_anti_real']
        # Target node columns we want to track (boolean activation)
        target_nodes = ['pi3k_node', 'akt_node', 'mek_node',
                        'node_cMYC', 'node_RSK', 'node_TCF',
                        'node_FOXO', 'node_Caspase8', 'node_Caspase9']
        read_cols = base_cols + target_nodes

        # Load only needed columns; Pandas silently skips missing ones
        raw_df = pd.read_csv(file_path, compression='gzip', usecols=lambda c: c in read_cols)

        if raw_df.empty or 'time' not in raw_df.columns:
            logging.warning(f"Skipping file {os.path.basename(file_path)}: empty or no 'time' column.")
            return None

        # Extract alive/apoptotic counts
        alive_apoptotic_df = raw_df.groupby('time')['current_phase'].value_counts().unstack(fill_value=0).reset_index()

        if 'apoptotic' not in alive_apoptotic_df.columns:
            alive_apoptotic_df['apoptotic'] = 0
        phase_cols = [c for c in alive_apoptotic_df.columns if c not in ['time', 'apoptotic']]
        alive_apoptotic_df['alive'] = alive_apoptotic_df[phase_cols].sum(axis=1)
        alive_apoptotic_df = alive_apoptotic_df[['time', 'alive', 'apoptotic']]
        
        # Extract cell rates and signals
        cell_rates_df = raw_df.groupby('time')[['apoptosis_rate', 'growth_rate']].mean().reset_index()
        cell_signals_df = raw_df.groupby('time')[['S_pro_real', 'S_anti_real']].mean().reset_index()

        # Calculate fraction of activated cells
        all_nodes_interest = target_nodes
        available_nodes = [node for node in all_nodes_interest if node in raw_df.columns]
        
        if not available_nodes:
            frac_df = pd.DataFrame({'time': raw_df['time'].unique()})
            for node in all_nodes_interest:
                frac_df[f'{node}_frac'] = np.nan
        else:
            for node in available_nodes:
                raw_df[f'{node}_activated'] = (raw_df[node] > 0.5).astype(int)
            
            activated_cols = [f'{node}_activated' for node in available_nodes]
            frac_activated = raw_df.groupby('time')[activated_cols].mean().reset_index()

            rename_dict = {f'{node}_activated': f'{node}_frac' for node in available_nodes}
            frac_df = frac_activated.rename(columns=rename_dict)
            
            for node in available_nodes:
                frac_df[f'{node}_frac'] *= 100

        return {
            "alive_apoptotic": alive_apoptotic_df,
            "cell_signals": cell_signals_df,
            "cell_rates": cell_rates_df,
            "activated_fractions": frac_df
        }

    except Exception as e:
        logging.error(f"Failed to read or process {os.path.basename(file_path)}. Error: {e}")
        return None

def aggregate_dynamics_for_group(group_df, exp_name, executor=None, max_workers=4):
    """
    Loads all simulation runs for a given group of parameters in parallel,
    and returns aggregated mean/std dynamics for multiple data types.
    """
    if group_df.empty:
        return None

    file_paths = []
    experiment_folder = f"experiments/{exp_name}"
    for _, row in group_df.iterrows():
        # ... (rest of file path logic is unchanged)
        instance_folder_parts = []
        if "iteration" in row.index and pd.notna(row['iteration']):
            instance_folder_parts.append(str(int(row['iteration'])))
        instance_folder_parts.append(str(int(row['individual'])))
        instance_folder_parts.append(str(int(row['replicate'])))
        instance_folder = f"instance_{'_'.join(instance_folder_parts)}"
        sim_file_path = os.path.join(experiment_folder, instance_folder, 'pcdl_total_info_sim.csv.gz')

        if os.path.exists(sim_file_path):
            file_paths.append(sim_file_path)

    if not file_paths:
        return None

    all_instance_data = {'alive_apoptotic': [], 'cell_signals': [], 'cell_rates': [], 'activated_fractions': []}
    
    exec_to_use = executor
    _local_pool = None
    if exec_to_use is None:
        _local_pool = ProcessPoolExecutor(max_workers=max_workers)
        exec_to_use = _local_pool

    try:
        results = list(exec_to_use.map(get_simulation_data, file_paths))
    finally:
        if _local_pool is not None:
            _local_pool.shutdown(wait=True)

    for sim_data in results:
        if sim_data:
            for key in all_instance_data:
                if key in sim_data and sim_data[key] is not None:
                    all_instance_data[key].append(sim_data[key])

    if not any(all_instance_data.values()):
        logging.warning(f"No valid simulation data could be processed for group in {exp_name}.")
        return None

    aggregated_stats = {}
    stats_map = {
        'alive_apoptotic': [('alive', 'alive'), ('apoptotic', 'apoptotic')],
        'cell_signals': [('S_pro_real', 'pro_survival'), ('S_anti_real', 'anti_survival')],
        'cell_rates': [('growth_rate', 'growth'), ('apoptosis_rate', 'apoptosis')],
        'activated_fractions': [
            (f'{node}_frac', node) for node in [
                'pi3k_node', 'akt_node', 'mek_node', 'node_cMYC', 'node_RSK', 
                'node_TCF', 'node_FOXO', 'node_Caspase8', 'node_Caspase9'
            ]
        ]
    }

    for data_key, fields in stats_map.items():
        all_dfs = all_instance_data.get(data_key, [])
        if not all_dfs:
            continue

        valid_dfs = [df for df in all_dfs if not df.empty and 'time' in df.columns]
        if not valid_dfs:
            continue

        master_time_index = pd.concat([df.set_index('time') for df in valid_dfs]).index.unique()
        
        aligned_dfs = []
        for df in valid_dfs:
            aligned_df = df.set_index('time').reindex(master_time_index).ffill().reset_index()
            aligned_dfs.append(aligned_df)
        
        if not aligned_dfs:
            continue

        combined_df = pd.concat(aligned_dfs, ignore_index=True)
        if combined_df.empty or 'time' not in combined_df.columns:
            continue

        agg_dict = {}
        for field, prefix in fields:
            if field in combined_df.columns:
                agg_dict[f'{prefix}_mean'] = (field, 'mean')
                agg_dict[f'{prefix}_std'] = (field, 'std')
        
        if agg_dict:
            aggregated_stats[data_key] = combined_df.groupby('time').agg(**agg_dict).reset_index()

    return aggregated_stats if aggregated_stats else None

def create_aggregated_dynamics_grid_plot(data, output_path, exp_name, control_min, control_max):
    """
    Creates a 4x3 summary grid plot for aggregated dynamics.
    Rows are dynamics types, columns are timing cases.
    """
    row_keys = ['target_activation', 'cell_signals', 'cell_rates', 'alive_apoptotic']
    drug_x_name = "AKTi" if "akt_mek" in exp_name else "PI3Ki"
    drug_y_name = "MEKi"
    col_order = [f'{drug_x_name} First', 'Simultaneous', f'{drug_y_name} First']
    data_keys_map = {col_order[0]: 'neg', col_order[1]: 'zero', col_order[2]: 'pos'}

    fig, axes = plt.subplots(4, 3, figsize=(14, 16), dpi=300, sharex=True, sharey='row')

    plot_configs = {
        'alive_apoptotic': {
            'ylabel': "Cell Count",
            'series': {
                'alive': {'label': 'Alive', 'color': CB_PALETTE['green']},
                'apoptotic': {'label': 'Apoptotic', 'color': CB_PALETTE['red']}
            }
        },
        'cell_rates': {
            'ylabel': "Rate (1/min)", 'ylim': (0, 0.0012),
            'series': {
                'growth': {'label': 'Growth', 'color': CB_PALETTE['green']},
                'apoptosis': {'label': 'Apoptosis', 'color': CB_PALETTE['red']}
            }
        },
        'cell_signals': {
            'ylabel': "Signal Value", 'ylim': (0, 1.05),
            'series': {
                'pro_survival': {'label': 'Pro-survival', 'color': CB_PALETTE['green']},
                'anti_survival': {'label': 'Anti-survival', 'color': CB_PALETTE['red']}
            }
        },
        'target_activation': {
            'ylabel': "% Activated Cells", 'ylim': (0, 101), 'series': {}
        }
    }
    # Dynamically set target activation series based on experiment
    if 'pi3k' in exp_name.lower():
        plot_configs['target_activation']['series']['pi3k_node'] = {'label': 'PI3K Active', 'color': CB_PALETTE['blue']}
        plot_configs['target_activation']['series']['mek_node'] = {'label': 'MEK Active', 'color': CB_PALETTE['orange']}
    elif 'akt' in exp_name.lower():
        plot_configs['target_activation']['series']['akt_node'] = {'label': 'AKT Active', 'color': CB_PALETTE['green']}
        plot_configs['target_activation']['series']['mek_node'] = {'label': 'MEK Active', 'color': CB_PALETTE['orange']}

    data_map = {
        'target_activation': 'activated_fractions', 'cell_signals': 'cell_signals',
        'cell_rates': 'cell_rates', 'alive_apoptotic': 'alive_apoptotic'
    }

    for i, row_key in enumerate(row_keys):
        plot_info = plot_configs[row_key]
        axes[i, 0].set_ylabel(plot_info['ylabel'], fontsize=16)

        for j, case in enumerate(col_order):
            ax = axes[i, j]
            if i == 0:
                ax.set_title(case, fontweight='bold', fontsize=18)

            data_key = data_keys_map[case]
            if data_key not in data or data[data_key] is None or data_map[row_key] not in data[data_key]:
                ax.text(0.5, 0.5, "Data Not Found", ha='center', va='center', alpha=0.5)
                continue
            
            data_df = data[data_key][data_map[row_key]].copy()
            
            # Backward-compat: older cached dataframes might still use 'live' prefix
            if row_key == 'alive_apoptotic' and 'alive_mean' not in data_df.columns and 'live_mean' in data_df.columns:
                data_df = data_df.rename(columns={
                    'live_mean': 'alive_mean',
                    'live_std': 'alive_std'
                })

            # Drop the initial t=0 point for clearer dynamics plots
            if 'time' in data_df.columns:
                data_df = data_df[data_df['time'] != 0].copy()

            # if row_key == 'alive_apoptotic' and control_min is not None and control_max is not None and (control_max - control_min) > 0:
            #     norm_range = control_max - control_min
            #     # Normalise 'alive' against the control growth range
            #     if 'alive_mean' in data_df.columns:
            #         data_df['alive_mean'] = ((data_df['alive_mean'] - control_min) / norm_range) * 100
            #     if 'alive_std' in data_df.columns:
            #         data_df['alive_std'] = (data_df['alive_std'] / norm_range) * 100
            #     
            #     # Normalise 'apoptotic' against the max control population size.
            #     if 'apoptotic_mean' in data_df.columns:
            #         data_df['apoptotic_mean'] = (data_df['apoptotic_mean'] / control_max) * 100
            #     if 'apoptotic_std' in data_df.columns:
            #         data_df['apoptotic_std'] = (data_df['apoptotic_std'] / control_max) * 100

            for series_key, props in plot_info['series'].items():
                mean_col, std_col = f"{series_key}_mean", f"{series_key}_std"
                # Fallback: support legacy 'live' column names if present
                alt_mean_col = mean_col.replace('alive', 'live') if series_key == 'alive' else None
                alt_std_col = std_col.replace('alive', 'live') if series_key == 'alive' else None
                if mean_col in data_df.columns or (alt_mean_col and alt_mean_col in data_df.columns):
                    _mcol = mean_col if mean_col in data_df.columns else alt_mean_col
                    _scol = std_col if std_col in data_df.columns else (alt_std_col if alt_std_col in data_df.columns else None)
                    ax.plot(data_df['time'], data_df[_mcol], label=props['label'], color=props['color'], linewidth=2.0)
                    if _scol and _scol in data_df.columns:
                        ax.fill_between(data_df['time'], 
                                        data_df[_mcol] - data_df[_scol].fillna(0),
                                        data_df[_mcol] + data_df[_scol].fillna(0),
                                        color=props['color'], alpha=0.2)
            
            if plot_info.get('ylim'):
                ax.set_ylim(plot_info['ylim'])
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=14)
            if i == len(row_keys) - 1:
                ax.set_xlabel('Time (min)', fontsize=16)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f'Aggregated Dynamics for {drug_x_name} + {drug_y_name} at High Dose', fontsize=22, weight='bold')
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved aggregated dynamics grid plot to {output_path}")

def create_survival_nodes_grid_plot(data, output_path, exp_name):
    """Creates a 2×3 grid for pro- and anti-survival node activation fractions."""
    row_info = {
        'pro': {
            'nodes': ['node_cMYC', 'node_RSK', 'node_TCF'],
            'labels': ['cMYC', 'RSK', 'TCF'],
            'colors': [CB_PALETTE['orange'], CB_PALETTE['green'], CB_PALETTE['blue']],
            'ylabel': 'Pro-survival (%)'
        },
        'anti': {
            'nodes': ['node_FOXO', 'node_Caspase8', 'node_Caspase9'],
            'labels': ['FOXO', 'Caspase8', 'Caspase9'],
            'colors': [CB_PALETTE['red'], CB_PALETTE['purple'], CB_PALETTE['grey']],
            'ylabel': 'Anti-survival (%)'
        }
    }
    col_order = [f'PI3K/AKT First', 'Simultaneous', 'MEKi First']
    data_keys_map = {'PI3K/AKT First': 'neg', 'Simultaneous': 'zero', 'MEKi First': 'pos'}

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), dpi=300, sharex=True, sharey='row')

    for i, (row_key, info) in enumerate(row_info.items()):
        axes[i, 0].set_ylabel(info['ylabel'], fontsize=16, fontweight='bold')
        for j, col_name in enumerate(col_order):
            ax = axes[i, j]
            if i == 0:
                ax.set_title(col_name, fontweight='bold', fontsize=18)
            dkey = data_keys_map[col_name]
            if dkey not in data or data[dkey] is None or 'activated_fractions' not in data[dkey]:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', alpha=0.5)
                continue
            df = data[dkey]['activated_fractions']
            if 'time' in df.columns:
                df = df[df['time'] != 0].copy()
            for node, label, color in zip(info['nodes'], info['labels'], info['colors']):
                mean_col = f'{node}_mean'
                std_col = f'{node}_std'
                if mean_col in df.columns:
                    ax.plot(df['time'], df[mean_col], label=label, color=color, linewidth=2)
                    if std_col in df.columns:
                        ax.fill_between(df['time'], df[mean_col]-df[std_col].fillna(0),
                                        df[mean_col]+df[std_col].fillna(0), color=color, alpha=0.2)
            ax.set_ylim(0, 101)
            if j == 2:
                ax.legend(fontsize=14)
            if i == 1:
                ax.set_xlabel('Time (min)', fontsize=16)

    plt.tight_layout()
    # Title removed as per visual simplification request
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.png','.svg'), format='svg', bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)

def create_super_summary_grid_plot(aggregated_results, output_path):
    """Builds a 4×6 grid combining both synergy experiments (AKTi-MEKi & PI3Ki-MEKi)
    across four scales (target activation, cell signals, cell rates, alive/apoptotic).

    Columns (6): for each combination we show three timing cases:
        [0] AKTi First (Δt<0)     [1] Simultaneous (Δt=0)     [2] MEKi First (Δt>0)
        [3] PI3Ki First          [4] Simultaneous             [5] MEKi First
    """

    # Determine which aggregated dict corresponds to which inhibitor axis
    akt_key = next((k for k in aggregated_results if 'akt_mek' in k), None)
    pi3k_key = next((k for k in aggregated_results if 'pi3k_mek' in k), None)

    if not akt_key or not pi3k_key:
        logging.error("Could not identify both AKTi-MEKi and PI3Ki-MEKi datasets. Super summary aborted.")
        return

    akt_data = aggregated_results[akt_key]
    pi3k_data = aggregated_results[pi3k_key]

    row_keys = ['target_activation', 'cell_signals', 'cell_rates', 'alive_apoptotic']
    fig, axes = plt.subplots(4, 6, figsize=(22, 11), dpi=300, sharex=True, sharey='row')

    col_titles = [
        'AKTi First', 'Simultaneous', 'MEKi First',
        'PI3Ki First', 'Simultaneous', 'MEKi First'
    ]
    col_cases = [
        ('neg', akt_data), ('zero', akt_data), ('pos', akt_data),
        ('neg', pi3k_data), ('zero', pi3k_data), ('pos', pi3k_data)
    ]

    # Prepare plotting config templates reusing colours
    base_plot_configs = {
        'alive_apoptotic': {
            'ylabel': "Cell Count",
            'series': {
                'alive': {'label': 'Alive', 'color': CB_PALETTE['green']},
                'apoptotic': {'label': 'Apoptotic', 'color': CB_PALETTE['red']}
            }
        },
        'cell_rates': {
            'ylabel': "Rate (1/min)", 'ylim': (0, 0.0012),
            'series': {
                'growth': {'label': 'Growth', 'color': CB_PALETTE['green']},
                'apoptosis': {'label': 'Apoptosis', 'color': CB_PALETTE['red']}
            }
        },
        'cell_signals': {
            'ylabel': "Signal Value", 'ylim': (0, 1.05),
            'series': {
                'pro_survival': {'label': 'Pro-survival', 'color': CB_PALETTE['green']},
                'anti_survival': {'label': 'Anti-survival', 'color': CB_PALETTE['red']}
            }
        }
    }

    data_map = {
        'target_activation': 'activated_fractions', 'cell_signals': 'cell_signals',
        'cell_rates': 'cell_rates', 'alive_apoptotic': 'alive_apoptotic'
    }

    for i, row_key in enumerate(row_keys):
        for j, (case_key, data_dict) in enumerate(col_cases):
            ax = axes[i, j]

            # Set column titles once on first row
            if i == 0:
                ax.set_title(col_titles[j], fontweight='bold', fontsize=18)

            # Set row ylabel once per row
            if j == 0:
                if row_key in base_plot_configs:
                    ax.set_ylabel(base_plot_configs[row_key]['ylabel'], fontsize=16)
                elif row_key == 'target_activation':
                    ax.set_ylabel('% Activated Cells', fontsize=16)

            # Fetch dataframe according to row
            if data_dict is None or case_key not in data_dict:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', alpha=0.5)
                continue

            if row_key in data_map:
                if data_map[row_key] not in data_dict[case_key]:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', alpha=0.5)
                    continue
                df = data_dict[case_key][data_map[row_key]].copy()

                # --- FIX: Explicitly handle legacy 'live' columns before plotting ---
                if 'live_mean' in df.columns and 'alive_mean' not in df.columns:
                    df.rename(columns={'live_mean': 'alive_mean'}, inplace=True)
                if 'live_std' in df.columns and 'alive_std' not in df.columns:
                    df.rename(columns={'live_std': 'alive_std'}, inplace=True)
                
                if 'time' in df.columns:
                    df = df[df['time'] != 0].copy()

                # Normalise alive/apoptotic if row is alive_apoptotic and bounds available
                # if row_key == 'alive_apoptotic' and '_ctrl_bounds' in data_dict:
                #     ctrl_min, ctrl_max = data_dict.get('_ctrl_bounds', (None, None))
                #     if ctrl_min is not None and ctrl_max is not None and (ctrl_max - ctrl_min) > 0:
                #         norm_range = ctrl_max - control_min
                #         # Normalise 'alive' against the control growth range
                #         if 'alive_mean' in df.columns:
                #             df['alive_mean'] = ((df['alive_mean'] - ctrl_min) / norm_range) * 100
                #         if 'alive_std' in df.columns:
                #             df['alive_std'] = (df['alive_std'] / norm_range) * 100
                #         
                #         # Normalise 'apoptotic' against the max control population size.
                #         if 'apoptotic_mean' in df.columns:
                #             df['apoptotic_mean'] = (df['apoptotic_mean'] / ctrl_max) * 100
                #         if 'apoptotic_std' in df.columns:
                #             df['apoptotic_std'] = (df['apoptotic_std'] / control_max) * 100

                plot_info = base_plot_configs[row_key] if row_key in base_plot_configs else None
                if plot_info:
                    for series_col, props in plot_info['series'].items():
                        mean_col = f"{series_col}_mean"
                        std_col = f"{series_col}_std"

                        if mean_col in df.columns:
                            ax.plot(df['time'], df[mean_col], label=props['label'], color=props['color'], linewidth=2)
                            if std_col in df.columns:
                                ax.fill_between(df['time'],
                                                df[mean_col] - df[std_col].fillna(0),
                                                df[mean_col] + df[std_col].fillna(0),
                                                color=props['color'], alpha=0.2)

                    if plot_info.get('ylim'):
                        ax.set_ylim(plot_info['ylim'])
                    if j == 5 and ax.get_legend_handles_labels()[0]:
                        ax.legend(fontsize=14, loc='center left', bbox_to_anchor=(1.02, 0.5))
                else:
                    # Handle target_activation specially
                    candidate_nodes = {
                        'pi3k_node': {'label': 'PI3K', 'color': CB_PALETTE['blue']},
                        'akt_node': {'label': 'AKT', 'color': CB_PALETTE['green']},
                        'mek_node': {'label': 'MEK', 'color': CB_PALETTE['orange']}
                    }
                    for node_key, props in candidate_nodes.items():
                        mean_col = f"{node_key}_mean"
                        std_col = f"{node_key}_std"
                        if mean_col in df.columns:
                            ax.plot(df['time'], df[mean_col], label=props['label'], color=props['color'], linewidth=2)
                            if std_col in df.columns:
                                ax.fill_between(df['time'],
                                                df[mean_col] - df[std_col].fillna(0),
                                                df[mean_col] + df[std_col].fillna(0),
                                                color=props['color'], alpha=0.2)
                    ax.set_ylim(0, 101)
                    if j == 5 and ax.get_legend_handles_labels()[0]:
                        ax.legend(fontsize=14, loc='center left', bbox_to_anchor=(1.02, 0.5))

            ax.tick_params(axis='both', labelsize=14)
            if i == len(row_keys) - 1:
                ax.set_xlabel('Time (min)', fontsize=16)

    plt.tight_layout(rect=[0.03, 0.04, 0.85, 0.95])
    fig.suptitle('Multi-scale Summary of Timing-dependent Synergy (D = 600)', fontsize=24, weight='bold')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved super-summary grid to {output_path}")


def create_super_summary_complete_grid_plot(aggregated_results, output_path):
    """Builds a 6×6 grid combining both synergy experiments (AKTi-MEKi & PI3Ki-MEKi)
    across six scales (pro/anti-survival nodes, target activation, cell signals, cell rates, alive/apoptotic).

    Columns (6): for each combination we show three timing cases:
        [0] AKTi First (Δt<0)     [1] Simultaneous (Δt=0)     [2] MEKi First (Δt>0)
        [3] PI3Ki First          [4] Simultaneous             [5] MEKi First
    """

    # Determine which aggregated dict corresponds to which inhibitor axis
    akt_key = next((k for k in aggregated_results if 'akt_mek' in k), None)
    pi3k_key = next((k for k in aggregated_results if 'pi3k_mek' in k), None)

    if not akt_key or not pi3k_key:
        logging.error("Could not identify both AKTi-MEKi and PI3Ki-MEKi datasets. Super summary aborted.")
        return

    akt_data = aggregated_results[akt_key]
    pi3k_data = aggregated_results[pi3k_key]

    row_keys = ['pro_survival_nodes', 'anti_survival_nodes', 'target_activation', 'cell_signals', 'cell_rates', 'alive_apoptotic']
    fig, axes = plt.subplots(6, 6, figsize=(24, 18), dpi=300, sharex=True, sharey='row')

    col_titles = [
        'AKTi First', 'Simultaneous', 'MEKi First',
        'PI3Ki First', 'Simultaneous', 'MEKi First'
    ]
    col_cases = [
        ('neg', akt_data), ('zero', akt_data), ('pos', akt_data),
        ('neg', pi3k_data), ('zero', pi3k_data), ('pos', pi3k_data)
    ]

    # Define color palettes for survival nodes
    PRO_SURVIVAL_COLORS = ['#006d2c', '#31a354', '#74c476']  # Shades of green
    ANTI_SURVIVAL_COLORS = ['#a50f15', '#de2d26', '#fb6a4a']   # Shades of red

    # Prepare plotting config templates reusing colours
    base_plot_configs = {
        'alive_apoptotic': {
            'ylabel': "Cell Count",
            'series': {
                'alive': {'label': 'Alive', 'color': CB_PALETTE['green']},
                'apoptotic': {'label': 'Apoptotic', 'color': CB_PALETTE['red']}
            }
        },
        'cell_rates': {
            'ylabel': "Rate (1/min)", 'ylim': (0, 0.0012),
            'series': {
                'growth': {'label': 'Growth', 'color': CB_PALETTE['green']},
                'apoptosis': {'label': 'Apoptosis', 'color': CB_PALETTE['red']}
            }
        },
        'cell_signals': {
            'ylabel': "Signal Value", 'ylim': (0, 1.05),
            'series': {
                'pro_survival': {'label': 'Pro-survival', 'color': CB_PALETTE['green']},
                'anti_survival': {'label': 'Anti-survival', 'color': CB_PALETTE['red']}
            }
        }
    }

    data_map = {
        'pro_survival_nodes': 'activated_fractions',
        'anti_survival_nodes': 'activated_fractions',
        'target_activation': 'activated_fractions',
        'cell_signals': 'cell_signals',
        'cell_rates': 'cell_rates',
        'alive_apoptotic': 'alive_apoptotic'
    }

    for i, row_key in enumerate(row_keys):
        for j, (case_key, data_dict) in enumerate(col_cases):
            ax = axes[i, j]

            # Set column titles once on first row
            if i == 0:
                ax.set_title(col_titles[j], fontweight='bold', fontsize=18)

            # Set row ylabel once per row
            if j == 0:
                if row_key in base_plot_configs:
                    ax.set_ylabel(base_plot_configs[row_key]['ylabel'], fontsize=16)
                elif row_key == 'target_activation':
                    ax.set_ylabel('% Activated Cells', fontsize=16)
                elif row_key == 'pro_survival_nodes':
                    ax.set_ylabel('Pro-Survival (%)', fontsize=16)
                elif row_key == 'anti_survival_nodes':
                    ax.set_ylabel('Anti-Survival (%)', fontsize=16)

            # Fetch dataframe according to row
            if data_dict is None or case_key not in data_dict:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', alpha=0.5)
                continue

            if row_key in data_map:
                if data_map[row_key] not in data_dict[case_key]:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', alpha=0.5)
                    continue
                df = data_dict[case_key][data_map[row_key]].copy()

                # --- FIX: Explicitly handle legacy 'live' columns before plotting ---
                if 'live_mean' in df.columns and 'alive_mean' not in df.columns:
                    df.rename(columns={'live_mean': 'alive_mean'}, inplace=True)
                if 'live_std' in df.columns and 'alive_std' not in df.columns:
                    df.rename(columns={'live_std': 'alive_std'}, inplace=True)
                
                if 'time' in df.columns:
                    df = df[df['time'] != 0].copy()

                plot_info = base_plot_configs.get(row_key)
                if plot_info:
                    for series_col, props in plot_info['series'].items():
                        mean_col = f"{series_col}_mean"
                        std_col = f"{series_col}_std"

                        if mean_col in df.columns:
                            ax.plot(df['time'], df[mean_col], label=props['label'], color=props['color'], linewidth=2)
                            if std_col in df.columns:
                                ax.fill_between(df['time'],
                                                df[mean_col] - df[std_col].fillna(0),
                                                df[mean_col] + df[std_col].fillna(0),
                                                color=props['color'], alpha=0.2)

                    if plot_info.get('ylim'):
                        ax.set_ylim(plot_info['ylim'])
                    if j == 5 and ax.get_legend_handles_labels()[0]:
                        ax.legend(fontsize=14, loc='center left', bbox_to_anchor=(1.02, 0.5))
                else:
                    # Handle special activation rows
                    candidate_nodes = {}
                    if row_key == 'target_activation':
                        candidate_nodes = {
                            'pi3k_node': {'label': 'PI3K', 'color': CB_PALETTE['blue']},
                            'akt_node': {'label': 'AKT', 'color': CB_PALETTE['green']},
                            'mek_node': {'label': 'MEK', 'color': CB_PALETTE['orange']}
                        }
                    elif row_key == 'pro_survival_nodes':
                        pro_nodes = ['node_cMYC', 'node_RSK', 'node_TCF']
                        pro_labels = ['cMYC', 'RSK', 'TCF']
                        candidate_nodes = {
                            node: {'label': label, 'color': color}
                            for node, label, color in zip(pro_nodes, pro_labels, PRO_SURVIVAL_COLORS)
                        }
                    elif row_key == 'anti_survival_nodes':
                        anti_nodes = ['node_FOXO', 'node_Caspase8', 'node_Caspase9']
                        anti_labels = ['FOXO', 'Caspase8', 'Caspase9']
                        candidate_nodes = {
                            node: {'label': label, 'color': color}
                            for node, label, color in zip(anti_nodes, anti_labels, ANTI_SURVIVAL_COLORS)
                        }

                    for node_key, props in candidate_nodes.items():
                        mean_col = f"{node_key}_mean"
                        std_col = f"{node_key}_std"
                        if mean_col in df.columns:
                            ax.plot(df['time'], df[mean_col], label=props['label'], color=props['color'], 
                                    linestyle=props.get('style', 'solid'), linewidth=2)
                            if std_col in df.columns:
                                ax.fill_between(df['time'],
                                                df[mean_col] - df[std_col].fillna(0),
                                                df[mean_col] + df[std_col].fillna(0),
                                                color=props['color'], alpha=0.2)
                    ax.set_ylim(0, 101)
                    if j == 5 and ax.get_legend_handles_labels()[0]:
                        ax.legend(fontsize=14, loc='center left', bbox_to_anchor=(1.02, 0.5))

            ax.tick_params(axis='both', labelsize=14)
            if i == len(row_keys) - 1:
                ax.set_xlabel('Time (min)', fontsize=16)

    plt.tight_layout(rect=[0.03, 0.04, 0.85, 0.95])
    fig.suptitle('Multi-scale Summary of Timing-dependent Synergy (D = 600)', fontsize=24, weight='bold')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved super-summary complete grid to {output_path}")


def process_experiment_dynamics(exp_name, executor=None, max_workers=4):
    """
    Main processing function for a single experiment, focusing on the high-dose case.
    """
    logging.info(f"--- Processing Experiment: {exp_name} ---")
    
    summary_path = f'results/sweep_summaries/final_summary_{exp_name}.csv'
    try:
        raw_df = pd.read_csv(summary_path)
        raw_df['experiment_name'] = exp_name  # Add experiment name for file path construction
    except FileNotFoundError:
        logging.error(f"Summary file not found for {exp_name}. Skipping.")
        return

    raw_df['delta_time'] = raw_df['user_parameters.drug_X_pulse_period'] - raw_df['user_parameters.drug_Y_pulse_period']
    
    # Get control bounds from the data itself
    control_min, control_max = get_control_bounds_from_data(raw_df)
    
    # --- Rest of the filtering logic ---
    late_pulse_threshold = 4000
    all_diff_coeffs = pd.unique(raw_df[['user_parameters.drug_X_diffusion_coefficient', 'user_parameters.drug_Y_diffusion_coefficient']].values.ravel('K'))
    high_dose = sorted([c for c in all_diff_coeffs if c != 6.0])[-1]
    low_dose = 6.0

    control_condition_exp = (
        (raw_df['user_parameters.drug_X_diffusion_coefficient'] == low_dose) &
        (raw_df['user_parameters.drug_Y_diffusion_coefficient'] == low_dose) &
        (raw_df['user_parameters.drug_X_pulse_period'] >= late_pulse_threshold) &
        (raw_df['user_parameters.drug_Y_pulse_period'] >= late_pulse_threshold)
    )
    control_runs_df = raw_df[control_condition_exp]
    
    if control_runs_df.empty:
        logging.error("Control data missing, cannot normalize or filter. Skipping.")
        return
    mean_control_alive = control_runs_df['FINAL_NUMBER_OF_ALIVE_CELLS'].mean()

    raw_df['percent_alive'] = (raw_df['FINAL_NUMBER_OF_ALIVE_CELLS'] / mean_control_alive) * 100

    synergy_condition_exp = (
        (raw_df['user_parameters.drug_X_diffusion_coefficient'] == high_dose) &
        (raw_df['user_parameters.drug_Y_diffusion_coefficient'] == high_dose) &
        (raw_df['user_parameters.drug_X_pulse_period'] == 4) &
        (raw_df['user_parameters.drug_Y_pulse_period'] == 4)
    )
    synergy_runs = raw_df[synergy_condition_exp]
    if synergy_runs.empty:
        logging.warning("Synergy data for benchmark not found. Filtering may be incomplete.")
        median_synergy_efficacy = 30 # Fallback value
    else:
        median_synergy_efficacy = synergy_runs['percent_alive'].median()
    
    drug_x_condition_exp = (
        (raw_df['user_parameters.drug_X_diffusion_coefficient'] == high_dose) &
        (raw_df['user_parameters.drug_Y_diffusion_coefficient'] == low_dose) &
        (raw_df['user_parameters.drug_X_pulse_period'] == 4) &
        (raw_df['user_parameters.drug_Y_pulse_period'] >= late_pulse_threshold)
    )
    drug_y_condition_exp = (
        (raw_df['user_parameters.drug_X_diffusion_coefficient'] == low_dose) &
        (raw_df['user_parameters.drug_Y_diffusion_coefficient'] == high_dose) &
        (raw_df['user_parameters.drug_X_pulse_period'] >= late_pulse_threshold) &
        (raw_df['user_parameters.drug_Y_pulse_period'] == 4)
    )
    single_drug_runs = raw_df[drug_x_condition_exp | drug_y_condition_exp]
    confounding_sd_individuals = single_drug_runs[single_drug_runs['percent_alive'] < median_synergy_efficacy]['individual'].unique()

    confounding_ctrl_individuals = raw_df[control_condition_exp & (raw_df['percent_alive'] < 80)]['individual'].unique()

    exclusion_list = np.union1d(confounding_sd_individuals, confounding_ctrl_individuals)
    
    if len(exclusion_list) > 0:
        logging.info(f"Excluding {len(exclusion_list)} confounding individuals.")
        summary_df = raw_df[~raw_df['individual'].isin(exclusion_list)].copy()
    else:
        logging.info("No confounding individuals found. Using all data.")
        summary_df = raw_df.copy()
    
    dose_to_plot = 600.0
    logging.info(f"--- Processing dose combination: D(X)={dose_to_plot}, D(Y)={dose_to_plot} ---")
    
    combo_df = summary_df[(summary_df['user_parameters.drug_X_diffusion_coefficient'] == dose_to_plot) & 
                          (summary_df['user_parameters.drug_Y_diffusion_coefficient'] == dose_to_plot)]

    neg_df = combo_df[combo_df['delta_time'] < 0]
    late_addition_threshold = 3000
    zero_df = combo_df[(combo_df['delta_time'] == 0) & (combo_df['user_parameters.drug_X_pulse_period'] < late_addition_threshold)]
    pos_df = combo_df[combo_df['delta_time'] > 0]
    
    logging.info(f"Found {len(neg_df)} runs for t<0, {len(zero_df)} for t=0, {len(pos_df)} for t>0.")

    labels_dfs = {'neg': neg_df, 'zero': zero_df, 'pos': pos_df}

    # Re-use shared executor if supplied; otherwise create a private pool
    _local_pool = None
    exec_to_use = executor
    if exec_to_use is None:
        _local_pool = ProcessPoolExecutor(max_workers=max_workers)
        exec_to_use = _local_pool

    aggregated_data = {}
    for label, gdf in labels_dfs.items():
        cached = load_cached_group(exp_name, label, dose_to_plot)
        if cached is not None:
            logging.info(f"Loaded cached aggregation for {exp_name}:{label} (n={len(gdf)})")
            aggregated_data[label] = cached
            continue
        # Cache miss → compute aggregation
        agg = aggregate_dynamics_for_group(gdf, exp_name, executor=exec_to_use, max_workers=max_workers)
        aggregated_data[label] = agg
        if agg:
            dump_cached_group(exp_name, label, dose_to_plot, agg)

        # Attach control bounds for downstream normalization (used in super-summary plots)
        aggregated_data['_ctrl_bounds'] = (control_min, control_max)

    if _local_pool is not None:
        _local_pool.shutdown(wait=True)

    output_dir = f"results/aggregated_timing_dynamics/{exp_name}"
    output_path = os.path.join(output_dir, f"aggregated_dynamics_grid_D{dose_to_plot}_filtered.png")
    
    create_aggregated_dynamics_grid_plot(aggregated_data, output_path, exp_name, control_min, control_max)

    create_survival_nodes_grid_plot(aggregated_data, os.path.join(output_dir, 'survival_nodes_grid_plot.png'), exp_name)

    # Return aggregated data for possible cross-experiment super-summary plotting
    return aggregated_data

def main():
    """
    Main function to run the analysis for all specified experiments.
    """
    parser = argparse.ArgumentParser(description="Aggregate timing-dynamics and plot grids.")
    parser.add_argument("--n_workers", type=int, default=min(15, mp.cpu_count()),
                        help="Maximum parallel workers to use (default: 15 or CPU count).")
    args = parser.parse_args()

    max_workers = args.n_workers
    
    experiment_names = [
        "synergy_sweep-akt_mek-2606-1819-4p_3D_drugtiming_synonly_consensus_hybrid_20",
        "synergy_sweep-pi3k_mek-2606-1819-4p_3D_drugtiming_synonly_consensus_hybrid_20"
    ]

    aggregated_results = {}

    # Create one shared pool for all experiments to minimise launch overhead
    with ProcessPoolExecutor(max_workers=max_workers) as shared_executor:
        for exp_name in experiment_names:
            aggregated_results[exp_name] = process_experiment_dynamics(exp_name, executor=shared_executor, max_workers=max_workers)

    # Create a combined super-summary grid if both experiments processed successfully
    if all(aggregated_results.values()):
        # Determine dimension (2D vs 3D) from first experiment name and tag the
        # super-summary output accordingly so separate runs don't overwrite.
        dims_tag = _dims_tag(experiment_names[0]) if experiment_names else 'out'
        output_super_path = _summary_output_path(dims_tag)
        create_super_summary_grid_plot(aggregated_results, output_super_path)
        
        # Create the new, more complete summary plot
        output_super_complete_path = f"results/aggregated_timing_dynamics/super_summary_complete_grid_{dims_tag}.png"
        create_super_summary_complete_grid_plot(aggregated_results, output_super_complete_path)
    else:
        logging.warning("Super-summary grid skipped: missing aggregated data for one or more experiments.")

# -----------------------------------------------------------------------------
# Utility helpers to keep output folders from different dimensional experiments
# neatly separated and avoid accidental overwrites when running 2D vs 3D jobs.
# -----------------------------------------------------------------------------

def _dims_tag(exp_name: str) -> str:
    """Return '2D' or '3D' based on markers in the experiment name."""
    return '2D' if '_2D_' in exp_name else '3D'

def _summary_output_path(dims_tag: str) -> str:
    """Build path for the super-summary file that includes the dimension tag."""
    return f"results/aggregated_timing_dynamics/super_summary_grid_{dims_tag}.png"

if __name__ == "__main__":
    main() 