#!/usr/bin/env python
# coding: utf-8

import re, os, sys, warnings
import numpy as np
import scipy as sp
from scipy import interpolate, integrate
import pandas as pd
import xml.dom.minidom
import matplotlib.pyplot as plt
import seaborn as sns
# if MN5
# import pcdl

# if Nord4
sys.path.append("/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/python/physicelldataloader/pcdl/")
sys.path.append("/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/python/physicelldataloader/")
import pcdl
from pyMCDSts import pyMCDSts
from pyMCDS import pyMCDS, es_coor_cell, es_coor_conc
# from minimal_mcds import MinimalMCDS as pyMCDSts


warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append("./python") # for running locally 
from multicellds import MultiCellDS
import pickle
from multiprocessing import Pool, cpu_count
import traceback

pd.options.mode.chained_assignment = None

def set_plotting_style():
    """Set consistent plotting style for publication-quality figures"""
    # Use updated style name
    plt.style.use('ggplot')  # or specifically 'seaborn-v0_8-white' for older look
    
    # Custom color palette (colorblind-friendly)
    colors = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', 
              '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9']
    
    # Publication-quality settings
    plt.rcParams.update({
        # Font settings
        'font.family': 'sans-serif',
        # 'font.sans-serif': ['DejaVu Sans', 'Helvetica'],
        'font.size': 12,
        
        # Axis settings
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.grid': True,
        'axes.linewidth': 1.5,
        
        # Tick settings
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        
        # Grid settings
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        
        # Line settings
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        
        # Legend settings
        'legend.fontsize': 12,
        'legend.frameon': False,
        'legend.handlelength': 2,
        
        # Figure settings
        'figure.dpi': 300,
        'figure.figsize': [10, 6],
        'figure.autolayout': True,
        
        # Save settings
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    return colors

def create_base_plot(ax, xlabel='Simulation time (min)', ylabel=None):
    """Apply consistent styling to any plot"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.set_xlim(0, 4200)
    
    ax.set_xlabel(xlabel, fontsize=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    return ax

def plot_molecular_model(df_cell_variables, list_of_variables, ax, threshold=0.5):

    for label in list_of_variables:
        y = df_cell_variables[label]
        time = df_cell_variables["time"]
        
        ax.plot(time, y, label=label + " state")

    ax.legend()
    ax.set_ylabel("Mean Boolean node state")
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    # ax.set_xlim((0,time.values[-1]))
    # ax.set_ylim((0,1.05))

def plot_cells(df_time_course, instance_folder, ax=None):
    """Plot cell phase transitions with publication quality"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax.clear()
        fig = ax.figure
    
    ax = create_base_plot(ax, ylabel='Number of cells')

    colors = set_plotting_style()
    
    # Data preparation
    df_subset = df_time_course[['time', 'current_phase']]
    df_aggregated = df_subset.groupby(['time', 'current_phase']).size().unstack(fill_value=0)
    df_aggregated = df_aggregated.reset_index()
    df_melted = pd.melt(df_aggregated, id_vars=['time'], 
                        var_name='current_phase', value_name='count')
    
    # Create plot
    sns.lineplot(data=df_melted, x='time', y='count', hue='current_phase', 
                palette=colors[:len(df_melted['current_phase'].unique())],
                linewidth=2.5, ax=ax)
    
    # Add treatment window
    drug_name = "drug_X"
    total_drug_addition_time = drug_time_addition_info(instance_folder, f"{drug_name}_pulse_duration")
    ax.axvspan(1280, int(1280 + total_drug_addition_time), color='grey', alpha=0.5)
        
    # Styling
    ax.legend(title='Cell Phase', bbox_to_anchor=(1.05, 1), 
             loc='upper left', borderaxespad=0)
    
    if ax is None:
        save_publication_plot(fig, instance_folder, "cellgrowth")
        plt.close(fig)
    
    return fig, ax


def calculate_density_statistics(df_subset):
    """
    Calculate mean and standard deviation for density columns
    
    Args:
        df_subset: DataFrame containing density columns and 'time'
        
    Returns:
        DataFrame with statistics grouped by time
    """
    # Group by time and calculate statistics
    df_stats = df_subset.groupby('time').agg(['mean', 'std'])
    
    # Flatten multi-index columns
    df_stats.columns = ['_'.join(col).strip() for col in df_stats.columns.values]
    
    # Reset index for plotting
    df_stats = df_stats.reset_index()
    
    # Melt for easier plotting
    df_melted = pd.melt(df_stats, 
                        id_vars=['time'], 
                        value_vars=[col for col in df_stats.columns if '_mean' in col],
                        var_name='density_type', 
                        value_name='density')
    
    # Add std values
    df_melted_std = pd.melt(df_stats,
                            id_vars=['time'],
                            value_vars=[col for col in df_stats.columns if '_std' in col],
                            var_name='density_type',
                            value_name='std')
    
    # Clean column names
    df_melted['density_type'] = df_melted['density_type'].str.replace('_mean', '')
    df_melted_std['density_type'] = df_melted_std['density_type'].str.replace('_std', '')
    
    # Merge dataframes
    return pd.merge(df_melted, df_melted_std, on=['time', 'density_type'])


def process_and_plot_density(df, instance_folder, drug_name, ax=None):
    """Plot drug density with publication quality"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax.clear()
        fig = ax.figure
    
    colors = set_plotting_style()
    
    # Data preparation
    density_cols = df.filter(regex=f'{drug_name}|internal_density|external_density')\
                    .filter(regex='^(?!.*oxygen)')\
                    .filter(regex=f'^(?!.*{drug_name})').columns
    df_subset = df[list(density_cols) + ['time']]
    
    # Calculate statistics
    df_stats = calculate_density_statistics(df_subset)
    
    # Plot
    for idx, (name, group) in enumerate(df_stats.groupby('density_type')):
        ax.plot(group['time'], group['density'], 
                color=colors[idx], label=name.replace('_mean', ''))
        ax.fill_between(group['time'], 
                       group['density'] - group['std'],
                       group['density'] + group['std'], 
                       color=colors[idx], alpha=0.2)
    
    # Add treatment window
    total_drug_addition_time = drug_time_addition_info(instance_folder, f"{drug_name}_pulse_duration")
    ax.axvspan(1280, int(1280 + total_drug_addition_time), color='grey', alpha=0.5)
    
    # Styling
    ax = create_base_plot(ax, ylabel='Drug density')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
    if ax is None:
        save_publication_plot(fig, instance_folder, f"densityplot_{drug_name}")
        plt.close(fig)
    
    return fig, ax

def process_and_plot_nodes(df, instance_folder, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    else:
        ax.clear()
        fig = ax.figure

    ax = create_base_plot(ax, ylabel='Node Value')

    # Filter out rows where 'current_phase' is 'alive'
    df = df[df['current_phase'] != 'alive']

    # Select columns containing 'node' but not 'anti', and 'time', nor "drug_"
    node_cols = df.filter(regex='node').filter(regex='^(?!.*anti)').filter(regex='^(?!.*drug_)').columns
    
    # Columns to explicitly exclude - write as a list of strings
    excluded_cols = list('node_BAD_mean', 'node_BAX_mean', 'node_BCL2_mean',
        'node_CytochromeC_mean', 'node_LEF_mean', 'node_PDK1_mean',
        'node_S6K_mean', 'node_mTOR_mean', 'node_p53_mean')

    print("these are the node columns: ", node_cols)
    print("these are the excluded columns: ", excluded_cols)
    
    # Filter out excluded columns and add time
    selected_cols = [col for col in node_cols if col not in excluded_cols] + ['time']

    print("these are the selected columns: ", selected_cols)
    
    df_subset = df[selected_cols]

    # Group by time and calculate mean and std for each node column
    df_aggregated = df_subset.groupby('time').agg(['mean', 'std'])

    # Flatten column names
    df_aggregated.columns = ['_'.join(col).strip() for col in df_aggregated.columns.values]

    # Reset index to make 'time' a column again
    df_aggregated = df_aggregated.reset_index()

    # Melt the dataframe to long format for seaborn
    df_melted = pd.melt(df_aggregated, id_vars=['time'], 
                        value_vars=[col for col in df_aggregated.columns if col.endswith('_mean')],
                        var_name='node_type', value_name='value')

    # Create a corresponding dataframe for standard deviation
    df_melted_std = pd.melt(df_aggregated, id_vars=['time'], 
                            value_vars=[col for col in df_aggregated.columns if col.endswith('_std')],
                            var_name='node_type', value_name='std')
    df_melted_std['node_type'] = df_melted_std['node_type'].str.replace('_std', '_mean')

    # Merge mean and std dataframes
    df_melted = pd.merge(df_melted, df_melted_std, on=['time', 'node_type'])

    # Create the plot
    # plt.figure(figsize=(12, 4), dpi=300)
    # colors = set_plotting_style()
    ax = sns.lineplot(data=df_melted, x='time', y='value', hue='node_type', ax=ax, linewidth=3, palette=colors)
    sns.despine(offset=0, trim=True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    # Add shaded region for standard deviation
    for name, group in df_melted.groupby('node_type'):
        ax.fill_between(group['time'], group['value'] - group['std'], 
                        group['value'] + group['std'], alpha=0.2)

    # Add shaded grey region from x=1200 to x=1240
    drug_name = "drug_X"
    total_drug_x_addition_time = drug_time_addition_info(instance_folder, f"{drug_name}_pulse_duration")
    ax.axvspan(1280, int(1280 + total_drug_x_addition_time), color='grey', alpha=0.5)

    # ax.set_title('Non-Anti Node Values Over Time (Excluding Alive Phase)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Node Value')
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    # Move legend outside the plot
    ax.legend(title='Node Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # plt.tight_layout()
    if ax is None:
        plt.savefig(os.path.join(instance_folder, "nodeplot.jpeg"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(instance_folder, "nodeplot.png"), dpi=300, transparent=True, bbox_inches="tight")
        print("Node plot obtained")
        plt.close(fig)
    return fig, ax

def get_min_max_wt_exp_curve(input_df):
    """
    Comes from the experimental data, will always need the same preprocessing
    """

    # get min and max values from control curve
    df_control_csv = input_df
    df_control = df_control_csv.reset_index(inplace=False)
    df_control.Time = df_control.Time / 60 # only needed when reading pickle data
    df_control['Average_Cell_Index'] = df_control.iloc[:,2:].mean(axis=1)
    df_control.Average_Cell_Index = df_control.Average_Cell_Index
    df_control = df_control[["Time", "Average_Cell_Index"]]
    df_control_sliced = df_control.loc[df_control["Time"] < 4200]

    ## 29.5.23: omitted fill_value argument

    try:
        f = interpolate.interp1d(df_control.Time, df_control.Average_Cell_Index, kind="cubic", fill_value="extrapolate")
        xnew_control_exp = np.arange(0, 4200, 40) # emulating PC simulation points 
        ynew_control_exp = f(xnew_control_exp)
    except ValueError:
        f = interpolate.interp1d(df_control.Time, df_control.Average_Cell_Index, kind="cubic", bounds_error=False, fill_value="extrapolate")
        xnew_control_exp = np.arange(0, 4200, 40) # emulating PC simulation points 
        ynew_control_exp = f(xnew_control_exp)
    
    max_value_wt_exp = ynew_control_exp.max()
    min_value_wt_exp = ynew_control_exp.min()

    return min_value_wt_exp, max_value_wt_exp

def normalize_exp_curve(exp_df, min, max):
    """
    This normalization will always be the same, too.
    """
    #  normalize any other column based on this: e.g. PI3K

    # df_drug_csv = pd.read_csv(os.path.join(exp_curves_folder, "PI103(0.70uM).csv"))
    df_drug = exp_df.reset_index(inplace=False)
    df_drug.Time = df_drug.Time / 60 # only needed when reading pickle data
    df_drug['Average_Cell_Index'] = df_drug.iloc[:,2:].mean(axis=1)
    df_drug.Average_Cell_Index = df_drug.Average_Cell_Index
    df_drug = df_drug[["Time", "Average_Cell_Index"]]

    df_drug = df_drug[ (df_drug['Time'] < 400.0) | (df_drug["Time"]>2500.0)]
    
    df_drug_sliced = df_drug.loc[df_drug["Time"] < 4200]

    try:
        f = interpolate.interp1d(df_drug.Time, df_drug.Average_Cell_Index, kind="cubic")
        xnew_drug_exp = np.arange(0, 4240, 40) # emulating PC simulation points 
        ynew_drug_exp = f(xnew_drug_exp)
    except ValueError:
        f = interpolate.interp1d(df_drug.Time, df_drug.Average_Cell_Index, kind="cubic", bounds_error=False)
        xnew_drug_exp = np.arange(0, 4240, 40) # emulating PC simulation points 
        ynew_drug_exp = f(xnew_drug_exp)

    # Then normalize based on WT
    # tmp_norm = df_drug_sliced.drop("Time", axis=1)
    tmp_norm = ((ynew_drug_exp-min)/(max-min)) * 100
    try:
        df_drug_exp_norm = pd.concat((pd.Series(xnew_drug_exp), pd.Series(tmp_norm)), axis=1)
    except ValueError:
        print("ERROR")

    df_drug_exp_norm.columns = ["Time", "Average_Cell_Index"]
    df_drug_exp_norm.fillna(0, inplace=True) # AKT has NaN's that have to be fixed


    return df_drug_exp_norm.Time, df_drug_exp_norm.Average_Cell_Index

def get_min_max_wt_sim_curve(input_df):
    """ 
    From the simgrowth curve of the control simulation, get the min, max and initial cell number.

    """
    initial_cell_number = input_df['alive'].iloc[0]
    min_value = input_df['alive'].min()
    max_value = input_df['alive'].max()

    print("for the WT simulation curve, the min, max and initial cell number are: ", min_value, max_value, initial_cell_number)

    return min_value, max_value, initial_cell_number     

def normalize_sim_curve(exp_drug_df, initial_cell_number, min, max):
    """ normalize simulation curve coming from the df_time_course dataframe from MCDS """

    df = exp_drug_df[exp_drug_df['current_phase'] != 'alive']
    # Calculate total number of cells for each time point
    df['alive_cell_number'] = df.groupby('time')['ID'].transform('nunique')
    # initial_cell_number = df.loc[0, 'alive_cell_number']

    # Now get the normalized column
    subset_df = df.loc[:, ['time', 'alive_cell_number']]
    subset_df['adjusted_alive_cell_number'] = subset_df['alive_cell_number'] - initial_cell_number
    max_value = subset_df['adjusted_alive_cell_number'].max()
    min_value = subset_df['adjusted_alive_cell_number'].min()
    subset_df['normalized_alive_cell_number'] = ((subset_df['adjusted_alive_cell_number'] - min) / (max - min)) * 100

    df_unique = subset_df.drop_duplicates()

    print(df_unique.head(15))

    return df_unique.time, df_unique.normalized_alive_cell_number

def normalize_simulation_growth_curve(sim_drug_df, initial_cell_number, min, max):
    """ 
    Identical to "normalize_sim_curve" but for the simulation_growth.csv file, which 
    contains 2 different columns: "time" and "alive". 
    This CSV comes from the function save_pcdl_timeseries() in this script
    """
    print("Normalizing simulation growth curve with simulated control")

    df = sim_drug_df.copy()

    # Use its own initial cell number to adjust
    # initial_cell_number_sim = df.loc[0, 'alive']
    # df['adjusted_alive'] = df['alive'] - initial_cell_number_sim

    # Calculate the range of possible values
    value_range = max - min

    # Normalize the 'alive' column
    df['normalized_alive'] = ((df['alive'] - min) / value_range) * 100

    # print(df.head(15))

    return df.time, df.normalized_alive

def normalize_sim_curve_control(input_df):
    """ normalize simulation curve coming from the df_time_course dataframe from MCDS """

    df = input_df[input_df['current_phase'] != 'alive']
    # Calculate total number of cells for each time point
    df['alive_cell_number'] = df.groupby('time')['ID'].transform('nunique')
    initial_cell_number = df.loc[0, 'alive_cell_number']

    # Now get the normalized column
    subset_df = df.loc[:, ['time', 'alive_cell_number']]
    subset_df['adjusted_alive_cell_number'] = subset_df['alive_cell_number'] - initial_cell_number

    max_value = subset_df['adjusted_alive_cell_number'].max()
    min_value = subset_df['adjusted_alive_cell_number'].min()
    subset_df['normalized_alive_cell_number'] = ((subset_df['adjusted_alive_cell_number'] - min_value) / (max_value - min_value)) * 100

    df_unique = subset_df.drop_duplicates()

    return df_unique.time, df_unique.normalized_alive_cell_number



def plot_curve_comparison(df_time_course, experimental_path, drug_name, instance_folder, ax=None):
    """Plot experimental vs simulation comparison with publication quality"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax.clear()
        fig = ax.figure
    
    colors = set_plotting_style()
    
    # first fetch experimental data & interpolate
    exp_wt_df = pd.read_csv(f'{experimental_path}/CTRL.csv') # WT

    if drug_name == "PI3K":
        exp_drug_df = pd.read_csv(f'{experimental_path}/PI103(0.70uM).csv') 
    elif drug_name == "MEK":
        exp_drug_df = pd.read_csv(f'{experimental_path}/PD0325901(35.00nM).csv') 
    elif drug_name == "TAK1":
        exp_drug_df = pd.read_csv(f'{experimental_path}/(5Z)-7-oxozeaenol(0.50uM).csv') 
    elif drug_name == "AKT":
        exp_drug_df = pd.read_csv(f'{experimental_path}/AKT_2.csv') 
    elif drug_name == "WT":
        exp_drug_df = pd.read_csv(f'{experimental_path}/CTRL.csv') 
    elif drug_name == "PI3K_MEK":
        exp_drug_df = pd.read_csv(f'{experimental_path}/PD0325901(17.50nM)+PI103(0.35uM).csv')
    elif drug_name == "AKT_MEK":
        exp_drug_df = pd.read_csv(f'{experimental_path}/AKT_MEK_final_ok.csv') 
    

    wt_min_exp, wt_max_exp = get_min_max_wt_exp_curve(exp_wt_df)
    x_exp, y_exp = normalize_exp_curve(exp_drug_df, wt_min_exp, wt_max_exp)

    # fetch simulated data
    sim_wt_df = pd.read_csv(f'{experimental_path}SIM_CTRL_CMA-1110-1637-5p.csv')

    # This comes from /home/oth/BSC/MN5mount/AGS/EMEWS/results/CMA_summaries/final_summary_CMA-21_08_2024-02:25:04-test_2p_CTRl/top_10.csv
    # Choosing the first one (lowest RMSE)

    if drug_name == "WT":
        x_sim, y_sim = normalize_sim_curve_control(df_time_course) # normalize it against itself
    else:
        # @oth (2.09.24) 
        # Used to be "normalize_sim_curve", but does not werk well for the "simulation_growth.csv" file
        # Now using "normalize_simulation_growth_curve"
        wt_min_sim, wt_max_sim, initial_cells = get_min_max_wt_sim_curve(sim_wt_df)
        # fetch the simulation growth curve from the df_time_course same folder
        simgrowth_reduced_df = pd.read_csv(os.path.join(instance_folder, "simulation_growth.csv"))
        x_sim, y_sim = normalize_simulation_growth_curve(simgrowth_reduced_df, initial_cells, wt_min_sim, wt_max_sim)

    # plotting with seaborn instead
    # Create a DataFrame for the experimental data
    df_exp = pd.DataFrame({
        'x': x_exp,
        'y': y_exp,
        'Type': 'Experimental'
    })

    # Create a DataFrame for the simulated data
    df_sim = pd.DataFrame({
        'x': x_sim,
        'y': y_sim,
        'Type': 'Simulation'
    })

    # Combine both DataFrames
    df_combined = pd.concat([df_exp, df_sim])

    # Create the plot
    ax.plot(df_combined['x'], df_combined['y'], 
            label='Experimental', color=colors[0], linestyle='-')
    ax.plot(df_combined['x'], df_combined['y'], 
            label='Simulation', color=colors[1], linestyle='--')
    
    # Add treatment window if not control
    if drug_name != "WT":
        drug_name = "drug_X"
        total_drug_addition_time = drug_time_addition_info(instance_folder, f"{drug_name}_pulse_duration")
        ax.axvspan(1280, int(1280 + total_drug_addition_time), color='grey', alpha=0.5)
    
    # Styling
    ax = create_base_plot(ax, ylabel='Relative cell number')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
    if ax is None:
        save_publication_plot(fig, instance_folder, "curve_comparison")
        plt.close(fig)
    
    return fig, ax

def get_drug_name(instance_folder, drug_name):
    doc = xml.dom.minidom.parse(os.path.join(instance_folder,"settings.xml"))
    custom_data = doc.getElementsByTagName(drug_name)
    drug_name = str(custom_data[0].firstChild.nodeValue)
    if drug_name != "null":
        drug_name = drug_name.split("anti_")[1]
        return drug_name
    else:
        return None

def drug_time_addition_info(instance_folder, drug_addition_time_tag):
    doc = xml.dom.minidom.parse(os.path.join(instance_folder, "settings.xml"))
    custom_data = doc.getElementsByTagName(drug_addition_time_tag)
    drug_addition_time = str(custom_data[0].firstChild.nodeValue)
    return round(float(drug_addition_time), 3)

def save_pcdl_timeseries(output_folder, instance_folder):
    " Saves a pickle with the TimeSeries object from pcdl, as well as some tabular data "
    mcdsts = pyMCDSts(output_path=output_folder, graph=False, verbose=False)
    print("mcdsts is", mcdsts)

    # @oth: This takes too much space in each instance folder
    # pickle_path = os.path.join(instance_folder, "pcdl_timeseries.pkl")
    # with open(pickle_path, 'wb') as a:
    #     pickle.dump(mcdsts, a)

    # also save ALL data in tabular format
    list_of_relevant_vars = list()
    all_data = pd.DataFrame()
    for mcds in mcdsts.get_mcds_list():
        frame_df = mcds.get_cell_df()
        frame_df.reset_index(inplace=True)
        list_of_relevant_vars.append(frame_df)

    all_data = pd.concat(list_of_relevant_vars, ignore_index=True) 

    print(all_data.head())

    # @oth: 10.03.25
    # Increased compression rate - used to be "gzip" and now it's "bz2"

    # all_data.to_csv(os.path.join(instance_folder, "pcdl_total_info_sim.csv.gz"), header=True, index=False, compression='gzip')
    all_data.to_csv(os.path.join(instance_folder, "pcdl_total_info_sim.csv.bz2"),
                header=True,
                index=False,
                compression={'method': 'bz2', 'compresslevel': 9})  # compresslevel 9 is maximum compression

    # only keep alive cells
    # @oth: #TODO complete this dictionary or find it somewhere in order to subset
    state_to_code = {
        'live': 5,
        'apoptotic': 100,
        'necrotic': 101
    }

    all_data_tmp = all_data.copy()
    all_data_tmp['current_phase_number'] = all_data_tmp['current_phase'].map(state_to_code)
    all_data_alive = all_data_tmp[all_data_tmp['current_phase_number'] <= 14]
    # Aggregate based on time and filter only by alive cells (current_phase <=14)
    time_aggregated_data_all = all_data.groupby('time').mean(numeric_only=True)
    time_aggregated_data_all.reset_index(inplace=True)
    time_aggregated_data_alive = all_data_alive.groupby('time').mean(numeric_only=True)
    time_aggregated_data_alive.reset_index(inplace=True)


    # This simulation growth csv is needed for the metrics computation
    simgrowth_df = all_data[all_data['current_phase'] != 'alive']
    # Calculate total number of cells for each time point
    simgrowth_df['alive'] = simgrowth_df.groupby('time')['ID'].transform('nunique')
    simgrowth_df.reset_index(inplace=True)
    simgrowth_reduced_df = simgrowth_df[["time", "alive"]]
    simgrowth_reduced_df.drop_duplicates(inplace=True)
    
    simgrowth_reduced_df.to_csv(os.path.join(instance_folder, 'simulation_growth.csv'), header=True, index=False)
    # time_aggregated_data_alive.to_csv( os.path.join(instance_folder, 'pcdl_celldata_alive_agg.csv'), header=True, index=False)
    # time_aggregated_data_all.to_csv( os.path.join(instance_folder, 'pcdl_celldata_all_agg.csv'), header=True, index=False)

    return all_data, time_aggregated_data_all, time_aggregated_data_alive

def save_tabular_data(output_folder, mcds, instance_folder_path):
    " Stores all cell variables as a .csv "

    # Save all cell variables
    df_time_course = mcds.get_cells_summary_frame()
    df_cell_variables = get_timeserie_mean(mcds, filter_alive=True)
    df_cell_variables_total = get_timeserie_mean(mcds, filter_alive=False)
    # df_cell_variables has all the relevant data needed for plotting
    df_cell_variables.to_csv( os.path.join(instance_folder_path, 'cells_timeseries_alive_mean.csv'), sep=",", header=True, index=False)
    df_cell_variables_total.to_csv( os.path.join(instance_folder_path, 'cells_timeseries_total_mean.csv'), sep=",", header=True, index=False)
    # Save alive plot
    alive_df = df_time_course[["time", "alive"]]
    alive_df.to_csv( os.path.join(instance_folder_path, 'simulation_growth.csv'), header=True, index=False)

    return df_time_course, df_cell_variables, df_cell_variables_total

def save_publication_plot(fig, instance_folder, plot_name):
    """Save plots in multiple formats with consistent settings"""
    plot_dir = os.path.join(instance_folder)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save as PNG
    # png_path = os.path.join(plot_dir, f"{plot_name}.png")
    # fig.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)
    
    # Save as JPEG
    jpeg_path = os.path.join(plot_dir, f"{plot_name}.jpeg")
    fig.savefig(jpeg_path, dpi=300, bbox_inches="tight")
    
    print(f"Saved publication plots for {plot_name} in {plot_dir}")

def create_combined_plot(df, instance_folder, experiment_type, experimental_data_path, drug_name):
    """Create publication-quality combined plots based on experiment type"""
    # Set consistent style
    colors = set_plotting_style()
    
    # Determine subplot configuration
    n_plots = {'control': 2, 'single': 4, 'combined': 5}
    fig_height = n_plots[experiment_type] * 4
    
    # Create figure
    fig, axs = plt.subplots(n_plots[experiment_type], 1, 
                           figsize=(15, fig_height), 
                           constrained_layout=True)
    

    print("the drug name is", drug_name)
    
    # Configure plot functions with required arguments
    plot_functions = {
        'control': [
            lambda d, i, ax: plot_cells(d, i, ax),
            lambda d, i, ax: plot_curve_comparison(d, experimental_data_path, drug_name, i, ax)
        ],
        'single': [
            lambda d, i, ax: plot_cells(d, i, ax),
            lambda d, i, ax: process_and_plot_density(d, i, "drug_X", ax),
            lambda d, i, ax: process_and_plot_nodes(d, i, ax),
            lambda d, i, ax: plot_curve_comparison(d, experimental_data_path, drug_name, i, ax)
        ],
        'combined': [
            lambda d, i, ax: plot_cells(d, i, ax),
            lambda d, i, ax: process_and_plot_density(d, i, "drug_X", ax),
            lambda d, i, ax: process_and_plot_density(d, i, "drug_Y", ax),
            lambda d, i, ax: process_and_plot_nodes(d, i, ax),
            lambda d, i, ax: plot_curve_comparison(d, experimental_data_path, drug_name, i, ax)
        ]
    }
    
    # Create plots

    for i, plot_func in enumerate(plot_functions[experiment_type]):
        try:
            _, axs[i] = plot_func(df, instance_folder, ax=axs[i])
        except Exception as e:
            print(f"Error in subplot {i}: {str(e)}")
            traceback.print_exc()
    
    # Save in multiple formats
    save_publication_plot(fig, instance_folder, "full_summary_plot")
    plt.close(fig)

def main():
    color_dict = {"alive": "g", "apoptotic": "r",  "necrotic":"k"}

    
    instance_folder  = sys.argv[1]
    drug_name = sys.argv[2]

    output_folder = os.path.join(instance_folder, "output")
    experimental_data_path = "/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/data/AGS_data/AGS_growth_data/output/csv/"
    drug_X_name = get_drug_name(instance_folder, "drug_X_target")
    drug_Y_name = get_drug_name(instance_folder, "drug_Y_target")

    #for the density plot, in order to fix curves that are plotted
    if drug_Y_name == "none" and drug_X_name == "none":
        experiment_type = "control"
    elif drug_Y_name == "none":
        experiment_type = "single"
    else:
        experiment_type = "combined"

    # Processing the data with PCTK
    # mcds = MultiCellDS(output_folder=output_folder)
    # df_time_course, df_cell_variables, df_cell_variables_total = save_tabular_data(output_folder, mcds, instance_folder)
    try:
        df_time_course, df_cell_variables_total, df_cell_variables = save_pcdl_timeseries(output_folder, instance_folder)
    except Exception as e:
        print(f"Error saving PCDL timeseries for {instance_folder}: {e}")
        return f"Failed to save PCDL timeseries for {instance_folder}: {e}\n Maybe NaN values in the XML?"
    
    if experiment_type != "control":
        plot_cells(df_time_course, instance_folder, ax=None)
        process_and_plot_density(df_time_course, instance_folder, "drug_X", ax=None)
        process_and_plot_density(df_time_course, instance_folder, "drug_Y", ax=None)
        plot_curve_comparison(df_time_course, experimental_data_path, drug_name, instance_folder, ax=None)
        process_and_plot_nodes(df_time_course, instance_folder, ax=None)
        create_combined_plot(df_time_course, instance_folder, experiment_type, experimental_data_path, drug_name)   
    else: # for control experiments, don't plot drug
        plot_cells(df_time_course, instance_folder, ax=None)
        plot_curve_comparison(df_time_course, experimental_data_path, drug_name, instance_folder, ax=None)
        process_and_plot_nodes(df_time_course, instance_folder, ax=None)
        create_combined_plot(df_time_course, instance_folder, experiment_type, experimental_data_path, drug_name)
    
    return "Summary script OK"


# For debugging purposes

def process_instance(instance_info):
    """Process a single instance with existing data check"""
    instance, experiment_folder = instance_info
    
    if not instance.startswith("instance_2_36_1"):  # Keep original filtering
        return
        
    print(f"Processing {instance}")
    instance_folder = os.path.join(experiment_folder, instance)
    output_folder = os.path.join(instance_folder, "output")
    experimental_data_path = "/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/data/AGS_data/AGS_growth_data/output/csv/"

    print("instance_folder is", instance_folder)
    print("output_folder is", output_folder)
    
    try:
        # Get drug information
        drug_X_name = get_drug_name(instance_folder, "drug_X_target")
        drug_Y_name = get_drug_name(instance_folder, "drug_Y_target")
        print("the drug names are", "for drug_X", drug_X_name, "for drug_Y", drug_Y_name)
        
        # Determine experiment type
        if drug_Y_name is None and drug_X_name is None:
            experiment_type = "control"
            drug_name = "WT"
        elif drug_Y_name is None and drug_X_name is not None:
            experiment_type = "single"
            drug_name = drug_X_name
        else:
            experiment_type = "combined"
            drug_name = f"{drug_X_name}_{drug_Y_name}"

        # Check for existing data files
        data_file = os.path.join(instance_folder, "pcdl_total_info_sim.csv.gz")
        simgrowth_file = os.path.join(instance_folder, "simulation_growth.csv")
        
        if os.path.exists(data_file) and os.path.exists(simgrowth_file):
            print(f"Loading existing data from {data_file}")
            df_time_course = pd.read_csv(data_file)
            simulation_growth_df = pd.read_csv(simgrowth_file)
        else:
            # Process and save new data if files don't exist
            print(f"Processing from scratch for {instance}")
            df_time_course, _, _ = save_pcdl_timeseries(output_folder, instance_folder)

        # Generate combined plot with error handling
        try:
            create_combined_plot(
                df_time_course,
                instance_folder,
                experiment_type,
                experimental_data_path,
                drug_name
            )
        except Exception as e:
            print(f"Combined plot error: {str(e)}")
            traceback.print_exc()

        return f"Processed {instance} with {len(df_time_course)} records"
        
    except FileNotFoundError as e:
        print(f"Missing file: {str(e)}")
        return f"Failed {instance}: File not found"
    except pd.errors.EmptyDataError:
        print(f"Empty data in {instance}")
        return f"Failed {instance}: Empty dataset"
    except Exception as e:
        print(f"Unexpected error: {traceback.format_exc()}")
        return f"Critical failure in {instance}: {str(e)}"



if __name__ == "__main__":
    experiment_folder = "./experiments/MEKi_CMA-0703-1706-19p_delayed_transient_rmse_postdrug_50gen/"
    
    # Create list of instances to process
    instances = [(instance, experiment_folder) for instance in os.listdir(experiment_folder)]
    
    # Use 75% of available CPUs
    # num_processes = max(1, int(cpu_count() * 0.5))
    num_processes = 1
    print(f"Running with {num_processes} processes")
    
    # Create pool and map processes
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_instance, instances)
        
    
    # Print results
    for result in results:
        if result:  # Only print non-None results
            print(result)
 

 
