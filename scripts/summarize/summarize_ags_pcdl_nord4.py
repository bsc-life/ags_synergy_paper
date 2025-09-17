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

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append("./python") # for running locally 
from multicellds import MultiCellDS
import pickle

pd.options.mode.chained_assignment = None

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
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    
    else:
        ax.clear()
        fig = ax.figure

    # Subset the dataframe to only include 'time' and 'cell_type' columns
    df_subset = df_time_course[['time', 'current_phase']]
    
    # Group by 'time' and 'cell_type', count occurrences, and unstack to create columns for each cell type
    df_aggregated = df_subset.groupby(['time', 'current_phase']).size().unstack(fill_value=0)
    
    # Reset index to make 'time' a column again
    df_aggregated = df_aggregated.reset_index()
    
    # Melt the dataframe to long format for seaborn
    df_melted = pd.melt(df_aggregated, id_vars=['time'], var_name='current_phase', value_name='count')
    
    # Create the plot
    sns.set_theme(style="whitegrid")
    ax = sns.lineplot(data=df_melted, x='time', y='count', hue='current_phase', linewidth=3, ax=ax)

    total_drug_addition_time = drug_time_addition_info(instance_folder, f"drug_X_pulse_duration")
    ax.axvspan(1280, int(1280 + total_drug_addition_time), color='grey', alpha=0.5, label='Drug X addition')

    ax.legend(title='Current Phase', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel('Number of cells', fontsize=20)
    # ax.set_title('Live cell plot', fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlim(0, 4200)

    # Showing legend
    ax.legend()
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    sns.despine(offset=0, trim=True)

    if ax is None:  
        plt.savefig(os.path.join(instance_folder, "cellgrowth.jpeg"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(instance_folder, "cellgrowth.png"), dpi=300, transparent=True, bbox_inches="tight")
        print("Alive/Apoptotic cells plot obtained")
        plt.close(fig)
    return fig, ax

def process_and_plot_density(df, instance_folder, drug_name, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    else:
        ax.clear()
        fig = ax.figure

    sns.set_theme(style="whitegrid")

    # Filter for the specified drug
    density_cols = df.filter(regex=f'{drug_name}|internal_density|external_density').filter(regex='^(?!.*oxygen)').filter(regex=f'^(?!.*{drug_name})').columns
    selected_cols = list(density_cols) + ['time']

    df_subset = df[selected_cols]

    # Group by time and calculate mean and std for each density column
    df_aggregated = df_subset.groupby('time').agg(['mean', 'std'])
    # Flatten column names
    df_aggregated.columns = ['_'.join(col).strip() for col in df_aggregated.columns.values]
    # Reset index to make 'time' a column again
    df_aggregated = df_aggregated.reset_index()
    # Melt the dataframe to long format for seaborn
    df_melted = pd.melt(df_aggregated, id_vars=['time'], 
                        value_vars=[col for col in df_aggregated.columns if col.endswith('_mean')],
                        var_name='density_type', value_name='density')
    # Create a corresponding dataframe for standard deviation
    df_melted_std = pd.melt(df_aggregated, id_vars=['time'], 
                            value_vars=[col for col in df_aggregated.columns if col.endswith('_std')],
                            var_name='density_type', value_name='std')
    df_melted_std['density_type'] = df_melted_std['density_type'].str.replace('_std', '_mean')
    # Merge mean and std dataframes
    df_melted = pd.merge(df_melted, df_melted_std, on=['time', 'density_type'])

    # Create the plot
    ax = sns.lineplot(data=df_melted, x='time', y='density', hue='density_type', ax=ax, linewidth=3)

    # Add shaded region for standard deviation
    for name, group in df_melted.groupby('density_type'):
        ax.fill_between(group['time'], group['density'] - group['std'], 
                        group['density'] + group['std'], alpha=0.2)
    
    total_drug_addition_time = drug_time_addition_info(instance_folder, f"{drug_name}_pulse_duration")
    ax.axvspan(1280, int(1280 + total_drug_addition_time), color='grey', alpha=0.5, label='Drug X addition')

    # ax.set_title(f'{drug_name} densities over time')
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel('Density', fontsize=20)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    # ax.legend(title='Density Type', fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlim(0, 4200)
    sns.despine(offset=0, trim=True)

    # Adjust layout and save the figure
    plt.tight_layout()

    if ax is None:
        plt.savefig(os.path.join(instance_folder, f"densityplot_{drug_name}.jpeg"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(instance_folder, f"densityplot_{drug_name}.png"), dpi=300, transparent=True, bbox_inches="tight")
        print(f"Density plot for {drug_name} saved")
        plt.close(fig)
    return fig, ax

def process_and_plot_nodes(df, instance_folder, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    else:
        ax.clear()
        fig = ax.figure
   

    # Filter out rows where 'current_phase' is 'alive'
    df = df[df['current_phase'] != 'alive']

    # Select columns containing 'node' but not 'anti', and 'time'
    node_cols = df.filter(regex='node').filter(regex='^(?!.*anti)').columns
    selected_cols = list(node_cols) + ['time']
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
    ax = sns.lineplot(data=df_melted, x='time', y='value', hue='node_type', ax=ax, linewidth=3)
    sns.despine(offset=0, trim=True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.set_xlim(0, 4200)

    # Add shaded region for standard deviation
    # for name, group in df_melted.groupby('node_type'):
    #     ax.fill_between(group['time'], group['value'] - group['std'], 
    #                     group['value'] + group['std'], alpha=0.2)

    # Add shaded grey region from x=1200 to x=1240
    total_drug_x_addition_time = drug_time_addition_info(instance_folder, "drug_X_pulse_duration")
    ax.axvspan(1280, int(1280 + total_drug_x_addition_time), color='grey', alpha=0.5, label='Drug X addition')

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

    # print(df_unique.head(15))

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
    if ax is None:
        fig, ax = plt.subplots(dpi=300)
    else:
        ax.clear()
        fig = ax.figure

    # sns.set_palette("muted")  # You can choose from seaborn's palettes or set your own
    sns.set_theme(style="whitegrid")
    ax = sns.lineplot(data=df_combined, x='x', y='y', hue='Type', style='Type', markers=True, palette="Set2", ax=ax, linewidth=3)

    # Add grid and other settings
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.legend(title='Curve Type')
    sns.despine(offset=0, trim=True)

    if drug_name != "WT":
        total_drug_x_addition_time = drug_time_addition_info(instance_folder, "drug_X_pulse_duration")
        ax.axvspan(1280, int(1280 + total_drug_x_addition_time), color='grey', alpha=0.5, label='Drug X addition')

    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel('Relative cell number', fontsize=20)
    # ax.set_title('Experimental and simulated curve comparison', fontsize=20)

    plt.xlim(0, 4200)

    # Adjust layout and save the figure
    # plt.tight_layout()
    # plt.savefig(os.path.join(instance_folder, "curve_comparison_seaborn.jpeg"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(instance_folder, "curve_comparison_seaborn.png"), dpi=300, transparent=True, bbox_inches="tight")
    print("Curve comparison plot obtained")
    # plt.close(fig)
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
    mcdsts = pyMCDSts(output_path = output_folder, graph=False, verbose=False)

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

    # @oth: 10.03.25
    # Increased compression rate - used to be "gzip" and now it's "bz2"

    # all_data.to_csv(os.path.join(instance_folder, "pcdl_total_info_sim.csv.gz"), header=True, index=False, compression='gzip')
    all_data.to_csv(os.path.join(instance_folder, "pcdl_total_info_sim.csv.gz"),
                header=True,
                index=False,
                compression={'method': 'gzip', 'compresslevel': 5})  # compresslevel 9 is maximum compression

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


def create_combined_plot(df, instance_folder, experiment_type, experimental_data_path, drug_name):
    """
    Creates a combined plot with different subplots based on experiment type.
    
    Args:
        df: DataFrame with simulation data
        instance_folder: Path to save outputs
        experiment_type: One of ['control', 'single', 'combined']
        experimental_data_path: Path to experimental data
        drug_name: Name of the drug being tested: PI3K, MEK, AKT or PI3K-MEK, AKT-MEK
    """
    # Determine number of subplots based on experiment type
    if experiment_type == 'control':
        n_plots = 2  # cells, nodes, curve comparison
        fig, axs = plt.subplots(n_plots, 1, figsize=(20, 18), dpi=300)
        
        # Generate only relevant plots
        _, axs[0] = plot_cells(df, instance_folder, ax=axs[0])
        _, axs[2] = plot_curve_comparison(df, experimental_data_path, drug_name, instance_folder, ax=axs[1])
        
    elif experiment_type == 'single':
        n_plots = 4  # cells, drug density, nodes, curve comparison
        fig, axs = plt.subplots(n_plots, 1, figsize=(20, 24), dpi=300)
        
        # Generate plots for single drug experiment
        _, axs[0] = plot_cells(df, instance_folder, ax=axs[0])
        _, axs[1] = process_and_plot_density(df, instance_folder, "drug_X", ax=axs[1])
        _, axs[2] = process_and_plot_nodes(df, instance_folder, ax=axs[2])
        _, axs[3] = plot_curve_comparison(df, experimental_data_path, drug_name, instance_folder, ax=axs[3])
        
    else:  # combined
        n_plots = 5  # all plots
        fig, axs = plt.subplots(n_plots, 1, figsize=(20, 30), dpi=300)
        
        # Generate all plots for combined drug experiment
        _, axs[0] = plot_cells(df, instance_folder, ax=axs[0])
        _, axs[1] = process_and_plot_density(df, instance_folder, "drug_X", ax=axs[1])
        _, axs[2] = process_and_plot_density(df, instance_folder, "drug_Y", ax=axs[2])
        _, axs[3] = process_and_plot_nodes(df, instance_folder, ax=axs[3])
        _, axs[4] = plot_curve_comparison(df, experimental_data_path, drug_name, instance_folder, ax=axs[4])

    # Common formatting
    fig.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    
    # Save figures in different formats
    save_path_base = os.path.join(instance_folder, 'full_summary_plot')
    
    # High quality PNG for presentations/publications
    plt.savefig(f"{save_path_base}.png", 
                dpi=300, 
                bbox_inches="tight", 
                transparent=True)
    
    # JPG for quick viewing/sharing
    plt.savefig(f"{save_path_base}.jpg", 
                dpi=100, 
                bbox_inches="tight", 
                quality=85)  # Reduced quality for smaller file size
    
    print(f"Full summary plot saved for {experiment_type} experiment")
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
    
    # if experiment_type != "control":
    #     # plot_cells(df_time_course, instance_folder, ax=None)
    #     # process_and_plot_density(df_time_course, instance_folder, "drug_X", ax=None)
    #     # process_and_plot_density(df_time_course, instance_folder, "drug_Y", ax=None)
    #     plot_curve_comparison(df_time_course, experimental_data_path, drug_name, instance_folder, ax=None)
        # process_and_plot_nodes(df_time_course, instance_folder, ax=None)
        # create_combined_plot(df_time_course, instance_folder, experiment_type, experimental_data_path, drug_name)   
    # else: # for control experiments, don't plot drug
    #     plot_cells(df_time_course, instance_folder, ax=None)
    #     plot_curve_comparison(df_time_course, experimental_data_path, drug_name, instance_folder, ax=None)
    #     process_and_plot_nodes(df_time_course, instance_folder, ax=None)
    #     create_combined_plot(df_time_course, instance_folder, experiment_type, experimental_data_path, drug_name)
    
    return "Summary script OK"


main()

