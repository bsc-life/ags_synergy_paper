#!/usr/bin/env python
# coding: utf-8

import re, os, sys, warnings
import numpy as np
import scipy as sp
from scipy import interpolate, integrate
import pandas as pd
import xml.dom.minidom
import matplotlib.pyplot as plt
import pcdl
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append("./python") # for running locally 
from multicellds import MultiCellDS
import pickle

pd.options.mode.chained_assignment = None


def get_timeserie_mean(mcds, filter_alive=True):
    time = []
    values = []
    filter_alive = True
    for t, df in mcds.cells_as_frames_iterator():
        time.append(t)
        df = df.iloc[:,3:]
        if filter_alive:
            mask = df['current_phase'] <= 14
            df = df[mask]
        values.append(df.mean(axis=0).values)

    cell_columns = df.columns.tolist()
    df = pd.DataFrame(values, columns=cell_columns)
    df['time'] = time
    return df[['time'] + cell_columns]

def get_timeserie_density(mcds, density_id, density_name='density', agg='sum'):
    data = []
    for t,m in mcds.microenvironment_as_matrix_iterator():
        value = -1
        if agg == 'sum':
            value = m[4 + density_id,:].sum()
        if agg == 'mean':
            value = m[4 + density_id,:].mean()
        data.append((t, value))
    df = pd.DataFrame(data=data, columns=['time', density_name])
    return df

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

def plot_cells(df_time_course, color_dict, ax):

    # Alive/Apoptotic/Necrotic vs Time
    for k in color_dict:
        ax.plot(df_time_course.time, df_time_course[k], "-", c=color_dict[k], label=k)

    # setting axes labels
    # ax.set_xlabel("time (min)")
    ax.set_ylabel("Nº of cells")

    # Showing legend
    ax.legend()
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    
    # print("Alive/Apoptotic cells plot obtained")
    
def plot_density(df_cell_variables, experiment_type, df_time_course, ax):

    # always plot Oxygen
    # ax.plot(df_time_course.time, df_cell_variables.oxygen_internal_density, "-", label="Drug X internal")
    # ax.plot(df_time_course.time, df_cell_variables.oxygen_external_density, "-", label="Drug X external")

    if experiment_type == "single":
        ax.plot(df_time_course.time, df_cell_variables.drug_X_internal_density, "-", label="Drug density internal")
        ax.plot(df_time_course.time, df_cell_variables.drug_X_external_density, "-", label="Drug density external")

    if experiment_type == "combined":
        ax.plot(df_time_course.time, df_cell_variables.drug_X_internal_density, "-", label="Drug X internal")
        ax.plot(df_time_course.time, df_cell_variables.drug_X_external_density, "-", label="Drug X external")

        ax.plot(df_time_course.time, df_cell_variables.drug_Y_internal_density, "-", label="Drug Y internal")
        ax.plot(df_time_course.time, df_cell_variables.drug_Y_external_density, "-", label="Drug Y external")
    

    ax.set_ylabel("Drug density (mM)")
    ax.legend()
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    # print("Saved density plot OK")

def get_min_max_wt_exp_curve(input_df):
    # get max value from control curve

    df_control_csv = input_df
    df_control = df_control_csv.reset_index(inplace=False)
    df_control.Time = df_control.Time / 60 # only needed when reading pickle data
    df_control['Average_Cell_Index'] = df_control.iloc[:,2:].mean(axis=1)
    df_control.Average_Cell_Index = df_control.Average_Cell_Index
    df_control = df_control[["Time", "Average_Cell_Index"]]
    df_control_sliced = df_control.loc[df_control["Time"] < 4200]

    try:
        f = interpolate.interp1d(df_control.Time, df_control.Average_Cell_Index, kind="cubic")
        xnew_control_exp = np.arange(0, 4200, 40) # emulating PC simulation points 
        ynew_control_exp = f(xnew_control_exp)
    except ValueError:
        f = interpolate.interp1d(df_control.Time, df_control.Average_Cell_Index, kind="cubic", bounds_error=False)
        xnew_control_exp = np.arange(0, 4200, 40) # emulating PC simulation points 
        ynew_control_exp = f(xnew_control_exp)

    max_value_wt_exp = ynew_control_exp.max()
    min_value_wt_exp = ynew_control_exp.min()

    return min_value_wt_exp, max_value_wt_exp

def normalize_exp_curve(exp_df, min, max):

    # df_drug_csv = pd.read_csv(os.path.join(exp_curves_folder, "PI103(0.70uM).csv"))
    df_drug = exp_df.reset_index(inplace=False)
    df_drug.Time = df_drug.Time / 60 # only needed when reading pickle data
    df_drug['Average_Cell_Index'] = df_drug.iloc[:,2:].mean(axis=1)
    df_drug.Average_Cell_Index = df_drug.Average_Cell_Index
    df_drug = df_drug[["Time", "Average_Cell_Index"]]
   
    # Interpolate avoiding technical artifacts on the curve
    # df_drug = df_drug[ (df_drug['Time'] < 1000.0) | (df_drug["Time"]>1600.0)] # does not work for MEK
    df_drug = df_drug[ (df_drug['Time'] < 400.0) | (df_drug["Time"]>2500.0)]
    df_drug_sliced = df_drug.loc[df_drug["Time"] < 4200]


    try:
        f = interpolate.interp1d(df_drug.Time, df_drug.Average_Cell_Index, kind="cubic", fill_value="extrapolate")
        xnew_drug_exp = np.arange(0, 4240, 40) # emulating PC simulation points 
        ynew_drug_exp = f(xnew_drug_exp)

    except ValueError:
        # f = interpolate.interp1d(df_drug.Time, df_drug.Average_Cell_Index, kind="cubic", fill_value="0")
        f = interpolate.interp1d(df_drug.Time, df_drug.Average_Cell_Index, kind="cubic", bounds_error=False, fill_value="extrapolate")
        xnew_drug_exp = np.arange(0, 4240, 40) # emulating PC simulation points 
        ynew_drug_exp = f(xnew_drug_exp)

    # Then normalize based on WT
    # tmp_norm = df_drug_sliced.drop("Time", axis=1)
    tmp_norm = ((ynew_drug_exp-min)/(max-min)) * 100
    df_drug_exp_norm = pd.concat((pd.Series(xnew_drug_exp), pd.Series(tmp_norm)), axis=1)
    df_drug_exp_norm.columns = ["Time", "Average_Cell_Index"]


    return df_drug_exp_norm.Time, df_drug_exp_norm.Average_Cell_Index

def get_min_max_wt_sim_curve(input_df):

    # df_merged_path = "/home/oth/BSC/my_PB_branch/drugsyn_pboss/PhysiBoSS/NEW_full_runs/2_naive_BM/MEK-AKT/output_1/full_data_merged.csv"
    # df_merged = pd.read_csv(input_df)
    df_merged = input_df.sort_values(["time", "ID"], ascending=True)

    # get "wrong" times
    df_merged_count_times = df_merged.groupby("time").count().reset_index()
    wrong_times = []
    for ix, time in enumerate(df_merged_count_times.ID):
        if time <= 150:
            wrong_times.append(df_merged_count_times.time[ix])

    wrong_times_df = pd.DataFrame(wrong_times, columns=["wrong_time"])
    df_merged.drop(df_merged[df_merged.time.isin(wrong_times_df.wrong_time)].index, axis=0, inplace=True)
    df_merged.drop(df_merged[df_merged.current_phase != 14.0].index, axis=0, inplace=True)
    number_of_agents_df = df_merged.groupby(["time", "current_phase"]).agg({"ID": max })
    number_of_agents_df = number_of_agents_df.reset_index()
    simulation_curve = number_of_agents_df.ID.to_numpy()

    initial_number_of_cells = simulation_curve[0]

    sim_norm_df = number_of_agents_df.copy()
    sim_norm_df.ID = sim_norm_df.ID - initial_number_of_cells
    simulation_curve_norm = sim_norm_df.ID.to_numpy()

    sim_curve_min = simulation_curve_norm.min()
    sim_curve_max = simulation_curve_norm.max()

    return sim_curve_min, sim_curve_max

def normalize_sim_curve(exp_drug_df, min, max):
    """ normalize simulation curve coming from the df_time_course dataframe from MCDS """
    
    exp_values = exp_drug_df["alive"].values
    exp_drug_df.alive = exp_drug_df.alive - exp_values[0]
    simulation_curve_norm_drug = exp_drug_df.alive.to_numpy()

    tmp_norm = exp_drug_df.drop("time", axis=1)
    tmp_norm = ((tmp_norm-min)/(max-min)) * 100 # scaling by fitted WT
    df_drug_sim_norm = pd.concat((tmp_norm, exp_drug_df.time), axis=1)

    return df_drug_sim_norm.time, df_drug_sim_norm.alive

def plot_curve_comparison(df_time_course, experimental_path, drug_name, ax):

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
    

    # elif drug_name == "PI3K_TAK1":
    #     exp_drug_df = pd.read_csv(f'{experimental_path}/PD0325901(17.50nM)+PI103(0.35uM).csv') 
    # elif drug_name == "AKT_TAK1":
    #     exp_drug_df = pd.read_csv(f'{experimental_path}/PD0325901(17.50nM)+PI103(0.35uM).csv') 



    wt_min_exp, wt_max_exp = get_min_max_wt_exp_curve(exp_wt_df)
    x_exp, y_exp = normalize_exp_curve(exp_drug_df, wt_min_exp, wt_max_exp)

    # fetch simulated data
    sim_wt_df = pd.read_csv(f'{experimental_path}/SIM_CTRL.csv')
    simulation_curve, simulation_time = df_time_course.alive, df_time_course.time

    wt_min_sim, wt_max_sim = get_min_max_wt_sim_curve(sim_wt_df)
    x_sim, y_sim = normalize_sim_curve(df_time_course, wt_min_sim, wt_max_sim)

    ax.plot(x_exp, y_exp, "r--", label="Experimental")
    ax.plot(x_sim, y_sim, "b-", label="Simulation")
    ax.set_ylabel("Number of alive cells")
    ax.legend()
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    # ax.set_xlabel("Simulation time (min)")

def save_simulated_growth_curve(df_time_course, output_folder):
    " Saves a .csv file with the time and amount of 'alive' cells within the simulation. "
    alive_df = df_time_course[["time", "alive"]]
    alive_df.to_csv(output_folder, header=True, index=False)

def get_drug_name(instance_folder, drug_name):
    doc = xml.dom.minidom.parse(os.path.join(instance_folder,"settings.xml"))
    custom_data = doc.getElementsByTagName(drug_name)
    drug_name = str(custom_data[0].firstChild.nodeValue)
    if drug_name != "none":
        drug_name = drug_name.split("anti_")[1]
        return drug_name
    else:
        return None

def save_pcdl_timeseries(output_folder, instance_folder):
    " Saves a pickle with the TimeSeries object from pcdl, as well as some tabular data "
    mcdsts = pcdl.TimeSeries(output_path = output_folder, graph=True)
    pickle_path = os.path.join(instance_folder, "pcdl_timeseries.pkl")
    with open(pickle_path, 'wb') as a:
        pickle.dump(mcdsts, a)

    # also save ALL data in tabular format
    list_of_relevant_vars = list()
    all_data = pd.DataFrame()
    for mcds in mcdsts.get_mcds_list():
        frame_df = mcds.get_cell_df()
        frame_df.reset_index(inplace=True)
        list_of_relevant_vars.append(frame_df)

    all_data = pd.concat(list_of_relevant_vars, ignore_index=True) 
    all_data.to_csv(os.path.join(instance_folder, "pcdl_total_info_sim.csv"), header=True, index=False)
    return all_data

def get_cell_df_time_aggregated()

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

def main():
    color_dict = {"alive": "g", "apoptotic": "r",  "necrotic":"k"}
    instance_folder  = sys.argv[1]
    drug_name = sys.argv[2]
    # AN OPTION to introduce the metric
    # metric = sys.argv[2]

    output_folder = os.path.join(instance_folder, "output")
    experimental_data_path = "/gpfs/projects/bsc08/MN4/bsc08/bsc08494/EMEWS/drug_synergy_emews/data/AGS_data/AGS_growth_data/output/csv/"
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
    mcds = MultiCellDS(output_folder=output_folder)
    df_time_course, df_cell_variables, df_cell_variables_total = save_tabular_data(output_folder, mcds, instance_folder)

    # Save internal BN node states
    list_of_variables = ['mek_node', 'pi3k_node', 'akt_node', 'tak1_node']

    
    if experiment_type != "control":
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), dpi=300, sharex=True, sharey=False) # original size is (12, 12)
        fig.suptitle('Pulse period: %.4f, Pulse duration: %.4f, [TNF]: %.4f' % (k1,k2,k3) )
        plot_cells(df_time_course, color_dict, axes[0])
        plot_density(df_cell_variables, experiment_type, df_time_course, axes[1])
        plot_curve_comparison(df_time_course, experimental_data_path, drug_name, axes[2])
        plot_molecular_model(df_cell_variables, list_of_variables, axes[3])
        axes[-1].set_label("Simulation time (min)")
    
    else: # for control experiments, don't plot drug
        fig, axes = plt.subplots(2, 1, figsize=(12, 12), dpi=300, sharex=True, sharey=False)
        plot_cells(df_time_course, color_dict, axes[0])
        plot_curve_comparison(df_time_course, experimental_data_path, drug_name, axes[1])
        axes[-1].set_label("Simulation time (min)")

    fig.tight_layout()
    
    # improve quality of plot
    fig.savefig(os.path.join(instance_folder, 'variables_vs_time.png'), dpi=300)
    fig.savefig(os.path.join(instance_folder, 'variables_vs_time.svg'), dpi=300)


main()



# For debugging purposes
# if __name__ == "__main__":

#     output_folder = "./experiments/v2_test_sweep/instance_4885_9/output"
#     instance_folder = "./experiments/v2_test_sweep/instance_4885_9"

#     # mcds = MultiCellDS(output_folder)
#     # df_time_course = mcds.get_cells_summary_frame()

#     save_pcdl_timeseries(output_folder, instance_folder)