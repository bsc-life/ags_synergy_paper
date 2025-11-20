# This script takes the topN curves (top 10, 20, 50, 100) from the CSV and averages them and does a lineplot
# Input are the CSV files from the CMA and GA runs


# TODO encapsulate the functions from "summarize_ags_pcdl.py" into a separate script and import from there

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import interpolate
from scipy import stats
from scipy.stats import pearsonr
import logging


# sys.path.append("scripts/summarize")
# from summarize_ags_pcdl import get_min_max_wt_exp_curve, normalize_exp_curve
def get_strategy_from_filename(filename):
    if "CMA" in filename:
        return "CMA"
    elif "GA" in filename:
        return "GA"
    elif "sweep" in filename:
        return "sweep"
    else:
        raise ValueError("Strategy not detected")

def get_drug_from_filename(filename):
    if "PI3K" in filename:
        return "PI3K"
    elif "MEK" in filename:
        return "MEK"
    elif "AKT" in filename:
        return "AKT"
    elif "CTRL" in filename:
        return "WT"
    elif "pi3k_mek" in filename:
        return "PI3K_MEK"
    elif "akt_mek" in filename:
        return "AKT_MEK"
    else:
        return "Unknown experiment name"


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

    return min_value, max_value, initial_cell_number     
     

def normalize_sim_curve(exp_drug_df, initial_cell_number, min, max):
    """ normalize simulation curve coming from the df_time_course dataframe from MCDS """

    df = exp_drug_df[exp_drug_df['current_phase'] != 'alive']
    # Calculate total number of cells for each time point
    df['alive_cell_number'] = df.groupby('time')['ID'].transform('nunique')
    initial_cell_number = df.loc[0, 'alive_cell_number']


    # Now get the normalized column
    subset_df = df.loc[:, ['time', 'alive_cell_number']]
    subset_df['adjusted_alive_cell_number'] = subset_df['alive_cell_number'] - initial_cell_number
    max_value = subset_df['adjusted_alive_cell_number'].max()
    min_value = subset_df['adjusted_alive_cell_number'].min()
    subset_df['normalized_alive_cell_number'] = ((subset_df['adjusted_alive_cell_number'] - min) / (max - min)) * 100

    df_unique = subset_df.drop_duplicates()

    return df_unique.time, df_unique.normalized_alive_cell_number

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
    # print("Normalizing simulation growth curve with simulated control")

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

def load_drug_data(experimental_data_path, drug_name):
    """
    Load and process experimental drug data based on the drug name.
    
    Args:
        experimental_data_path (str): Path to the experimental data directory
        drug_name (str): Name of the drug to load
        
    Returns:
        tuple: (DataFrame, numpy array, numpy array) containing the drug data, mean values, and standard deviations
    """
    if drug_name == "PI3K":
        exp_drug_df = pd.read_csv(f'{experimental_data_path}/PI103(0.70uM).csv') 
    elif drug_name == "MEK":
        exp_drug_df = pd.read_csv(f'{experimental_data_path}/PD0325901(35.00nM).csv') 
    elif drug_name == "TAK1":
        exp_drug_df = pd.read_csv(f'{experimental_data_path}/(5Z)-7-oxozeaenol(0.50uM).csv') 
    elif drug_name == "AKT":
        exp_drug_df = pd.read_csv(f'{experimental_data_path}/AKT_2.csv') 
    elif drug_name == "WT":
        exp_drug_df = pd.read_csv(f'{experimental_data_path}/CTRL.csv') 
    elif drug_name == "PI3K_MEK":
        exp_drug_df = pd.read_csv(f'{experimental_data_path}/PD0325901(17.50nM)+PI103(0.35uM).csv')
    elif drug_name == "AKT_MEK":
        exp_drug_df = pd.read_csv(f'{experimental_data_path}/AKT_MEK_final_ok.csv') 
    else:
        raise ValueError(f"Unknown drug name: {drug_name}")
        
    y_exp = exp_drug_df.iloc[:, 1:].mean(axis=1)  # Mean across all replicates
    y_exp_std = exp_drug_df.iloc[:, 1:].std(axis=1)  # SD for each timepoint
    
    return exp_drug_df, y_exp, y_exp_std

def get_experiment_color_scheme():
    """
    Returns a professional color scheme for different experiment types.
    Colors are chosen to be publication-quality, colorblind-friendly, and print-friendly.
    """
    return {
        'control': {
            'exp': '#808080',  # Medium grey for experimental
            'sim': '#A9A9A9'   # Lighter grey for simulation
        },
        'PI3Ki': {
            'exp': '#1E88E5',  # Professional blue for experimental
            'sim': '#64B5F6'   # Lighter blue for simulation
        },
        'MEKi': {
            'exp': '#2E7D32',  # Professional green for experimental
            'sim': '#81C784'   # Lighter green for simulation
        },
        'AKTi': {
            'exp': '#7B1FA2',  # Professional purple for experimental
            'sim': '#BA68C8'   # Lighter purple for simulation
        },
        'pi3k_mek': {
            'exp': '#F57C00',  # Professional orange for experimental
            'sim': '#FFB74D'   # Lighter orange for simulation
        },
        'akt_mek': {
            'exp': '#E65100',  # Darker orange for experimental
            'sim': '#FF8F00'   # Lighter orange for simulation
        }
    }

def get_experiment_colors(exp_name):
    """
    Helper function to get colors for a specific experiment type.
    """
    color_scheme = get_experiment_color_scheme()
    exp_type = exp_name.lower()
    
    if exp_type == "wt" or exp_type == "ctrl" or 'control' in exp_type:
        return color_scheme['control']
    elif ('pi3k_mek' in exp_type) or ('pi3kmek' in exp_type):
        return color_scheme['pi3k_mek']
    elif ('akt_mek' in exp_type) or ('aktmek' in exp_type):
        return color_scheme['akt_mek']
    elif 'pi3k' in exp_type:
        return color_scheme['PI3Ki']
    elif 'mek' in exp_type:
        return color_scheme['MEKi']
    elif 'akt' in exp_type:
        return color_scheme['AKTi']
    else:
        return color_scheme['control']  # Default to control colors

def process_top_10(csv_path, experiment_name, experimental_data_path, drug_name, top_n, nodrug=False):
    # Read the top 10 CSV
    top_10_df = pd.read_csv(csv_path)

    # Get the experiment folder from the csv path
    experiment_folder = os.path.join("experiments", experiment_name)

    # For the best fittings run, the WT reference was SIM_CTRL_CMA-1110-1637-5p.csv
    sim_wt_df = pd.read_csv(f'{experimental_data_path}SIM_CTRL_CMA-1110-1637-5p.csv')

    # Initialize a list to store all simulation growth dataframes
    all_sim_growths = []
    
    # Process each row in the top 10 CSV
    for _, row in top_10_df.iterrows():
        if "iteration" in row.index:
            instance_folder = f"instance_{int(row['iteration'])}_{int(row['individual'])}_{int(row['replicate'])}"
        else:
            instance_folder = f"instance_{int(row['individual'])}_{int(row['replicate'])}"

        # for different summary files, we can either have the gzip or the bz2 compressed versions
        full_path_gzip = os.path.join(experiment_folder, instance_folder, 'pcdl_total_info_sim.csv.gz')
        full_path_bz2 = os.path.join(experiment_folder, instance_folder, 'pcdl_total_info_sim.csv.bz2')
        
        if os.path.exists(full_path_gzip):
            full_path = full_path_gzip
        elif os.path.exists(full_path_bz2):
            full_path = full_path_bz2
        else:
            raise FileNotFoundError(f"Neither {full_path_gzip} nor {full_path_bz2} exists.")

            
        if os.path.exists(full_path):
            try:
                full_sim_df = pd.read_csv(full_path, compression='gzip')
            except Exception as e:
                full_sim_df = pd.read_csv(full_path, compression='bz2')
            if drug_name == "WT":
                x_sim, y_sim = normalize_sim_curve_control(full_sim_df)
            else:
                wt_min_sim, wt_max_sim, initial_cells = get_min_max_wt_sim_curve(sim_wt_df)
                simgrowth_reduced_df = pd.read_csv(os.path.join(experiment_folder, instance_folder, "simulation_growth.csv"))
                x_sim, y_sim = normalize_simulation_growth_curve(simgrowth_reduced_df, initial_cells, wt_min_sim, wt_max_sim)  

            sim_growth = pd.DataFrame(columns=["time", "alive"])
            sim_growth["time"] = x_sim
            sim_growth["alive"] = y_sim
            sim_growth['instance'] = instance_folder
            all_sim_growths.append(sim_growth)
    
    combined_sim_growth = pd.concat(all_sim_growths, ignore_index=True)
    
    sim_stats = combined_sim_growth.groupby('time').agg({'alive': ['mean', 'std']}).reset_index()
    sim_stats.columns = ['time', 'mean_alive', 'std_alive']
    
    exp_wt_df = pd.read_csv(f'{experimental_data_path}/CTRL.csv')

    # Load drug data using the new function
    exp_drug_df, y_exp, y_exp_std = load_drug_data(experimental_data_path, drug_name)
    
    wt_min_exp, wt_max_exp = get_min_max_wt_exp_curve(exp_wt_df)
    x_exp, y_exp = normalize_exp_curve(exp_drug_df, wt_min_exp, wt_max_exp)

    # compute the SD of the y_exp 
    # y_exp_std = np.std(y_exp)
    # Calculate Pearson correlation coefficient and p-value
    # cut the simulation data to the experiment length
    y_sim = sim_stats["mean_alive"][:len(y_exp)]
    pearson_corr, p_value = pearsonr(y_sim, y_exp)

    # Save the Pearson correlation results to a .txt file
    correlation_results_path = os.path.join(os.path.dirname(csv_path), f'pearson_correlation_{drug_name}_top{top_n}.txt')
    with open(correlation_results_path, 'w') as f:
        f.write(f'Pearson correlation: {pearson_corr}\n')
        f.write(f'p-value: {p_value}\n')

    # Plotting with Cell Systems aesthetics
    if nodrug:
        plt.figure(figsize=(10, 4), dpi=300)
    else:
        plt.figure(figsize=(4, 4), dpi=300)
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    
    # Use sans-serif font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Liberation Sans', 
                                      'FreeSans', 'Arial', 'sans-serif']
    
    # Plot experimental data as scatterplot
    if not nodrug:
        if "AKT" in drug_name or "AKT_MEK" in drug_name:
            # slice the std data to the same length as the experimental data
            y_exp_std = y_exp_std[:len(y_exp) - 1]  # Exclude the last point
            x_exp = x_exp[:-1]  # Exclude the last point
            y_exp = y_exp[:-1]  # Exclude the last point

            # Plot the mean experimental line
            plt.plot(x_exp[::3], y_exp[::3],  
                    label='Experimental',
                    color='#808080',
                    marker='o',  
                    markersize=4,
                    linewidth=1.0,
                    zorder=3)

            # Add shaded area for standard deviation
            plt.fill_between(x_exp[::3], 
                            (y_exp - y_exp_std)[::3],  # Lower bound
                            (y_exp + y_exp_std)[::3],  # Upper bound
                            color='#808080',
                            alpha=0.9,  # Transparency of the shaded area
                            zorder=2)

        else: # for PI3K, MEK
            # slice the std data to the same length as the experimental data
            y_exp_std = y_exp_std[:len(y_exp)]

            # Plot the mean experimental line
            plt.plot(x_exp[::3], y_exp[::3],  
                    label='Experimental',
                    color='#808080',
                    marker='o',  
                    markersize=4,
                    linewidth=1.0,
                    zorder=3)

            # Add shaded area for standard deviation
            plt.fill_between(x_exp[::3], 
                            (y_exp - y_exp_std)[::3],  # Lower bound
                            (y_exp + y_exp_std)[::3],  # Upper bound
                            color='#808080',
                            alpha=0.9,  # Transparency of the shaded area
                            zorder=2)
    
    else:
        # slice the std data to the same length as the experimental data
        y_exp_std = y_exp_std[:len(y_exp)]

        # Plot the mean experimental line
        plt.plot(x_exp[::3], y_exp[::3],  
                label='Experimental',
                color='#808080',
                marker='o',  
                markersize=4,
                linewidth=1.0,
                zorder=3)

        # Add shaded area for standard deviation
        plt.fill_between(x_exp[::3], 
                        (y_exp - y_exp_std)[::3],  # Lower bound
                        (y_exp + y_exp_std)[::3],  # Upper bound
                        color='#808080',
                        alpha=0.9,  # Transparency of the shaded area
                        zorder=2)

    # Plot simulation data as a line with updated styling
    plt.plot(sim_stats['time'], 
            sim_stats['mean_alive'],
            label='Simulation (mean)', 
            color='#4B4BFF',  # Matching blue from other plots
            linewidth=1.0,  # Thin line to match style
            alpha=1.0,
            zorder=2)
    
    # Add confidence interval
    plt.fill_between(sim_stats['time'], 
                    sim_stats['mean_alive'] - sim_stats['std_alive'],
                    sim_stats['mean_alive'] + sim_stats['std_alive'],
                    alpha=0.3,  # Reduced alpha to match style
                    color='#4B4BFF',
                    zorder=1)

    # add the treatment window
    if not nodrug:
        plt.axvspan(1280, 1292, 
                    color='#FF6B6B', 
                    alpha=1.0,
                    zorder=0,
                    label='Treatment')
 
    # Get current axis
    ax = plt.gca()
    
    # Customize the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)

    # Labels and ticks with updated styling
    plt.xlabel('Simulation Time (min)', fontsize=13, fontweight="bold")
    plt.ylabel('Relative Cell Count', fontsize=13, fontweight="bold")
    
    # Adjust tick parameters
    ax.tick_params(axis='both', width=0.8, length=2, labelsize=12, colors='black')
    
    # Set axis limits
    plt.xlim(0, 4200)
    plt.ylim(0, None)
    
    # Legend with updated styling
    plt.legend(frameon=True, 
            fontsize=12,
            bbox_to_anchor=(0.5, -0.15),  # Places legend below the plot
            loc='upper center',  # Centers the legend
            ncol=2)  # Optional: arranges items in 2 columns

    # Layout
    # plt.tight_layout()

    # Save the plot in both PNG and SVG formats with xlim
    plot_path_png = os.path.join(os.path.dirname(csv_path), 
                                f'growth_comparison_{drug_name}_top{top_n}.png')
    plot_path_svg = os.path.join(os.path.dirname(csv_path), 
                                f'growth_comparison_{drug_name}_top{top_n}.svg')
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight', transparent=True)
    # plt.savefig(plot_path_svg, bbox_inches='tight', transparent=True)
    logging.info(f"Top N plot saved to {plot_path_png}")
    plt.close()
    return sim_stats, x_exp, y_exp

def process_last_gen(experiment_name, experimental_data_path, drug_name):
    """
    Process the last generation of a given experiment, similar to process_top_10 but for all individuals
    in the final generation.
    """
    # Get the experiment folder
    experiment_folder = os.path.join("experiments", experiment_name)
    
    # For the best fittings run, the WT reference was SIM_CTRL_CMA-1110-1637-5p.csv
    sim_wt_df = pd.read_csv(f'{experimental_data_path}SIM_CTRL_CMA-1110-1637-5p.csv')

    # get the strategy from the experiment name
    strategy = get_strategy_from_filename(experiment_name)

    # Find the summary folder
    summary_folder = f'results/{strategy}_summaries/'
    
    # Read the summary file (both GA and CMA store their data in similar CSV format)
    summary_file = os.path.join(summary_folder, f'final_summary_{experiment_name}.csv')
    summary_df = pd.read_csv(summary_file)
    
    # Get the last generation
    last_gen = summary_df['iteration'].max()
    last_gen_df = summary_df[summary_df['iteration'] == last_gen]

    # Initialize a list to store all simulation growth dataframes
    all_sim_growths = []
    
    # Process each individual in the last generation
    for _, row in last_gen_df.iterrows():
        if "iteration" in row.index:
            instance_folder = f"instance_{int(row['iteration'])}_{int(row['individual'])}_{int(row['replicate'])}"
        else:
            instance_folder = f"instance_{int(row['individual'])}_{int(row['replicate'])}"
        
        # Check for both compression types
        full_path_gzip = os.path.join(experiment_folder, instance_folder, 'pcdl_total_info_sim.csv.gz')
        full_path_bz2 = os.path.join(experiment_folder, instance_folder, 'pcdl_total_info_sim.csv.bz2')
        
        if os.path.exists(full_path_gzip):
            full_path = full_path_gzip
        elif os.path.exists(full_path_bz2):
            full_path = full_path_bz2
        else:
            continue  # Skip if file doesn't exist

        try:
            try:
                full_sim_df = pd.read_csv(full_path, compression='gzip')
            except Exception:
                full_sim_df = pd.read_csv(full_path, compression='bz2')
                
            if drug_name == "WT":
                x_sim, y_sim = normalize_sim_curve_control(full_sim_df)
            else:
                wt_min_sim, wt_max_sim, initial_cells = get_min_max_wt_sim_curve(sim_wt_df)
                simgrowth_reduced_df = pd.read_csv(os.path.join(experiment_folder, instance_folder, "simulation_growth.csv"))
                x_sim, y_sim = normalize_simulation_growth_curve(simgrowth_reduced_df, initial_cells, wt_min_sim, wt_max_sim)  

            sim_growth = pd.DataFrame(columns=["time", "alive"])
            sim_growth["time"] = x_sim
            sim_growth["alive"] = y_sim
            sim_growth['instance'] = instance_folder
            all_sim_growths.append(sim_growth)
            
        except Exception as e:
            print(f"Error processing {instance_folder}: {str(e)}")
            continue

    # Combine all growth curves
    combined_sim_growth = pd.concat(all_sim_growths, ignore_index=True)
    
    # Calculate statistics
    sim_stats = combined_sim_growth.groupby('time').agg({'alive': ['mean', 'std']}).reset_index()
    sim_stats.columns = ['time', 'mean_alive', 'std_alive']
    
    # Process experimental data
    exp_wt_df = pd.read_csv(f'{experimental_data_path}/CTRL.csv')
    exp_drug_df, y_exp, y_exp_std = load_drug_data(experimental_data_path, drug_name)
    
    wt_min_exp, wt_max_exp = get_min_max_wt_exp_curve(exp_wt_df)
    x_exp, y_exp = normalize_exp_curve(exp_drug_df, wt_min_exp, wt_max_exp)

    # Calculate correlation
    y_sim = sim_stats["mean_alive"][:len(y_exp)]
    pearson_corr, p_value = pearsonr(y_sim, y_exp)

    # Save correlation results
    correlation_results_path = os.path.join(summary_folder, f'pearson_correlation_{drug_name}_lastgen.txt')
    with open(correlation_results_path, 'w') as f:
        f.write(f'Pearson correlation: {pearson_corr}\n')
        f.write(f'p-value: {p_value}\n')

    # Create plot
    plt.figure(figsize=(4, 4), dpi=300)
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Liberation Sans', 
                                      'FreeSans', 'Arial', 'sans-serif']
    
    # Plot experimental data
    if "AKT" in drug_name or True:  # Using same plotting style for all drugs
        y_exp_std = y_exp_std[:len(y_exp)]
        plt.plot(x_exp[::3], y_exp[::3],  
                label='Experimental',
                color='#808080',
                marker='o',  
                markersize=4,
                linewidth=1.0,
                zorder=3)

        plt.fill_between(x_exp[::3], 
                        (y_exp - y_exp_std)[::3],
                        (y_exp + y_exp_std)[::3],
                        color='#808080',
                        alpha=0.9,
                        zorder=2)

    # Plot simulation data
    plt.plot(sim_stats['time'], 
            sim_stats['mean_alive'],
            label='Simulation (mean)', 
            color='#4B4BFF',
            linewidth=1.0,
            alpha=1.0,
            zorder=2)
    
    plt.fill_between(sim_stats['time'], 
                    sim_stats['mean_alive'] - sim_stats['std_alive'],
                    sim_stats['mean_alive'] + sim_stats['std_alive'],
                    alpha=0.3,
                    color='#4B4BFF',
                    zorder=1)

    # Add treatment window
    plt.axvspan(1280, 1292, 
                color='#FF6B6B', 
                alpha=1.0,
                zorder=0,
                label='Treatment')

    # Customize plot
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)

    plt.xlabel('Simulation Time (min)', fontsize=13, fontweight="bold")
    plt.ylabel('Relative Cell Count', fontsize=13, fontweight="bold")
    
    ax.tick_params(axis='both', width=0.8, length=2, labelsize=12, colors='black')
    
    plt.xlim(0, 4200)
    plt.ylim(0, None)
    
    plt.legend(frameon=True, 
              fontsize=12,
              bbox_to_anchor=(0.5, -0.15),
              loc='upper center',
              ncol=2)

    # Save plots
    plot_path_png = os.path.join(summary_folder, f"final_summary_{experiment_name}", f'growth_comparison_{drug_name}_lastgen.png')
    plot_path_svg = os.path.join(summary_folder, f"final_summary_{experiment_name}", f'growth_comparison_{drug_name}_lastgen.svg')
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight', transparent=True)
    # plt.savefig(plot_path_svg, bbox_inches='tight', transparent=True)
    logging.info(f"Last generation plot saved to {plot_path_png}")
    
    plt.close()
    return sim_stats, x_exp, y_exp

def combine_and_plot_experiments(experiment_info, output_path):
    # Set up the figure with high-quality settings
    plt.figure(figsize=(4, 4), dpi=300)
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    
    # Use sans-serif font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Liberation Sans', 
                                      'FreeSans', 'Arial', 'sans-serif']
    
    # Get current axis
    ax = plt.gca()
    
    # Iterate through each experiment and plot
    for idx, exp in enumerate(experiment_info):
        sim_stats, x_exp, y_exp = exp["data"]
        y_exp_sd = np.std(y_exp)
        
        # Get colors for this experiment type - use drug_name for consistent color mapping
        colors = get_experiment_colors(exp["drug_name"])

        # Exclude the last point for AKT experimental data
        if "AKT" in exp["drug_name"] or "AKT_MEK" in exp["drug_name"]:
            x_exp = x_exp[:-1]
            y_exp = y_exp[:-1]
            # Recalculate y_exp_sd on the shortened y_exp if needed
            y_exp_sd = np.std(y_exp)

        # Plot experimental data with reduced alpha
        plt.plot(x_exp[::3], y_exp[::3], 
                label=f'{exp["drug_name"]} (exp)', 
                color=colors['exp'],
                marker='o',
                markersize=4,
                linewidth=1.0,
                alpha=0.6,  # Reduced alpha for experimental lines
                zorder=1)
        
        # Plot simulation data
        plt.plot(sim_stats['time'], sim_stats['mean_alive'],
                label=f'{exp["short_name"]} (sim)', 
                color=colors['sim'],
                linewidth=1.5,
                alpha=0.9,
                zorder=3)
        
        # Add confidence intervals with reduced alpha
        plt.fill_between(sim_stats['time'], 
                        sim_stats['mean_alive'] - sim_stats['std_alive'],
                        sim_stats['mean_alive'] + sim_stats['std_alive'],
                        alpha=0.15,  # Reduced alpha for confidence intervals
                        color=colors['sim'],
                        zorder=2)

    # Add treatment period indicator
    plt.axvspan(1280, 1292, 
                color='#FF6B6B', 
                alpha=0.3,
                zorder=0,
                label='Treatment')

    # Customize the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)
    
    # Set axis labels
    plt.xlabel('Simulation Time (min)', 
              fontsize=13, 
              fontweight="bold")
    plt.ylabel('Relative Cell Count', 
              fontsize=13, 
              fontweight="bold")
    
    # Customize ticks
    ax.tick_params(axis='both', width=0.8, length=2, labelsize=12, colors='black')
    
    # Set axis limits
    plt.xlim(0, 4200)
    plt.ylim(0, None)
    
    # Legend with updated styling
    plt.legend(frameon=True, 
              fontsize=12,
              bbox_to_anchor=(0.5, -0.15),  # Places legend below the plot
              loc='upper center',  # Centers the legend
              ncol=2)  # Optional: arranges items in 2 columns
    
    # Save plots
    plt.savefig(f'{output_path}_combined.png', 
                dpi=300, 
                bbox_inches='tight',
                transparent=True)
    
    plt.close()
    logging.info(f"Combined plot saved to {output_path}")

def side_by_side_experiment_plots(experiment_info, output_path):
    """
    Create side-by-side plots comparing experimental and simulation growth curves.
    """
    # Set up the figure with high-quality settings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    
    # Use sans-serif font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Liberation Sans', 
                                      'FreeSans', 'Arial', 'sans-serif']
    
    # Plot experimental data on left subplot
    for idx, exp in enumerate(experiment_info):
        sim_stats, x_exp, y_exp = exp["data"]
        y_exp_sd = np.std(y_exp)
        
        # Get colors for this experiment type
        colors = get_experiment_colors(exp["drug_name"])

        # Exclude the last point for AKT experimental data
        if "AKT" in exp["drug_name"] or "AKT_MEK" in exp["drug_name"]:
            x_exp = x_exp[:-1]
            y_exp = y_exp[:-1]
            # Recalculate y_exp_sd on the shortened y_exp if needed
            y_exp_sd = np.std(y_exp)

        # Experimental data (left subplot) - add alpha
        ax1.plot(x_exp[::3], y_exp[::3], 
                label=f'{exp["drug_name"]}', 
                color=colors['exp'],
                marker='o',
                markersize=4,
                linewidth=1.0,
                alpha=0.6,  # Added alpha
                zorder=1)
        
        # Simulation data (right subplot)
        ax2.plot(sim_stats['time'], sim_stats['mean_alive'],
                label=f'{exp["short_name"]}', 
                color=colors['sim'],
                linewidth=1.5,
                alpha=0.9,
                zorder=3)
        ax2.fill_between(sim_stats['time'], 
                        sim_stats['mean_alive'] - sim_stats['std_alive'],
                        sim_stats['mean_alive'] + sim_stats['std_alive'],
                        alpha=0.15,  # Reduced alpha
                        color=colors['sim'],
                        zorder=2)

    # Add treatment period indicator to both plots
    ax1.axvspan(1280, 1292, color='#FF6B6B', alpha=1.0, zorder=0, label='Treatment')
    ax2.axvspan(1280, 1292, color='#FF6B6B', alpha=1.0, zorder=0, label='Treatment')

    # Customize both subplots
    for ax in [ax1, ax2]:
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.7)
        ax.spines['bottom'].set_linewidth(0.7)
        
        # Set axis limits
        ax.set_xlim(0, 4200)
        ax.set_ylim(0, None)
        
        # Customize ticks
        ax.tick_params(axis='both', width=0.8, length=2, labelsize=12, colors='black')

    # Set titles and labels
    ax1.set_title('Experimental Data', fontsize=13, fontweight="bold", pad=10)
    ax2.set_title('Simulation Data', fontsize=13, fontweight="bold", pad=10)
    
    # Set axis labels
    ax1.set_xlabel('Time (min)', fontsize=13, fontweight="bold")
    ax2.set_xlabel('Time (min)', fontsize=13, fontweight="bold")
    ax1.set_ylabel('Relative Cell Count', fontsize=13, fontweight="bold")
    ax2.set_ylabel('Relative Cell Count', fontsize=13, fontweight="bold")

    # Add legends
    ax1.legend(frameon=True, 
              fontsize=12,
              bbox_to_anchor=(0.5, -0.15),
              loc='upper center',
              ncol=2)
    ax2.legend(frameon=True, 
              fontsize=12,
              bbox_to_anchor=(0.5, -0.15),
              loc='upper center',
              ncol=2)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{output_path}_side_by_side.png', 
                dpi=300, 
                bbox_inches='tight',
                transparent=True)
    
    plt.close()
    logging.info(f"Side-by-side plot saved to {output_path}")


# single drug experiment fittings
best_control_experiment_name = "CTRL_CMA-1110-1637-5p"
# best_pi3k_experiment_name = "PI3Ki_CMA-2502-0113-18p_delayed_transient_rmse_postdrug_50gen" # "PI3Ki_CMA-1410-1014-12p_rmse_final_50gen"
# best_mek_experiment_name = "MEKi_CMA-2502-0113-18p_delayed_transient_rmse_postdrug_50gen" # "MEKi_CMA-1410-1026-12p_rmse_final_50gen"
# best_mek_experiment_name = "MEKi_CMA-1410-1026-12p_rmse_final_50gen"
# best_akt_experiment_name = "AKTi_CMA-1002-0147-8p_linear_mapping" # "AKTi_CMA-1710-0934-12p_rmse_final_50gen"

# OK FINAL EXPERIMENTS
best_pi3k_experiment_name   = "PI3Ki_GA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen" # "PI3Ki_CMA-1410-1014-12p_rmse_final_50gen"
best_mek_experiment_name    = "MEKi_GA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen" # "MEKi_CMA-1410-1026-12p_rmse_final_50gen"
best_akt_experiment_name    = "AKTi_GA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen" # "AKTi_CMA-1710-0934-12p_rmse_final_50gen"

top_n = "1p"


# Define the experiments
control_single_drug_experiment = [
    {
        "name": best_control_experiment_name,
        "top_n": 10,
        "strategy": get_strategy_from_filename(best_control_experiment_name),
        "drug_name": get_drug_from_filename(best_control_experiment_name),
        "short_name": "CTRL_CMA_top10"
    }
]

pi3k_single_drug_experiment = [
    {
        "name": best_control_experiment_name,
        "top_n": 10,
        "strategy": get_strategy_from_filename(best_control_experiment_name),
        "drug_name": get_drug_from_filename(best_control_experiment_name),
        "short_name": "ctrlcma_t10"
    },
    {
        "name": best_pi3k_experiment_name,
        "top_n": 10,
        "strategy": get_strategy_from_filename(best_pi3k_experiment_name),
        "drug_name": get_drug_from_filename(best_pi3k_experiment_name),
        "short_name": "pi3kcma_18p_0704_1815_top10"
    }
]

mek_single_drug_experiment = [
    {
        "name": best_control_experiment_name,
        "top_n": 10,
        "strategy": get_strategy_from_filename(best_control_experiment_name),
        "drug_name": get_drug_from_filename(best_control_experiment_name),
        "short_name": "ctrlcma_t10"
    },
    {
        "name": best_mek_experiment_name,
        "top_n": 10,
        "strategy": get_strategy_from_filename(best_mek_experiment_name),
        "drug_name": get_drug_from_filename(best_mek_experiment_name),
        "short_name": "mekcma_18p_0704_1815_top10"
    }
]

akt_single_drug_experiment = [
    {
        "name": best_control_experiment_name,
        "top_n": 10,
        "strategy": get_strategy_from_filename(best_control_experiment_name),
        "drug_name": get_drug_from_filename(best_control_experiment_name),
        "short_name": "ctrlcma_t10"
    },
    {
        "name": best_akt_experiment_name,
        "top_n": 10,
        "strategy": get_strategy_from_filename(best_akt_experiment_name),
        "drug_name": get_drug_from_filename(best_akt_experiment_name),
        "short_name": "aktcma_18p_0704_1815_top10"
    }
]


# Combine and plot
# Create the folder if it does not exist
output_dir = 'results/exp_sim_comparisons_topN_single_drug'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#########################
# SINGLE DRUG EXPERIMENTS
#########################

experimental_data_path = "/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/data/AGS_data/AGS_growth_data/output/csv/"

################# Control #################

# control_single_drug_experiment_info = []
# # Process each experiment
# for experiment in control_single_drug_experiment:
#     csv_path = f'results/{experiment["strategy"]}_summaries/final_summary_{experiment["name"]}/top_{experiment["top_n"]}.csv'
#     print("processing experiment", experiment)
    
#     # Call process_top_10 for each experiment
#     exp_data = process_top_10(csv_path, experiment["name"], experimental_data_path, experiment["drug_name"], experiment["top_n"], nodrug=True)

#     # Append the experiment info
#     control_single_drug_experiment_info.append({
#         "name": experiment["name"],
#         "top_n": experiment["top_n"],
#         "strategy": experiment["strategy"],
#         "drug_name": experiment["drug_name"],
#         "short_name": experiment["short_name"],
#         "data": exp_data
#     })

# # Call the modified combine_and_plot_experiments function
# output_path = os.path.join(output_dir, f'topN_growth_sim_exp_comparison_{"_AND_".join([exp["short_name"] + "_" + exp["name"] + "_" + str(exp["top_n"]) for exp in control_single_drug_experiment_info])}.png')
# combine_and_plot_experiments(control_single_drug_experiment_info, output_path)


################# PI3K #################

# pi3k_single_drug_experiment_info = []
# # Process each experiment
# for experiment in pi3k_single_drug_experiment:
#     csv_path = f'results/{experiment["strategy"]}_summaries/final_summary_{experiment["name"]}/top_{experiment["top_n"]}.csv'
#     print("processing experiment", experiment)
    
#     # Call process_top_10 for each experiment
#     exp_data = process_top_10(csv_path, experiment["name"], experimental_data_path, experiment["drug_name"], experiment["top_n"])
    
#     # Add last generation processing
#     if "CTRL" not in experiment["name"]:
#         last_gen_data = process_last_gen(experiment["name"], experimental_data_path, experiment["drug_name"])
#     else:
#         last_gen_data = None
    
#     # Append both to experiment info
#     pi3k_single_drug_experiment_info.append({
#         "name": experiment["name"],
#         "top_n": experiment["top_n"],
#         "strategy": experiment["strategy"],
#         "drug_name": experiment["drug_name"],
#         "short_name": experiment["short_name"],
#         "data": exp_data,
#         "last_gen_data": last_gen_data
#     })

# # Call the modified combine_and_plot_experiments function
# # output_path = os.path.join(output_dir, f'topN_growth_sim_exp_comparison_{"_AND_".join([exp["short_name"] + "_" + exp["name"] + "_" + str(exp["top_n"]) for exp in pi3k_single_drug_experiment_info])}.png')
# output_path = os.path.join(output_dir, f'topN_growth_sim_exp_{"_AND_".join([exp["short_name"] + "_" + str(exp["top_n"]) for exp in pi3k_single_drug_experiment_info])}.png')
# combine_and_plot_experiments(pi3k_single_drug_experiment_info, output_path)
# side_by_side_experiment_plots(pi3k_single_drug_experiment_info, output_path)


################# MEK #################

# mek_single_drug_experiment_info = []
# # Process each experiment
# for experiment in mek_single_drug_experiment:
#     csv_path = f'results/{experiment["strategy"]}_summaries/final_summary_{experiment["name"]}/top_{experiment["top_n"]}.csv'
#     print("processing experiment", experiment)
    
#     # Call process_top_10 for each experiment
#     exp_data = process_top_10(csv_path, experiment["name"], experimental_data_path, experiment["drug_name"], experiment["top_n"])

#     # Add last generation processing
#     if "CTRL" not in experiment["name"]:
#         last_gen_data = process_last_gen(experiment["name"], experimental_data_path, experiment["drug_name"])
#     else:
#         last_gen_data = None
    
#     # Append both to experiment info
#     mek_single_drug_experiment_info.append({
#         "name": experiment["name"],
#         "top_n": experiment["top_n"],
#         "strategy": experiment["strategy"],
#         "drug_name": experiment["drug_name"],
#         "short_name": experiment["short_name"],
#         "data": exp_data,
#         "last_gen_data": last_gen_data
#         })
    
# # Call the modified combine_and_plot_experiments function
# # output_path = os.path.join(output_dir, f'topN_growth_sim_exp_comparison_{"_AND_".join([exp["short_name"] + "_" + exp["name"] + "_" + str(exp["top_n"]) for exp in mek_single_drug_experiment_info])}.png')
# output_path = os.path.join(output_dir, f'topN_growth_sim_exp_{"_AND_".join([exp["short_name"] + "_" + str(exp["top_n"]) for exp in mek_single_drug_experiment_info])}.png')
# combine_and_plot_experiments(mek_single_drug_experiment_info, output_path)
# side_by_side_experiment_plots(mek_single_drug_experiment_info, output_path)

# ################# AKT #################

# akt_single_drug_experiment_info = []
# # Process each experiment
# for experiment in akt_single_drug_experiment:
#     csv_path = f'results/{experiment["strategy"]}_summaries/final_summary_{experiment["name"]}/top_{experiment["top_n"]}.csv'
#     print("processing experiment", experiment)
    
#     # Call process_top_10 for each experiment
#     exp_data = process_top_10(csv_path, experiment["name"], experimental_data_path, experiment["drug_name"], experiment["top_n"])

#     # Add last generation processing
#     if "CTRL" not in experiment["name"]:
#         last_gen_data = process_last_gen(experiment["name"], experimental_data_path, experiment["drug_name"])
#     else:
#         last_gen_data = None
    
#     # Append the experiment info
#     akt_single_drug_experiment_info.append({
#         "name": experiment["name"],
#         "top_n": experiment["top_n"],
#         "strategy": experiment["strategy"],
#         "drug_name": experiment["drug_name"],
#         "short_name": experiment["short_name"],
#         "data": exp_data,
#         "last_gen_data": last_gen_data
#     })

# # Call the modified combine_and_plot_experiments function
# # output_path = os.path.join(output_dir, f'topN_growth_sim_exp_comparison_{"_AND_".join([exp["short_name"] + "_" + exp["name"] + "_" + str(exp["top_n"]) for exp in akt_single_drug_experiment_info])}.png')
# output_path = os.path.join(output_dir, f'topN_growth_sim_exp_{"_AND_".join([exp["short_name"] + "_" + str(exp["top_n"]) for exp in akt_single_drug_experiment_info])}.png')
# combine_and_plot_experiments(akt_single_drug_experiment_info, output_path)
# side_by_side_experiment_plots(akt_single_drug_experiment_info, output_path)

########################
# SYNERGY EXPERIMENTS #
########################

best_pi3kmek_combined_experiment_name = "synergy_sweep-pi3k_mek-1104-2212-18p_transient_delayed_uniform_5k_10p"
best_pi3kmek_PI3K_singledrug_experiment_name = "synergy_sweep-pi3k_mek-1104-2212-18p_PI3K_transient_delayed_uniform_5k_singledrug"
best_pi3kmek_MEK_singledrug_experiment_name = "synergy_sweep-pi3k_mek-1104-2212-18p_MEK_transient_delayed_uniform_5k_singledrug"

best_aktmek_combined_experiment_name = "synergy_sweep-akt_mek-1204-1639-18p_transient_delayed_uniform_postdrug_RMSE_5k"
best_aktmek_AKT_singledrug_experiment_name = "synergy_sweep-akt_mek-1104-2212-18p_AKT_transient_delayed_uniform_5k_singledrug"
best_aktmek_MEK_singledrug_experiment_name = "synergy_sweep-akt_mek-1104-2212-18p_MEK_transient_delayed_uniform_5k_singledrug"


# Linear mapping experiments
# best_pi3kmek_combined_experiment_name = "synergy_sweep-pi3k_mek-1102-1909-8p_linear_mapping_uniform_5k"
# best_pi3kmek_PI3K_singledrug_experiment_name = "synergy_sweep-pi3k_mek-1102-1909-8p_PI3K_linear_mapping_singledrug"
# best_pi3kmek_MEK_singledrug_experiment_name = "synergy_sweep-pi3k_mek-1102-1909-8p_MEK_linear_mapping_singledrug"


# best_aktmek_combined_experiment_name = "synergy_sweep-akt_mek-1102-1909-8p_linear_mapping_uniform_5k"
# best_aktmek_AKT_singledrug_experiment_name = "synergy_sweep-akt_mek-1102-1909-8p_AKT_linear_mapping_singledrug"
# best_aktmek_MEK_singledrug_experiment_name = "synergy_sweep-akt_mek-1102-1909-8p_MEK_linear_mapping_singledrug"


pi3k_mek_synergy_experiments = [
    {
        "name": best_control_experiment_name,
        "top_n": 10,
        "strategy": get_strategy_from_filename(best_control_experiment_name),
        "drug_name": get_drug_from_filename(best_control_experiment_name),
        "short_name": "CTRL_CMA_top10"
    },
    {
        "name": best_pi3kmek_PI3K_singledrug_experiment_name,
        "top_n": 10,
        "strategy": get_strategy_from_filename(best_pi3kmek_PI3K_singledrug_experiment_name),
        "drug_name": get_drug_from_filename(best_pi3kmek_PI3K_singledrug_experiment_name),
        "short_name": "PI3K_sweep_top10"
    },
    {
        "name": best_pi3kmek_MEK_singledrug_experiment_name,
        "top_n": 10,
        "strategy": get_strategy_from_filename(best_pi3kmek_MEK_singledrug_experiment_name),
        "drug_name": get_drug_from_filename(best_pi3kmek_MEK_singledrug_experiment_name),
        "short_name": "MEK_sweep_top10"
    },
    {
        "name": best_pi3kmek_combined_experiment_name,
        "top_n": 10,
        "strategy": get_strategy_from_filename(best_pi3kmek_combined_experiment_name),
        "drug_name": get_drug_from_filename(best_pi3kmek_combined_experiment_name),
        "short_name": "PI3KMEK_top10_final_rmse"
    }
]

akt_mek_synergy_experiments = [
    {
        "name": best_control_experiment_name,
        "top_n": 10,
        "strategy": get_strategy_from_filename(best_control_experiment_name),
        "drug_name": get_drug_from_filename(best_control_experiment_name),
        "short_name": "CTRL_CMA_top10"
    },
    {
        "name": best_aktmek_AKT_singledrug_experiment_name,
        "top_n": 10,
        "strategy": get_strategy_from_filename(best_aktmek_AKT_singledrug_experiment_name),
        "drug_name": get_drug_from_filename(best_aktmek_AKT_singledrug_experiment_name),
        "short_name": "AKT_sweep_top10"
    },
    {
        "name": best_aktmek_MEK_singledrug_experiment_name,
        "top_n": 10,
        "strategy": get_strategy_from_filename(best_aktmek_MEK_singledrug_experiment_name),
        "drug_name": get_drug_from_filename(best_aktmek_MEK_singledrug_experiment_name),
        "short_name": "MEK_sweep_top10"
    },
    {
        "name": best_aktmek_combined_experiment_name,
        "top_n": 10,
        "strategy": get_strategy_from_filename(best_aktmek_combined_experiment_name),
        "drug_name": get_drug_from_filename(best_aktmek_combined_experiment_name),
        "short_name": "AKTMEK_top10_final_rmse"
    }
]

# Combine and plot
# Create the folder if it does not exist
output_dir = 'results/exp_sim_comparisons_topN_synergy_experiments'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


####################
# PI3K MEK SYNERGY #
####################

# experiment_info_pi3k_mek_synergy = []
# # Process each experiment
# for experiment in pi3k_mek_synergy_experiments:
#     csv_path = f'results/{experiment["strategy"]}_summaries/final_summary_{experiment["name"]}/top_{experiment["top_n"]}.csv'
#     print("processing experiment", experiment)
#     experimental_data_path = "/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/data/AGS_data/AGS_growth_data/output/csv/"

    
#     exp_data = process_top_10(csv_path, experiment["name"], experimental_data_path, experiment["drug_name"], experiment["top_n"])
    
#     # Append the experiment info
#     experiment_info_pi3k_mek_synergy.append({
#         "name": experiment["name"],
#         "top_n": experiment["top_n"],
#         "strategy": experiment["strategy"],
#         "drug_name": experiment["drug_name"],
#         "short_name": experiment["short_name"],
#         "data": exp_data
#     })
# # # Call the modified combine_and_plot_experiments function
# output_path = os.path.join(output_dir, f'combined_growth_comparison_{"_AND_".join([exp["short_name"] + "_top_" + str(exp["top_n"]) for exp in experiment_info_pi3k_mek_synergy])}.png')
# combine_and_plot_experiments(experiment_info_pi3k_mek_synergy, output_path)
# side_by_side_experiment_plots(experiment_info_pi3k_mek_synergy, output_path)


####################
# AKT MEK SYNERGY #
####################

experiment_info_akt_mek_synergy = []
# Process each experiment
for experiment in akt_mek_synergy_experiments:
    csv_path = f'results/{experiment["strategy"]}_summaries/final_summary_{experiment["name"]}/top_{experiment["top_n"]}.csv'
    print("processing experiment", experiment)
    experimental_data_path = "/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/data/AGS_data/AGS_growth_data/output/csv/"
    
    # Call process_top_10 for each experiment
    exp_data = process_top_10(csv_path, experiment["name"], experimental_data_path, experiment["drug_name"], experiment["top_n"])
    
    # Append the experiment info
    experiment_info_akt_mek_synergy.append({
        "name": experiment["name"],
        "top_n": experiment["top_n"],
        "strategy": experiment["strategy"],
        "drug_name": experiment["drug_name"],
        "short_name": experiment["short_name"],
        "data": exp_data
    })

# Call the modified combine_and_plot_experiments function
output_path = os.path.join(output_dir, f'combined_growth_comparison_{"_AND_".join([exp["short_name"] for exp in experiment_info_akt_mek_synergy])}.png')
combine_and_plot_experiments(experiment_info_akt_mek_synergy, output_path)
side_by_side_experiment_plots(experiment_info_akt_mek_synergy, output_path)

