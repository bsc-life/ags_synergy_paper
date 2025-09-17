import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# This script plots a dose response curve for a given dose-response experiment for single-drug experiments with AGS 
# takes the top 100 CSV, plots the log(M) of drug conceentration vs the last cell index point 

def plot_dose_response_curve(top_100_csv):
    # read the top 100 CSV
    top_100_df = pd.read_csv(top_100_csv)

    # filter only relevant columns: user_parameters.drug_X_pulse_concentration and FINAL_CELL_INDEX_VALUE (which is always the last column)
    top_100_df = top_100_df[['user_parameters.drug_X_pulse_concentration', top_100_df.columns[-1]]]
    
    # the pulse concentration is in mM - convert to M 
    top_100_df['user_parameters.drug_X_pulse_concentration'] = top_100_df['user_parameters.drug_X_pulse_concentration'] * 1e-3
    # and then on to log scale 
    top_100_df['log(M)'] = np.log10(top_100_df['user_parameters.drug_X_pulse_concentration'])


    # print(top_100_df.head())

    # plot the log(M) of drug conceentration vs the last cell index point 
    output_dir = os.path.join('results', 'dose_response_curves')
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(top_100_df['log(M)'], top_100_df[top_100_df.columns[-2]])
    plt.xlabel('log(M)')
    plt.ylabel('Alive cells')
    
    plt.savefig(os.path.join(output_dir, "dose_response_curve_log_" + top_100_csv.split('/')[-1] + ".png"), dpi=300, transparent=False)
    print(f"Dose response curve plotted for experiment: {top_100_csv}\n")


def hill_function(x, bottom, top, ec50, hill):
    """Hill function for dose-response curves in normal scale
    Parameters:
    -----------
    x : array-like
        Drug concentration (in M)
    bottom : float
        Minimum response
    top : float
        Maximum response
    ec50 : float
        Half maximal effective concentration (in M)
    hill : float
        Hill coefficient (steepness)
    
    Returns:
    --------
    response : array-like
        Response values
    """
    return bottom + (top - bottom) / (1 + (ec50/x)**hill)

def plot_dose_response_curve_with_fit(top_100_csv):
    # read and process data as before
    top_100_df = pd.read_csv(top_100_csv)
    top_100_df = top_100_df[['user_parameters.drug_X_pulse_concentration', top_100_df.columns[-1]]]
    top_100_df['user_parameters.drug_X_pulse_concentration'] = top_100_df['user_parameters.drug_X_pulse_concentration'] * 1e-3

    cell_count_col = top_100_df.columns[-1]
    zero_conc_df = top_100_df[top_100_df['user_parameters.drug_X_pulse_concentration'] == 0]

    if not zero_conc_df.empty:
        # If there are multiple entries for zero concentration, take the mean as the baseline
        min_effect = zero_conc_df[cell_count_col].mean()
    else:
        # If no zero concentration point, use the max cell count as a robust baseline
        print("Warning: Zero concentration point not found. Using maximum cell count as baseline for normalization.")
        min_effect = top_100_df[cell_count_col].max()

    print("Initial number of cells: ", min_effect)

    # Use a with block to handle potential divide-by-zero warnings for log10
    with np.errstate(divide='ignore'):
        top_100_df['log(M)'] = np.log10(top_100_df['user_parameters.drug_X_pulse_concentration'])

    # What we need for the curve fit is the normalized alive cells
    top_100_df['effect'] = np.abs(top_100_df[cell_count_col]) / min_effect

    # subset the top_100_df to only include the log(M) and effect columns
    top_100_df = top_100_df[['log(M)', 'effect']]

    # Remove rows where log(M) is not finite (i.e., from concentration = 0)
    top_100_df = top_100_df[np.isfinite(top_100_df['log(M)'])]

    # sort the dataframe by the log(M) column
    top_100_df = top_100_df.sort_values(by='log(M)')

    print(top_100_df.head())

    
    # Fit the curve
    x_data = top_100_df['log(M)']
    y_data = top_100_df['effect']


    # build a new dataframe with the x_data and y_data
    df = pd.DataFrame({'x': x_data, 'y': y_data})
    
    # Check if there are enough data points for a meaningful fit
    if len(x_data) < 4:
        print("Could not fit curve - not enough data points. Plotting raw data.")
        plt.figure(figsize=(5, 5), dpi=300)
        plt.scatter(x_data, y_data, alpha=0.5, label='Data')
        plt.xlabel('log(M)')
        plt.ylabel('Effect')
        plt.legend()
        output_dir = os.path.join('results', 'dose_response_curves')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "dose_response_curve_log_fit_" + top_100_csv.split('/')[-1] + ".png"), 
                    dpi=300, transparent=False, bbox_inches='tight')
        plt.close()
        return None

    # store the dataframe in a csv
    
    try:
        p0 = [
            min(y_data),        # bottom
            max(y_data),        # top
            np.median(x_data),  # log_ec50 (already in log scale)
            1.0                 # hill
        ]
        
        # Add bounds to prevent unrealistic parameters
        bounds = (
            [0, 0.5, -10, 1.0],          # lower bounds
            [1.0, 1.5, -1, 10000]        # upper bounds
        )
        
        # Fit curve with bounds
        popt, _ = curve_fit(hill_function, x_data, y_data, p0=p0)
        
        # Generate smooth curve
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = hill_function(x_fit, *popt)
        

        # Create plot
        plt.figure(figsize=(5, 5), dpi=300)
        plt.scatter(x_data, y_data, alpha=0.5, label='Data')
        plt.plot(x_fit, y_fit, 'r-', label='Fitted curve')
        plt.axvline(x=popt[2], color='k', linestyle='--', label='EC50 simulation')
        plt.xlabel('log(M)')
        plt.ylabel('Effect')
        plt.legend()
        
        # Add EC50 value to plot
        ec50_text = f'EC50 = {10**popt[2]:.2e} M'
        plt.text(0.05, 0.95, ec50_text, transform=plt.gca().transAxes, 
                verticalalignment='top')
        
        output_dir = os.path.join('results', 'dose_response_curves')   
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "dose_response_curve_log_fit_" + top_100_csv.split('/')[-1] + ".png"), 
                    dpi=300, transparent=False, bbox_inches='tight')
        print(f"Dose response curve plotted for experiment: {top_100_csv}")
        print(f"Fitted parameters: Bottom={popt[0]:.2f}, Top={popt[1]:.2f}, EC50={10**popt[2]:.2e}, Hill={popt[3]:.2f}")
        print("\n")
        
        return popt
            
    except RuntimeError:
        print("Could not fit curve - Doing a simple plot of the data")
        return None
  
def plot_theoretical_hill_curve(popt, x_range=None):
    """Plot theoretical Hill curve using fitted parameters
    Parameters:
    -----------
    popt : array-like
        Fitted parameters [bottom, top, log_ec50, hill]
    x_range : tuple, optional
        (min_x, max_x) for plotting range in log M
    """
    # If no range specified, use a reasonable range around EC50
    if x_range is None:
        log_ec50 = popt[2]
        x_range = (log_ec50 - 3, log_ec50 + 3)  # 3 log units each way
    
    # Generate smooth curve
    x_theoretical = np.linspace(x_range[0], x_range[1], 1000)
    y_theoretical = hill_function(x_theoretical, *popt)
    
    # Create plot
    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(x_theoretical, y_theoretical, 'b-', label='Hill function')
    
    # Add EC50 line
    plt.axvline(x=popt[2], color='gray', linestyle='--', alpha=0.5, label='EC50')
    plt.axhline(y=(popt[0] + popt[1])/2, color='gray', linestyle='--', alpha=0.5)
    
    # Labels and title
    plt.xlabel('log(M)')
    plt.ylabel('Response')
    plt.title(f'Theoretical Hill Curve\nEC50={10**popt[2]:.2e}M, Hill={popt[3]:.2f}')
    plt.legend()
    
    # Save plot
    output_dir = os.path.join('results', 'dose_response_curves')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "theoretical_hill_curve.png"), 
                dpi=300, transparent=False, bbox_inches='tight')
    plt.close()



def plot_drug_death_relation(top_100_csv):
    # read the top 100 CSV
    top_100_df = pd.read_csv(top_100_csv)

    # filter only relevant columns: user_parameters.drug_X_pulse_concentration and FINAL_CELL_INDEX_VALUE
    top_100_df = top_100_df[['user_parameters.drug_X_pulse_concentration', top_100_df.columns[-1]]]
    # the pulse concentration is in mM - convert to M 
    top_100_df['drug (M)'] = top_100_df['user_parameters.drug_X_pulse_concentration'] * 1e-3
    
    # plot the drug concentration vs the last cell index point 
    output_dir = os.path.join('results', 'dose_response_curves')
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(top_100_df['drug (M)'], top_100_df[top_100_df.columns[-2]])
    plt.xlabel('drug (M)')
    plt.ylabel('Alive cells')
    plt.savefig(os.path.join(output_dir, "dose_response_curve_linear_scale_" + top_100_csv.split('/')[-1] + ".png"), dpi=300, transparent=False)
    print(f"Dose response curve plotted for experiment: {top_100_csv}")

def plot_correlation(top_100_csv):
    # read the top 100 CSV
    top_100_df = pd.read_csv(top_100_csv)

    # filter only relevant columns
    top_100_df = top_100_df[['user_parameters.drug_X_pulse_concentration', top_100_df.columns[-1]]]
    # convert to M
    top_100_df['drug (M)'] = top_100_df['user_parameters.drug_X_pulse_concentration'] * 1e-3
    
    # Calculate correlations
    pearson_corr, pearson_p = stats.pearsonr(top_100_df['drug (M)'], 
                                            top_100_df[top_100_df.columns[-1]])
    spearman_corr, spearman_p = stats.spearmanr(top_100_df['drug (M)'], 
                                               top_100_df[top_100_df.columns[-1]])
    
    # Create plot
    plt.figure(figsize=(5, 5), dpi=300)
    
    # Plot scatter points
    plt.scatter(top_100_df['drug (M)'], top_100_df[top_100_df.columns[-1]], 
               alpha=0.6, label='Data points')
    
    # Add correlation information
    corr_text = (f'Pearson r = {pearson_corr:.3f} (p = {pearson_p:.2e})\n'
                 f'Spearman ρ = {spearman_corr:.3f} (p = {spearman_p:.2e})')
    plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('drug (M)')
    plt.ylabel('Alive cells')
    
    # Optional: add trend line
    z = np.polyfit(top_100_df['drug (M)'], top_100_df[top_100_df.columns[-2]], 1)
    p = np.poly1d(z)
    plt.plot(top_100_df['drug (M)'], p(top_100_df['drug (M)']), 
            "r--", alpha=0.8, label='Trend line')
    
    plt.legend()
    
    # Save plot
    output_dir = os.path.join('results', 'dose_response_curves')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "dose_response_curve_linear_fit_" + top_100_csv.split('/')[-1] + ".png"), 
                dpi=300, transparent=False, bbox_inches='tight')
    
    print(f"Dose response curve plotted for experiment: {top_100_csv}")
    print(f"Pearson correlation: {pearson_corr:.3f} (p = {pearson_p:.2e})")
    print(f"Spearman correlation: {spearman_corr:.3f} (p = {spearman_p:.2e})")
    print("\n")


if __name__ == "__main__":
    sweep_summaries_folder = "results/sweep_summaries/"
    for file in os.listdir(sweep_summaries_folder):
        if "DR" in file and "1606" in file and os.path.isdir(os.path.join(sweep_summaries_folder, file)):
            # print("Plotting dose response curve for experiment: ", file)
            top_100_csv = os.path.join(sweep_summaries_folder, file, "top_100.csv")
            plot_dose_response_curve(top_100_csv)
            hill_params = plot_dose_response_curve_with_fit(top_100_csv)
            if hill_params is not None:
                plot_theoretical_hill_curve(hill_params)
            plot_drug_death_relation(top_100_csv)
            plot_correlation(top_100_csv)

        elif "DR" in file and "1606" in file and file.endswith(".csv"):
            full_path = os.path.join(sweep_summaries_folder, file)
            print("This is the full experiment csv: ", full_path)
            plot_dose_response_curve(full_path)
            hill_params = plot_dose_response_curve_with_fit(full_path)
            if hill_params is not None:
                plot_theoretical_hill_curve(hill_params)
            plot_drug_death_relation(full_path)
            plot_correlation(full_path)