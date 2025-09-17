import pandas as pd
import os, sys
import numpy as np
import hillfit
import matplotlib.pyplot as plt
import neutcurve
from scipy.optimize import curve_fit


# This script loads the PDF-extracted data for a given drug-response curve
drug_name = "AKTi" # PD0 for MEK, AKTi for AKT


# Load the data (there's 3 files for each drug-response curve)
experiment_path = f"./data/AGS_data/AGS_growth_data/drug_response_curves/{drug_name}/points_from_pdf/" 


replicate_csvs = []

for file in os.listdir(experiment_path):
    if file.endswith(".csv") and f"{drug_name}" in file and "rep" in file:
        tmp_df = pd.read_csv(os.path.join(experiment_path, file))
        tmp_df.columns = ["drug_concentration_log", "cell_index"]
        # get linear scale of drug concentration
        tmp_df['drug_concentration'] = np.power(10, tmp_df['drug_concentration_log'])
        # normalize cell index to 0-1
        tmp_df['cell_index'] = (tmp_df['cell_index'] - tmp_df['cell_index'].min()) / (tmp_df['cell_index'].max() - tmp_df['cell_index'].min())
        # get the "drug effect", which is the inverse of the cell index
        tmp_df['drug_effect'] = 1 - tmp_df['cell_index']
        # tmp_df = tmp_df[['drug_concentration', 'cell_index']]
        # add a constant factor to the drug concentration 
        # tmp_df['drug_concentration'] = tmp_df['drug_concentration'] + 10
        # then reverse to log again
        tmp_df['drug_concentration_log'] = np.log10(tmp_df['drug_concentration'])

        
        conversion_factor_PI3K, conversion_factor_AKT, conversion_factor_MEK = 10**6, 10**6, 10**9
        # tmp_df['drug_concentration'] = tmp_df['drug_concentration'] * conversion_factor_PI3K # convert to uM
        replicate_csvs.append(tmp_df)

# Join all 3 dataframes in one, averaging all three columns 
# Combine all dataframes and calculate mean and standard deviation
merged_df = pd.concat(replicate_csvs)
averaged_df = merged_df.groupby(merged_df.index).agg({
    'drug_concentration': ['mean', 'std'],
    'drug_concentration_log': ['mean', 'std'],
    'cell_index': ['mean', 'std'],
    'drug_effect': ['mean', 'std']
})
# Flatten column names
averaged_df.columns = ['_'.join(col).strip() for col in averaged_df.columns.values]

# Rename columns for clarity
averaged_df = averaged_df.rename(columns={
    'drug_concentration_mean': 'drug_concentration',
    'drug_concentration_std': 'drug_concentration_std',
    'drug_concentration_log_mean': 'drug_concentration_log',
    'drug_concentration_log_std': 'drug_concentration_log_std',
    'cell_index_mean': 'cell_index',
    'cell_index_std': 'cell_index_std',
    'drug_effect_mean': 'drug_effect',
    'drug_effect_std': 'drug_effect_std'
})

# Reset index for the final dataframe
averaged_df = averaged_df.reset_index(drop=True)
# save the averaged dataframe to a CSV file
averaged_df.to_csv(os.path.join(experiment_path, f"{drug_name}_DR_averaged_linear_uM.csv"), index=False, header=True)

# Quick plot of the data
# Add vertical line at EC50 and 1/2 EC50
pi3k_EC50 = 0.683
log_pi3k_EC50 = np.log10(pi3k_EC50)

plt.plot(averaged_df['drug_concentration_log'], averaged_df['drug_effect'], 'o')
# also plot the variance for each point
plt.errorbar(averaged_df['drug_concentration_log'], averaged_df['drug_effect'], yerr=averaged_df['drug_effect_std'], fmt='o', capsize=5)
# plt.axvline(x=pi3k_EC50, color='r', linestyle='--', label='EC50')
# plt.axvline(x=pi3k_EC50 / 2, color='g', linestyle='--', label='1/2 EC50')
plt.xlabel('Drug concentration (log10 uM)')
plt.ylabel('Cell index')
plt.show()
plt.savefig(os.path.join(experiment_path, f"{drug_name}_hill_fit_mtpltlib.png"), dpi=300)


def hill_equation(x, ec50, hill_coef, bottom, top):
    return bottom + (top - bottom) / (1 + (10**ec50 / 10**x)**hill_coef)

# Hill function fitting
x_data = averaged_df['drug_concentration_log'].values
y_data = averaged_df['drug_effect'].values

# Remove any infinite or NaN values
valid_indices = np.isfinite(x_data) & np.isfinite(y_data)
x_data = x_data[valid_indices]
y_data = y_data[valid_indices]

# Print some diagnostic information
# print("x_data range:", x_data.min(), x_data.max())
# print("y_data range:", y_data.min(), y_data.max())

try:
    # Set bounds for the parameters
    # Order: [EC50, Hill coefficient, bottom, top]
    lower_bounds = [x_data.min(), 0.1, 0, 0.5]
    upper_bounds = [x_data.max(), 5, 0.5, 1.1]

    # Set initial guess
    p0 = [np.mean([x_data.min(), x_data.max()]), 1.0, np.min(y_data), np.max(y_data)]

    # Perform curve fitting
    popt, _ = curve_fit(hill_equation, x_data, y_data, p0=p0, bounds=(lower_bounds, upper_bounds))

    # Extract parameters
    EC50_log, Hill_coefficient, min_response, max_response = popt
    EC50 = 10**EC50_log
    EC25 = EC50 / 2
    EC25_log = np.log10(EC25)

    print(f"Fitted parameters: log(EC50)= {EC50_log}, EC50={EC50}, Hill coefficient={Hill_coefficient}, min response={min_response}, max response={max_response}")

    # Generate points for the fitted curve
    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = hill_equation(x_fit, *popt)
    # Plot the data points and fitted curve (log scale)
    plt.figure(figsize=(10, 6))
    # plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_fit, y_fit, 'r-', label='Fitted Hill curve')
    plt.axvline(x=EC50_log, color='r', linestyle='--', label='EC50')
    plt.axvline(x=EC25_log, color='g', linestyle='--', label='1/2 EC50')
    plt.xlabel('Log Drug Concentration')
    plt.ylabel('Drug Effect')
    plt.title(f'Drug Response Curve for {drug_name} (Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(experiment_path, f"{drug_name}_DR_hill_curve_log.png"), dpi=300)
    plt.close()


    # Plot the data points and fitted curve (linear scale)
    plt.figure(figsize=(10, 6))
    x_data_linear = 10**x_data
    x_data_linear_uM = x_data_linear * 10E-6
    x_fit_linear_uM = (10**x_fit) * 10E-6
    # plt.scatter(x_data_linear_uM, y_data, label='Data')
    plt.plot(x_fit_linear_uM, y_fit, 'r-', label='Fitted Hill curve')
    plt.axvline(x=EC50 * 10E-6, color='r', linestyle='--', label='EC50')
    plt.axvline(x=EC25 * 10E-6, color='g', linestyle='--', label='1/2 EC50')
    plt.xlabel('Drug Concentration (uM)')
    plt.ylabel('Drug Effect')
    plt.title(f'Drug Response Curve for {drug_name} (Linear Scale)')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')  # Use log scale for x-axis to spread out the points
    # save the plot
    plt.savefig(os.path.join(experiment_path, f"{drug_name}_DR_hill_curve_linear.png"), dpi=300)
    plt.close()

 
    # Save Hill curve parameters
    hill_params_df = pd.DataFrame({
        'Parameter': ['EC50', 'Hill_coefficient', 'min_response', 'max_response'],
        'Value': [EC50, Hill_coefficient, min_response, max_response]
    })
    hill_params_df.to_csv(os.path.join(experiment_path, f"{drug_name}_DR_hill_params.csv"), index=False)

    # Save the fitted curve data
    fit_df = pd.DataFrame({'drug_concentration_log': x_fit, 'drug_effect': y_fit})
    fit_df['drug_concentration'] = 10**fit_df['drug_concentration_log']  # Add linear concentration
    fit_df.to_csv(os.path.join(experiment_path, f"{drug_name}_DR_fitted_curve.csv"), index=False)


    # print value for EC50 and 1/2 EC50 and Hill coefficient
    print(f"EC50: {EC50} (M), {EC50 * 1E6} (uM)")
    print(f"1/2 EC50: {EC25} (M), {EC25 * 1E6} (uM)")
    print(f"Hill coefficient: {Hill_coefficient}")

except Exception as e:
    print(f"Error during Hill fitting: {e}")
    print("Try adjusting the initial guesses or bounds for the Hill function parameters.")
    
    # Print additional diagnostic information
    print("Number of valid data points:", len(x_data))
    print("x_data:", x_data)
    print("y_data:", y_data)