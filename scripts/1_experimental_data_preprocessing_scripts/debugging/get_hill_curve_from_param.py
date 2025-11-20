# Script that, given the Km (EC50) and Hill coefficient of an experimental drug-response curve, 
# plots a Hill curve with different values of Hill coefficient

import numpy as np
import matplotlib.pyplot as plt
import os

# Use scipy to fit the Hill curve
from scipy.optimize import curve_fit

# Define the Hill function
def hill_function(x, top, bottom, midpoint, hill_coefficient):
    return top + (bottom - top) / (1 + (x / midpoint)**hill_coefficient)

# This script loads the PDF-extracted data for a given drug-response curve
drug_name = "PD0" # PD0 for MEK, AKTi for AKT


# Load the data (there's 3 files for each drug-response curve)
experiment_path = f"./data/AGS_data/AGS_growth_data/drug_response_curves/{drug_name}/" 




# then input parameters
EC50 = 0.031 # uM
Hill_coefficient = 0.58

# Plot the Hill curve
x = np.linspace(0, 0.1, 100)
y = hill_function(x, 1, 0, EC50, Hill_coefficient)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f'Hill coefficient = {Hill_coefficient}')

# Calculate y values for EC50 and EC50/2
y_EC50 = hill_function(EC50, 1, 0, EC50, Hill_coefficient)
y_EC50_half = hill_function(EC50/2, 1, 0, EC50, Hill_coefficient)

# Add vertical lines at EC50 and 1/2 EC50
plt.axvline(x=EC50, color='r', linestyle='--', label=f'EC50 = {EC50} uM')
plt.axvline(x=EC50/2, color='g', linestyle='--', label=f'1/2 EC50 = {EC50/2} uM')

# Add horizontal lines for y values at EC50 and 1/2 EC50
plt.axhline(y=y_EC50, color='r', linestyle=':', label=f'y at EC50 = {y_EC50:.2f}')
plt.axhline(y=y_EC50_half, color='g', linestyle=':', label=f'y at 1/2 EC50 = {y_EC50_half:.2f}')

plt.title(f'Hill Curve for {drug_name} with Hill coefficient = {Hill_coefficient}')
plt.xlabel('Drug Concentration (uM)')
plt.ylabel('Drug Effect')
plt.legend()

# Save the plot
save_path = os.path.join(experiment_path, f'Hill_curves_{drug_name}')
if not os.path.exists(save_path):
    os.makedirs(save_path)

plt.savefig(os.path.join(experiment_path, f'{drug_name}_Hill_curve_H{Hill_coefficient}.png'))
plt.show()