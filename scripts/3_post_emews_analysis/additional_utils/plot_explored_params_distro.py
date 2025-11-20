# this script plots, for the full summary of a given EMEWS run experiment, the distribution of all explored parameters

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



experiment_name = "MEKi_CMA-1603-2034-19p_delayed_transient_rmse_postdrug_50gen"
if "CMA" in experiment_name:
    summaries_path = f"results/CMA_summaries"
if "GA" in experiment_name:
    summaries_path = f"results/GA_summaries"


full_summary_csv_path = f"{summaries_path}/final_summary_{experiment_name}.csv"
results_path = f"{summaries_path}/final_summary_{experiment_name}/parameters_distributions/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

df = pd.read_csv(full_summary_csv_path)

print(df.columns)

# plot the distribution of all explored parameters in a grid of plots
num_cols = 3  # Number of columns in the grid
num_rows = (len(df.columns) + num_cols - 1) // num_cols  # Calculate number of rows needed

plt.figure(figsize=(25, 4 * num_rows))
for i, col in enumerate(df.columns):
    plt.subplot(num_rows, num_cols, i + 1)  # Create a subplot for each parameter
    sns.histplot(df, x=col, bins=20, alpha=0.5)  # Use alpha for transparency
    plt.title(col)
    plt.xlabel('Parameter Value')
    plt.ylabel('Frequency')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig(f"{results_path}/all_parameters_distribution.png", dpi=300, bbox_inches='tight')
print(f"Saved all parameters distribution to {results_path}/all_parameters_distribution.png")
plt.close()



