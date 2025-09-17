import pandas as pd
import json
import os
import numpy as np

def obtain_param_distribution(csv_file):
    print(f"Processing {csv_file}")
    # Load the data from the CSV file
    df = pd.read_csv(csv_file)

    # Initialize a dictionary to hold the statistics
    param_stats = {}

    # Identify the columns to exclude
    objective_function = df.columns[-1]  # Always the last column
    exclude_columns = ["individual", "iteration", "replicate", objective_function]
    param_columns = [col for col in df.columns if col not in exclude_columns]

    # Calculate statistics for each parameter
    for param in param_columns:
        stats = {
            "mean": df[param].mean(),
            "std": df[param].std(),
            "min": df[param].min(),
            "max": df[param].max(),
            "count": df[param].count()
        }
        param_stats[param] = stats
    
    # Custom JSON encoder to handle NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    # Write the statistics to a JSON file
    # The name of the csv file is the name of the experiment
    experiment_name = os.path.basename(os.path.dirname(csv_file))

    # also add the "top_N" to the name of the json file
    top_N = os.path.basename(csv_file).split("_")[1].split(".")[0]
    json_output_file = f"{experiment_name}_param_distribution_top_{top_N}.json"

    # Save it in the same path as the csv file
    with open(os.path.join(os.path.dirname(csv_file), json_output_file), 'w') as json_file:
        json.dump(param_stats, json_file, indent=4, cls=NumpyEncoder)

    print(f"Parameter distribution statistics saved to {json_output_file}")


# Define the base directories for CMA and GA summaries
cma_base_dir = "./results/CMA_summaries"
ga_base_dir = "./results/GA_summaries"
sweep_base_dir = "./results/sweep_summaries"

# Iterate through all sub-folders in the GA summaries directory
for subdir, _, files in os.walk(ga_base_dir):
    if subdir != ga_base_dir:  # Ensure we are not in the base directory
        for file in files:
            if file.endswith(".csv"):
                csv_file_path = os.path.join(subdir, file)
                obtain_param_distribution(csv_file_path)

# Iterate through all sub-folders in the CMA summaries directory and avoid csv files in the "CMA_summaries" folder
for subdir, _, files in os.walk(cma_base_dir):
    if subdir != cma_base_dir:  # Ensure we are not in the base directory
        for file in files:
            if file.endswith(".csv"):
                csv_file_path = os.path.join(subdir, file)
                obtain_param_distribution(csv_file_path)

# Iterate through all sub-folders in the sweep summaries directory
for subdir, _, files in os.walk(sweep_base_dir):
    if subdir != sweep_base_dir:  # Ensure we are not in the base directory
        for file in files:
            if file.endswith(".csv"):
                csv_file_path = os.path.join(subdir, file)
                obtain_param_distribution(csv_file_path)


