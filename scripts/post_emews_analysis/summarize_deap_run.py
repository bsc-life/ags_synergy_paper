# EMEWS DEAP GA/CMA-ES RUN SUMMARY
# A quick summary of the population after an EMEWS run

import os, sys, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def argparser():
    parser = argparse.ArgumentParser(description='Summarize DEAP run')
    parser.add_argument('--single_experiment', '-s', type=str, help='Path to the single experiment folder')
    parser.add_argument('--all_experiments_folder', '-exp', default="/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/experiments", type=str, help='Path to the parent experiments folder')
    parser.add_argument('--run_all_experiments', '-a', action='store_true', help='Summarize all experiments')
    return parser.parse_args()

# Obtain a CSV of all instances, store it in the experiment and in the results folder
def get_experiment_summary(single_experiment_folder):
    json_results_array = []

    print("this is the single experiment folder: ", single_experiment_folder)

    experiment_title = single_experiment_folder.split("/")[-1] # for naming the output file

    # Check if there are any instance directories
    instance_dirs = [dir for dir in os.listdir(single_experiment_folder) if dir.startswith("instance")]
    if not instance_dirs:
        print(f"No instance directories found in {single_experiment_folder}")
        return
    
    for instance in instance_dirs:
        json_files = [dir for dir in os.listdir(os.path.join(single_experiment_folder, instance)) if dir.endswith("summary.json")]
        if not json_files:
            continue
            
        for json_res in json_files:
            with open(os.path.join(single_experiment_folder, instance, json_res)) as fh:
                try:
                    lines = [i.rstrip() for i in fh.readlines()]
                    test_line = json.loads(lines[0])
                    json_results_array.append(test_line)
                except Exception as e:
                    print(f"Could not read file: {os.path.join(single_experiment_folder, instance, json_res)}")
                    print(f"Error: {str(e)}")
                    continue
    
    # Check if we have any results to process
    if not json_results_array:
        print(f"No valid JSON data found in {single_experiment_folder}")
        return
    
    final_title = "final_summary_" + experiment_title + ".csv"
    
    try:
        final_df = pd.DataFrame.from_dict(json_results_array, orient="columns")
        
        # Check if columns are string type before using str accessor
        if not final_df.empty and isinstance(final_df.columns, pd.Index):
            # Convert columns to string if needed
            final_df.columns = final_df.columns.astype(str)
            # delete the "cell_definitions.cell_definition.custom_data" from the column names
            final_df.columns = final_df.columns.str.replace('cell_definitions.cell_definition.custom_data.', '')
        
        # Determine output locations
        final_df.to_csv(os.path.join(single_experiment_folder, final_title), float_format='%.10f', header=True, index=False)

        emews_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))
        print("this is the experiment title: ", experiment_title)

        if "sweep" in experiment_title:
            final_destination_path = os.path.join(emews_folder_path, "results", "sweep_summaries")
        elif "GA" in experiment_title:
            final_destination_path = os.path.join(emews_folder_path, "results", "GA_summaries")
        elif "CMA" in experiment_title:
            final_destination_path = os.path.join(emews_folder_path, "results", "CMA_summaries")
        else:
            print(f"Unknown experiment type for {experiment_title}, saving to generic location")
            final_destination_path = os.path.join(emews_folder_path, "results", "other_summaries")
            os.makedirs(final_destination_path, exist_ok=True)

        final_df.to_csv(os.path.join(final_destination_path, final_title), float_format='%.10f', header=True, index=False)

        # then get the top 10, 20, 50, 100 best results
        obtain_best_summary_results(os.path.join(final_destination_path, final_title))
        print("Top 10, 20, 50, 100, 200 best results obtained!")
        print("Top 1%, 5%, 10%, 25 percentages best results obtained!")
    
    except Exception as e:
        print(f"Error processing data from {single_experiment_folder}: {str(e)}")
        return

def get_all_experiments_summary(all_experiments_folder):
    for file in os.listdir(all_experiments_folder):
        file_path = os.path.join(all_experiments_folder, file)
        
        # Skip non-directories
        if not os.path.isdir(file_path):
            print(f"Skipping non-directory: {file}")
            continue
            
        # Skip directories that start with numbers or underscores (likely special folders)
        if file.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_')):
            print(f"Skipping special directory: {file}")
            continue

        # if the summary already exists in the results folder, skip the experiment
        results_folder = os.path.join(os.path.dirname(__file__), "..", "..", "results")
        if os.path.exists(os.path.join(results_folder, f"final_summary_{file}.csv")):
            print(f"Skipping experiment {file} because summary already exists in the results folder")
            continue
            
        try:
            get_experiment_summary(os.path.join(all_experiments_folder, file))
            print(f"Summary for {file} obtained!")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

def obtain_best_summary_results(experiment_summary_csv):
    df = pd.read_csv(experiment_summary_csv)
    # sort by last column (RMSE_SK) in ascending order 
    df = df.sort_values(by=df.columns[-1], ascending=True)

    # Fixed number selections
    top_10 = df.head(10)
    top_20 = df.head(20)
    top_50 = df.head(50)
    top_100 = df.head(100)
    top_200 = df.head(200)

    # Percentage-based selections
    total_rows = len(df)
    top_1p = df.head(int(total_rows * 0.01))  # top 1%
    top_5p = df.head(int(total_rows * 0.05))  # top 5%
    top_10p = df.head(int(total_rows * 0.10)) # top 10%
    top_25p = df.head(int(total_rows * 0.25)) # top 25%

    # Create output directory
    experiment_folder = os.path.dirname(experiment_summary_csv)
    experiment_folder = os.path.join(experiment_folder, os.path.basename(experiment_summary_csv).split(".csv")[0])
    os.makedirs(experiment_folder, exist_ok=True)

    # Save fixed number selections
    top_10.to_csv(os.path.join(experiment_folder, "top_10.csv"), float_format='%.10f', header=True, index=False)
    top_20.to_csv(os.path.join(experiment_folder, "top_20.csv"), float_format='%.10f', header=True, index=False)
    top_50.to_csv(os.path.join(experiment_folder, "top_50.csv"), float_format='%.10f', header=True, index=False)
    top_100.to_csv(os.path.join(experiment_folder, "top_100.csv"), float_format='%.10f', header=True, index=False)
    top_200.to_csv(os.path.join(experiment_folder, "top_200.csv"), float_format='%.10f', header=True, index=False)

    # Save percentage-based selections
    top_1p.to_csv(os.path.join(experiment_folder, "top_1p.csv"), float_format='%.10f', header=True, index=False)
    top_5p.to_csv(os.path.join(experiment_folder, "top_5p.csv"), float_format='%.10f', header=True, index=False)
    top_10p.to_csv(os.path.join(experiment_folder, "top_10p.csv"), float_format='%.10f', header=True, index=False)
    top_25p.to_csv(os.path.join(experiment_folder, "top_25p.csv"), float_format='%.10f', header=True, index=False)

def main():

    args = argparser()

    if args.run_all_experiments:
        print("Obtaining summary for all experiments...")
        get_all_experiments_summary(args.all_experiments_folder)
        print("Summary for all experiments obtained!")
    else:
        print("this is the single experiment: ", args.single_experiment)
        # delete trailing slash if it exists
        if args.single_experiment.endswith('/'):
            args.single_experiment = args.single_experiment[:-1]
            print("this is the single experiment: ", args.single_experiment)
        get_experiment_summary(args.single_experiment)
        print("Summary for single experiment obtained!")

if __name__ == "__main__":
    main()
