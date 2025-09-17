#!/usr/bin/env python
# coding: utf-

import json, os
import argparse
import numpy as np
from numpy.random import normal
from numpy.random import uniform
import math

from sklearn.model_selection import ParameterGrid

MODES = ("uniform", "normal", "grid", "logscale", "hybrid", "structured_hybrid")


##### If we wish to plot the points
# from collections import defaultdict
# import matplotlib.pyplot as plt
# mypoints = defaultdict(list)
#####

# For the PI3K-MEK synergy experiment:
# out path data/JSON/sweep/sweep_txt/12p_PI3K_MEK


def create_parser():
    parser = argparse.ArgumentParser(description="Parameter grid generator to run model exploration sweep")
    parser.add_argument("param_json", action="store", help="JSON file having param name as key and a dic with ref values")
    parser.add_argument("--out", action="store", help="output name", default=None)
    parser.add_argument("--mode", action="store", help="Sampling mode", choices=MODES, default='grid')
    parser.add_argument("--size", action="store", type=int, help="Total values for each parameter (in uniform mode, the total number of points)")
    return parser



# Loading the path to the JSON param sweep files
param_json_folder_path = "results/comparing_top_distributions/"

# Best fittings for normal, Hill-shaped first calibration

# best_result_pi3k_mek_synergy_params_path = param_json_folder_path + "sweep_combined_PI3Ki_CMA-1410-1014-12p_rmse_final_50gen_and_MEKi_CMA-1410-1026-12p_rmse_final_50gen.json"
# best_result_pi3kmek_PI3K_singledrug_params_path = param_json_folder_path + "PI3Ki_CMA-1410-1014-12p_rmse_final_50gen_single_drug_sweep_combined_PI3Ki_CMA-1410-1014-12p_rmse_final_50gen_MEKi_CMA-1410-1026-12p_rmse_final_50gen.json"
# best_result_pi3kmek_MEK_singledrug_params_path = param_json_folder_path + "MEKi_CMA-1410-1026-12p_rmse_final_50gen_single_drug_sweep_combined_PI3Ki_CMA-1410-1014-12p_rmse_final_50gen_MEKi_CMA-1410-1026-12p_rmse_final_50gen.json"

# best_result_akt_mek_synergy_params_path = param_json_folder_path + "sweep_combined_AKTi_CMA-1710-0934-12p_rmse_final_50gen_and_MEKi_CMA-1410-1026-12p_rmse_final_50gen.json"
# best_result_aktmek_AKT_singledrug_params_path = param_json_folder_path + "AKTi_CMA-1710-0934-12p_rmse_final_50gen_single_drug_sweep_combined_AKTi_CMA-1710-0934-12p_rmse_final_50gen_MEKi_CMA-1410-1026-12p_rmse_final_50gen.json"
# best_result_aktmek_MEK_singledrug_params_path = param_json_folder_path + "MEKi_CMA-1410-1026-12p_rmse_final_50gen_single_drug_sweep_combined_AKTi_CMA-1710-0934-12p_rmse_final_50gen_MEKi_CMA-1410-1026-12p_rmse_final_50gen.json"


# Best fittings for linear mapping
# best_result_pi3k_mek_synergy_params_path = param_json_folder_path + "sweep_combined_PI3Ki_CMA-1002-0147-8p_linear_mapping_and_MEKi_CMA-1002-0147-8p_linear_mapping.json"
# best_result_pi3kmek_PI3K_singledrug_params_path = param_json_folder_path + "PI3Ki_CMA-1002-0147-8p_linear_mapping_single_drug_sweep_combined_PI3Ki_CMA-1002-0147-8p_linear_mapping_MEKi_CMA-1002-0147-8p_linear_mapping.json"
# best_result_pi3kmek_MEK_singledrug_params_path = param_json_folder_path + "MEKi_CMA-1002-0147-8p_linear_mapping_single_drug_sweep_combined_PI3Ki_CMA-1002-0147-8p_linear_mapping_MEKi_CMA-1002-0147-8p_linear_mapping.json"

# best_result_akt_mek_synergy_params_path = param_json_folder_path + "sweep_combined_AKTi_CMA-1002-0147-8p_linear_mapping_and_MEKi_CMA-1002-0147-8p_linear_mapping.json"
# best_result_aktmek_AKT_singledrug_params_path = param_json_folder_path + "AKTi_CMA-1002-0147-8p_linear_mapping_single_drug_sweep_combined_AKTi_CMA-1002-0147-8p_linear_mapping_MEKi_CMA-1002-0147-8p_linear_mapping.json"
# best_result_aktmek_MEK_singledrug_params_path = param_json_folder_path + "MEKi_CMA-1002-0147-8p_linear_mapping_single_drug_sweep_combined_AKTi_CMA-1002-0147-8p_linear_mapping_MEKi_CMA-1002-0147-8p_linear_mapping.json"

# Best fittings for transient delayed effect
pi3k_exp_info = "0704-1815-18p_delayed_transient_rmse_postdrug_25gen"
mek_exp_info = "0704-1815-18p_delayed_transient_rmse_postdrug_25gen"
akt_exp_info = "0704-1815-18p_delayed_transient_rmse_postdrug_25gen"
top_n = "10p"

best_result_pi3k_mek_synergy_params_path = param_json_folder_path + f"sweep_combined_PI3Ki_CMA-{pi3k_exp_info}_and_MEKi_CMA-{mek_exp_info}_top_{top_n}.json"
best_result_pi3kmek_PI3K_singledrug_params_path = param_json_folder_path + f"PI3Ki_CMA-{pi3k_exp_info}_single_drug_sweep_combined_PI3Ki_CMA-{pi3k_exp_info}_MEKi_CMA-{mek_exp_info}_top_{top_n}.json"
best_result_pi3kmek_MEK_singledrug_params_path = param_json_folder_path + f"MEKi_CMA-{mek_exp_info}_single_drug_sweep_combined_PI3Ki_CMA-{pi3k_exp_info}_MEKi_CMA-{mek_exp_info}_top_{top_n}.json"

best_result_akt_mek_synergy_params_path = param_json_folder_path + f"sweep_combined_AKTi_CMA-{akt_exp_info}_and_MEKi_CMA-{mek_exp_info}_top_{top_n}.json"
best_result_aktmek_AKT_singledrug_params_path = param_json_folder_path + f"AKTi_CMA-{akt_exp_info}_single_drug_sweep_combined_AKTi_CMA-{akt_exp_info}_MEKi_CMA-{mek_exp_info}_top_{top_n}.json"
best_result_aktmek_MEK_singledrug_params_path = param_json_folder_path + f"MEKi_CMA-{mek_exp_info}_single_drug_sweep_combined_AKTi_CMA-{akt_exp_info}_MEKi_CMA-{mek_exp_info}_top_{top_n}.json"

# output path
output_path = "./data/JSON/sweep/sweep_txt/"

def generate_sweep(param_json_path, mode, size, output_path):

    output_path = os.path.join(output_path, param_json_path.split("/")[-1].split(".")[0] + "_" + mode + "_" + str(size) + ".txt")
    
    params = {}
    with open(param_json_path) as fh:
        params = json.load(fh)

    grid = {}
    if mode == "uniform":
        for i in range(size):
            grid = {}  # Reset grid for each iteration
            for k, v in params.items():
                # print("this is v: ", v)
                # print("this is k: ", k)
                grid[k] = uniform(v['min'], v['max'])

            if output_path is not None:
                with open(output_path, 'a') as fh:  # Append to the output file
                    print(json.dumps(grid), file=fh)
            else:
                print(json.dumps(grid))
                
    elif mode == "normal":  # Not Used Yet
        for k, v in params.items():
            grid[k] = normal(loc=v['loc'], scale=v['scale'], size=args.size)

    elif mode == "grid":
        grid = {k: np.linspace(v['min'], v['max'], size) for k, v in params.items()}
        if output_path is not None:
            with open(output_path, 'w') as fh:
                for p in ParameterGrid(grid):
                    print(json.dumps(p), file=fh)
        else:
            for p in ParameterGrid(grid):
                print(json.dumps(p))

    elif mode == "logscale":
        # Generate values that maintain the same leading digit across orders of magnitude
        param_values = {}
        
        for k, v in params.items():
            min_val = v['min']
            max_val = v['max']
            
            # Get the leading digit and magnitude of the minimum value
            log_min = math.log10(min_val)
            log_max = math.log10(max_val)
            min_magnitude = math.floor(log_min)
            leading_digit = min_val / (10 ** min_magnitude)
            
            # Generate values with the same leading digit across orders of magnitude
            values = []
            current_magnitude = min_magnitude
            current_value = min_val
            
            while current_value <= max_val:
                values.append(current_value)
                current_magnitude += 1
                current_value = leading_digit * (10 ** current_magnitude)
            
            # Make sure to include the max value if it's not already in the list
            if values[-1] < max_val:
                values.append(max_val)
                
            param_values[k] = values
            
            # Print debug info
            print(f"Parameter {k}: {values}")
        
        # Generate all combinations using ParameterGrid
        if output_path is not None:
            with open(output_path, 'w') as fh:
                for p in ParameterGrid(param_values):
                    print(json.dumps(p), file=fh)
        else:
            for p in ParameterGrid(param_values):
                print(json.dumps(p))

        print("saved to ", output_path)

    elif mode == "hybrid":
        # In this mode, `size` is the total number of parameter sets to generate.
        # We perform random sampling: uniform for consensus, log-uniform for others.
        logscale_keywords = ["diffusion_coefficient", "pulse_period"]

        if output_path is not None:
            fh = open(output_path, 'w')
        
        for i in range(size):
            instance = {}
            for k, v in params.items():
                min_val, max_val = v['min'], v['max']

                if any(keyword in k for keyword in logscale_keywords):
                    # Sample from a log-uniform distribution
                    if min_val <= 0 or max_val <= 0:
                         print(f"Warning: Parameter '{k}' has non-positive bounds ({min_val}, {max_val}). Falling back to uniform sampling.")
                         instance[k] = uniform(min_val, max_val)
                    else:
                         instance[k] = 10**uniform(math.log10(min_val), math.log10(max_val))
                else:
                    # Sample from a standard uniform distribution for consensus parameters
                    instance[k] = uniform(min_val, max_val)
            
            # Write the generated instance
            if output_path is not None:
                print(json.dumps(instance), file=fh)
            else:
                print(json.dumps(instance))

        if output_path is not None:
            fh.close()
            print(f"Hybrid sweep with {size} points saved to {output_path}")

    elif mode == "structured_hybrid":
        # In this mode, `size` is the number of random samples per grid point.
        # We create a grid for strategic params (diffusion/timing) and sample
        # randomly from consensus params at each grid point.
        strategic_params = {}
        consensus_params = {}
        strategic_keywords = ["diffusion_coefficient", "pulse_period"]

        for k, v in params.items():
            if any(keyword in k for keyword in strategic_keywords):
                strategic_params[k] = v
            else:
                consensus_params[k] = v

        # 1. Create the grid for strategic parameters
        strategic_grid_values = {}
        log_points = 4  # e.g., 6, 60, 600, 6000

        for k, v in strategic_params.items():
            min_val, max_val = v['min'], v['max']
            if min_val <= 0 or max_val <= 0:
                print(f"Warning: Strategic parameter '{k}' has non-positive bounds. Using linspace.")
                strategic_grid_values[k] = np.linspace(min_val, max_val, log_points)
            else:
                strategic_grid_values[k] = np.logspace(np.log10(min_val), np.log10(max_val), num=log_points)
        
        strategic_grid = ParameterGrid(strategic_grid_values)
        
        num_grid_points = len(list(ParameterGrid(strategic_grid_values)))
        n_samples_per_point = size
        total_points = num_grid_points * n_samples_per_point
        
        print(f"Generating structured hybrid sweep...")
        print(f"Strategic grid has {num_grid_points} points.")
        print(f"Generating {n_samples_per_point} random samples per grid point.")
        print(f"Total points to be generated: {total_points}")

        # 2. Iterate grid and sample consensus parameters
        if output_path is not None:
            # It's better to open the file once and append
            with open(output_path, 'w') as fh:
                for grid_point in strategic_grid:
                    for _ in range(n_samples_per_point):
                        # Start with the fixed strategic parameters from the grid
                        final_instance = {k: v.item() if hasattr(v, 'item') else v for k, v in grid_point.items()}
                        
                        # Sample consensus parameters randomly
                        for k_c, v_c in consensus_params.items():
                            final_instance[k_c] = uniform(v_c['min'], v_c['max'])
                        
                        # Write the complete instance to file
                        print(json.dumps(final_instance), file=fh)
        else: # If no output path, print to stdout
            for grid_point in strategic_grid:
                for _ in range(n_samples_per_point):
                    final_instance = {k: v.item() if hasattr(v, 'item') else v for k, v in grid_point.items()}
                    for k_c, v_c in consensus_params.items():
                        final_instance[k_c] = uniform(v_c['min'], v_c['max'])
                    print(json.dumps(final_instance))

        if output_path is not None:
            print(f"Structured hybrid sweep with {total_points} points saved to {output_path}")




# do pi3k mek synergy
# for file in [best_result_pi3k_mek_synergy_params_path, best_result_pi3kmek_PI3K_singledrug_params_path, best_result_pi3kmek_MEK_singledrug_params_path]:
#     generate_sweep(file, "uniform", 5000, output_path)

# # do akt mek synergy
# for file in [best_result_akt_mek_synergy_params_path, best_result_aktmek_AKT_singledrug_params_path, best_result_aktmek_MEK_singledrug_params_path]:
#     generate_sweep(file, "uniform", 5000, output_path)


# lowpulse_drugs_path = ["data/JSON/sweep/sweep_3p_synergy_frequency_lowpulse.json"]
# drugaddition_3D_path = ["data/JSON/sweep/sweep_2p_3D_drugaddition.json"]
# drugaddition_3D_drugtiming_path = ["data/JSON/sweep/sweep_4p_3D_drugaddition_drugtiming.json"]
# drugaddition_3D_layerdepth_path = ["data/JSON/sweep/sweep_4p_3D_drugaddition_layerdepth.json"]
# drugaddition_3D_drugtiming_layerdepth_path = ["data/JSON/sweep/sweep_6p_3D_drugaddition_drugtiming_layerdepth.json"]


drugaddition_3D_drugtiming_path = ["data/JSON/sweep/sweep_consensus_pi3k_mek_top1p.json", "data/JSON/sweep/sweep_consensus_akt_mek_top1p.json"]

for file in drugaddition_3D_drugtiming_path:
    # Generate a structured sweep: 256 strategic points, 100 random samples each.
    generate_sweep(file, "structured_hybrid", 20, output_path)



# dose response curve JSONs
# pi3k_DR_path = "data/JSON/sweep/sweep_1p_dose_response_curve_PI3K.json"
# mek_DR_path = "data/JSON/sweep/sweep_1p_dose_response_curve_MEK.json"
# akt_DR_path = "data/JSON/sweep/sweep_1p_dose_response_curve_AKT.json"

# # do dose response curve
# for file in [pi3k_DR_path, mek_DR_path, akt_DR_path]:
#     generate_sweep(file, "uniform", 100, output_path)

