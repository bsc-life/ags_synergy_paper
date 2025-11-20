import seaborn as sns
import matplotlib.pyplot as plt
import os, sys, glob
# from surface_plot import load_and_preprocess_data # TODO: THIS SHOULD NOT BE HERE, make a separate file for this
import pandas as pd
import numpy as np
# This script plots the convergence of the EMEWS CMA and GA run 
# Input is the folder with the results of the CMA / GA


def detect_strategy_from_filename(filename):
    if "CMA" in filename:
        return "CMA"
    elif "GA" in filename:
        return "GA"
    else:
        raise ValueError("Strategy not detected")


def filter_df_for_convergence(df, objective_metric_name):
    """
    Keep only the iteration, replicate, individual and RMSE_SK_POSTDRUG columns    
    and compute the mean RMSE_SK_POSTDRUG for each generation as well as the standard deviation
    """
    # group by generation and replicate and compute the mean RMSE_SK_POSTDRUG for each generation as well as the standard deviation
    df = df.groupby(["iteration", "replicate"]).agg({f"{objective_metric_name}": ["mean", "std"]}).reset_index()
    # rename the columns to mean_RMSE_SK_POSTDRUG and std_RMSE_SK_POSTDRUG
    df.columns = ["iteration", "replicate", f"mean_{objective_metric_name}", f"std_{objective_metric_name}"]
    # return the dataframe
    return df

def plot_convergence(*dataframes):
    """
    Plots the average RMSE_SK_POSTDRUG for each generation for one or more algorithms.
    Cell Systems-quality styling.
    """
    # Set up the figure with high-quality settings
    plt.figure(figsize=(4, 4), dpi=300)  # Changed to match other plots
    sns.set_context("paper", font_scale=0.8)
    sns.set_style("ticks")
    
    # Use sans-serif font to match other plots
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Liberation Sans', 
                                      'FreeSans', 'Arial', 'sans-serif']
    
    dataframes_names = []
    
    for summary in dataframes:
        # Skip files that don't exist
        if not os.path.exists(summary):
            print(f"Warning: File not found: {summary} - skipping")
            continue
            
        input_csv_name = os.path.basename(summary)
        input_csv_name = os.path.splitext(input_csv_name)[0]
        input_csv_name = input_csv_name.split("final_summary_")[1]
        dataframes_names.append(input_csv_name)

        # Process data
        df = pd.read_csv(summary)
        objective_metric_name = df.columns[-1]
        df = filter_df_for_convergence(df, objective_metric_name)
        df["iteration"] = df["iteration"].astype(int)
        
        # Compute statistics
        grouped_df = df.groupby('iteration').agg(
            mean_metric=(f"mean_{objective_metric_name}", 'mean'),
            std_metric=(f"std_{objective_metric_name}", 'mean')
        ).reset_index()
        
        # Updated color selection to match Cell Systems style
        palette = ['#4B4BFF', '#FF4B4B']  # Blue and Red to match SHAP plots
        color_index = dataframes_names.index(input_csv_name) % len(palette)
        color = palette[color_index]

        # Determine plot label
        strategy_name = next((s for s in ["GA", "CMA", "sweep"] if s in input_csv_name), "")
        drug_name = next((d for d in ["CTRL", "PI3K", "MEK", "AKT"] if d in input_csv_name), "")
        simple_name = f"{strategy_name} Control" if drug_name == "CTRL" else f"{strategy_name} {drug_name}"

        # Plot error bars with thinner lines
        plt.errorbar(
            x=grouped_df['iteration'], 
            y=grouped_df['mean_metric'], 
            fmt='o', 
            label=None,
            capsize=2,
            capthick=0.7,
            markersize=2,
            color=color,
            elinewidth=0.7
        )
        
        # Plot line with adjusted width
        sns.lineplot(
            x='iteration', 
            y='mean_metric', 
            data=grouped_df, 
            label=simple_name,
            color=color,
            linewidth=0.7
        )
        
        # Plot confidence interval with adjusted alpha
        plt.fill_between(
            grouped_df['iteration'], 
            grouped_df['mean_metric'] - grouped_df['std_metric'], 
            grouped_df['mean_metric'] + grouped_df['std_metric'], 
            alpha=0.1,  # Reduced alpha to match style
            color=color
        )
    
    # If no valid dataframes were processed, exit
    if not dataframes_names:
        print("No valid dataframes to plot. Exiting.")
        plt.close()
        return
    
    # Customize the plot
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)
    
    # Set axis limits
    plt.ylim(bottom=0)
    plt.xlim(left=1)
    if "CTRL" in input_csv_name:
        plt.xlim(right=24)

    # Labels and ticks with updated styling
    plt.xlabel("Generation", fontsize=11, fontweight="bold")
    plt.ylabel(f"{objective_metric_name}", fontsize=11, fontweight="bold")
    
    # Adjust tick parameters
    ax.tick_params(axis='both', width=0.8, length=2, labelsize=12, colors='black')
    
    # Legend with updated styling
    plt.legend(frameon=False, fontsize=8, loc='upper right')
    
    # Layout
    plt.tight_layout()
    
    # Save plots
    dataframes_names = "_".join(dataframes_names)
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           "results", "convergence_plots", f"{dataframes_names}")
    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(os.path.join(save_dir, f"convergence_plot_{dataframes_names}.png"), 
                dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(save_dir, f"convergence_plot_{dataframes_names}.svg"), 
                format='svg', bbox_inches='tight', transparent=True)
    print(f"Convergence plot saved to {save_dir}/convergence_plot_{dataframes_names}.png")
    plt.close()

if __name__ == "__main__":

    # fetch experiment folders
    experiment_folders = "./experiments"
    cma_experiment_names = [os.path.basename(folder) for folder in glob.glob(os.path.join(experiment_folders, "*")) if "CMA" in os.path.basename(folder)]

    for exp_name in cma_experiment_names:
        print(f"Processing {exp_name}")
        experiment_1_name = exp_name
        experiment_2_name = experiment_1_name.replace("CMA", "GA")
        experiment_1_type = detect_strategy_from_filename(experiment_1_name)
        experiment_2_type = detect_strategy_from_filename(experiment_2_name)
        
        summary_cma = f"results/{experiment_1_type}_summaries/final_summary_{experiment_1_name}.csv"
        summary_ga = f"results/{experiment_2_type}_summaries/final_summary_{experiment_2_name}.csv"
        
        # Check if files exist before attempting to plot
        if os.path.exists(summary_cma):
            plot_convergence(summary_cma)
        else:
            print(f"Warning: CMA summary file not found: {summary_cma}")
            
        if os.path.exists(summary_ga):
            plot_convergence(summary_ga)
        else:
            print(f"Warning: GA summary file not found: {summary_ga}")
            
        # Only plot comparison if both files exist
        if os.path.exists(summary_cma) and os.path.exists(summary_ga):
            plot_convergence(summary_cma, summary_ga)
        else:
            print(f"Warning: Cannot create comparison plot because one or both files are missing")



