import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def lowpulse_scatterplot(df, filename):
    # Create the scatter plot
    sns.set_context("paper")
    plt.figure(figsize=(6, 4))
    # remove the "user_parameters" from the column names if they have it
    df.columns = df.columns.str.replace('user_parameters.', '')
    # Calculate the 1st percentile threshold
    threshold = np.percentile(df['FINAL_CELL_INDEX_VALUE'], 1)
    # Create binary alpha values based on threshold
    metric_values = df['FINAL_CELL_INDEX_VALUE']
    alpha_values = np.where(metric_values <= threshold, 1.0, 0.1)
    # Print some information about the selection
    print(f"Threshold value (1st percentile): {threshold:.2f}")
    print(f"Number of highlighted points: {sum(metric_values <= threshold)}")
    scatter = plt.scatter(df['drug_X_pulse_duration'], 
                         df['fraction_of_concentration'],
                         c=df['FINAL_CELL_INDEX_VALUE'],
                         cmap='magma',
                         alpha=alpha_values)

    # Add colorbar
    plt.colorbar(scatter, label='Final Cell Index Value')

    # Customize the plot
    plt.xlabel('Drug X Pulse Duration')
    plt.ylabel('Fraction of Concentration')
    plt.title('Parameter Space Exploration\nHighlighting Lowest 5% Cell Index Values')

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "MED_plots")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{filename}_top5percent.png"), dpi=300, bbox_inches='tight')
    print(f"Saved plot to {os.path.join(save_dir, f'{filename}_top5percent.png')}")


if __name__ == "__main__":
    sweep_summary_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "sweep_summaries")

    for file in os.listdir(sweep_summary_path):
        if "MED" in file and file.endswith(".csv"):
            df = pd.read_csv(os.path.join(sweep_summary_path, file))
            lowpulse_scatterplot(df, file.split(".")[0])