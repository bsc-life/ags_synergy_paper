import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
# this script, for a given synergy sweep, for each instance, it obtains the minimum number of alive cells and the time point at which this occurs
# the minimum number of alive cells is the minimum number of alive cells in the last 100 time points, except for the first time point


experiment_name = "synergy_sweep-pi3k_mek-0205-1442-2p_3D_singledrugparams"
experiment_folder_path = f"experiments/{experiment_name}"

alive_cells_df = pd.DataFrame(columns=["individual", "replicate", "min_alive_cells", "time_point", "drug_X_diffusion_coefficient", "drug_Y_diffusion_coefficient"])

print("Experiment folder: ", experiment_folder_path)

results = []

def find_first_descent(growth_df):
    # Assumes growth_df is sorted by time and has an "alive" column
    alive = growth_df["alive"].values
    # Compute the difference between consecutive points
    diff = np.diff(alive)
    # Find where the difference goes from positive to negative
    for i in range(1, len(diff)):
        if diff[i-1] > 0 and diff[i] < 0:
            # Return the time and alive value at the local maximum
            return growth_df.iloc[i]["time"], growth_df.iloc[i]["alive"]
    # If no descent is found, return the last point
    return growth_df.iloc[-1]["time"], growth_df.iloc[-1]["alive"]

for instance_folder in os.listdir(experiment_folder_path):
    if instance_folder.startswith("instance_"):
        # print("Reading instance: ", instance_folder)
        instance_folder_path = os.path.join(experiment_folder_path, instance_folder)
        instance_info = instance_folder.split("instance_")[-1]
        individual = instance_info.split("_")[0]
        replicate = instance_info.split("_")[-1]
        # print(individual, replicate)

        # read the summary json file
        summary_json_path = os.path.join(instance_folder_path, "sim_summary.json")
        # convert to a dictionary
        summary_dict = json.load(open(summary_json_path))
        # get the diffusion coefficient for drug X and drug Y 
        drug_X_diffusion_coefficient = summary_dict["user_parameters.drug_X_diffusion_coefficient"]
        drug_Y_diffusion_coefficient = summary_dict["user_parameters.drug_Y_diffusion_coefficient"]


        growth_csv_path = os.path.join(instance_folder_path, "simulation_growth.csv")
        growth_df = pd.read_csv(growth_csv_path)
        # filter out first row
        growth_df = growth_df.iloc[1:]
        # Find the first descent
        time_point, alive_at_descent = find_first_descent(growth_df)

        results.append({
            "individual": individual,
            "replicate": replicate,
            "alive_at_first_descent": alive_at_descent,
            "time_point": time_point,
            "drug_X_diffusion_coefficient": drug_X_diffusion_coefficient,
            "drug_Y_diffusion_coefficient": drug_Y_diffusion_coefficient
        })

# save the dataframe in the folder where this script is located
save_path = os.path.join(os.path.dirname(__file__), "min_alive_cells_sweep_results", f"{experiment_name}_min_alive_cells.csv")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
alive_cells_df = pd.DataFrame(results)
alive_cells_df.to_csv(save_path, index=False)
print("Saved alive cells dataframe to: ", save_path)


# draw a histogram of the time_point column
plt.figure(figsize=(10, 6))
sns.histplot(alive_cells_df["time_point"], bins=100, kde=True)
plt.title("Histogram of Time Points")
plt.xlabel("Time Point")
plt.ylabel("Frequency")
plt.savefig(os.path.join(os.path.dirname(__file__), "min_alive_cells_sweep_results", f"{experiment_name}_time_point_histogram.png"))
plt.show()

# Get all unique diffusion coefficient combinations, sorted for consistent subplot order
diff_combos = alive_cells_df[["drug_X_diffusion_coefficient", "drug_Y_diffusion_coefficient"]].drop_duplicates()
diff_combos = diff_combos.sort_values(["drug_X_diffusion_coefficient", "drug_Y_diffusion_coefficient"]).values.tolist()

# Set global matplotlib parameters for Nature-style
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['figure.dpi'] = 100

nrows = ncols = 4  # 16 subplots (4x4 grid)
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12), sharex=True, sharey=True)

for idx, (x_diff, y_diff) in enumerate(diff_combos):
    row = idx // ncols
    col = idx % ncols
    ax = axes[row, col]

    # Find all instances for this diffusion combo
    combo_instances = alive_cells_df[
        (alive_cells_df["drug_X_diffusion_coefficient"] == x_diff) &
        (alive_cells_df["drug_Y_diffusion_coefficient"] == y_diff)
    ]

    # Collect all growth curves for this combo
    growth_curves = []
    time_points = None
    for _, instance in combo_instances.iterrows():
        instance_folder = f"instance_{instance['individual']}_{instance['replicate']}"
        instance_folder_path = os.path.join(experiment_folder_path, instance_folder)
        growth_csv_path = os.path.join(instance_folder_path, "simulation_growth.csv")
        growth_df = pd.read_csv(growth_csv_path)
        growth_df = growth_df.iloc[1:]  # skip first row
        if time_points is None:
            time_points = growth_df["time"].values
        growth_curves.append(growth_df["alive"].values)

    if growth_curves:
        growth_curves = np.array(growth_curves)
        mean_curve = np.mean(growth_curves, axis=0)
        std_curve = np.std(growth_curves, axis=0)
        # Use a distinct color for mean and a lighter shade for the fill
        ax.plot(time_points, mean_curve, color='#1f77b4', lw=2.5, zorder=3)
        ax.fill_between(time_points, mean_curve - std_curve, mean_curve + std_curve, 
                        color='#1f77b4', alpha=0.18, zorder=2)
        ax.set_title(f"X: {x_diff}, Y: {y_diff}", fontsize=14, pad=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=6, width=1.2)
    else:
        ax.set_visible(False)

# Shared axis labels
fig.text(0.5, 0.01, "Simulation Time (min)", ha='center', va='center', fontsize=20)
fig.text(0.01, 0.5, "Number of alive cells", ha='center', va='center', rotation='vertical', fontsize=20)

# plt.suptitle("Averaged Growth Curves by Drug Diffusion Combination", y=0.995, fontsize=22, fontweight='bold')
plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.06, wspace=0.25, hspace=0.25)

# Save at high DPI for publication
save_dir = os.path.join(os.path.dirname(__file__), "min_alive_cells_sweep_results")
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, f"{experiment_name}_growth_curves_by_diffusion.png"), dpi=600, bbox_inches='tight')
print(f"Saved growth curves to: {os.path.join(save_dir, f'{experiment_name}_growth_curves_by_diffusion.png')}")
plt.show()






