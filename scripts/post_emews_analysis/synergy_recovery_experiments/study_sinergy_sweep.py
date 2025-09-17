import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# this script provides a first analysis of the 3D AGS model and the synergy sweep for different
# diffusion coefficients


########################################################
# FOR AKT-MEK SYNERGY SWEEP
########################################################

# first we need to obtain the control numbers of alive cells for the 3D AGS model under two different conditions


sweep_summaries_path = "results/sweep_summaries/"
positive_control_name = "synergy_sweep-akt_mek-3D-0205-1608-positive_control"
negative_control_name = "synergy_sweep-3D-0205-1608-control_nodrug"
synergy_sweep_2p_name = "synergy_sweep-akt_mek-0505-1910-2p_3D_singledrugparams"
# synergy_sweep_2p_name = "synergy_sweep-pi3k_mek-0205-1608-2p_3D_singledrugparams"





# synergy_sweep_6p_name = "synergy_sweep-akt_mek-2504-1825-6p_3D_drugaddition_drugtiming_layerdepth"

# 1. positive control with drug diffusion coefficient which was employed in the previous AGS 2D synergy experiments
# print(f"positive_control_name: {positive_control_name}")
# print("--------------------------------")
# final_summary_positive_control = pd.read_csv(f"{sweep_summaries_path}/final_summary_{positive_control_name}.csv")
# # print(final_summary_positive_control.head())
# # the last column is the objective function, we get the mean and std of this one
# mean_final_number_of_alive_cells_positive_control = round(final_summary_positive_control.iloc[:, -1].mean())
# std_final_number_of_alive_cells_positive_control = round(final_summary_positive_control.iloc[:, -1].std(), 2)
# print(f"POSITIVE CONTROL mean: {mean_final_number_of_alive_cells_positive_control}")
# print(f"POSITIVE CONTROL std: {std_final_number_of_alive_cells_positive_control}")
# print("--------------------------------")


# 2. negative control with no drug
final_summary_negative_control = pd.read_csv(f"{sweep_summaries_path}/final_summary_{negative_control_name}.csv")
# print(final_summary_negative_control.head())
# the last column is the objective function, we get the mean and std of this one
mean_final_number_of_alive_cells_negative_control = round(final_summary_negative_control.iloc[:, -1].mean())
std_final_number_of_alive_cells_negative_control = round(final_summary_negative_control.iloc[:, -1].std(), 2)
print(f"NEGATIVE CONTROL mean: {mean_final_number_of_alive_cells_negative_control}")
print(f"NEGATIVE CONTROL std: {std_final_number_of_alive_cells_negative_control}")
print("--------------------------------\n")


#################################################################################################
# Reading the singledrug control experiments
#################################################################################################

pi3k_single_drug_name = "synergy_sweep-akt_mek-3D-0505-1910-logscale_singledrug_akt"
mek_single_drug_name = "synergy_sweep-akt_mek-3D-0505-1910-logscale_singledrug_mek"


# pi3k_single_drug_name = "synergy_sweep-pi3k_mek-3D-0505-0218-logscale_singledrug_pi3k"
# mek_single_drug_name = "synergy_sweep-pi3k_mek-3D-0505-0218-logscale_singledrug_mek"


pi3k_single_drug = pd.read_csv(f"{sweep_summaries_path}/final_summary_{pi3k_single_drug_name}.csv")
mek_single_drug = pd.read_csv(f"{sweep_summaries_path}/final_summary_{mek_single_drug_name}.csv")

print(pi3k_single_drug.head())
print(mek_single_drug.head())

# we need to obtain a dictionary of the diffusion coefficient and the average final number of alive cells
pi3k_grouped = pi3k_single_drug.groupby(pi3k_single_drug.columns[0]).agg({pi3k_single_drug.columns[-1]: 'mean'})
pi3k_single_drug_dict = pi3k_grouped[pi3k_single_drug.columns[-1]].to_dict()

mek_grouped = mek_single_drug.groupby(mek_single_drug.columns[0]).agg({mek_single_drug.columns[-1]: 'mean'})
mek_single_drug_dict = mek_grouped[mek_single_drug.columns[-1]].to_dict()

print("--------------------------------")
print(f"PI3K single drug dict: {pi3k_single_drug_dict}")
print(f"MEK single drug dict: {mek_single_drug_dict}")
print("--------------------------------\n")


#################################################################################################
# Analysis of 2p synergy sweep: Drug diffusion ratios
#################################################################################################

synergy_sweep_2p = pd.read_csv(f"{sweep_summaries_path}/final_summary_{synergy_sweep_2p_name}.csv")
# last column is the final number of alive cells, which we can normalize to the negative control (no drug)
synergy_sweep_2p['norm_final_cellcount'] = synergy_sweep_2p.iloc[:, -1] / mean_final_number_of_alive_cells_negative_control
# delete the "user_parameters." from the columns which have it 
synergy_sweep_2p.columns = synergy_sweep_2p.columns.str.replace('user_parameters.', '')
# delete columns "individual" and "replica"
synergy_sweep_2p = synergy_sweep_2p.drop(columns=['individual', 'replicate'])

print(synergy_sweep_2p.head(50))

# 1st column is the drug X diffusion coefficient
# 2nd column is the drug Y diffusion coefficient
# 3rd column is the final number of alive cells
# 4th column is the normalized final number of alive cells


# Explicitly aggregate the data from replicates
# Group by diffusion coefficients and calculate statistics
aggregated_data = synergy_sweep_2p.groupby(['drug_X_diffusion_coefficient', 'drug_Y_diffusion_coefficient']).agg({
    'FINAL_NUMBER_OF_ALIVE_CELLS': ['mean', 'std']
}).reset_index()

# Flatten the column hierarchy for easier access
aggregated_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in aggregated_data.columns.values]

# Create a pivot table for the heatmap using the mean values
pivot_mean = aggregated_data.pivot_table(
    index='drug_Y_diffusion_coefficient',  # Y-axis
    columns='drug_X_diffusion_coefficient',  # X-axis
    values='FINAL_NUMBER_OF_ALIVE_CELLS_mean'  # Color values (mean of replicates)
)

# Create a pivot table for the standard deviation
pivot_std = aggregated_data.pivot_table(
    index='drug_Y_diffusion_coefficient',  # Y-axis
    columns='drug_X_diffusion_coefficient',  # X-axis
    values='FINAL_NUMBER_OF_ALIVE_CELLS_std'  # Standard deviation values
)

# Normalize mean and std by the negative control mean
pivot_mean_norm = pivot_mean / mean_final_number_of_alive_cells_negative_control
pivot_std_norm = pivot_std / mean_final_number_of_alive_cells_negative_control

# Create new annotations for normalized values (mean ± std, rounded to 2 decimals)
pivot_annotations_norm = pivot_mean_norm.copy()
for i in pivot_mean_norm.index:
    for j in pivot_mean_norm.columns:
        mean_val = pivot_mean_norm.loc[i, j]
        std_val = pivot_std_norm.loc[i, j]
        pivot_annotations_norm.loc[i, j] = f"{mean_val:.2f} ± {std_val:.2f}"

# Get sorted indices to ensure proper ordering
x_order = sorted(aggregated_data['drug_X_diffusion_coefficient'].unique())
y_order = sorted(aggregated_data['drug_Y_diffusion_coefficient'].unique(), reverse=True)  # Reversed order

# Set up the figure (compact, square cells)
plt.figure(figsize=(7, 6), dpi=300)

# Use a diverging colormap for better contrast (like Bliss)
cmap = sns.diverging_palette(240, 10, as_cmap=True, center="light")

# Create the heatmap with improved aesthetics
ax = sns.heatmap(
    pivot_mean_norm.loc[y_order, x_order],
    cmap=cmap,
    annot=pivot_annotations_norm.loc[y_order, x_order],
    fmt="",
    annot_kws={"size": 9, "weight": "bold"},
    linewidths=0.5,
    square=True,
    cbar_kws={
        'label': 'Normalized Mean Number of Alive Cells',
        'shrink': 0.8,
        'aspect': 10,
        'pad': 0.01
    },
    center=0.5  # Since normalized, 1.0 = control, <1 = fewer alive cells
)

# Adjust axis labels for readability
plt.xlabel('Drug X Diffusion Coefficient (μm²/min)', fontsize=10, fontweight='bold')
plt.ylabel('Drug Y Diffusion Coefficient (μm²/min)', fontsize=10, fontweight='bold')

# Optional: Add a professional title
# plt.title('Cell Count Heatmap (Mean ± Std Dev)', fontsize=12, fontweight='bold')

# Rotate tick labels for better fit
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)

# Tighter layout
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Save the figure
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = f"{script_dir}/synergy_plots_2D"
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(f"{save_path}/diffusion_heatmap_with_std_normalized_{synergy_sweep_2p_name}.png", dpi=600, bbox_inches='tight', transparent=False)
plt.show()

print(f"Heatmap with standard deviations saved to {save_path}")



#################################################################################################
# Bliss synergy analysis for diffusion coefficient combinations
#################################################################################################

# Prepare a DataFrame for Bliss scores
bliss_scores = []

for idx, row in aggregated_data.iterrows():
    x_diff = row['drug_X_diffusion_coefficient']
    y_diff = row['drug_Y_diffusion_coefficient']
    observed = row['FINAL_NUMBER_OF_ALIVE_CELLS_mean']

    # Get single-drug effects at the same diffusion coefficients
    pi3k_single = pi3k_single_drug_dict.get(x_diff, None)
    mek_single = mek_single_drug_dict.get(y_diff, None)
    if pi3k_single is None or mek_single is None:
        continue

    # Normalize single-drug and observed values to negative control (if not already)
    E_A = pi3k_single / mean_final_number_of_alive_cells_negative_control
    E_B = mek_single / mean_final_number_of_alive_cells_negative_control
    E_AB = observed / mean_final_number_of_alive_cells_negative_control

    # Bliss independence model: E_bliss = E_A * E_B
    # Bliss score: (expected - observed); negative = synergy, positive = antagonism
    bliss_expected = E_A * E_B
    bliss_score = bliss_expected - E_AB

    bliss_scores.append({
        'drug_X_diffusion_coefficient': x_diff,
        'drug_Y_diffusion_coefficient': y_diff,
        'bliss_score': bliss_score
    })

# Convert to DataFrame
bliss_df = pd.DataFrame(bliss_scores)

# Create a pivot table for the heatmap
pivot_bliss = bliss_df.pivot_table(
    index='drug_Y_diffusion_coefficient',
    columns='drug_X_diffusion_coefficient',
    values='bliss_score'
)

# Get sorted indices for axes
x_order_bliss = sorted(bliss_df['drug_X_diffusion_coefficient'].unique())
y_order_bliss = sorted(bliss_df['drug_Y_diffusion_coefficient'].unique(), reverse=True)

# Plot the Bliss score heatmap - improved version
plt.figure(figsize=(7, 6), dpi=300)  # More compact size

# Create custom colormap with stronger contrast for synergy/antagonism
# Deep blue for synergy, white for neutral, dark red for antagonism
cmap = sns.diverging_palette(240, 10, as_cmap=True, center="light")

# Format annotations to be cleaner
def format_bliss(val):
    return f"{val:.2f}" if abs(val) >= 0.1 else ""

# Create the heatmap with improved aesthetics
ax_bliss = sns.heatmap(
    pivot_bliss.loc[y_order_bliss, x_order_bliss],
    cmap=cmap,
    center=0,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 9, "weight": "bold"},
    linewidths=0.5,
    square=True,  # Makes cells square for better proportions
    cbar_kws={
        'label': 'Bliss Score',
        'shrink': 0.8,
        'aspect': 10,
        'pad': 0.01
    }
)

# Adjust axis labels for readability
plt.xlabel('Drug X Diffusion Coefficient (μm²/min)', fontsize=10, fontweight='bold')
plt.ylabel('Drug Y Diffusion Coefficient (μm²/min)', fontsize=10, fontweight='bold')

# Professional title
# plt.title('Bliss Synergy Map', fontsize=12, fontweight='bold')

# Add explanatory text
plt.figtext(0.5, 0.01, 'Negative values (blue) = Synergy | Positive values (red) = Antagonism', 
           ha='center', fontsize=8, style='italic')

# Rotate tick labels for better fit
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)

# Tighter layout
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Save the figure with improved quality
plt.savefig(f"{save_path}/bliss_heatmap_{synergy_sweep_2p_name}.png", 
           dpi=600, bbox_inches='tight', transparent=False)
plt.show()

print(f"Bliss synergy heatmap saved to {save_path}")



#################################################################################################
# Analysis of 6p synergy sweep: Drug diffusions, membrane thickness, and drug addition time
#################################################################################################


# synergy_sweep_2p = pd.read_csv(f"{sweep_summaries_path}/final_summary_{synergy_sweep_6p_name}.csv")
# print(synergy_sweep_6p.columns)
# synergy_sweep_6p['diff_ratio_X_Y'] = synergy_sweep_6p.iloc[:, 0] / synergy_sweep_6p.iloc[:, 1]  # Ratio of diffusion coefficients
# synergy_sweep_6p['pulse_ratio_X_Y'] = synergy_sweep_6p.iloc[:, 2] / synergy_sweep_6p.iloc[:, 3]  # Ratio of pulse periods
# synergy_sweep_6p['membrane_ratio_X_Y'] = synergy_sweep_6p.iloc[:, 4] / synergy_sweep_6p.iloc[:, 5]  # Ratio of membrane lengths

# # Keep all relevant columns instead of just two
# synergy_sweep_6p = synergy_sweep_6p[['FINAL_NUMBER_OF_ALIVE_CELLS', 'diff_ratio_X_Y', 
#                                      'pulse_ratio_X_Y', 'membrane_ratio_X_Y']]




# After your existing code that creates the ratios
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

# # Calculate how close each point is to the positive control
# # Lower values mean closer to the positive control
# synergy_sweep_6p['distance_to_positive_control'] = abs(synergy_sweep_6p['FINAL_NUMBER_OF_ALIVE_CELLS'] - 
#                                                     mean_final_number_of_alive_cells_positive_control)

# # Normalize this distance for better color mapping (0 = closest to positive control)
# max_distance = synergy_sweep_6p['distance_to_positive_control'].max()
# synergy_sweep_6p['normalized_closeness'] = 1 - (synergy_sweep_6p['distance_to_positive_control'] / max_distance)

# # Create 3D plot
# fig = plt.figure(figsize=(12, 10), dpi=300)
# ax = fig.add_subplot(111, projection='3d')

# # Create scatter plot with color mapping
# scatter = ax.scatter(synergy_sweep_6p['diff_ratio_X_Y'], 
#                      synergy_sweep_6p['pulse_ratio_X_Y'],
#                      synergy_sweep_6p['membrane_ratio_X_Y'],
#                      c=synergy_sweep_6p['normalized_closeness'],
#                      cmap='viridis',  # Use a colormap: viridis, plasma, inferno, magma, etc.
#                      s=50,            # Point size
#                      alpha=0.8,       # Transparency
#                      edgecolor='black',
#                      linewidth=0.5)

# # Add colorbar
# cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
# cbar.set_label('Similarity to Positive Control\n(1=perfect match)', rotation=270, labelpad=20, fontsize=12)

# # Add labels
# ax.set_xlabel('Diffusion Ratio (X/Y)', fontsize=12, fontweight='bold')
# ax.set_ylabel('Pulse Period Ratio (X/Y)', fontsize=12, fontweight='bold')
# ax.set_zlabel('Membrane Length Ratio (X/Y)', fontsize=12, fontweight='bold')

# # Add a title
# plt.title('Parameter Space Exploration for Synergistic Effects', fontsize=14, fontweight='bold', pad=20)

# # Add a reference point for the positive control (for context)
# ax.text2D(0.05, 0.95, f"Positive Control Cell Count: {mean_final_number_of_alive_cells_positive_control}", 
#           transform=ax.transAxes, fontsize=10)

# # Improve visualization
# ax.grid(True)
# ax.view_init(elev=30, azim=45)  # Set viewing angle

# # Save the figure
# script_dir = os.path.dirname(os.path.abspath(__file__))
# save_path = f"{script_dir}/synergy_plots_3D"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# plt.savefig(f"{save_path}/3D_parameter_space_{synergy_sweep_6p_name}.png")

# plt.tight_layout()
# plt.show()

# # Optional: Create 2D projections for easier interpretation
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

# # Plot each pair of ratios
# scatter1 = axs[0].scatter(synergy_sweep_6p['diff_ratio_X_Y'], 
#                          synergy_sweep_6p['pulse_ratio_X_Y'],
#                          c=synergy_sweep_6p['normalized_closeness'],
#                          cmap='viridis', s=70, alpha=0.8)
# axs[0].set_xlabel('Diffusion Ratio (X/Y)')
# axs[0].set_ylabel('Pulse Period Ratio (X/Y)')
# axs[0].grid(True)

# scatter2 = axs[1].scatter(synergy_sweep_6p['diff_ratio_X_Y'], 
#                          synergy_sweep_6p['membrane_ratio_X_Y'],
#                          c=synergy_sweep_6p['normalized_closeness'],
#                          cmap='viridis', s=70, alpha=0.8)
# axs[1].set_xlabel('Diffusion Ratio (X/Y)')
# axs[1].set_ylabel('Membrane Length Ratio (X/Y)')
# axs[1].grid(True)

# scatter3 = axs[2].scatter(synergy_sweep_6p['pulse_ratio_X_Y'], 
#                          synergy_sweep_6p['membrane_ratio_X_Y'],
#                          c=synergy_sweep_6p['normalized_closeness'],
#                          cmap='viridis', s=70, alpha=0.8)
# axs[2].set_xlabel('Pulse Period Ratio (X/Y)')
# axs[2].set_ylabel('Membrane Length Ratio (X/Y)')
# axs[2].grid(True)

# # Add a colorbar
# cbar = fig.colorbar(scatter1, ax=axs, pad=0.01)
# cbar.set_label('Similarity to Positive Control', rotation=270, labelpad=20)

# plt.tight_layout()
# plt.savefig(f"{save_path}/2D_projections_{synergy_sweep_6p_name}.png")
# plt.show()

# ########################################################
# # FOR PI3K-MEK SYNERGY SWEEP
# ########################################################

# nrows = ncols = 4  # 16 subplots (4x4 grid)
# fig, axes = plt.subplots(nrows, ncols, figsize=(16, 16), sharex=True, sharey=True)  # smaller figure

# for idx, (x_diff, y_diff) in enumerate(diff_combos):
#     row = idx // ncols
#     col = idx % ncols
#     ax = axes[row, col]

#     # Find all instances for this diffusion combo
#     combo_instances = alive_cells_df[
#         (alive_cells_df["drug_X_diffusion_coefficient"] == x_diff) &
#         (alive_cells_df["drug_Y_diffusion_coefficient"] == y_diff)
#     ]

#     # Collect all growth curves for this combo
#     growth_curves = []
#     time_points = None
#     for _, instance in combo_instances.iterrows():
#         instance_folder = f"instance_{instance['individual']}_{instance['replicate']}"
#         instance_folder_path = os.path.join(experiment_folder_path, instance_folder)
#         growth_csv_path = os.path.join(instance_folder_path, "simulation_growth.csv")
#         growth_df = pd.read_csv(growth_csv_path)
#         growth_df = growth_df.iloc[1:]  # skip first row
#         if time_points is None:
#             time_points = growth_df["time"].values
#         growth_curves.append(growth_df["alive"].values)

#     if growth_curves:
#         growth_curves = np.array(growth_curves)
#         mean_curve = np.mean(growth_curves, axis=0)
#         std_curve = np.std(growth_curves, axis=0)
#         ax.plot(time_points, mean_curve, color='blue', label='Mean')
#         ax.fill_between(time_points, mean_curve - std_curve, mean_curve + std_curve, color='blue', alpha=0.3, label='Std')
#         ax.set_title(f"X: {x_diff}, Y: {y_diff}", fontsize=14)
#         ax.tick_params(labelsize=12)
#     else:
#         ax.set_visible(False)

# # Add shared axis labels
# fig.text(0.5, 0.04, "Simulation Time (min)", ha='center', va='center', fontsize=18)
# fig.text(0.04, 0.5, "Number of alive cells", ha='center', va='center', rotation='vertical', fontsize=18)

# plt.suptitle("Averaged Growth Curves by Drug Diffusion Combination", y=1.01, fontsize=20)
# plt.tight_layout(rect=[0.07, 0.06, 1, 0.97])  # leave space for axis labels and title

# save_dir = os.path.join(os.path.dirname(__file__), "min_alive_cells_sweep_results")
# os.makedirs(save_dir, exist_ok=True)
# plt.savefig(os.path.join(save_dir, f"{experiment_name}_growth_curves_by_diffusion.png"))
# print(f"Saved growth curves to: {os.path.join(save_dir, f'{experiment_name}_growth_curves_by_diffusion.png')}")
# plt.show()

