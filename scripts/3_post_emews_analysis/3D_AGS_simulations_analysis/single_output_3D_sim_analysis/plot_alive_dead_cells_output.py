import pcdl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# output directory is two directories up
output_folder_name = "output_drugY_test"
output_dir = f"/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/model/PhysiBoSS/{output_folder_name}"

mcdsts = pcdl.pyMCDSts(output_path = output_dir, graph=False, verbose=False)

list_of_relevant_vars = list()
all_data = pd.DataFrame()
for mcds in mcdsts.get_mcds_list():
    frame_df = mcds.get_cell_df()
    frame_df.reset_index(inplace=True)
    list_of_relevant_vars.append(frame_df)

all_data = pd.concat(list_of_relevant_vars, ignore_index=True)

### Plot 1: Alive and dead cells over time ###
# explore columns from all_data with "current_phase" in the name
current_phase_data = all_data[['time', 'ID'] + [col for col in all_data.columns if 'current_phase' in col]]

# count the number of alive cells over time
alive_cells = current_phase_data[current_phase_data['current_phase'] == "live"]
apoptotic_cells = current_phase_data[current_phase_data['current_phase'] == "apoptotic"]
necrotic_cells = current_phase_data[current_phase_data['current_phase'] == "necrotic"]

# aggregate by time and count the number of alive cells
alive_cells_agg = alive_cells.groupby('time').count()
apoptotic_cells_agg = apoptotic_cells.groupby('time').count()
necrotic_cells_agg = necrotic_cells.groupby('time').count()

# Create a large figure with subplots (3x3 grid)
plt.figure(figsize=(24, 20))

# Get S_anti and S_pro data
S_anti_pro_data = all_data[['time', 'ID', 'S_anti_extended', 'S_pro_extended']]

# Plot 1: Alive and dead cells (top left)
plt.subplot(3, 3, 1)
sns.lineplot(x='time', y='ID', data=alive_cells_agg)
sns.scatterplot(x='time', y='ID', data=alive_cells_agg, label='alive')
sns.lineplot(x='time', y='ID', data=apoptotic_cells_agg)
sns.scatterplot(x='time', y='ID', data=apoptotic_cells_agg, label='apoptotic')
sns.lineplot(x='time', y='ID', data=necrotic_cells_agg, label='necrotic')
sns.scatterplot(x='time', y='ID', data=necrotic_cells_agg, label='necrotic')
plt.axvspan(1280, 1292, color='gray', alpha=0.2, label='Treatment window')
plt.xlabel('Simulation time (min)', fontsize=12)
plt.ylabel('Number of cells', fontsize=12)
plt.title('Cell Population Dynamics', fontsize=14)
plt.legend()
plt.xlim(0, 4200)

### Plot 2: Apoptosis node states analysis ###
# subset all_data to only include the "node" columns + time
node_data = all_data[['time'] + [col for col in all_data.columns if 'node' in col]]

# First identify the base names that have _x and _y suffixes
base_names = []
for col in node_data.columns:
    if col.endswith('_x'):
        base_name = col[:-2]  # Remove '_x'
        if f"{base_name}_y" in node_data.columns:
            base_names.append(base_name)

# First convert float columns to boolean, then perform logical OR
for base_name in base_names:
    # Convert float columns to boolean (True if > 0, False otherwise)
    x_col = node_data[f"{base_name}_x"].fillna(0).astype(bool)
    y_col = node_data[f"{base_name}_y"].fillna(0).astype(bool)
    
    # Perform logical OR and convert back to int
    node_data[base_name] = (x_col | y_col).astype(int)
    
    # Drop the original _x and _y columns
    node_data = node_data.drop([f"{base_name}_x", f"{base_name}_y"], axis=1)

print("Node columns:", node_data.columns)

# Plot 2: S_anti trajectories (top middle)
plt.subplot(3, 3, 2)
sns.lineplot(data=S_anti_pro_data, 
            x='time', 
            y='S_anti_extended',
            units='ID',
            estimator=None,
            alpha=0.3,
            color='blue')
sns.lineplot(data=S_anti_pro_data,
            x='time',
            y='S_anti_extended',
            color='red',
            linewidth=1,
            label='Population average')
plt.axvspan(1280, 1292, color='gray', alpha=0.5, label='Treatment window')
plt.xlabel('Simulation time (min)', fontsize=12)
plt.ylabel('S_anti', fontsize=12)
plt.title('Anti-survival Signal', fontsize=14)
plt.ylim(0, 1)
plt.legend()

# Plot 3: S_pro trajectories (top right)
plt.subplot(3, 3, 3)
sns.lineplot(data=S_anti_pro_data, 
            x='time', 
            y='S_pro_extended',
            units='ID',
            estimator=None,
            alpha=0.3,
            color='blue')
sns.lineplot(data=S_anti_pro_data,
            x='time',
            y='S_pro_extended',
            color='red',
            linewidth=1,
            label='Population average')
plt.axvspan(1280, 1292, color='gray', alpha=0.5, label='Treatment window')
plt.xlabel('Simulation time (min)', fontsize=12)
plt.ylabel('S_pro', fontsize=12)
plt.title('Pro-survival Signal', fontsize=14)
plt.ylim(0, 1)
plt.legend()

### Plot 4: Growth node states analysis ###
# subset all_data to only include the growth node columns + time (excluding drug_X)
growth_node_data = all_data[['time'] + [col for col in all_data.columns if 'node' in col and not 'drug_X' in col]]

# First identify the base names that have _x and _y suffixes
base_names = []
for col in growth_node_data.columns:
    if col.endswith('_x'):
        base_name = col[:-2]  # Remove '_x'
        if f"{base_name}_y" in growth_node_data.columns:
            base_names.append(base_name)

# First convert float columns to boolean, then perform logical OR
for base_name in base_names:
    # Convert float columns to boolean (True if > 0, False otherwise)
    x_col = growth_node_data[f"{base_name}_x"].fillna(0).astype(bool)
    y_col = growth_node_data[f"{base_name}_y"].fillna(0).astype(bool)
    
    # Perform logical OR and convert back to int
    growth_node_data[base_name] = (x_col | y_col).astype(int)
    
    # Drop the original _x and _y columns
    growth_node_data = growth_node_data.drop([f"{base_name}_x", f"{base_name}_y"], axis=1)

print("Growth node columns:", growth_node_data.columns)

# Plot 4: Apoptosis nodes (middle left)
plt.subplot(3, 3, 4)
sns.lineplot(x='time', y='node_FOXO', data=node_data, label='FOXO')
sns.lineplot(x='time', y='node_Caspase8', data=node_data, label='Caspase8')
sns.lineplot(x='time', y='node_Caspase9', data=node_data, label='Caspase9')
sns.lineplot(x='time', y='anti_pi3k_node', data=node_data, label='anti_PI3K')
sns.lineplot(x='time', y='anti_mek_node', data=node_data, label='anti_MEK')
sns.lineplot(x='time', y='anti_akt_node', data=node_data, label='anti_AKT')
plt.axvspan(1280, 1292, color='gray', alpha=0.6)
plt.ylim(0, 1)
plt.xlabel('Simulation time (min)', fontsize=12)
plt.ylabel('Node State', fontsize=12)
plt.title('Apoptosis Node States', fontsize=14)
plt.legend()

# Plot 5: Growth nodes (middle middle)
plt.subplot(3, 3, 5)
sns.lineplot(x='time', y='node_cMYC', data=growth_node_data, label='cMYC')
sns.lineplot(x='time', y='node_TCF', data=growth_node_data, label='TCF')
sns.lineplot(x='time', y='node_RSK', data=growth_node_data, label='RSK')
sns.lineplot(x='time', y='pi3k_node', data=growth_node_data, label='PI3K')
sns.lineplot(x='time', y='mek_node', data=growth_node_data, label='MEK')
sns.lineplot(x='time', y='akt_node', data=growth_node_data, label='AKT')
plt.axvspan(1280, 1292, color='gray', alpha=0.6)
plt.ylim(0, 1)
plt.xlabel('Simulation time (min)', fontsize=12)
plt.ylabel('Node State', fontsize=12)
plt.title('Growth Node States', fontsize=14)
plt.legend()

### Plot 5: Apoptosis and Growth rates ###
# Get apoptosis and growth data
apoptosis_data = all_data[['time', 'apoptosis_rate', 'max_apoptosis_rate', 'apoptosis_rate_basal']]
growth_data = all_data[['time', 'growth_rate', 'basal_growth_rate']]

# Plot 6: Apoptosis rate (middle right)
plt.subplot(3, 3, 6)
sns.lineplot(x='time', y='apoptosis_rate', data=apoptosis_data, label='apoptosis_rate')
plt.axvspan(1280, 1292, color='gray', alpha=0.2, label='Treatment window')
plt.xlabel('Simulation time (min)', fontsize=12)
plt.ylabel('Apoptosis rate', fontsize=12)
plt.title('Apoptosis Rate', fontsize=14)
plt.ylim(apoptosis_data['apoptosis_rate_basal'].unique()[0], 
         apoptosis_data['max_apoptosis_rate'].unique()[0])
plt.legend()

# Plot 7: Growth rate (bottom left)
plt.subplot(3, 3, 7)
sns.lineplot(x='time', y='growth_rate', data=growth_data, label='growth_rate')
plt.axvspan(1280, 1292, color='gray', alpha=0.2, label='Treatment window')
plt.xlabel('Simulation time (min)', fontsize=12)
plt.ylabel('Growth rate', fontsize=12)
plt.title('Growth Rate', fontsize=14)
plt.legend()

### Plot 8: Drug-Target complex concentration ###
# Get drug-target complex data for both drugs
kinetic_data = all_data[['time', 'drug_X_DT_conc', 'drug_Y_DT_conc']]

# Plot 8: Drug-Target complex (bottom middle)
plt.subplot(3, 3, 8)
sns.lineplot(x='time', y='drug_X_DT_conc', data=kinetic_data, label='drug_X_DT_conc')
sns.lineplot(x='time', y='drug_Y_DT_conc', data=kinetic_data, label='drug_Y_DT_conc')
plt.axvspan(1280, 1292, color='gray', alpha=0.2, label='Treatment window')
plt.xlabel('Simulation time (min)', fontsize=12)
plt.ylabel('Drug-Target complex conc.', fontsize=12)
plt.title('Drug-Target Complex', fontsize=14)
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined plot
output_dir_combined = "/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/scripts/post_emews_analysis/single_output_analysis/results/combined_analysis"
os.makedirs(output_dir_combined, exist_ok=True)
plt.savefig(os.path.join(output_dir_combined, "combined_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved combined analysis plot to {output_dir_combined}/combined_analysis.png")

# Keep the individual plots as well
# Save the first plot
output_dir_cells = "/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/scripts/post_emews_analysis/single_output_analysis/results/alive_dead_cells_output"
os.makedirs(output_dir_cells, exist_ok=True)
plt.savefig(os.path.join(output_dir_cells, "alive_dead_cells_output.png"))
plt.close()

# Save the second plot
output_dir_nodes = "/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/scripts/post_emews_analysis/single_output_analysis/results/tmp_apoptosis_nodes"
os.makedirs(output_dir_nodes, exist_ok=True)
plt.savefig(f'{output_dir_nodes}/apoptosis_nodes.png')
plt.close()

# Save the third plot
output_dir_growth = "/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/scripts/post_emews_analysis/single_output_analysis/results/tmp_growth_nodes"
os.makedirs(output_dir_growth, exist_ok=True)
plt.savefig(f'{output_dir_growth}/growth_nodes.png')
plt.close()

# Save S_anti trajectories
plt.savefig(os.path.join(output_dir, "S_anti_trajectories.png"))
plt.close()

# Save S_pro trajectories
plt.savefig(os.path.join(output_dir, "S_pro_trajectories.png"))
plt.close()

# Save apoptosis rate plot
plt.savefig(os.path.join(output_dir, "apoptosis_rate.png"))
plt.close()

# Save growth rate plot
plt.savefig(os.path.join(output_dir, "growth_rate.png"))
plt.close()

# Save drug-target complex plot
plt.savefig(os.path.join(output_dir, "drug_target_complex.png"))
plt.close()

print(f"Saved cell count plot to {output_dir}/alive_dead_cells_output.png")
print(f"Saved apoptosis nodes plot to {output_dir}/apoptosis_nodes.png")
print(f"Saved growth nodes plot to {output_dir}/growth_nodes.png")
print(f"Saved S_anti trajectories to {output_dir}/S_anti_trajectories.png")
print(f"Saved S_pro trajectories to {output_dir}/S_pro_trajectories.png")
print(f"Saved apoptosis rate plot to {output_dir}/apoptosis_rate.png")
print(f"Saved growth rate plot to {output_dir}/growth_rate.png")
print(f"Saved drug-target complex plot to {output_dir}/drug_target_complex.png")

