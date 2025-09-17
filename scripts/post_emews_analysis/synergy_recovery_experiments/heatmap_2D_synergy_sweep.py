

# this script plots a heatmap of the synergy sweep for the 2D synergy sweep where we test different diffusion coefficients

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 


sweep_summaries_path = "results/sweep_summaries/"
synergy_sweep_2p_name = "synergy_sweep-akt_mek-2304-1404-2p_3D_drugaddition"
synergy_sweep_2p = pd.read_csv(f"{sweep_summaries_path}/final_summary_{synergy_sweep_2p_name}.csv")
# round values to 2 decimal places
synergy_sweep_2p = synergy_sweep_2p.round(2)
# synergy_sweep_2p['dx_dy_ratio'] = synergy_sweep_2p.iloc[:, 0] / synergy_sweep_2p.iloc[:, 1]

# create a heatmap of the synergy sweep, with the drug_X diffusion coefficient on the x-axis and the drug_Y diffusion coefficient on the y-axis


# Set style for publication-quality plot
# plt.style.use('seaborn-whitegrid')
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

# Create figure with publication dimensions
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# Create pivot table for heatmap
pivot_data = synergy_sweep_2p.pivot_table(
    values='FINAL_NUMBER_OF_ALIVE_CELLS',
    index=synergy_sweep_2p.iloc[:, 1],  # drug_Y diffusion coefficient
    columns=synergy_sweep_2p.iloc[:, 0],  # drug_X diffusion coefficient
    aggfunc='mean'
)

# Create heatmap with enhanced styling
sns.heatmap(pivot_data,
            cmap='RdYlBu_r',  # Red-Yellow-Blue reversed colormap
            annot=True,  # Show values
            # fmt='.0f',  # Format values as integers
            cbar_kws={'label': 'Final Cell Count'},
            square=True,  # Make cells square
            linewidths=0.1,  # Add grid lines
            linecolor='white')

# Customize axes
ax.set_xlabel('Drug X Diffusion Coefficient', fontsize=14, fontweight='bold')
ax.set_ylabel('Drug Y Diffusion Coefficient', fontsize=14, fontweight='bold')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add title
plt.title('Synergy Sweep Heatmap', fontsize=16, fontweight='bold')

# Save figure
plt.savefig(f'heatmap_{synergy_sweep_2p_name}.png', dpi=300, bbox_inches='tight')







