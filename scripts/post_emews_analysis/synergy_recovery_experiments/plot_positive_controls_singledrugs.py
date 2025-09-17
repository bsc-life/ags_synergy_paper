import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# IMPORTANT: Use Agg backend for headless environments
import matplotlib
matplotlib.use('Agg')

# REMOVE scienceplots dependency - it requires LaTeX which isn't available
# import scienceplots
# plt.style.use(['science', 'nature'])

# Use minimal, server-friendly settings
plt.style.use('default')  # Reset to default style
plt.rcParams.update({
    # Use fonts that are always available on servers
    'font.family': 'DejaVu Sans',
    'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'FreeSans', 'sans-serif'],
    
    # Disable LaTeX rendering
    'text.usetex': False,
    
    # Basic styling
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    
    # Figure settings
    'figure.figsize': (4.5, 4.0),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',  # Use PNG instead of PDF on servers
})

# Experiment names
nodrug_experiment_name = "synergy_sweep-3D-3004-0121-control_nodrug"
pi3k_experiment_name = "synergy_sweep-pi3k_mek-3D-0505-0218-logscale_singledrug_pi3k"
mek_experiment_name = "synergy_sweep-pi3k_mek-3D-0505-0218-logscale_singledrug_mek"
akt_experiment_name = "synergy_sweep-akt_mek-3D-0505-1910-logscale_singledrug_akt"
# Add the combined drug experiment for AKT+MEK synergy at D=600
akt_mek_combined_experiment_name = "synergy_sweep-akt_mek-1606-0214-4p_3D_drugtiming"
# Add the combined drug experiment for PI3K+MEK synergy at D=600 from the timing experiment
pi3k_mek_combined_experiment_name = "synergy_sweep-pi3k_mek-1606-0214-4p_3D_drugtiming"

pi3k_experiment_folder_path = f"experiments/{pi3k_experiment_name}"
mek_experiment_folder_path = f"experiments/{mek_experiment_name}"
akt_experiment_folder_path = f"experiments/{akt_experiment_name}"
akt_mek_combined_folder_path = f"experiments/{akt_mek_combined_experiment_name}"
pi3k_mek_combined_folder_path = f"experiments/{pi3k_mek_combined_experiment_name}"

# Dictionary to store growth curves by diffusion coefficient for each drug
pi3k_growth_by_diffusion = defaultdict(list)
mek_growth_by_diffusion = defaultdict(list)
akt_growth_by_diffusion = defaultdict(list)
# Add a dictionary for the combined experiment data
akt_mek_combined_growth_by_diffusion = defaultdict(list)
pi3k_mek_combined_growth_by_diffusion = defaultdict(list)

def process_experiment(experiment_folder_path, growth_by_diffusion, is_control=False):
    """Process a single drug experiment and store growth curves by diffusion coefficient"""
    print(f"Processing experiment: {os.path.basename(experiment_folder_path)}")
     
    for instance_folder in os.listdir(experiment_folder_path):
        if instance_folder.startswith("instance_"):
            instance_folder_path = os.path.join(experiment_folder_path, instance_folder)
            instance_info = instance_folder.split("instance_")[-1]
            replicate = instance_info.split("_")[-1]

            drug_diffusion_coefficient = 0.0 # Default for control case

            if not is_control:
                # For drug experiments, read the diffusion coefficient from the summary file
                summary_json_path = os.path.join(instance_folder_path, "sim_summary.json")
                try:
                    with open(summary_json_path, 'r') as f:
                        summary_dict = json.load(f)
                    drug_diffusion_coefficient = summary_dict["user_parameters.drug_X_diffusion_coefficient"]
                except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                    print(f"  Could not read diffusion coefficient for {instance_folder}, skipping: {e}")
                    continue
            
            # Read growth curve data
            growth_csv_path = os.path.join(instance_folder_path, "simulation_growth.csv")
            if os.path.exists(growth_csv_path):
                try:
                    growth_df = pd.read_csv(growth_csv_path)
                    # Store the growth curve data
                    growth_by_diffusion[drug_diffusion_coefficient].append({
                        'time': growth_df['time'].values,
                        'cells': growth_df['alive'].values,
                        'replicate': replicate
                    })
                    print(f"  Added growth curve for diffusion coefficient {drug_diffusion_coefficient}, replicate {replicate}")
                except Exception as e:
                    print(f"  Error reading growth CSV {growth_csv_path}: {e}")
            else:
                print(f"  Growth CSV not found: {growth_csv_path}")

def process_combined_experiment(experiment_folder_path, growth_by_diffusion, target_diff, is_timing_exp=False):
    """
    Process a combined drug experiment, filtering for specific conditions.
    - For all experiments, filters for a symmetric diffusion coefficient (target_diff).
    - If is_timing_exp is True, it also filters for EARLY simultaneous drug additions (delta_time=0 and low pulse period).
    """
    print(f"Processing combined experiment: {os.path.basename(experiment_folder_path)} for D={target_diff}")
     
    if not os.path.exists(experiment_folder_path):
        print(f"  WARNING: Experiment folder not found at {experiment_folder_path}. Skipping.")
        return

    for instance_folder in os.listdir(experiment_folder_path):
        if instance_folder.startswith("instance_"):
            instance_folder_path = os.path.join(experiment_folder_path, instance_folder)
            
            summary_json_path = os.path.join(instance_folder_path, "sim_summary.json")
            try:
                with open(summary_json_path, 'r') as f:
                    summary_dict = json.load(f)
                
                x_diff = summary_dict.get("user_parameters.drug_X_diffusion_coefficient")
                y_diff = summary_dict.get("user_parameters.drug_Y_diffusion_coefficient")
                
                # Condition 1: Must be the symmetric scenario we want
                if x_diff != target_diff or y_diff != target_diff:
                    continue

                # Condition 2: If it's a timing experiment, ensure it's an EARLY simultaneous addition
                if is_timing_exp:
                    x_pulse = summary_dict.get("user_parameters.drug_X_pulse_period", -1)
                    y_pulse = summary_dict.get("user_parameters.drug_Y_pulse_period", -2) # Use different defaults to fail inequality
                    
                    # We want simultaneous (x_pulse == y_pulse) and early (x_pulse < 1000)
                    if not (x_pulse == y_pulse and x_pulse < 1000):
                        continue
                
                # If all conditions pass, read growth curve data
                growth_csv_path = os.path.join(instance_folder_path, "simulation_growth.csv")
                if os.path.exists(growth_csv_path):
                    growth_df = pd.read_csv(growth_csv_path)
                    growth_by_diffusion[target_diff].append({
                        'time': growth_df['time'].values,
                        'cells': growth_df['alive'].values,
                        'replicate': instance_folder.split("instance_")[-1]
                    })
                    print(f"  Added valid combined growth curve for D={target_diff}, from {instance_folder}")

            except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                print(f"  Could not process {instance_folder}, skipping: {e}")
                continue

# Process drug experiments
process_experiment(pi3k_experiment_folder_path, pi3k_growth_by_diffusion)
process_experiment(mek_experiment_folder_path, mek_growth_by_diffusion)
process_experiment(akt_experiment_folder_path, akt_growth_by_diffusion)
# Process the combined experiments with correct filtering
# BOTH are now timing experiments and need the appropriate flag
process_combined_experiment(akt_mek_combined_folder_path, akt_mek_combined_growth_by_diffusion, target_diff=600.0, is_timing_exp=True)
process_combined_experiment(pi3k_mek_combined_folder_path, pi3k_mek_combined_growth_by_diffusion, target_diff=600.0, is_timing_exp=True)

# Process no-drug experiment explicitly as a control
nodrug_experiment_folder_path = f"experiments/{nodrug_experiment_name}"
nodrug_growth_by_diffusion = defaultdict(list)
process_experiment(nodrug_experiment_folder_path, nodrug_growth_by_diffusion, is_control=True)

def plot_combined_growth_curves(growth_by_diffusion, drug_name, save_path):
    """Plot all growth curves in a single figure, colored by diffusion coefficient"""
    diffusion_values = sorted(growth_by_diffusion.keys())
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    
    # Use standard colors that work across all environments
    colors = ['blue', 'green', 'red', 'purple']
    if len(diffusion_values) > 4:
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(diffusion_values)))
    
    # Store handles for legend
    handles = []
    
    # Plot mean curves for each diffusion coefficient
    for i, diff_coef in enumerate(diffusion_values):
        curves = growth_by_diffusion[diff_coef]
        
        if curves:
            # Find the common time points across all replicates
            min_length = min(len(curve['time']) for curve in curves)
            mean_cells = np.mean([curve['cells'][:min_length] for curve in curves], axis=0)
            std_cells = np.std([curve['cells'][:min_length] for curve in curves], axis=0)
            time_points = curves[0]['time'][:min_length]
            
            # Format diffusion coefficient display
            if diff_coef >= 1000:
                diff_label = f"{int(diff_coef/1000)}k"
            else:
                diff_label = f"{int(diff_coef)}"
            
            # Plot mean curve with thicker line
            line, = ax.plot(
                time_points,
                mean_cells,
                color=colors[i % len(colors)],
                linewidth=1.5,
                label=f"{diff_label}"
            )
            handles.append(line)
            
            # Add shaded area for standard deviation
            ax.fill_between(
                time_points,
                mean_cells - std_cells,
                mean_cells + std_cells,
                color=colors[i % len(colors)],
                alpha=0.15
            )
    
    # Add labels
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Cell Count")
    
    # Add vertical line at drug administration time
    ax.axvline(x=0, color='red', linestyle='-', linewidth=1, alpha=0.7)
    ax.text(50, ax.get_ylim()[1]*0.95, "Drug addition", color='red', fontsize=8, va='top')
    
    # Create legend
    ax.legend(handles=handles, 
              title="Diffusion Coefficient (μm²/min)",
              frameon=True,
              loc='best', 
              fontsize=8)
    
    # Add title instead of panel label
    ax.set_title(f"{drug_name} Inhibitor Growth Curves", fontsize=11)
    
    # IMPORTANT: Use plt.tight_layout() with no arguments
    # This avoids the LaTeX error
    plt.tight_layout()
    
    # Save with appropriate settings for SSH environment
    save_path_svg = save_path.replace('.png', '.svg')
    plt.savefig(f"{save_path}", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path_svg}", dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")
    plt.close()

# Create the output directory
save_dir = os.path.join(os.path.dirname(__file__), "positive_controls_singledrugs_results")
os.makedirs(save_dir, exist_ok=True)

# Plot and save the combined growth curves for each drug
plot_combined_growth_curves(
    pi3k_growth_by_diffusion, 
    "PI3K", 
    os.path.join(save_dir, "pi3k_combined_growth_curves.png")
)

plot_combined_growth_curves(
    mek_growth_by_diffusion, 
    "MEK", 
    os.path.join(save_dir, "mek_combined_growth_curves.png")
)

plot_combined_growth_curves(
    akt_growth_by_diffusion, 
    "AKT", 
    os.path.join(save_dir, "akt_combined_growth_curves.png")
)

# Now add a call to plot the no-drug data individually
plot_combined_growth_curves(
    nodrug_growth_by_diffusion, 
    "No Drug", 
    os.path.join(save_dir, "nodrug_combined_growth_curves.png")
)

print("Plotting complete!")

# For combined panel figure, use similar SSH-friendly settings
def create_combined_panel_figure(save_dir):
    """Create a multi-panel figure with Q1 systems biology journal styling"""
    
    # Use a standard color palette inspired by Cell Systems
    # These are SSH-safe colors that closely match journal aesthetics
    palette = {
        0.0: '#000000',      # black for no-drug
        6.0: '#4472C4',      # blue
        60.0: '#70AD47',     # green
        600.0: '#ED7D31',    # orange
        6000.0: '#9E480E'    # brown
    }
    
    # Create figure with wider aspect ratio for 4 panels instead of 3
    # Add sharey=True to share Y axis across all panels
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    
    # Add gray background for that Cell Systems look
    fig.patch.set_facecolor('#F5F5F5')
    
    # Include No Drug as the first panel
    drug_names = ['No Drug', 'PI3K', 'MEK', 'AKT']
    drug_data = [nodrug_growth_by_diffusion, pi3k_growth_by_diffusion, mek_growth_by_diffusion, akt_growth_by_diffusion]
    panel_labels = ['A', 'B', 'C', 'D']  # Updated to include no-drug panel
    
    # Find the global min and max for all cell counts to ensure consistent y-limit if needed
    global_min_cells = float('inf')
    global_max_cells = 0
    
    for growth_data in drug_data:
        for diff_coef in growth_data:
            curves = growth_data[diff_coef]
            for curve in curves:
                min_cells = np.min(curve['cells'])
                max_cells = np.max(curve['cells'])
                global_min_cells = min(global_min_cells, min_cells)
                global_max_cells = max(global_max_cells, max_cells)
    
    # Add 10% padding to the top for better visualization
    global_max_cells *= 1.1
    
    for i, (drug_name, growth_data) in enumerate(zip(drug_names, drug_data)):
        ax = axes[i]
        ax.set_facecolor('#F9F9F9')  # Lighter gray for plot area
        diffusion_values = sorted(growth_data.keys())
        
        # Add gridlines (common in systems biology journals)
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Make spines (borders) thinner and lighter
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('#888888')
        
        # Ensure all plots have the same y-limits
        ax.set_ylim(0, global_max_cells)
        
        # Draw curves - for no-drug, there might be only one curve
        for j, diff_coef in enumerate(diffusion_values):
            curves = growth_data[diff_coef]
            
            if curves:
                # Calculate mean and std
                min_length = min(len(curve['time']) for curve in curves)
                mean_cells = np.mean([curve['cells'][:min_length] for curve in curves], axis=0)
                std_cells = np.std([curve['cells'][:min_length] for curve in curves], axis=0)
                time_points = curves[0]['time'][:min_length]
                
                # Format diffusion coefficient display for legend
                # Skip diffusion labels for no-drug panel
                if drug_name == "No Drug":
                    diff_label = "Control"
                elif diff_coef >= 1000:
                    diff_label = f"{int(diff_coef/1000)}k"
                else:
                    diff_label = f"{int(diff_coef)}"
                
                # Plot mean curve with more professional styling
                line, = ax.plot(
                    time_points,
                    mean_cells,
                    color=palette.get(diff_coef, 'black'),  # Default to black if color not in palette
                    linewidth=2.0,
                    label=f"{diff_label}"
                )
                
                # Add shaded area for standard deviation with more subtle styling
                ax.fill_between(
                    time_points,
                    mean_cells - std_cells,
                    mean_cells + std_cells,
                    color=palette.get(diff_coef, 'black'),
                    alpha=0.12
                )
        
        # Add vertical line at drug administration time (except for no-drug panel)
        if drug_name != "No Drug":
            ax.axvline(x=0, color='#D62728', linestyle='-', linewidth=1, alpha=0.7)
        
        # Label axes in systems biology style (first plot has y-label)
        if i == 0:
            ax.set_ylabel("Cell Count", fontsize=10, fontweight='bold')
        
        ax.set_xlabel("Time (min)", fontsize=10)
        
        # Add panel label in top-left with Cell Systems style
        ax.text(-0.18, 1.05, panel_labels[i], transform=ax.transAxes, 
                fontsize=14, fontweight='bold')
        
        # Add drug name as title with cleaner typography
        title_text = "Control" if drug_name == "No Drug" else f"{drug_name} Inhibitor"
        ax.set_title(title_text, fontsize=11, pad=10)
        
        # Professional tick styling
        ax.tick_params(direction='out', length=4, width=0.8, colors='#444444')
        
        # Special case for no-drug panel - add legend if it's the only curve
        if drug_name == "No Drug":
            ax.legend(loc='upper left', frameon=True, fontsize=9)
    
    # Create a shared legend for diffusion coefficients (exclude from no-drug panel)
    # Get handles and labels from the last panel (AKT)
    handles, labels = axes[-1].get_legend_handles_labels()
    
    # Only show legend if we have diffusion data
    if handles:
        legend_labels = [f"{label} μm²/min" for label in labels]
        fig.legend(handles, legend_labels, 
                loc='upper center', 
                bbox_to_anchor=(0.5, 0.10),  # Centered and with space above
                ncol=4,
                frameon=True,
                fontsize=9,
                title="Diffusion Coefficient")
    
    # Apply tight layout with carefully chosen spacing - more room for legend
    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.25, wspace=0.18)
    
    # Save the multi-panel figure
    multi_panel_path = os.path.join(save_dir, "combined_drugs_panel_figure.png")
    # Save to PDF for publication
    multi_panel_pdf_path = os.path.join(save_dir, "FIG_suppmat_singledrug_effect_diff_coeff.pdf")
    multi_panel_svg_path = os.path.join(save_dir, "FIG_suppmat_singledrug_effect_diff_coeff.svg")
    plt.savefig(multi_panel_pdf_path, dpi=400, bbox_inches='tight')
    plt.savefig(multi_panel_path, dpi=400, bbox_inches='tight')
    plt.savefig(multi_panel_svg_path, dpi=400, bbox_inches='tight')
    
    print(f"Saved multi-panel figure to: {multi_panel_path}")
    plt.close()

# Additionally create a combined panel figure
create_combined_panel_figure(save_dir)

def plot_control_vs_treatment_curves(nodrug_data, pi3k_data, mek_data, akt_data, akt_mek_combined_data, pi3k_mek_combined_data, save_dir):
    """
    Creates a single-panel plot comparing the no-drug control against
    the three single-drug treatments and the combined treatment at a diffusion coefficient of 600.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    data_to_plot = {
        'Control': nodrug_data.get(0.0, []),
        'PI3K Inhibitor': pi3k_data.get(600.0, []),
        'MEK Inhibitor': mek_data.get(600.0, []),
        'AKT Inhibitor': akt_data.get(600.0, []),
        'AKT + MEK': akt_mek_combined_data.get(600.0, []),
        'PI3K + MEK': pi3k_mek_combined_data.get(600.0, [])
    }

    colors = {
        'Control': '#000000',
        'PI3K Inhibitor': '#4472C4',
        'MEK Inhibitor': '#ED7D31',
        'AKT Inhibitor': '#70AD47',
        'AKT + MEK': '#5E3C99',  # A distinct purple for the combination
        'PI3K + MEK': '#C44E52'  # A distinct red/crimson for the other combo
    }

    max_y = 0
    
    for label, curves in data_to_plot.items():
        if not curves:
            print(f"Warning: No data for '{label}' in control vs treatment plot.")
            continue

        min_length = min(len(curve['time']) for curve in curves)
        mean_cells = np.mean([curve['cells'][:min_length] for curve in curves], axis=0)
        std_cells = np.std([curve['cells'][:min_length] for curve in curves], axis=0)
        time_points = curves[0]['time'][:min_length]

        max_y = max(max_y, np.max(mean_cells + std_cells))

        ax.plot(
            time_points,
            mean_cells,
            color=colors.get(label, 'black'),
            linewidth=2.0,
            label=label
        )
        
        ax.fill_between(
            time_points,
            mean_cells - std_cells,
            mean_cells + std_cells,
            color=colors.get(label, 'black'),
            alpha=0.12
        )

    ax.set_ylim(0, max_y * 1.1)
    ax.axvline(x=0, color='#D62728', linestyle='-', linewidth=1, alpha=0.7)
    
    ax.set_xlabel("Time (min)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Cell Count", fontsize=10, fontweight='bold')
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=9)
    ax.tick_params(axis='both', width=0.8, length=2, labelsize=9, colors='black')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_png = os.path.join(save_dir, "control_vs_treatment_comparison.png")
    output_pdf = os.path.join(save_dir, "FIG_suppmat_control_vs_treatment_comparison.pdf")
    output_svg = os.path.join(save_dir, "control_vs_treatment_comparison.svg")
    plt.savefig(output_png, dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(output_svg, dpi=300, bbox_inches='tight', transparent=True)
    
    print(f"Saved control vs treatment plot to: {output_png}")
    plt.close()

# Create the new control vs treatment plot
plot_control_vs_treatment_curves(
    nodrug_growth_by_diffusion,
    pi3k_growth_by_diffusion,
    mek_growth_by_diffusion,
    akt_growth_by_diffusion,
    akt_mek_combined_growth_by_diffusion,
    pi3k_mek_combined_growth_by_diffusion,
    save_dir
)