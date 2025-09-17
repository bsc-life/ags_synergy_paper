import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_drug_concentration_effects(gi50_conc=1.0):
    """
    Create a simplified plot showing drug concentration effects on cell fate
    for graphical abstract, maintaining linear dependencies
    """
    # Set up the plot with clean, minimal style and square aspect
    plt.figure(figsize=(4, 4), dpi=300)  # Made figure smaller
    sns.set_context("paper", font_scale=1.0)  # Reduced font scale
    sns.set_style("white")
    
    # Define key thresholds
    apoptosis_threshold = 5.0 * gi50_conc
    necrosis_threshold = 10.0 * gi50_conc
    max_apoptosis_rate = 0.01
    lethal_apoptosis_rate = 2.0 * max_apoptosis_rate
    max_necrosis_rate = 1.0
    
    # Generate concentration range
    drug_concentrations = np.linspace(0, 12 * gi50_conc, 1000)
    
    # Calculate death rates with linear transitions
    apoptosis_rates = np.zeros_like(drug_concentrations)
    necrosis_rates = np.zeros_like(drug_concentrations)
    
    for i, conc in enumerate(drug_concentrations):
        if conc >= necrosis_threshold:
            necrosis_rates[i] = max_necrosis_rate
            apoptosis_rates[i] = 0
        elif conc >= apoptosis_threshold:
            fraction = (conc - apoptosis_threshold) / (necrosis_threshold - apoptosis_threshold)
            necrosis_rates[i] = fraction * max_necrosis_rate
            apoptosis_rates[i] = lethal_apoptosis_rate
        elif conc > gi50_conc:
            fraction = (conc - gi50_conc) / (apoptosis_threshold - gi50_conc)
            apoptosis_rates[i] = max_apoptosis_rate + fraction * (lethal_apoptosis_rate - max_apoptosis_rate)
    
    # Create the plot
    ax = plt.gca()
    
    # Plot death rates with thicker lines
    plt.plot(drug_concentrations, apoptosis_rates, '-', color='#E41A1C', 
             label='Apoptosis', linewidth=3.0)  # Increased line width
    plt.plot(drug_concentrations, necrosis_rates, '-', color='#000000', 
             label='Necrosis', linewidth=3.0)  # Increased line width
    
    # Add key threshold lines with uniform color and increased width
    plt.axvline(x=gi50_conc, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    plt.axvline(x=apoptosis_threshold, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    plt.axvline(x=necrosis_threshold, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Simplified labels with smaller font
    plt.xlabel("Drug Concentration", fontsize=10, fontweight="bold")
    plt.ylabel("Cell Death Rate", fontsize=10, fontweight="bold")
    
    # Clean up the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.0)  # Thicker spine
    ax.spines['bottom'].set_linewidth(2.0)  # Thicker spine
    
    # Set axis limits and ticks
    plt.ylim(0, 1.1)
    plt.xlim(0, 12 * gi50_conc)
    plt.xticks([0, gi50_conc, apoptosis_threshold, necrosis_threshold], 
              ['0', 'GI50', '5×GI50', '10×GI50'], fontsize=9)
    plt.yticks(fontsize=9)
    
    # Add minimal legend with smaller font
    plt.legend(loc='upper left', frameon=False, fontsize=9)
    
    plt.tight_layout()
    
    # Save in high resolution
    plt.savefig("drug_concentration_effects.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig("drug_concentration_effects.svg", format='svg', bbox_inches='tight')
    
    # plt.show()
    
    return plt

def plot_drug_phenotype_schematic(gi50_conc=1.0):
    # Thresholds
    apoptosis_threshold = 5.0 * gi50_conc
    necrosis_threshold = 10.0 * gi50_conc

    # Plot setup
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=300)
    ax.set_xlim(0, 12 * gi50_conc)
    ax.set_ylim(0, 3)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(['Proliferation', 'Apoptosis', 'Necrosis'], fontsize=12, fontweight='bold')
    ax.set_xlabel('Drug Concentration', fontsize=12, fontweight='bold')
    ax.set_xticks([0, gi50_conc, apoptosis_threshold, necrosis_threshold, 12*gi50_conc])
    ax.set_xticklabels(['0', 'GI50', '5×GI50', '10×GI50', ''], fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Draw colored regions
    ax.axvspan(0, gi50_conc, ymin=0, ymax=1/3, color='#009E73', alpha=0.3, label='Proliferation')
    ax.axvspan(gi50_conc, apoptosis_threshold, ymin=1/3, ymax=2/3, color='#E41A1C', alpha=0.3, label='Apoptosis')
    ax.axvspan(apoptosis_threshold, necrosis_threshold, ymin=2/3, ymax=1, color='#000000', alpha=0.2, label='Necrosis')
    ax.axvspan(necrosis_threshold, 12*gi50_conc, ymin=2/3, ymax=1, color='#000000', alpha=0.5)

    # Add arrows or text for transitions
    ax.annotate('↑ Apoptosis', xy=(gi50_conc+1, 1.1), fontsize=10, color='#E41A1C', fontweight='bold')
    ax.annotate('↑ Necrosis', xy=(apoptosis_threshold+1, 2.1), fontsize=10, color='#000000', fontweight='bold')

    # Optional: vertical lines for thresholds
    for x in [gi50_conc, apoptosis_threshold, necrosis_threshold]:
        ax.axvline(x, color='gray', linestyle='--', linewidth=1.2)

    plt.tight_layout()
    plt.savefig("drug_phenotype_schematic.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig("drug_phenotype_schematic.svg", format='svg', bbox_inches='tight')
    plt.show()

def plot_drug_dual_yaxis_schematic(gi50_conc=1.0):
    # Thresholds
    apoptosis_threshold = 5.0 * gi50_conc
    necrosis_threshold = 10.0 * gi50_conc
    max_apoptosis_rate = 0.01
    lethal_apoptosis_rate = 2.0 * max_apoptosis_rate
    max_necrosis_rate = 1.0

    # Drug concentration range
    drug_conc = np.linspace(0, 12 * gi50_conc, 1000)
    apoptosis_rate = np.zeros_like(drug_conc)
    necrosis_rate = np.zeros_like(drug_conc)

    # Fill in rates according to C++ logic
    for i, conc in enumerate(drug_conc):
        if conc >= necrosis_threshold:
            necrosis_rate[i] = max_necrosis_rate
            apoptosis_rate[i] = 0
        elif conc >= apoptosis_threshold:
            fraction = (conc - apoptosis_threshold) / (necrosis_threshold - apoptosis_threshold)
            necrosis_rate[i] = fraction * max_necrosis_rate
            apoptosis_rate[i] = lethal_apoptosis_rate
        elif conc > gi50_conc:
            fraction = (conc - gi50_conc) / (apoptosis_threshold - gi50_conc)
            apoptosis_rate[i] = max_apoptosis_rate + fraction * (lethal_apoptosis_rate - max_apoptosis_rate)
            necrosis_rate[i] = 0
        else:
            apoptosis_rate[i] = 0
            necrosis_rate[i] = 0

    # Plot
    fig, ax1 = plt.subplots(figsize=(4, 4), dpi=300)
    ax2 = ax1.twinx()

    # Apoptosis rate (left y-axis) - colored line, black axis
    apoptosis_color = '#009E73'  # Colorblind-friendly green
    ax1.plot(drug_conc, apoptosis_rate, color=apoptosis_color, linewidth=3, label='Apoptosis rate')
    ax1.set_ylabel('Apoptosis rate', fontsize=9, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(0, lethal_apoptosis_rate * 1.2)

    # Necrosis rate (right y-axis) - colored line, black axis
    ax2.plot(drug_conc, necrosis_rate, color='black', linewidth=3, label='Necrosis rate')
    ax2.set_ylabel('Necrosis rate', fontsize=9, fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, max_necrosis_rate * 1.2)

    # X axis
    ax1.set_xlabel('Drug Concentration', fontsize=11, fontweight='bold', color='black')
    ax1.set_xlim(0, 12 * gi50_conc)
    ax1.set_xticks([0, gi50_conc, apoptosis_threshold, necrosis_threshold])
    ax1.set_xticklabels(['0', 'GI50', '5×GI50', '10×GI50'], fontsize=10, color='black')
    ax1.tick_params(axis='x', labelcolor='black')

    # Threshold lines
    for x in [gi50_conc, apoptosis_threshold, necrosis_threshold]:
        ax1.axvline(x, color='gray', linestyle='--', linewidth=1.2, zorder=0)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False, fontsize=9)

    # Style: all axes and ticks in black
    for spine in ax1.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2.0)
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig("drug_dual_yaxis_schematic.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig("drug_dual_yaxis_schematic.svg", format='svg', bbox_inches='tight')
    plt.show()

# Create the plot
plot_drug_concentration_effects()
plot_drug_phenotype_schematic()
plot_drug_dual_yaxis_schematic()