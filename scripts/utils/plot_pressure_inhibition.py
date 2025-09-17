import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_growth_inhibition():
    """
    Create a schematic plot showing growth rate inhibition as a function of pressure
    using a negative Hill function with more pronounced sigmoidal effect
    """
    # Set up the plot with clean, minimal style
    plt.figure(figsize=(3, 3), dpi=300)  # Made figure more compact
    sns.set_context("paper", font_scale=1.0)
    sns.set_style("white")
    
    # Define Hill function parameters
    max_growth_rate = 0.02  # 1/min
    min_growth_rate = 0.0   # 1/min
    k = 5.0                 # pressure at half-maximal inhibition
    n = 4.0                 # Hill coefficient for pronounced sigmoid
    
    # Generate pressure range
    pressure = np.linspace(0, 15, 1000)
    
    # Calculate growth rate using negative Hill function
    growth_rate = max_growth_rate - (max_growth_rate - min_growth_rate) * (pressure**n / (k**n + pressure**n))
    
    # Create the plot
    ax = plt.gca()
    
    # Plot growth rate with thicker line
    plt.plot(pressure, growth_rate, '-', color='#377EB8', 
             linewidth=4.0)  # Increased line width
    
    # Add horizontal line at max growth rate
    plt.axhline(y=max_growth_rate, color='gray', linestyle='--', 
                alpha=0.7, linewidth=2.0)  # Increased line width
    
    # Simplified labels with larger font
    plt.xlabel("Pressure (a.u.)", fontsize=12, fontweight="bold")
    plt.ylabel("Growth Rate (1/min)", fontsize=12, fontweight="bold")
    
    # Clean up the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.5)  # Thicker spine
    ax.spines['bottom'].set_linewidth(2.5)  # Thicker spine
    
    # Set axis limits and ticks
    plt.ylim(0, max_growth_rate * 1.1)
    plt.xlim(0, 15)
    plt.xticks([])  # Remove x-tick values
    plt.yticks([])  # Remove y-tick values
    
    plt.tight_layout()
    
    # Save in high resolution with transparent background
    plt.savefig("growth_inhibition.png", dpi=300, bbox_inches='tight', 
                transparent=True)
    plt.savefig("growth_inhibition.svg", format='svg', bbox_inches='tight', 
                transparent=True)
    
    return plt

# Create the plot
plot_growth_inhibition()
