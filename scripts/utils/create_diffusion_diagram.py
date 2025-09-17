import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from skimage import measure

def draw_domain_box(ax, lims):
    """Draws a wireframe box around the simulation domain."""
    x_lim, y_lim, z_lim = lims
    points = np.array([
        [x_lim[0], y_lim[0], z_lim[0]], [x_lim[1], y_lim[0], z_lim[0]],
        [x_lim[1], y_lim[1], z_lim[0]], [x_lim[0], y_lim[1], z_lim[0]],
        [x_lim[0], y_lim[0], z_lim[1]], [x_lim[1], y_lim[0], z_lim[1]],
        [x_lim[1], y_lim[1], z_lim[1]], [x_lim[0], y_lim[1], z_lim[1]],
    ])
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
        [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    for edge in edges:
        ax.plot3D(points[edge, 0], points[edge, 1], points[edge, 2],
                  color="black", alpha=0.8, lw=0.8)

def plot_drug_gradient(ax, concentration_data, color, plot_lims):
    """
    Helper function to plot multiple isosurfaces for a single drug on a given 3D axis,
    creating a volumetric gradient effect.
    """
    x_lim, y_lim, z_lim = plot_lims

    # Generate many levels and corresponding alphas for a smooth gradient effect
    isosurface_levels = np.linspace(0.9, 0.1, 15)
    alphas = np.linspace(0.35, 0.01, 15)
    
    x = np.linspace(x_lim[0], x_lim[1], 50)
    y = np.linspace(y_lim[0], y_lim[1], 50)
    z = np.linspace(z_lim[0], z_lim[1], 50)

    for level, alpha in zip(isosurface_levels, alphas):
        try:
            verts, faces, _, _ = measure.marching_cubes(
                concentration_data, level=level,
                spacing=(x[1] - x[0], y[1] - y[0], z[1] - z[0])
            )
            verts += [x_lim[0], y_lim[0], z_lim[0]]
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                            color=color, alpha=alpha, lw=0)
        except Exception:
            # Skip if a surface cannot be generated (e.g., data is all below the level)
            continue
            
    # --- Styling ---
    draw_domain_box(ax, plot_lims) # Draw the outer box
    
    # Turn off all axis lines, labels, and panes for a clean diagram
    ax.set_axis_off()

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.view_init(elev=20, azim=-65)

def create_diffusion_diagram():
    """
    Generates an illustrative side-by-side 3D diagram of two drugs diffusing 
    into a simulation domain, showing their concentration gradients.
    """
    
    # --- 1. Simulation & Visualization Parameters ---
    time_release_X = 0
    time_release_Y = 50
    time_snapshot = 100
    diffusion_coeff_X = 5.0
    diffusion_coeff_Y = 1.5
    domain_lims = ((-50, 50), (-50, 50), (0, 100))
    penetration_scale_factor = 2.0 # Factor to exaggerate diffusion depth
    color_X = '#009E73'  # Publication-quality Teal
    color_Y = '#D55E00'  # Publication-quality Orange
    
    # --- 2. Model the Diffusion ---
    time_diffused_X = time_snapshot - time_release_X
    time_diffused_Y = time_snapshot - time_release_Y

    x = np.linspace(domain_lims[0][0], domain_lims[0][1], 50)
    y = np.linspace(domain_lims[1][0], domain_lims[1][1], 50)
    z = np.linspace(domain_lims[2][0], domain_lims[2][1], 50)
    X, Y, Z = np.meshgrid(x, y, z)

    penetration_X = np.sqrt(diffusion_coeff_X * time_diffused_X) * penetration_scale_factor
    penetration_Y = np.sqrt(diffusion_coeff_Y * time_diffused_Y) * penetration_scale_factor

    concentration_X = np.exp(-(domain_lims[2][1] - Z) / penetration_X)
    concentration_Y = np.exp(-(domain_lims[2][1] - Z) / penetration_Y)

    # --- 3. Create the 3D Plot ---
    fig = plt.figure(figsize=(18, 9))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # Plot Drug X on the first subplot
    plot_drug_gradient(ax1, concentration_X, color_X, domain_lims)

    # Plot Drug Y on the second subplot
    plot_drug_gradient(ax2, concentration_Y, color_Y, domain_lims)

    # --- 4. Annotate and Finalize ---
    fig.suptitle("Conceptual Diagram of Drug Diffusion", fontsize=20, fontweight='bold')

    # Create a custom legend for the entire figure
    legend_elements = [
        mpatches.Patch(color=color_X, alpha=0.6, label=f'Drug X (High Diffusion, D={diffusion_coeff_X})'),
        mpatches.Patch(color=color_Y, alpha=0.6, label=f'Drug Y (Low Diffusion, D={diffusion_coeff_Y})')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.90), fontsize=14, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for titles

    # --- 5. Save and Show ---
    output_path = "diffusion_diagram_publication"
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(f"{output_path}.svg", format='svg', bbox_inches='tight', transparent=True)
    
    # Deactivated interactive showing for remote execution
    # plt.show()

    print(f"Diagram saved to {output_path}.png and {output_path}.svg")

if __name__ == '__main__':
    create_diffusion_diagram() 