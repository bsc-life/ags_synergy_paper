import pandas as pd
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
import numpy as np
import seaborn as sns

def get_evolutionary_strategy_from_experiment_name(experiment_name):
    # Extract the evolutionary strategy from the experiment name
    # Assuming the format is EVO_STRATEGY-DATE-EXPERIMENT_NAME
    # and the strategy is either CMA or GA
    if 'CMA' in experiment_name:
        return 'CMA'
    elif 'GA' in experiment_name:
        return 'GA'
    elif 'sweep' in experiment_name:
        return 'sweep'
    else:
        return "unknown strategy"
    

def hill_function(x_values, hill_coeff, K_half, V_max=1):
    """
    Calculate the Hill function value.

    Parameters:
    - x_values: The input values (e.g., concentration) as a numpy array.
    - hill_coeff: The Hill coefficient (n).
    - K_half: The half-maximal effective concentration (K).
    - V_max: The maximum response (default is 1).

    Returns:
    - The calculated Hill function value as a numpy array.
    """

    return (V_max * (x_values ** hill_coeff)) / (K_half ** hill_coeff + (x_values ** hill_coeff))

def obtain_hill_function_params_from_instance_number(experiment_name, instance_number):
    # this is in the experiments folder
    # find the file with the instance number in the name
    # fetch the XML and extract the hill_coeff_growth, K_half_growth, hill_coeff_apoptosis, K_half_apoptosis
    # return the values as a dictionary

    settings_path = f'experiments/{experiment_name}/instance_{instance_number}/settings.xml'
    tree = ET.parse(settings_path)
    root = tree.getroot()

    # print all element names and subelement names
    for elem in root.iter():
        if elem.tag == 'hill_coeff_growth':
            hill_coeff_growth = elem.text
        elif elem.tag == 'K_half_growth':
            K_half_growth = elem.text
        elif elem.tag == 'hill_coeff_apoptosis':
            hill_coeff_apoptosis = elem.text
        elif elem.tag == 'K_half_apoptosis':
            K_half_apoptosis = elem.text
        elif elem.tag == 'w1_growth':
            w1_growth = elem.text
        elif elem.tag == 'w2_growth':
            w2_growth = elem.text
        elif elem.tag == 'w3_growth':
            w3_growth = elem.text
        elif elem.tag == "basal_growth_rate":
            basal_growth_rate = elem.text
        elif elem.tag == 'w1_apoptosis':
            w1_apoptosis = elem.text
        elif elem.tag == 'w2_apoptosis':
            w2_apoptosis = elem.text
        elif elem.tag == 'w3_apoptosis':
            w3_apoptosis = elem.text
        elif elem.tag == 'apoptosis_rate_basal':
            apoptosis_rate_basal = elem.text
        elif elem.tag == 'max_apoptosis_rate':
            max_apoptosis_rate = elem.text
    
    # Print the extracted values
    # print(f'Hill coeff growth: {hill_coeff_growth}, K half growth: {K_half_growth}, Hill coeff apoptosis: {hill_coeff_apoptosis}, K half apoptosis: {K_half_apoptosis}')

    settings_dict = {
        'hill_coeff_growth': hill_coeff_growth,
        'K_half_growth': K_half_growth,
        'hill_coeff_apoptosis': hill_coeff_apoptosis,
        'K_half_apoptosis': K_half_apoptosis,
        'w1_growth': w1_growth,
        'w2_growth': w2_growth,
        'w3_growth': w3_growth,
        'basal_growth_rate': basal_growth_rate,
        'w1_apoptosis': w1_apoptosis,
        'w2_apoptosis': w2_apoptosis,
        'w3_apoptosis': w3_apoptosis,
        'apoptosis_rate_basal': apoptosis_rate_basal,
        'max_apoptosis_rate': max_apoptosis_rate
    }

    return settings_dict

def obtain_instance_number_from_experiment_name(row):
    if 'iteration' in row.index:
        instance_number = '_'.join(row[['iteration', 'individual', 'replicate']].astype(int).astype(str))
    else:
        instance_number = '_'.join(row[['individual', 'replicate']].astype(int).astype(str))
    return instance_number
    
def get_top_instances_dataframe(experiment_name, n=10):
    strategy = get_evolutionary_strategy_from_experiment_name(experiment_name)
    csv_file = f'results/{strategy}_summaries/final_summary_{experiment_name}/top_{n}.csv'
    df = pd.read_csv(csv_file)

    instance_numbers = df.apply(obtain_instance_number_from_experiment_name, axis=1)
    hill_params_list = instance_numbers.apply(lambda instance: obtain_hill_function_params_from_instance_number(experiment_name, instance))

    # Create a new DataFrame to store the parameters
    hill_params_df = pd.DataFrame(hill_params_list.tolist())

    # Combine the original DataFrame with the new parameters DataFrame
    df_top_n = pd.concat([df.reset_index(drop=True), hill_params_df.reset_index(drop=True)], axis=1)

    return df_top_n


def plot_growth_hill_function(experiment_name, output_path, top):
    # Data preparation
    top_instances = get_top_instances_dataframe(experiment_name, top)
    top_instances = top_instances.astype(float)
    growth_columns = [col for col in top_instances.columns if 'growth' in col]
    growth_top_instances = top_instances[growth_columns]
    growth_top_instances = growth_top_instances.loc[:, ~growth_top_instances.columns.duplicated()]

    # Set up the figure with high-quality settings
    plt.figure(figsize=(4, 4), dpi=300)
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    
    # Use universal sans-serif font stack
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Liberation Sans', 
                                      'FreeSans', 'Arial', 'sans-serif']

    # Define x_values and plot parameters
    x_values = np.linspace(0, 1, 1000)
    
    # Create color gradient from dark to light blue
    colors = plt.cm.Blues(np.linspace(0.5, 1, len(growth_top_instances)))
    
    # Plot transfer functions
    n = len(growth_top_instances)
    for index, (row, color) in enumerate(zip(growth_top_instances.iterrows(), colors)):
        try:
            _, row_data = row
            hill_coeff_growth = float(row_data['hill_coeff_growth'])
            K_half_growth = float(row_data['K_half_growth'])
            V_max = float(row_data['basal_growth_rate'])

            # Calculate Hill function values
            hill_values = ((V_max * (x_values ** hill_coeff_growth)) / 
                         (K_half_growth ** hill_coeff_growth + 
                          (x_values ** hill_coeff_growth)))

            # Plot with improved visibility
            plt.plot(x_values, hill_values, 
                    color=color,
                    linewidth=1.5,  # Increased line width
                    alpha=0.8,      # Increased base opacity
                    zorder=n-index)

        except ValueError as e:
            print(f"Error converting values for row {index}: {e}")
            continue

    # Customize the plot
    ax = plt.gca()
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make remaining spines thicker
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Set limits
    plt.xlim(0, 1)
    plt.ylim(0, V_max)
    
    # Adjust labels with larger font sizes
    plt.xlabel('Boolean readouts convex sum', 
              fontsize=11,  # Increased from 8
              fontweight="bold")
    plt.ylabel(r'$r_{\gamma}$', 
              fontsize=24,  # Increased from 8
              fontweight="bold")
    
    # Adjust tick parameters
    ax.tick_params(axis='both', 
                  width=1.0,      # Thicker ticks
                  length=4,       # Longer ticks
                  labelsize=12,   # Larger tick labels
                  pad=3)
    
    plt.tight_layout()
    
    # Save plots
    save_path_png = f'{output_path}/growth_TF_{experiment_name}_top_{top}.png'
    save_path_svg = f'{output_path}/growth_TF_{experiment_name}_top_{top}.svg'
    
    plt.savefig(save_path_png, 
                dpi=300, 
                bbox_inches='tight',
                transparent=True)
    plt.savefig(save_path_svg, 
                format='svg',
                bbox_inches='tight',
                transparent=True)
    print(f'Saved figures to {save_path_png} and {save_path_svg}')
    plt.close()

def plot_growth_hill_function_with_weights(experiment_name, output_path, top):
    """
    Plot the growth Hill function curves for the top instances and mark the intersections 
    with their respective growth weights (w1_growth, w2_growth, w3_growth).

    The Hill function is defined as:
        f(x) = (V_max * x^n) / (K_half^n + x^n)
    where:
      - n = hill_coeff_growth,
      - K_half = K_half_growth, and
      - V_max = basal_growth_rate

    If the weights are normalized between 0 and 1 (with 1 corresponding to V_max),
    the intersection is computed from:
        f(x) = w * V_max  =>  (V_max * x^n)/(K_half^n+x^n)= w * V_max.
    Canceling V_max (assumed nonzero) and rearranging yields:
        x = K_half * (w/(1-w))^(1/n)
    Note that the computation is only valid for weights 0 < w < 1.

    Parameters:
      - experiment_name: Name of the experiment (used to locate the data).
      - output_path: Directory to save the output plot.
      - top: Number of top instances to plot.
    """
    # Data preparation
    top_instances = get_top_instances_dataframe(experiment_name, top)
    top_instances = top_instances.astype(float)
    
    # Select columns that include growth-related parameters and weights.
    growth_columns = [col for col in top_instances.columns if 'growth' in col]
    growth_top_instances = top_instances[growth_columns]
    growth_top_instances = growth_top_instances.loc[:, ~growth_top_instances.columns.duplicated()]
    
    # Set up plot appearance
    plt.figure(figsize=(4, 4), dpi=300)
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [
        'DejaVu Sans', 'Helvetica', 'Liberation Sans', 
        'FreeSans', 'Arial', 'sans-serif'
    ]
    
    # Domain for the Hill function evaluation
    x_values = np.linspace(0, 1, 1000)
    
    # Create a color gradient for the different instances
    colors = plt.cm.Blues(np.linspace(0.5, 1, len(growth_top_instances)))
    n = len(growth_top_instances)
    
    # Loop through each top instance.
    for index, (row, color) in enumerate(zip(growth_top_instances.iterrows(), colors)):
        try:
            _, row_data = row
            hill_coeff_growth = float(row_data['hill_coeff_growth'])
            K_half_growth = float(row_data['K_half_growth'])
            V_max = float(row_data['basal_growth_rate'])
            
            # Calculate the Hill function values.
            hill_values = (V_max * (x_values ** hill_coeff_growth)) / (
                K_half_growth ** hill_coeff_growth + (x_values ** hill_coeff_growth)
            )
            
            # Plot the Hill function curve for this instance.
            plt.plot(x_values, hill_values, 
                     color=color,
                     linewidth=1.5,
                     alpha=0.8,
                     zorder=n-index)
            
            # Retrieve the growth weights from the instance.
            w1 = float(row_data['w1_growth'])
            w2 = float(row_data['w2_growth'])
            w3 = float(row_data['w3_growth'])
            growth_weights = [w1, w2, w3]
            
            # For each normalized weight in (0, 1), compute and mark the intersection point.
            for w in growth_weights:
                # Only compute if 0 < w < 1 (each weight is normalized).
                if w <= 0 or w >= 1:
                    continue
                # Solve f(x) = w * V_max.
                # Derivation leads to: x_intersect = K_half * ((w/(1-w))^(1/hill_coeff_growth))
                x_intersect = K_half_growth * ((w / (1 - w)) ** (1.0 / hill_coeff_growth))
                # Plot only if x_intersect falls within the [0,1] domain.
                if 0 <= x_intersect <= 1:
                    plt.plot(x_intersect, w * V_max, marker='o', color=color, markersize=4, zorder=n-index+1)
                    plt.vlines(x_intersect, 0, w * V_max, colors=color, linestyles='dashed', linewidth=1, alpha=0.6)
        except ValueError as e:
            print(f"Error converting values for row {index}: {e}")
            continue
    
    # Customize the plot aesthetics.
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, V_max)
    plt.xlabel('Boolean readouts convex sum', fontsize=11, fontweight="bold")
    plt.ylabel(r'$r_{\gamma}$', fontsize=24, fontweight="bold")
    ax.tick_params(axis='both', width=1.0, length=4, labelsize=12, pad=3)
    plt.tight_layout()
    
    # Save the resulting plots.
    save_path_png = f'{output_path}/growth_TF_with_weights_{experiment_name}_top_{top}.png'
    save_path_svg = f'{output_path}/growth_TF_with_weights_{experiment_name}_top_{top}.svg'
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight', transparent=True)
    print(f'Saved figures to {save_path_png} and {save_path_svg}')
    plt.close()



def plot_weights_intersections_with_linear_fit(experiment_name, output_path, top):
    """
    Plot only the intersection points of the growth weights on the Hill function for the 
    top instances and then fit a linear regression to these points.

    For each instance, the Hill function is defined as:
        f(x) = (V_max * x^n) / (K_half^n + x^n)
    with:
        - n          = hill_coeff_growth,
        - K_half     = K_half_growth,
        - V_max      = basal_growth_rate.
    
    For a normalized weight w (with 0 < w < 1), the intersection is defined through:
        f(x) = w * V_max
    which leads to:
        x = K_half * (w/(1 - w))^(1/n)
        y = w * V_max

    This function performs the following steps:
      1. For each top instance, compute the (x, y) intersection points for w1_growth, w2_growth, and w3_growth.
      2. Create a scatter plot of all the intersection points.
      3. Fit a linear regression line (y = slope*x + intercept) to these points.
      4. Create a second figure displaying the scatter points along with the linear fit.
      5. Save both plots as PNG and SVG files.
      6. Return the coordinates and the linear fit parameters.

    Parameters:
      - experiment_name: Name of the experiment (used to locate the data).
      - output_path: Directory to save the output plots.
      - top: Number of top instances to process.
    """
    # Data preparation
    top_instances = get_top_instances_dataframe(experiment_name, top)
    top_instances = top_instances.astype(float)
    
    # Select columns which include growth-related parameters and weights.
    growth_columns = [col for col in top_instances.columns if 'growth' in col]
    growth_top_instances = top_instances[growth_columns]
    growth_top_instances = growth_top_instances.loc[:, ~growth_top_instances.columns.duplicated()]

    intersection_points = []  # list of (x, y) coordinates

    # For each instance, calculate intersection points from the normalized growth weights.
    for index, (row, _) in enumerate(growth_top_instances.iterrows()):
        try:
            hill_coeff_growth = float(growth_top_instances.at[index, 'hill_coeff_growth'])
            K_half_growth = float(growth_top_instances.at[index, 'K_half_growth'])
            V_max = float(growth_top_instances.at[index, 'basal_growth_rate'])
            
            # Retrieve the three growth weights.
            weights = []
            for key in ['w1_growth', 'w2_growth', 'w3_growth']:
                value = float(growth_top_instances.at[index, key])
                weights.append(value)
            
            # For each weight, compute the intersection if weight is valid (0 < w < 1).
            for w in weights:
                if w <= 0 or w >= 1:
                    continue
                # Using the derivation: x = K_half * (w/(1-w))^(1/n) and y = w*V_max
                x_intersect = K_half_growth * ((w / (1 - w)) ** (1.0 / hill_coeff_growth))
                y_intersect = w * V_max
                # Only include the point if x falls in the expected domain.
                if 0 <= x_intersect <= 1:
                    intersection_points.append((x_intersect, y_intersect))
        except ValueError as e:
            print(f"Error converting values for row {index}: {e}")
            continue

    if not intersection_points:
        print("No valid intersection points found.")
        return

    # Separate the points into x and y components.
    x_points, y_points = zip(*intersection_points)

    # ---------------------------
    # Plot 1: Scatter plot of intersection points only.
    plt.figure(figsize=(4, 4), dpi=300)
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    plt.scatter(x_points, y_points, color="blue", s=30, alpha=0.8)
    plt.xlabel("x (Intersection Point)", fontsize=11, fontweight="bold")
    plt.ylabel("y (w * V_max)", fontsize=11, fontweight="bold")
    plt.title("Intersection Points of Growth Weights", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, max(y_points)*1.1)
    plt.tight_layout()
    
    scatter_png = f"{output_path}/weights_intersections_points_{experiment_name}_top_{top}.png"
    scatter_svg = f"{output_path}/weights_intersections_points_{experiment_name}_top_{top}.svg"
    plt.savefig(scatter_png, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(scatter_svg, format="svg", bbox_inches="tight", transparent=True)
    print(f"Saved scatter plot to {scatter_png} and {scatter_svg}")
    plt.close()

    # ---------------------------
    # Plot 2: Scatter plot with a linear fit line.
    # Fit a linear regression line (1st degree polynomial) to the intersection points.
    slope, intercept = np.polyfit(x_points, y_points, 1)
    x_fit = np.linspace(0, 1, 100)
    y_fit = slope * x_fit + intercept

    plt.figure(figsize=(4, 4), dpi=300)
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    plt.scatter(x_points, y_points, color="blue", s=30, alpha=0.8, label="Data Points")
    plt.plot(x_fit, y_fit, color="red", linewidth=2, label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")
    plt.xlabel("x (Intersection Point)", fontsize=11, fontweight="bold")
    plt.ylabel("y (w * V_max)", fontsize=11, fontweight="bold")
    plt.title("Linear Fit to Intersection Points", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, max(y_points)*1.1)
    plt.tight_layout()

    linear_fit_png = f"{output_path}/weights_intersections_linear_fit_{experiment_name}_top_{top}.png"
    linear_fit_svg = f"{output_path}/weights_intersections_linear_fit_{experiment_name}_top_{top}.svg"
    plt.savefig(linear_fit_png, dpi=300, bbox_inches="tight", transparent=True)
    plt.savefig(linear_fit_svg, format="svg", bbox_inches="tight", transparent=True)
    print(f"Saved linear fit plot to {linear_fit_png} and {linear_fit_svg}")
    plt.close()
    
    # Optionally return the data and fit parameters.
    return {
        "intersection_points": intersection_points,
        "linear_fit": {"slope": slope, "intercept": intercept},
        "scatter_plot": scatter_png,
        "linear_fit_plot": linear_fit_png
    }


def plot_individual_instance_weights_linear_fit(experiment_name, output_path, top):
    """
    For each top instance (from get_top_instances_dataframe), compute the intersection
    points of the normalized growth weights with the Hill function. Each intersection
    is computed using:
        x_intersect = K_half * (w/(1-w))^(1/hill_coeff_growth)
        y_intersect = w * V_max
    where:
      - hill_coeff_growth is the Hill coefficient,
      - K_half is K_half_growth, and
      - V_max is basal_growth_rate.
    
    For each instance, two plots are generated:
      1. A scatter plot of the intersection points.
      2. A scatter plot overlaid with a linear regression fit (if at least two points exist).
    
    If a straight line cannot be fit to an instance's points, a message is printed.
    
    The function saves these plots as PNG and SVG files (one pair per instance) and returns
    a dictionary with the intersection coordinates and the linear fit parameters for each instance.
    
    Parameters:
      - experiment_name: Name of the experiment (used to locate the data).
      - output_path: Directory to save the output plots.
      - top: Number of top instances to process.
    """
    # Data preparation: extract the top instances and convert them to floats.
    top_instances = get_top_instances_dataframe(experiment_name, top)
    top_instances = top_instances.astype(float)
    
    # Select columns including growth-related parameters and weights.
    growth_columns = [col for col in top_instances.columns if 'growth' in col]
    growth_top_instances = top_instances[growth_columns]
    growth_top_instances = growth_top_instances.loc[:, ~growth_top_instances.columns.duplicated()]
    
    # Dictionary to store results for each instance.
    instance_results = {}
    
    # Loop over each instance: each row is processed individually.
    for instance_id, row in growth_top_instances.iterrows():
        try:
            hill_coeff_growth = float(row['hill_coeff_growth'])
            K_half_growth = float(row['K_half_growth'])
            V_max = float(row['basal_growth_rate'])
            
            # Retrieve the three growth weights.
            weights = []
            for key in ['w1_growth', 'w2_growth', 'w3_growth']:
                value = float(row[key])
                weights.append(value)
                
            # Compute intersection points for weights in (0, 1)
            intersections = []
            for w in weights:
                if w <= 0 or w >= 1:
                    continue
                # Compute using the formula: x = K_half * (w/(1-w))^(1/hill_coeff_growth), y = w * V_max.
                x_intersect = K_half_growth * ((w / (1 - w)) ** (1.0 / hill_coeff_growth))
                y_intersect = w * V_max
                # Only include the point if it falls within the expected domain.
                if 0 <= x_intersect <= 1:
                    intersections.append((x_intersect, y_intersect))
            
            if not intersections:
                print(f"[Verbose] Instance {instance_id}: No valid intersection points found.")
                print(f"Instance {instance_id}: {row}")
                continue
            
            # Separate intersection points into x and y components.
            x_vals, y_vals = zip(*intersections)
            
            # ---------------------------
            # Plot 1: Scatter plot of intersection points only.
            plt.figure(figsize=(4, 4), dpi=300)
            sns.set_context("paper", font_scale=1.2)
            sns.set_style("ticks")
            plt.scatter(x_vals, y_vals, color="blue", s=40, alpha=0.8)
            plt.xlabel("x (Intersection Position)", fontsize=11, fontweight="bold")
            plt.ylabel("y (w * V_max)", fontsize=11, fontweight="bold")
            plt.title(f"Instance {instance_id}: Intersection Points", fontsize=12, fontweight="bold")
            plt.xlim(0, 1)
            plt.ylim(0, max(y_vals) * 1.1)
            plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
            plt.tight_layout()
            scatter_file_png = f"{output_path}/instance_{instance_id}_weights_scatter.png"
            scatter_file_svg = f"{output_path}/instance_{instance_id}_weights_scatter.svg"
            # plt.savefig(scatter_file_png, dpi=300, bbox_inches="tight", transparent=True)
            # plt.savefig(scatter_file_svg, format="svg", bbox_inches="tight", transparent=True)
            plt.close()
            
            # ---------------------------
            # Plot 2: Scatter with linear regression fit.
            plt.figure(figsize=(4, 4), dpi=300)
            sns.set_context("paper", font_scale=1.2)
            sns.set_style("ticks")
            plt.scatter(x_vals, y_vals, color="blue", s=40, alpha=0.8, label="Intersection Points")
            
            # Fit a line if there are at least 2 points.
            if len(x_vals) >= 2:
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                x_fit = np.linspace(0, 1, 100)
                y_fit = slope * x_fit + intercept
                plt.plot(x_fit, y_fit, color="red", linewidth=2, label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")
            else:
                slope, intercept = None, None
                plt.text(0.5, max(y_vals) * 0.8, "Not enough points for linear fit", 
                         horizontalalignment="center", color="red")
                print(f"[Verbose] Instance {instance_id}: Not enough intersection points for linear fit (found {len(x_vals)} point).")
            
            plt.xlabel("x (Intersection Position)", fontsize=11, fontweight="bold")
            plt.ylabel("y (w * V_max)", fontsize=11, fontweight="bold")
            plt.title(f"Instance {instance_id}: Linear Fit of Intersections", fontsize=12, fontweight="bold")
            plt.xlim(0, 1)
            plt.ylim(0, max(y_vals) * 1.1)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
            plt.tight_layout()
            linear_file_png = f"{output_path}/instance_{instance_id}_weights_linear_fit.png"
            linear_file_svg = f"{output_path}/instance_{instance_id}_weights_linear_fit.svg"
            # plt.savefig(linear_file_png, dpi=300, bbox_inches="tight", transparent=True)
            # plt.savefig(linear_file_svg, format="svg", bbox_inches="tight", transparent=True)
            plt.close()
            
            # Store the results to return.
            instance_results[instance_id] = {
                "intersection_points": intersections,
                "linear_fit": {"slope": slope, "intercept": intercept},
                "scatter_plot": scatter_file_png,
                "linear_fit_plot": linear_file_png
            }
        except ValueError as e:
            print(f"[Verbose] Error processing instance {instance_id}: {e}")
            continue
    
    return instance_results


def plot_combined_instance_weights_linear_fit(experiment_name, output_path, top):
    """
    Create a combined figure (2 rows x 5 columns) of linear fits for the intersection points
    of the normalized growth weights with the Hill function for the top instances.
    
    For each instance, the intersection points are computed using:
        x_intersect = K_half * (w/(1-w))^(1/hill_coeff_growth)
        y_intersect = w * V_max
    where:
      - hill_coeff_growth is the Hill coefficient,
      - K_half is K_half_growth, and
      - V_max is basal_growth_rate.
    
    For each instance, if at least two intersection points exist, a linear regression is performed
    and the regression line is plotted over the scatter points. The plot itself does not display any legends.
    
    If a straight line cannot be fit (e.g. fewer than two valid intersection points), a verbose message
    is printed and an annotation is added to the subplot.
    
    The complete figure is saved in both PNG and SVG formats.
    
    Parameters:
      - experiment_name: Name of the experiment (used to locate the data).
      - output_path: Directory to save the output figure.
      - top: Number of top instances to process (should be 10 to fill a 2x5 grid).
    
    Returns:
      - A dictionary with intersection points and linear fit parameters per instance.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Data preparation: extract the top instances and convert to float.
    top_instances = get_top_instances_dataframe(experiment_name, top)
    top_instances = top_instances.astype(float)

    # Select only columns that include growth-related parameters and weights.
    growth_columns = [col for col in top_instances.columns if 'growth' in col]
    growth_top_instances = top_instances[growth_columns]
    growth_top_instances = growth_top_instances.loc[:, ~growth_top_instances.columns.duplicated()]

    # Set up the combined figure: 2 rows and 5 columns.
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), dpi=300)
    axes = axes.flatten()

    # Dictionary to store results for each instance.
    instance_results = {}
    
    # Loop over each instance (row). A maximum of 10 instances will be plotted.
    for idx, (instance_id, row) in enumerate(growth_top_instances.iterrows()):
        if idx >= len(axes):
            break  # Ensure we do not exceed the available subplots.
        ax = axes[idx]
        try:
            hill_coeff_growth = float(row['hill_coeff_growth'])
            K_half_growth = float(row['K_half_growth'])
            V_max = float(row['basal_growth_rate'])
            
            # Retrieve the three growth weights.
            weights = []
            for key in ['w1_growth', 'w2_growth', 'w3_growth']:
                value = float(row[key])
                weights.append(value)
                
            # Compute intersection points for weights in the valid range (0 < w < 1).
            intersections = []
            for w in weights:
                if w <= 0 or w >= 1:
                    continue
                # Compute:
                #   x_intersect = K_half * (w/(1-w))^(1/hill_coeff_growth)
                #   y_intersect = w * V_max
                x_intersect = K_half_growth * ((w / (1 - w)) ** (1.0 / hill_coeff_growth))
                y_intersect = w * V_max
                if 0 <= x_intersect <= 1:
                    intersections.append((x_intersect, y_intersect))
            
            if not intersections:
                print(f"[Verbose] Instance {instance_id}: No valid intersection points found.")
                ax.text(0.5, 0.5, "No valid points", horizontalalignment="center", color="red", fontsize=10)
                ax.set_title(f"Instance {instance_id}", fontsize=10, fontweight="bold")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, V_max)
                ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
                continue
            
            # Separate intersections into x and y components.
            x_vals, y_vals = zip(*intersections)
            
            # Plot the intersection points.
            sns.set_context("paper", font_scale=1.2)
            sns.set_style("ticks")
            ax.scatter(x_vals, y_vals, color="blue", s=40, alpha=0.8)
            
            # Attempt to perform a linear regression if there are at least 2 points.
            if len(x_vals) >= 2:
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                x_fit = np.linspace(0, 1, 100)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, color="red", linewidth=2)
            else:
                slope, intercept = None, None
                ax.text(0.5, max(y_vals) * 0.8, "Not enough points for linear fit", 
                        horizontalalignment="center", color="red", fontsize=10)
                print(f"[Verbose] Instance {instance_id}: Not enough intersection points for linear fit (found {len(x_vals)} point).")
            
            # Set axis limits and title.
            ax.set_xlabel("x", fontsize=10)
            ax.set_ylabel("y", fontsize=10)
            ax.set_title(f"Instance {instance_id}", fontsize=10, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, V_max * 1.1)
            ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
            
            # Save results for this instance.
            instance_results[instance_id] = {
                "intersection_points": intersections,
                "linear_fit": {"slope": slope, "intercept": intercept},
            }
        except ValueError as e:
            print(f"[Verbose] Error processing instance {instance_id}: {e}")
            continue

    # Remove any unused subplots.
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    
    # Save the combined figure.
    combined_png = f"{output_path}/combined_instance_weights_linear_fit_{experiment_name}_top_{top}.png"
    combined_svg = f"{output_path}/combined_instance_weights_linear_fit_{experiment_name}_top_{top}.svg"
    plt.savefig(combined_png, dpi=300, bbox_inches="tight", transparent=True)
    # plt.savefig(combined_svg, format="svg", bbox_inches="tight", transparent=True)
    print(f"Saved combined plot to {combined_png} and {combined_svg}")
    plt.close()

    return {
        "instance_results": instance_results,
        "combined_png": combined_png,
        "combined_svg": combined_svg
    }


def plot_apoptosis_function(experiment_name, output_path, top):
    # Data preparation
    top_instances = get_top_instances_dataframe(experiment_name, top)
    top_instances = top_instances.astype(float)
    apoptosis_columns = [col for col in top_instances.columns if 'apoptosis' in col]
    apoptosis_top_instances = top_instances[apoptosis_columns]
    apoptosis_top_instances = apoptosis_top_instances.loc[:, ~apoptosis_top_instances.columns.duplicated()]

    # Set up the figure with high-quality settings
    plt.figure(figsize=(4, 4), dpi=300)
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    
    # Use universal sans-serif font stack
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Liberation Sans', 
                                      'FreeSans', 'Arial', 'sans-serif']

    # Define x_values and plot parameters
    x_values = np.linspace(0, 1, 1000)
    
    # Create color gradient from dark to light purple
    colors = plt.cm.Purples(np.linspace(0.5, 1, len(apoptosis_top_instances)))
    
    # Plot transfer functions
    n = len(apoptosis_top_instances)
    for index, (row, color) in enumerate(zip(apoptosis_top_instances.iterrows(), colors)):
        try:
            _, row_data = row
            hill_coeff_apoptosis = float(row_data['hill_coeff_apoptosis'])
            K_half_apoptosis = float(row_data['K_half_apoptosis'])
            V_max_apoptosis = float(row_data['max_apoptosis_rate'])
            basal_apoptosis_rate = float(row_data['apoptosis_rate_basal'])

            # Calculate apoptosis values
            apoptosis_values = ((V_max_apoptosis * (x_values ** hill_coeff_apoptosis)) / 
                              (K_half_apoptosis ** hill_coeff_apoptosis + 
                               (x_values ** hill_coeff_apoptosis)) + 
                              basal_apoptosis_rate)

            # Plot with improved visibility
            plt.plot(x_values, apoptosis_values, 
                    color=color,
                    linewidth=1.5,  # Increased line width
                    alpha=0.8,      # Increased base opacity
                    zorder=n-index)

        except ValueError as e:
            print(f"Error converting values for row {index}: {e}")
            continue

    # Customize the plot
    ax = plt.gca()
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make remaining spines thicker
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Set limits
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    
    # Adjust labels with larger font sizes
    plt.xlabel('Boolean readouts convex sum', 
              fontsize=11,  # Increased from 8
              fontweight="bold")
    plt.ylabel(r'$r_{\alpha}$', 
              fontsize=24,  # Increased from 8
              fontweight="bold")
    
    # Set y-axis maximum to 0.007
    plt.ylim(0, 0.007)
    # Adjust tick parameters
    ax.tick_params(axis='both', 
                  width=1.0,      # Thicker ticks
                  length=4,       # Longer ticks
                  labelsize=12,   # Larger tick labels
                  pad=3)
    
    plt.tight_layout()
    
    # Save plots
    save_path_png = f'{output_path}/apoptosis_TF_{experiment_name}_top_{top}.png'
    save_path_svg = f'{output_path}/apoptosis_TF_{experiment_name}_top_{top}.svg'
    
    plt.savefig(save_path_png, 
                dpi=300, 
                bbox_inches='tight',
                transparent=True)
    plt.savefig(save_path_svg, 
                format='svg',
                bbox_inches='tight',
                transparent=True)
    print(f'Saved figures to {save_path_png} and {save_path_svg}')
    plt.close()


if __name__ == "__main__":
    n_top = 10
    # PI3K experiment
    pi3k_experiment_name = 'PI3Ki_CMA-1410-1014-12p_rmse_final_50gen'
    saving_path = f'./results/hill_functions_plots/TF_{pi3k_experiment_name}_top_{n_top}'

    # Create directory if it does not exist
    directory = os.path.dirname(saving_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # plot_growth_hill_function(pi3k_experiment_name, directory, n_top)
    # plot_combined_instance_weights_linear_fit(pi3k_experiment_name, directory, n_top)
    # plot_apoptosis_function(pi3k_experiment_name, directory, n_top)

    # # MEKi experiment
    meki_experiment_name = 'MEKi_CMA-1410-1026-12p_rmse_final_50gen'
    saving_path = f'./results/hill_functions_plots/TF_{meki_experiment_name}_top_{n_top}'

    # Create directory if it does not exist
    directory = os.path.dirname(saving_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


    # plot_growth_hill_function(meki_experiment_name, directory, n_top)
    plot_combined_instance_weights_linear_fit(meki_experiment_name, directory, n_top)
    # plot_apoptosis_function(meki_experiment_name, directory, n_top)

    # # AKTi experiment
    # akti_experiment_name = 'AKTi_CMA-1710-0934-12p_rmse_final_50gen'
    # saving_path = f'./results/hill_functions_plots/TF_{akti_experiment_name}_top_{n_top}'

    # # Create directory if it does not exist
    # directory = os.path.dirname(saving_path)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    # plot_growth_hill_function(akti_experiment_name, directory, n_top)
    # plot_apoptosis_function(akti_experiment_name, directory, n_top)



    # # PI3K + MEK experiment
    # # Combined experiment
    # pi3k_mek_combined_experiment_name = 'synergy_sweep-pi3k_mek-1610-1455-13p_uniform_5k'
    # saving_path = f'./results/hill_functions_plots/TF_{pi3k_mek_combined_experiment_name}_top_{n_top}'

    # # Create directory if it does not exist
    # directory = os.path.dirname(saving_path)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    # plot_growth_hill_function(pi3k_mek_combined_experiment_name, directory, n_top)
    # plot_apoptosis_function(pi3k_mek_combined_experiment_name, directory, n_top)

    # # Just PI3K from the combined experiment
    # pi3k_mek_PI3Ksingledrug_experiment_name = 'synergy_sweep-pi3k_mek-1510-1508-12p_PI3K_singledrug'
    # saving_path = f'./results/hill_functions_plots/TF_{pi3k_mek_PI3Ksingledrug_experiment_name}_top_{n_top}'

    # # Create directory if it does not exist
    # directory = os.path.dirname(saving_path)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    # plot_growth_hill_function(pi3k_mek_PI3Ksingledrug_experiment_name, directory, n_top)
    # plot_apoptosis_function(pi3k_mek_PI3Ksingledrug_experiment_name, directory, n_top)

    # # Just MEK from the combined experiment
    # pi3k_mek_MEKsingledrug_experiment_name = 'synergy_sweep-pi3k_mek-1510-1508-12p_MEK_singledrug'
    # saving_path = f'./results/hill_functions_plots/TF_{pi3k_mek_MEKsingledrug_experiment_name}_top_{n_top}'

    # # Create directory if it does not exist
    # directory = os.path.dirname(saving_path)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    # plot_growth_hill_function(pi3k_mek_MEKsingledrug_experiment_name, directory, n_top)
    # plot_apoptosis_function(pi3k_mek_MEKsingledrug_experiment_name, directory, n_top)



    # # AKTi + MEK experiment
    # akti_mek_combined_experiment_name = 'synergy_sweep-akt_mek-1810-1101-13p_uniform_5k'
    # saving_path = f'./results/hill_functions_plots/TF_{akti_mek_combined_experiment_name}_top_{n_top}'

    # # Create directory if it does not exist
    # directory = os.path.dirname(saving_path)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    # plot_growth_hill_function(akti_mek_combined_experiment_name, directory, n_top)
    # plot_apoptosis_function(akti_mek_combined_experiment_name, directory, n_top)

    # # Just AKTi from the combined experiment
    # akti_mek_AKTisingledrug_experiment_name = 'synergy_sweep-akt_mek-1810-1101-12p_AKT_singledrug'
    # saving_path = f'./results/hill_functions_plots/TF_{akti_mek_AKTisingledrug_experiment_name}_top_{n_top}'

    # # Create directory if it does not exist
    # directory = os.path.dirname(saving_path)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    # plot_growth_hill_function(akti_mek_AKTisingledrug_experiment_name, directory, n_top)
    # plot_apoptosis_function(akti_mek_AKTisingledrug_experiment_name, directory, n_top)

    # # Just MEK from the combined experiment
    # akti_mek_MEKsingledrug_experiment_name = 'synergy_sweep-akt_mek-1810-1502-12p_MEK_singledrug'
    # saving_path = f'./results/hill_functions_plots/TF_{akti_mek_MEKsingledrug_experiment_name}_top_{n_top}'

    # # Create directory if it does not exist
    # directory = os.path.dirname(saving_path)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    # plot_growth_hill_function(akti_mek_MEKsingledrug_experiment_name, directory, n_top)
    # plot_apoptosis_function(akti_mek_MEKsingledrug_experiment_name, directory, n_top)
