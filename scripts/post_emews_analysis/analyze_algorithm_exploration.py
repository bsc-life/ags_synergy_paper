import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def analyze_parameter_space(cma_results_df, output_dir, strategy_name):
    # First, identify parameter columns
    metric_columns = cma_results_df.columns[-1]
    non_param_columns = ['individual', 'replicate', 'iteration', 'fitness', 'time', metric_columns]
    param_columns = [col for col in cma_results_df.columns if col not in non_param_columns]
    
    print(f"Analyzing the following parameters: {param_columns}")
    
    # Calculate grid dimensions based on number of parameters
    n_params = len(param_columns)
    n_rows = int(np.ceil(np.sqrt(n_params)))
    n_cols = int(np.ceil(n_params / n_rows))
    
    # 1. Parameter Distribution Analysis
    print(f"Plotting parameter distributions for {n_params} parameters")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.ravel()  # Flatten the axes array
    
    for idx, param in enumerate(param_columns):
        # Get parameter range
        param_min = cma_results_df[param].min()
        param_max = cma_results_df[param].max()
        
        # Filter out the exact minimum and maximum values
        filtered_data = cma_results_df[
            (cma_results_df[param] != param_min) & 
            (cma_results_df[param] != param_max)
        ]
        
        # Calculate the percentage of data points removed
        total_points = len(cma_results_df)
        filtered_points = len(filtered_data)
        removed_percentage = ((total_points - filtered_points) / total_points) * 100
        
        # Create histogram with filtered data
        sns.histplot(data=filtered_data, x=param, ax=axes[idx], color='black')
        
        # Add information about the filtering to the title
        axes[idx].set_title(f'{param} Distribution\n({removed_percentage:.1f}% boundary points removed)', 
                           fontsize=14)
        axes[idx].tick_params(axis='x', rotation=45, labelsize=12)
        axes[idx].tick_params(axis='y', labelsize=12)
        
        # Add vertical lines for min and max with labels
        axes[idx].axvline(x=param_min, color='red', linestyle='--', alpha=0.5)
        axes[idx].axvline(x=param_max, color='red', linestyle='--', alpha=0.5)
        
        # Add text annotations for min and max values
        axes[idx].text(param_min, axes[idx].get_ylim()[1], f'min={param_min:.2e}', 
                      rotation=90, va='top', ha='right', color='red')
        axes[idx].text(param_max, axes[idx].get_ylim()[1], f'max={param_max:.2e}', 
                      rotation=90, va='top', ha='left', color='red')
    
    # Hide empty subplots if any
    for idx in range(len(param_columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'parameter_distributions_{strategy_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Generation Evolution
    print(f"Plotting parameter evolution for {n_params} parameters")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.ravel()
    
    for idx, param in enumerate(param_columns):
        sns.scatterplot(data=cma_results_df, x='iteration', y=param, ax=axes[idx], color='black')
        axes[idx].set_title(f'{param} Evolution', fontsize=14)
        axes[idx].tick_params(axis='x', labelsize=12)
        axes[idx].tick_params(axis='y', labelsize=12)
    
    # Hide empty subplots if any
    for idx in range(len(param_columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'parameter_evolution_{strategy_name}.png'), dpi=300)
    plt.close()

    # 3. Correlation Analysis
    print(f"Plotting parameter correlations for {n_params} parameters")
    correlation_matrix = cma_results_df[param_columns].corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix.abs(),  # Use absolute values of correlations
                annot=True, 
                cmap='YlOrBr',  # Sequential colormap: Yellow to Orange to Brown
                vmin=0,
                vmax=1,
                fmt='.2f',
                annot_kws={'size': 12},
                square=True)  # Make cells square for better visualization
    
    plt.title('Parameter Correlations', fontsize=18)
    plt.xlabel('Parameters', fontsize=16)
    plt.ylabel('Parameters', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'parameter_correlations_{strategy_name}.png'), dpi=300)
    plt.close()

    # 4. Parameter Space Coverage
    print(f"Plotting parameter pairwise for {n_params} parameters")
    g = sns.PairGrid(cma_results_df[param_columns], height=3)  # Increase subplot size
    g.map_upper(sns.scatterplot, color='black', alpha=0.5)
    g.map_lower(sns.kdeplot, color='black', alpha=0.7)
    g.map_diag(sns.histplot, color='black')
    
    # Increase and rotate x-axis labels
    for ax in g.axes[-1,:]:
        ax.set_xticklabels(ax.get_xticklabels(), 
                          rotation=45,
                          ha='right',
                          fontsize=14)
    
    # Increase and rotate y-axis labels
    for ax in g.axes[:,0]:
        ax.set_yticklabels(ax.get_yticklabels(),
                          rotation=45,
                          ha='right',
                          fontsize=14)
    
    # Adjust layout to prevent label cutoff
    plt.savefig(os.path.join(output_dir, f'parameter_pairwise_{strategy_name}.png'), 
                dpi=300,
                bbox_inches='tight')
    plt.close()

    # 5. Diversity Analysis
    print(f"Plotting parameter diversity for {n_params} parameters")
    diversity_metrics = cma_results_df.groupby('iteration')[param_columns].agg(['std', 'mean'])
    
    plt.figure(figsize=(10, 8))
    
    colors = sns.color_palette('Set2', n_colors=len(param_columns))
    line_styles = ['-', '--'] * (len(param_columns) // 2 + 1)
    
    for i, param in enumerate(param_columns):
        # Calculate normalized standard deviation
        std_values = diversity_metrics[param]['std']
        mean_values = diversity_metrics[param]['mean']
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-10
        normalized_std = std_values / (mean_values.abs().mean() + epsilon)
        
        # Plot normalized values
        plt.scatter(diversity_metrics.index, 
                   normalized_std,
                   color=colors[i],
                   alpha=0.3, 
                   s=30)
        plt.plot(diversity_metrics.index, 
                 normalized_std,
                 label=param, 
                 color=colors[i],
                 linestyle=line_styles[i % 2],
                 linewidth=2)
    
    plt.title('Normalized Parameter Diversity Over Iterations', fontsize=16)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Normalized Standard Deviation', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    
    # Set y-axis limits for normalized values
    plt.ylim(0, 1)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.legend(bbox_to_anchor=(1.05, 1), 
              loc='upper left', 
              borderaxespad=0.,
              fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'parameter_diversity_{strategy_name}.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

    # 6. Summary Statistics
    summary_stats = {
        'parameter_ranges': cma_results_df[param_columns].agg(['min', 'max', 'mean', 'std']).to_dict(),
        'final_iteration_stats': cma_results_df[cma_results_df['iteration'] == cma_results_df['iteration'].max()][param_columns].describe().to_dict(),
        'correlation_strength': float(correlation_matrix.abs().mean().mean())
    }
    
    # Save summary stats as text file instead of trying to convert to DataFrame
    with open(os.path.join(output_dir, f'summary_statistics_{strategy_name}.txt'), 'w') as f:
        f.write("Parameter Ranges:\n")
        for param, stats in summary_stats['parameter_ranges'].items():
            f.write(f"\n{param}:\n")
            for stat, value in stats.items():
                f.write(f"  {stat}: {value}\n")
        
        f.write("\nFinal Iteration Statistics:\n")
        for param, stats in summary_stats['final_iteration_stats'].items():
            f.write(f"\n{param}:\n")
            for stat, value in stats.items():
                f.write(f"  {stat}: {value}\n")
        
        f.write(f"\nMean Absolute Correlation: {summary_stats['correlation_strength']:.3f}\n")
    
    return summary_stats

def compare_top_performers(cma_results_df, ga_results_df, output_dir):
    """Compare the top 10% performers between CMA and GA strategies."""
    
    # Colors for each strategy
    cma_color = '#1f77b4'  # Nice blue
    ga_color = '#d62728'   # Nice red
    
    # Get parameter columns (assuming both dataframes have the same structure)
    metric_column = cma_results_df.columns[-1]
    non_param_columns = ['individual', 'replicate', 'iteration', 'fitness', 'time', metric_column]
    param_columns = [col for col in cma_results_df.columns if col not in non_param_columns]
    
    # Get top 10% for each strategy
    cma_top_10 = cma_results_df.nsmallest(int(len(cma_results_df) * 0.1), metric_column)
    ga_top_10 = ga_results_df.nsmallest(int(len(ga_results_df) * 0.1), metric_column)
    
    # Calculate grid dimensions
    n_params = len(param_columns)
    n_rows = int(np.ceil(np.sqrt(n_params)))
    n_cols = int(np.ceil(n_params / n_rows))
    
    # 1. Parameter Distribution Analysis for top 10%
    # Normalized histograms given that top 10% for GA and CMA have different number of individuals
    print("Plotting normalized parameter distributions for top 10% performers")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.ravel()
    
    for idx, param in enumerate(param_columns):
        # Plot normalized histograms
        sns.histplot(data=cma_top_10, x=param, ax=axes[idx], color=cma_color, alpha=0.5, 
                    label='CMA', stat="density")
        sns.histplot(data=ga_top_10, x=param, ax=axes[idx], color=ga_color, alpha=0.5, 
                    label='GA', stat="density")
        
        axes[idx].set_title(f'{param} Distribution\n(Top 10% performers - Density normalized)', fontsize=14)
        axes[idx].tick_params(axis='x', rotation=45, labelsize=12)
        axes[idx].tick_params(axis='y', labelsize=12)
        axes[idx].legend()
    
    # Hide empty subplots
    for idx in range(len(param_columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_10_parameter_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Generation Evolution for top 10%
    print("Plotting parameter evolution for top 10% performers")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.ravel()
    
    for idx, param in enumerate(param_columns):
        # Plot CMA evolution
        sns.scatterplot(data=cma_top_10, x='iteration', y=param, ax=axes[idx], 
                       color=cma_color, alpha=0.5, label='CMA')
        # Plot GA evolution
        sns.scatterplot(data=ga_top_10, x='iteration', y=param, ax=axes[idx],
                       color=ga_color, alpha=0.5, label='GA')
        
        axes[idx].set_title(f'{param} Evolution\n(Top 10% performers)', fontsize=14)
        axes[idx].tick_params(axis='x', labelsize=12)
        axes[idx].tick_params(axis='y', labelsize=12)
        axes[idx].legend()
    
    # Hide empty subplots
    for idx in range(len(param_columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_10_parameter_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Pairwise plots for top 10%
    print("Plotting pairwise parameters for top 10% performers")
    
    # Create a combined dataframe with a 'strategy' column
    cma_top_10['strategy'] = 'CMA'
    ga_top_10['strategy'] = 'GA'
    combined_top_10 = pd.concat([cma_top_10, ga_top_10])
    
    # Create the PairGrid
    g = sns.PairGrid(combined_top_10, hue='strategy', vars=param_columns, height=3)
    
    # Map different plots to upper and lower triangles
    g.map_upper(sns.scatterplot, alpha=0.5)
    g.map_lower(sns.kdeplot, alpha=0.5)
    g.map_diag(sns.histplot, alpha=0.5)
    
    # Add the legend with proper customization
    g.add_legend(title="Strategy", fontsize=12)
    
    # Get the legend object after it's been created
    legend = g.fig.legends[0]
    
    # Update legend title font size
    legend.set_title("Strategy", prop={"size": 12})
    
    # Update legend handle colors if needed (should be automatic based on hue)
    handles = legend.legendHandles
    for i, handle in enumerate(handles):
        if i == 0:  # First handle (CMA)
            handle.set_color(cma_color)
        elif i == 1:  # Second handle (GA)
            handle.set_color(ga_color)
    
    # Rotate labels
    for ax in g.axes[-1,:]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)
    for ax in g.axes[:,0]:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha='right', fontsize=14)
    
    plt.savefig(os.path.join(output_dir, 'top_10_parameter_pairwise.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    drugs = ["PI3Ki", "MEKi", "AKTi"]
    
    for drug in drugs:
        # Load CMA results
        cma_name = f"{drug}_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen"
        ga_name = f"{drug}_GA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen"
        
        try:
            cma_results_df = pd.read_csv(f'results/CMA_summaries/final_summary_{cma_name}.csv')
            ga_results_df = pd.read_csv(f'results/GA_summaries/final_summary_{ga_name}.csv')
            
            # Create output directory for comparison
            output_dir = f'results/evolutionary_algorithm_analysis/CMA_GA_comparison_{drug}'
            os.makedirs(output_dir, exist_ok=True)

            print(f"Running the comparison analysis for the {drug} drug")
            
            # Run the comparison analysis
            # compare_top_performers(cma_results_df, ga_results_df, output_dir)
            analyze_parameter_space(cma_results_df, output_dir, "CMA")
            analyze_parameter_space(ga_results_df, output_dir, "GA")
            
        except Exception as e:
            print(f"Error processing {drug}: {str(e)}")
            continue