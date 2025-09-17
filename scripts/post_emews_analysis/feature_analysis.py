import seaborn as sns
import matplotlib.pyplot as plt
import os, sys
# from scripts.post_emews_analysis.plot_3d_dimred import load_and_preprocess_data_single # TODO: THIS SHOULD NOT BE HERE, make a separate file for this
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed


def load_and_preprocess_data_single(file_path):
    """
    Load and preprocess data from a CSV file for Random Forest training and SHAP analysis.

    Args:
        file_path (str): Path to the CSV file containing the data.

    Returns:
        tuple: (scaled_features, target, metadata, columns)
            - scaled_features: Scaled feature data ready for model training.
            - target: The target variable (objective function).
            - metadata: DataFrame containing metadata columns.
            - columns: List of feature column names.
    """
    # Load the data from the CSV file
    df = pd.read_csv(file_path)

    # Identify metadata columns - check which ones exist in the dataframe
    metadata_columns = []
    if 'individual' in df.columns:
        metadata_columns.append('individual')
    if 'iteration' in df.columns:
        metadata_columns.append('iteration')
    if 'replicate' in df.columns:
        metadata_columns.append('replicate')

    metadata = df[metadata_columns]

    # Assume the last column is the target (objective function)
    target = df.iloc[:, -1]

    # Exclude metadata and target columns to get feature columns
    feature_columns = df.columns.difference(metadata_columns + [df.columns[-1]])
    features = df[feature_columns]

    # Extract column names for features
    columns = features.columns.tolist()

    # Preprocess the features
    # Here we use standard scaling, but this can be adjusted based on the data characteristics
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Optionally, split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

    # Metadata can include any additional information you want to extract or keep track of
    metadata = {
        'file_path': file_path,
        'num_samples': len(df),
        'num_features': len(columns)
    }

    return scaled_features, target, metadata, columns


def plot_correlation_matrix(df, features, filename):
    print("plotting correlation matrix")
    # Calculate the correlation matrix
    metric = df.columns[-1]
    corr_matrix = df[features + [metric]].corr()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create a heatmap for the lower triangle
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(f'Correlation Matrix of Parameters and {metric}')
    
    # Save the plot
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "feature_analysis", "correlation_matrix_heatmap")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'correlation_heatmap_{filename}_{metric}.png'), dpi=300, bbox_inches='tight')
    print(f"Correlation matrix heatmap saved to {os.path.join(save_dir, f'correlation_heatmap_{filename}_{metric}.png')}")
    
    # Write out the correlation coefficients to a CSV file
    correlation_df = corr_matrix[features + [metric]].iloc[:-1, -1]  # Get correlations with the metric
    correlation_df.to_csv(os.path.join(save_dir, f'correlation_heatmap_{filename}_{metric}.csv'), header=True)
    plt.close()


def plot_pairwise_relationships(df, features, filename):
    print("plotting pairwise relationships")
    # Include the metric as the last column
    metric = df.columns[-1]
    pair_data = df[features + [metric]]

    # Create a pair plot
    plt.figure(figsize=(12, 10))
    sns.pairplot(pair_data, diag_kind='kde', markers='o', plot_kws={'alpha': 0.5})
    plt.suptitle(f'Pairwise Relationships of Parameters and {metric}', y=1.02)
    
    # Save the pair plot
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "feature_analysis", "pairwise_relationships")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'pairplots_{filename}_{metric}.png'), dpi=300, bbox_inches='tight')
    print(f"Pairwise relationships plot saved to {os.path.join(save_dir, f'pairplots_{filename}_{metric}.png')}")
    plt.close()


importances = None
def plot_feature_importance(df, features, filename):
    # Define the target variable
    metric = df.columns[-1]
    X = df[features]
    y = df[metric]

    # Fit a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Get feature importances
    importances = model.feature_importances_

    # Create a DataFrame for feature importances
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Set up the figure with high-quality settings
    plt.figure(figsize=(3, 3), dpi=300)
    sns.set_context("paper", font_scale=0.8)
    sns.set_style("ticks")
    
    # Create the bar plot
    ax = sns.barplot(x='Importance', 
                    y='Feature', 
                    data=importance_df,
                    color='#2E8B57',  # Sea green color
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5)
    
    # Customize the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', width=0.5, length=3)
    
    # Use serif font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Liberation Sans', 
                                      'FreeSans', 'Arial', 'sans-serif']

    
    plt.xlabel("Feature Importance")
    # delete y label
    plt.gca().set_ylabel('')
    
    plt.tight_layout()
    
    # Save the plot
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           "results", "feature_analysis", "basic_feature_importance")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'feature_importance_{filename}_{metric}.png'), 
                dpi=300, 
                bbox_inches='tight',
                transparent=True)
    # save in SVG format
    plt.savefig(os.path.join(save_dir, f'feature_importance_{filename}_{metric}.svg'), 
                format='svg',
                dpi=300, 
                bbox_inches='tight',
                transparent=True)
    print(f"Feature importance plot saved to {os.path.join(save_dir, f'feature_importance_{filename}_{metric}.svg')}")
    plt.close()

def plot_shap_importance(df, features, filename, sample_size=1000, n_estimators=50):
    """
    Plot SHAP importance with optimizations for large datasets.
    Shows both magnitude and direction of SHAP values.
    
    Args:
        df: DataFrame with simulation results
        features: List of feature names
        filename: Output filename
        sample_size: Number of samples to use (default: 1000)
        n_estimators: Number of trees in Random Forest (default: 50)
    """
    print("Calculating SHAP importance with optimized processing")
    
    # Prepare data
    X = df[features]
    metric = df.columns[-1]
    y = df[metric]
    
    # Subsample if necessary
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X = X.iloc[idx]
        y = y.iloc[idx]
    
    # Convert to numpy for faster processing
    X_values = X.values
    y_values = y.values
    
    # Fit Random Forest with optimized parameters
    model = RandomForestRegressor(n_estimators=n_estimators, 
                                random_state=42,
                                n_jobs=-1)  # Use all CPU cores
    model.fit(X_values, y_values)
    
    # Calculate SHAP values efficiently
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_values)
    
    # Calculate mean SHAP values (keeping signs)
    mean_shap = shap_values.mean(0)
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': features,
        'SHAP Value': mean_shap
    })
    importance_df = importance_df.sort_values('SHAP Value', key=abs)
    
    # Set up the figure
    plt.figure(figsize=(2.5, 4), dpi=300)
    sns.set_context("paper", font_scale=0.8)
    sns.set_style("ticks")
    
    # Use sans-serif font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Liberation Sans', 
                                      'FreeSans', 'Arial', 'sans-serif']
    
    # Create diverging colors based on SHAP value signs
    colors = ['#FF4B4B' if x > 0 else '#4B4BFF' for x in importance_df['SHAP Value']]
    
    # Plot bars
    ax = plt.gca()
    bars = ax.barh(importance_df['Feature'], 
                  importance_df['SHAP Value'],
                  color=colors,
                  alpha=0.8,
                  height=0.7)
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    plt.xticks(rotation=45, ha='right')  # ha='right' aligns the rotated labels nicely

    
    # Customize the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', width=0.8, length=2, labelsize=10, colors='black')  # Increased label size
    
    # Add labels
    plt.xlabel("Mean SHAP Value", 
              fontsize=11,
              fontweight='bold')
    plt.gca().set_ylabel('')
    
    plt.tight_layout()

    # Save plots
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           "results", "feature_analysis", "shap_importance")
    os.makedirs(save_dir, exist_ok=True)
    
    base_path = os.path.join(save_dir, f'shap_importance_{filename}_{metric}')
    plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(f'{base_path}.svg', format='svg', bbox_inches='tight', transparent=True)
    print(f"SHAP importance plot saved to {base_path}.png")
    plt.close()

    # Save the SHAP importance dataframe to a CSV file
    importance_df.to_csv(os.path.join(save_dir, f'shap_importance_{filename}_{metric}.csv'), index=False)
    print(f"SHAP importance CSV saved to {os.path.join(save_dir, f'shap_importance_{filename}_{metric}.csv')}")

def plot_permutation_importance(df, features, filename, sample_size=1000, n_estimators=50, n_repeats=30):
    """
    Plot permutation importance with optimizations for large datasets.
    Compact square version with black colors for publication figures.
    """
    print("Calculating permutation importance with optimized processing")
    
    # Prepare data
    X = df[features]
    metric = df.columns[-1]
    y = df[metric]
    
    # Subsample if necessary
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X = X.iloc[idx]
        y = y.iloc[idx]
    
    # Convert to numpy for faster processing
    X_values = X.values
    y_values = y.values
    
    # Fit Random Forest
    model = RandomForestRegressor(n_estimators=n_estimators, 
                                random_state=42,
                                n_jobs=-1)
    model.fit(X_values, y_values)
    
    # Calculate permutation importance
    result = permutation_importance(model, X_values, y_values, 
                                  n_repeats=n_repeats, 
                                  random_state=42,
                                  n_jobs=-1)
    
    # Normalize importance values
    normalized_importance = result.importances_mean / result.importances_mean.sum()
    
    # Create DataFrame for plotting
    perm_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': normalized_importance
    }).sort_values('Importance', ascending=True)

    # Set up the figure
    plt.figure(figsize=(2, 4), dpi=300)
    sns.set_context("paper", font_scale=0.8)
    sns.set_style("white")
    
    # Use sans-serif font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # Plot bars
    ax = plt.gca()
    bars = ax.barh(perm_importance_df['Feature'],
                  perm_importance_df['Importance'],
                  color='black',
                  alpha=1.0,
                  height=0.7)  # Reduced height for compactness
    
    # Customize the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', width=0.8, length=2, labelsize=12, colors='black')  # Increased label size
    
    # Add labels
    plt.xlabel("Norm. PI", fontsize=11, color='black')
    plt.gca().set_ylabel('')
        
    # Add light grid lines
    ax.grid(True, axis='x', linestyle='--', alpha=0.2, color='black', linewidth=0.5)
    
    # Adjust layout to be more square
    plt.tight_layout()  # Reduced padding

    # Save plots
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           "results", "feature_analysis", "permutation_importance")
    os.makedirs(save_dir, exist_ok=True)
    
    base_path = os.path.join(save_dir, f'permutation_importance_{filename}_{metric}')
    plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(f'{base_path}.svg', format='svg', bbox_inches='tight', transparent=True)
    print(f"Permutation importance plot saved to {base_path}.png")
    plt.close()

    # Save the permutation importance dataframe to a CSV file
    csv_path = os.path.join(save_dir, f'permutation_importance_{filename}_{metric}.csv')
    perm_importance_df.to_csv(csv_path, index=False)
    print(f"Permutation importance CSV saved to {csv_path}")

    

def plot_partial_dependence(df, features, filename, sample_size=1000, n_estimators=50, grid_resolution=20):
    """
    Plot partial dependence with optimizations for large datasets.
    
    Args:
        df: DataFrame with simulation results
        features: List of feature names
        filename: Output filename
        sample_size: Number of samples to use (default: 1000)
        n_estimators: Number of trees in Random Forest (default: 50)
        grid_resolution: Number of points for PDP grid (default: 20)
    """
    print("Calculating partial dependence with optimized processing")
    
    # Prepare data
    X = df[features]
    metric = df.columns[-1]
    y = df[metric]
    
    # Subsample if necessary
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X = X.iloc[idx]
        y = y.iloc[idx]
    
    # Convert to numpy for faster processing
    X_values = X.values
    y_values = y.values
    
    # Fit Random Forest with optimized parameters
    model = RandomForestRegressor(n_estimators=n_estimators, 
                                random_state=42,
                                n_jobs=-1)  # Use all CPU cores
    model.fit(X_values, y_values)
    
    # Set up the plot grid
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(12, 4*n_rows),
                            dpi=300)
    sns.set_context("paper", font_scale=0.8)
    sns.set_style("ticks")
    
    # Use sans-serif font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Liberation Sans', 
                                      'FreeSans', 'Arial', 'sans-serif']
    
    # Convert axes to 1D array if it's a single row
    if n_rows == 1:
        axes = np.array([axes])
    
    # Flatten axes array for easier iteration
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Calculate and plot PDP for each feature
    for idx, (feature, ax) in enumerate(zip(features, axes_flat)):
        try:
            # Calculate PDP with optimized grid resolution
            pdp = partial_dependence(
                model, 
                X, 
                [features.index(feature)],
                kind='average',
                grid_resolution=grid_resolution
            )
            
            # Extract values
            grid_points = pdp['grid_values'][0]
            pd_values = pdp['average'][0]
            
            # Plot with optimized styling
            axes_flat[idx].plot(grid_points, pd_values, 
                              color='#2E8B57',
                              linewidth=1,
                              zorder=2)
            
            # Add light grid
            axes_flat[idx].grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=1)
            
            # Customize subplot
            axes_flat[idx].spines['top'].set_visible(False)
            axes_flat[idx].spines['right'].set_visible(False)
            axes_flat[idx].spines['left'].set_linewidth(0.5)
            axes_flat[idx].spines['bottom'].set_linewidth(0.5)
            
            # Adjust tick parameters
            axes_flat[idx].tick_params(axis='both', 
                                     width=0.5, 
                                     length=3, 
                                     labelsize=8)
            
            # Labels
            axes_flat[idx].set_xlabel(feature, fontsize=8, fontweight='bold')
            if idx % n_cols == 0:
                axes_flat[idx].set_ylabel('Partial dependence', fontsize=8, fontweight='bold')
                
        except Exception as e:
            print(f"Error plotting PDP for {feature}: {str(e)}")
            axes_flat[idx].text(0.5, 0.5, 'Error calculating PDP',
                              ha='center', va='center')
    
    # Remove empty subplots
    for idx in range(len(features), len(axes_flat)):
        fig.delaxes(axes_flat[idx])
    
    plt.tight_layout()
    
    # Save plots
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           "results", "feature_analysis", "partial_dependence")
    os.makedirs(save_dir, exist_ok=True)
    
    base_path = os.path.join(save_dir, f'pdp_grid_{filename}_{metric}')
    plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(f'{base_path}.svg', format='svg', bbox_inches='tight', transparent=True)
    print(f"Partial dependence grid plot saved to {base_path}.png")
    plt.close()

def analyze_feature_interactions(df, features, filename):
    """Analyze feature interactions using SHAP interaction values and PDPs"""
    print("Analyzing feature interactions")
    
    # Prepare data
    X = df[features]
    metric = df.columns[-1]
    y = df[metric]
    
    # Fit Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Calculate SHAP interaction values
    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(X)
    
    # Calculate mean absolute interaction values
    mean_interactions = np.abs(shap_interaction_values).mean(0)
    
    # Create interaction matrix plot
    plt.figure(figsize=(4, 3))
    sns.set_context("paper", font_scale=0.8)
    sns.set_style("ticks")
    
    # Plot interaction matrix
    im = plt.imshow(mean_interactions, cmap='RdBu')
    
    # Customize plot
    plt.xticks(range(len(features)), features, rotation=45, ha='right', fontsize=7)
    plt.yticks(range(len(features)), features, fontsize=7)
    cbar = plt.colorbar(im, label='Mean |SHAP interaction value|')
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Mean |SHAP interaction value|', size=7)
    
    plt.tight_layout()
    
    # Save interaction matrix
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           "results", "feature_analysis", "interactions")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'shap_interactions_{filename}.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'shap_interactions_{filename}.svg'), 
                format='svg', bbox_inches='tight')
    plt.close()
    
    # Get top N interactions
    n_top = 5
    interaction_strength = mean_interactions.copy()
    np.fill_diagonal(interaction_strength, 0)  # Remove self-interactions
    top_pairs = []
    
    for _ in range(n_top):
        max_idx = np.unravel_index(np.argmax(interaction_strength), interaction_strength.shape)
        if interaction_strength[max_idx] > 0:
            top_pairs.append((features[max_idx[0]], features[max_idx[1]]))
            interaction_strength[max_idx] = 0
            interaction_strength[max_idx[1], max_idx[0]] = 0

    # Plot PDPs for top interactions
    for feat1, feat2 in top_pairs:
        plt.figure(figsize=(4, 3))
        sns.set_context("paper", font_scale=0.8)
        
        try:
            # Calculate partial dependence with new API
            pdp_result = partial_dependence(
                model, X, 
                features=[features.index(feat1), features.index(feat2)],
                kind='average',
                grid_resolution=20
            )
            
            # Extract grid values and PDP values
            grid_values = pdp_result['grid_values']
            pdp_values = pdp_result['average']
            
            # Create meshgrid for plotting
            XX, YY = np.meshgrid(grid_values[0], grid_values[1])
            Z = pdp_values.T
            
            # Plot contour
            plt.contourf(XX, YY, Z, levels=15, cmap='RdBu_r')
            cbar = plt.colorbar(label=f'{metric}')
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label(metric, size=7)
            
            # Customize plot
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            
            plt.xlabel(feat1, fontsize=8, fontweight="bold")
            plt.ylabel(feat2, fontsize=8, fontweight="bold")
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            
            # Add actual data points
            plt.scatter(X[feat1], X[feat2], c='black', s=1, alpha=0.1)
            
            plt.tight_layout()
            
            # Save PDP
            plt.savefig(os.path.join(save_dir, f'pdp_{feat1}_{feat2}_{filename}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'pdp_{feat1}_{feat2}_{filename}.svg'), 
                        format='svg', bbox_inches='tight')
            
        except Exception as e:
            print(f"Error creating PDP for {feat1} vs {feat2}: {str(e)}")
            
        finally:
            plt.close()

    return top_pairs


def _run_bootstrap_iteration(X, y, features, seed):
    """Helper function for a single bootstrap iteration."""
    # Create a bootstrap sample using a dedicated RandomState for reproducibility
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(X), size=len(X), replace=True)
    X_boot = X.iloc[indices]
    y_boot = y.iloc[indices]

    # Fit a new model on the bootstrap sample
    # n_jobs is set to 1 to avoid over-subscription when this is run in parallel
    model_boot = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=1)
    model_boot.fit(X_boot, y_boot)

    # Calculate SHAP values
    explainer_boot = shap.TreeExplainer(model_boot)
    shap_values_boot = explainer_boot.shap_values(X_boot)

    # Return mean absolute SHAP for each feature
    return np.abs(shap_values_boot).mean(axis=0)


def stability_analysis(df, features, filename, n_bootstrap=100):
    """
    Performs a stability analysis of SHAP feature importances using bootstrapping.
    This process is parallelized to run faster on multi-core machines.
    
    Args:
        df (pd.DataFrame): The input dataframe with features and a target column.
        features (list): The list of feature names.
        filename (str): The base filename for saving outputs.
        n_bootstrap (int): The number of bootstrap samples to create.
    """
    print(f"--- Performing SHAP Stability Analysis with {n_bootstrap} bootstraps (in parallel) ---")
    
    X = df[features]
    metric = df.columns[-1]
    y = df[metric]
    
    # Use joblib to parallelize the bootstrap iterations
    # n_jobs=-1 uses all available CPU cores.
    # pre_dispatch='n_jobs' and batch_size=1 ensures jobs are sent out
    # at a steady rate, making the progress bar update smoothly for long tasks.
    bootstrap_shap_values = Parallel(n_jobs=-1, pre_dispatch='n_jobs', batch_size=1)(
        delayed(_run_bootstrap_iteration)(X, y, features, i) 
        for i in range(n_bootstrap)
    )
    
    # Convert list of arrays to a DataFrame
    shap_stability_df = pd.DataFrame(bootstrap_shap_values, columns=features)
    
    # --- Visualization ---
    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("whitegrid")
    
    # Sort features by the median importance for clearer plotting
    median_importances = shap_stability_df.median().sort_values(ascending=False)
    sorted_features = median_importances.index
    
    # Create the boxplot
    sns.boxplot(data=shap_stability_df[sorted_features], orient='h', palette='viridis_r', fliersize=2)
    
    plt.xlabel("Mean Absolute SHAP Value (Feature Importance)", fontweight='bold')
    plt.ylabel("Features", fontweight='bold')
    plt.title(f"SHAP Feature Importance Stability ({n_bootstrap} Bootstraps)", fontweight='bold')
    plt.tight_layout()

    # --- Save Results ---
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           "results", "feature_analysis", "stability_analysis")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save plot
    plot_path = os.path.join(save_dir, f'shap_stability_{filename}_{metric}')
    plt.savefig(f'{plot_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{plot_path}.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    # Save the raw stability data
    csv_path = os.path.join(save_dir, f'shap_stability_data_{filename}_{metric}.csv')
    shap_stability_df.to_csv(csv_path, index=False)
    
    print(f"SHAP stability analysis complete. Results saved in {save_dir}")

def analyze_by_generation(df, columns, filename):
    # Get unique generations (iterations)
    generations = df['iteration'].unique()
    
    # Sort generations to process them in order
    generations.sort()
    
    # Select generations to analyze:
    # First generation, every 10th generation, and last generation
    selected_generations = []
    
    # Add first generation
    selected_generations.append(generations[0])
    
    # Add every 10th generation
    for gen in generations[::10]:
        if gen not in selected_generations:  # Avoid duplicates
            selected_generations.append(gen)
    
    # Add last generation if not already included
    if generations[-1] not in selected_generations:
        selected_generations.append(generations[-1])
    
    # Sort the selected generations
    selected_generations.sort()
    
    print(f"Processing {len(selected_generations)} generations out of {len(generations)} total for {filename}")
    print(f"Selected generations: {selected_generations}")
    
    for gen in selected_generations:
        print(f"\nAnalyzing generation {gen}")
        # Subset the dataframe for this generation
        gen_df = df[df['iteration'] == gen].copy()
        # Create a generation-specific filename
        gen_filename = f"{filename}_gen_{gen}"
        # Run comprehensive analysis on this generation's data
        comprehensive_feature_analysis(gen_df, columns, gen_filename)



def plot_feature_importance_evolution(df, features, filename):
    print("plotting feature importance evolution")
    metric = df.columns[-1]
    generations = sorted(df['iteration'].unique())
    
    # Store importances for each generation
    importance_over_time = []
    
    for gen in generations:
        gen_df = df[df['iteration'] == gen]
        X = gen_df[features]
        y = gen_df[metric]
        
        # Fit Random Forest for this generation
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Store importances with generation info
        importance_dict = {'Generation': gen}
        importance_dict.update({feat: imp for feat, imp in zip(features, model.feature_importances_)})
        importance_over_time.append(importance_dict)
    
    # Convert to DataFrame for plotting
    importance_df = pd.DataFrame(importance_over_time)
    
    plt.figure(figsize=(15, 6))
    sns.set_context("paper")
    sns.set_style("white")
    # Option 4: Create a custom palette by combining different color schemes
    palette = (sns.color_palette("Set1", 6) + 
              sns.color_palette("deep", 6) + 
              sns.color_palette("Set3", 12))


    ax = importance_df.plot(x='Generation', 
                          y=features,
                          kind='bar',
                          stacked=True,
                          color=palette,
                          width=0.8)
    
    # Option 1: Show fewer ticks (e.g., every 5th generation)
    tick_positions = range(0, len(generations), 5)
    plt.xticks(tick_positions, [generations[i] for i in tick_positions], rotation=0)

    # set y axis limits to 0-1
    plt.ylim(0, 1)
    
    # Option 2 (alternative): Rotate labels 45 degrees
    # plt.xticks(range(len(generations)), generations, rotation=45, ha='right')
    
    # plt.title(f'Feature Importance Evolution Over Generations for {metric}')
    # bold the labels
    plt.xlabel('Generation', fontweight='bold', fontsize=14) 
    plt.ylabel('Feature Importance', fontweight='bold', fontsize=14)
    plt.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # Save the plot
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           "results", "feature_analysis", "feature_importance_evolution")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'feature_importance_evolution_{filename}_{metric}.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Feature importance evolution plot saved to {os.path.join(save_dir, f'feature_importance_evolution_{filename}_{metric}.png')}")
    plt.close()


def process_simulation_file(file_path):
    print("Processing:", file_path)
    # Read only necessary columns
    df = pd.read_csv(file_path, usecols=['iteration'] + features + [metric])
    
    filename = os.path.basename(file_path).split(".csv")[0]
    plot_shap_importance_evolution(df, features, filename, 
                                 sample_size=1000,  # Adjust based on your needs
                                 n_estimators=50)   # Adjust based on your needs


def plot_shap_importance_evolution(df, features, filename, sample_size=1000, n_estimators=50):
    """
    Plot SHAP importance evolution with optimizations for large datasets.
    Shows absolute importance values as a stacked bar plot.
    """
    print("Plotting SHAP importance evolution with optimized processing")
    metric = df.columns[-1]
    generations = sorted(df['iteration'].unique())
    
    # Pre-allocate storage for SHAP values
    importance_over_time = []
    
    # Convert to numpy arrays for faster processing
    X_full = df[features].values
    y_full = df[metric].values
    
    for gen in generations:
        # Get indices for current generation
        gen_mask = df['iteration'] == gen
        X_gen = X_full[gen_mask]
        y_gen = y_full[gen_mask]
        
        # Subsample if necessary
        if len(X_gen) > sample_size:
            idx = np.random.choice(len(X_gen), sample_size, replace=False)
            X_gen = X_gen[idx]
            y_gen = y_gen[idx]
        
        # Fit Random Forest with fewer trees
        model = RandomForestRegressor(n_estimators=n_estimators, 
                                    random_state=42,
                                    n_jobs=-1)
        model.fit(X_gen, y_gen)
        
        # Calculate SHAP values efficiently
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_gen)
        
        # Calculate mean absolute SHAP values
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        # Normalize values to sum to 1
        normalized_values = mean_shap_values / mean_shap_values.sum()
        
        # Store results
        importance_dict = {
            'Generation': gen,
            **{feat: imp for feat, imp in zip(features, normalized_values)}
        }
        importance_over_time.append(importance_dict)
    
    # Convert to DataFrame for plotting
    importance_df = pd.DataFrame(importance_over_time)
    
    # Create the plot
    plt.figure(figsize=(15, 6))
    sns.set_context("paper")
    sns.set_style("white")
    
    # Create a custom palette
    palette = (sns.color_palette("Set1", 6) + 
              sns.color_palette("deep", 6) + 
              sns.color_palette("Set3", 12))
    
    # Create stacked bar plot
    ax = importance_df.plot(x='Generation',
                          y=features,
                          kind='bar',
                          stacked=True,
                          color=palette,
                          width=0.8)
    
    # Show fewer ticks
    tick_positions = range(0, len(generations), 5)
    plt.xticks(tick_positions, [generations[i] for i in tick_positions], rotation=0)
    
    # Set y axis limits to 0-1
    plt.ylim(0, 1)
    
    # Labels
    plt.xlabel('Generation', fontweight='bold', fontsize=12)
    plt.ylabel('Normalized |SHAP value|', fontweight='bold', fontsize=12)
    
    # Legend
    plt.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plots
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           "results", "feature_analysis", "shap_importance_evolution")
    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(os.path.join(save_dir, f'shap_evolution_{filename}_{metric}.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'shap_evolution_{filename}_{metric}.svg'), 
                format='svg', bbox_inches='tight')
    print(f"SHAP importance evolution plot saved to {os.path.join(save_dir, f'shap_evolution_{filename}_{metric}.png')}")
    plt.close()


def plot_permutation_importance_evolution(df, features, filename, sample_size=1000, n_estimators=50, n_repeats=30):
    """
    Plot permutation importance evolution with optimizations for large datasets.
    Shows normalized importance values as a stacked bar plot.
    
    Args:
        df: DataFrame with simulation results
        features: List of feature names
        filename: Output filename
        sample_size: Number of samples to use per generation (default: 1000)
        n_estimators: Number of trees in Random Forest (default: 50)
        n_repeats: Number of permutation repeats (default: 30)
    """
    print("Calculating permutation importance evolution with optimized processing")
    metric = df.columns[-1]
    generations = sorted(df['iteration'].unique())
    
    # Store importances for each generation
    importance_over_time = []
    
    # Process each generation with progress bar
    for gen in generations:
        gen_df = df[df['iteration'] == gen]
        X = gen_df[features]
        y = gen_df[metric]
        
        # Subsample if necessary
        if len(X) > sample_size:
            idx = np.random.choice(len(X), sample_size, replace=False)
            X = X.iloc[idx]
            y = y.iloc[idx]
        
        # Convert to numpy for faster processing
        X_values = X.values
        y_values = y.values
        
        # Fit Random Forest with optimized parameters
        model = RandomForestRegressor(n_estimators=n_estimators, 
                                    random_state=42,
                                    n_jobs=-1)  # Use all CPU cores
        model.fit(X_values, y_values)
        
        # Calculate permutation importance efficiently
        result = permutation_importance(model, X_values, y_values, 
                                      n_repeats=n_repeats, 
                                      random_state=42,
                                      n_jobs=-1)  # Parallel processing
        
        # Normalize importances
        normalized_importance = result.importances_mean / result.importances_mean.sum()
        
        # Store results
        importance_dict = {
            'Generation': gen,
            **{feat: imp for feat, imp in zip(features, normalized_importance)}
        }
        importance_over_time.append(importance_dict)
    
    # Convert to DataFrame for plotting
    importance_df = pd.DataFrame(importance_over_time)
    
    # Create the plot
    plt.figure(figsize=(15, 6))
    sns.set_context("paper")
    sns.set_style("white")
    
    # Create color palette
    palette = (sns.color_palette("Set1", 6) + 
              sns.color_palette("deep", 6) + 
              sns.color_palette("Set3", 12))
    
    # Create stacked bar plot
    ax = importance_df.plot(x='Generation',
                          y=features,
                          kind='bar',
                          stacked=True,
                          color=palette,
                          width=0.8)
    
    # Show fewer ticks
    tick_positions = range(0, len(generations), 5)
    plt.xticks(tick_positions, [generations[i] for i in tick_positions], rotation=0)
    
    # Set y axis limits to 0-1
    plt.ylim(0, 1)
    
    # Labels
    plt.xlabel('Generation', fontweight='bold', fontsize=14)
    plt.ylabel('Normalized Permutation Importance', fontweight='bold', fontsize=14)
    
    # Legend
    plt.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plots
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           "results", "feature_analysis", "permutation_importance_evolution")
    os.makedirs(save_dir, exist_ok=True)
    
    base_path = os.path.join(save_dir, f'permutation_importance_evolution_{filename}_{metric}')
    plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base_path}.svg', format='svg', bbox_inches='tight')
    print(f"Permutation importance evolution plot saved to {base_path}.png")
    plt.close()



def transform_parameter_names(df, columns):
    """
    Transform parameter names to LaTeX mathematical notation.
    Handles both experimental and control parameter sets.
    Returns transformed DataFrame and column list.
    """
    # Detect if we're dealing with control case by checking for characteristic columns
    is_control = any('cell_spacing' in col or 'Hill_coeff_pressure' in col for col in columns)
    
    if is_control:
        name_mapping = {
            'user_parameters.cell_spacing': 'Cell spacing',
            'user_parameters.new_intracellular_dt': 'Modified Intracel. dt',
            'user_parameters.new_scaling': 'Modified scaling',
            'hill_coeff_pressure': r'$H_{p}$',
            'pressure_half': r'$K_{A,p}$',
            'user_parameters.tumor_radius': 'Tumor radius',
            'basal_growth_rate': r'$r_{\gamma,basal}$'
        }
    else:
        name_mapping = {
            # Experimental case mappings
            'K_half_growth': r'$K_{A,\gamma}$',
            'K_half_apoptosis': r'$K_{A,\alpha}$',
            'hill_coeff_growth': r'$H_{\gamma}$',
            'hill_coeff_apoptosis': r'$H_{\alpha}$',
            'w_pro_cMYC': r'$\omega_{\gamma, cMYC}$',
            'w_pro_TCF': r'$\omega_{\gamma, TCF}$',
            'w_pro_RSK': r'$\omega_{\gamma, RSK}$',
            'w_anti_FOXO': r'$\omega_{\alpha, FOXO}$',
            'w_anti_Caspase8': r'$\omega_{\alpha, Casp8}$',
            'w_anti_Caspase9': r'$\omega_{\alpha, Casp9}$',
            'drug_X_permeability': r'$k_{drug,X}$',
            'drug_X_node_rate_scaling': r'$scaling_{drug, X}$',
            'drug_X_node_rate_threshold': r'$threshold_{drug, X}$',
            'drug_Y_permeability': r'$k_{drug,Y}$',
            'drug_Y_node_rate_scaling': r'$scaling_{drug, Y}$',
            'drug_Y_node_rate_threshold': r'$threshold_{drug, Y}$',
            'max_apoptosis_rate': r'$r_{\alpha, max}$',
            'response_rate_apoptosis': r'$r_{\alpha, response}$',
            'response_rate_growth': r'$r_{\gamma, response}$',
            'user_parameters.new_intracellular_dt': r'$dt_{intracellular}$',
            'user_parameters.new_scaling': r'$scaling_{intracellular}$',
            'drug_X_kon': r'$K_{on,drug,X}$',
            'drug_X_koff': r'$K_{off,drug,X}$',
            'drug_Y_kon': r'$K_{on,drug,Y}$',
            'drug_Y_koff': r'$K_{off,drug,Y}$',
            'drug_Y_target_concentration': r'$[drug,Y]_{target}$'
        }
    
    # Create new DataFrame with renamed columns
    df_transformed = df.copy()
    columns_transformed = []
    
    for col in columns:
        if col in name_mapping:
            # Rename in DataFrame
            if col in df_transformed.columns:
                df_transformed = df_transformed.rename(columns={col: name_mapping[col]})
            # Add to transformed columns list
            columns_transformed.append(name_mapping[col])
        else:
            # Keep original name if no mapping exists
            columns_transformed.append(col)
            print(f"Warning: No mapping found for column '{col}'")
    
    return df_transformed, columns_transformed

def comprehensive_feature_analysis(df, features, filename):
    # plot_correlation_matrix(df, list(columns), filename)
    # plot_pairwise_relationships(df, features, filename)
    # plot_feature_importance(df, features, filename)
    plot_shap_importance(df, features, filename)
    # plot_permutation_importance(df, features, filename)
    # plot_partial_dependence(df, features, filename)
    # analyze_feature_interactions(df, features, filename)
    stability_analysis(df, features, filename)

def plot_shap_importance_exclude_intracel_scaling(df, features, filename, sample_size=1000, n_estimators=50):
    """
    Plot SHAP importance, summary, and dependence plots, excluding intracellular_dt and scaling.
    """
    import shap
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import seaborn as sns

    # Exclude any feature containing 'intracellular' or 'scaling' (case-insensitive)
    filtered_features = [col for col in features if 'intracellular' not in col.lower() and 'scaling' not in col.lower()]
    X = df[filtered_features]
    metric = df.columns[-1]
    y = df[metric]

    # Subsample if necessary
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X = X.iloc[idx]
        y = y.iloc[idx]

    # Fit Random Forest
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Calculate mean SHAP values (keeping signs)
    mean_shap = shap_values.mean(0)
    importance_df = pd.DataFrame({'Feature': filtered_features, 'SHAP Value': mean_shap})
    importance_df = importance_df.sort_values('SHAP Value', key=abs)

    # Set up the figure
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "feature_analysis", "shap_importance_exclude_intracel_scaling")
    os.makedirs(save_dir, exist_ok=True)
    base_path = os.path.join(save_dir, f'shap_importance_{filename}_{metric}_no_intracel_scaling')

    # SHAP bar plot
    plt.figure(figsize=(2.5, 4), dpi=300)
    sns.set_context("paper", font_scale=0.8)
    sns.set_style("ticks")
    colors = ['#FF4B4B' if x > 0 else '#4B4BFF' for x in importance_df['SHAP Value']]
    ax = plt.gca()
    bars = ax.barh(importance_df['Feature'], importance_df['SHAP Value'], color=colors, alpha=0.8, height=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel("Mean SHAP Value", fontsize=11, fontweight='bold')
    plt.gca().set_ylabel('')
    plt.tight_layout()
    plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(f'{base_path}.svg', format='svg', bbox_inches='tight', transparent=True)
    plt.close()
    importance_df.to_csv(f'{base_path}.csv', index=False)

    # SHAP summary (beeswarm) plot
    plt.figure(figsize=(6, 4))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(f'{base_path}_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()

    # SHAP dependence plots for top 3 features
    top_features = importance_df['Feature'].iloc[-3:]
    for feat in top_features:
        plt.figure(figsize=(4, 3))
        shap.dependence_plot(feat, shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(f'{base_path}_dependence_{feat}.png', dpi=300, bbox_inches='tight')
        plt.close()

def comprehensive_feature_analysis_exclude_intracel_scaling(df, features, filename):
    plot_shap_importance_exclude_intracel_scaling(df, features, filename)





def main():
    """
    Main function to run feature analysis on specific, predefined experiments.
    """
    # Define specific experiments to analyze, matching global_sensitivity_analysis.py
    cma_experiments = [
        "PI3Ki_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen",
        "MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen",
        "AKTi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen"
    ]

    sweep_experiments = [
        "synergy_sweep-pi3k_mek-1104-2212-18p_transient_delayed_uniform_5k_10p",
        "synergy_sweep-pi3k_mek-1104-2212-18p_PI3K_transient_delayed_uniform_5k_singledrug",
        "synergy_sweep-pi3k_mek-1104-2212-18p_MEK_transient_delayed_uniform_5k_singledrug",
        "synergy_sweep-akt_mek-1204-1639-18p_transient_delayed_uniform_postdrug_RMSE_5k",
        "synergy_sweep-akt_mek-1104-2212-18p_AKT_transient_delayed_uniform_5k_singledrug",
        "synergy_sweep-akt_mek-1104-2212-18p_MEK_transient_delayed_uniform_5k_singledrug"
    ]

    # Base directories
    cma_base_dir = "./results/CMA_summaries"
    sweep_base_dir = "./results/sweep_summaries"
    
    # Combine all experiment definitions
    all_exp_definitions = []
    for exp_name in cma_experiments:
        all_exp_definitions.append({'name': exp_name, 'base_dir': cma_base_dir})
    for exp_name in sweep_experiments:
        all_exp_definitions.append({'name': exp_name, 'base_dir': sweep_base_dir})

    # Process all selected experiments
    for exp in all_exp_definitions:
        exp_name = exp['name']
        base_dir = exp['base_dir']
        
        # Construct the path to the final summary CSV file
        filename_csv = f"final_summary_{exp_name}.csv"
        file_path = os.path.join(base_dir, filename_csv)
        
        print(f"\n--- Processing Experiment: {exp_name} ---")
        
        if os.path.exists(file_path):
            print(f"Found file: {file_path}")
            
            # Load data and identify feature columns
            df = pd.read_csv(file_path)
            
            # Identify feature columns (excluding metadata and target)
            metadata_cols = [col for col in ['individual', 'iteration', 'replicate'] if col in df.columns]
            target_col = df.columns[-1]
            feature_cols = df.columns.difference(metadata_cols + [target_col]).tolist()
            
            # Transform parameter names for plotting
            df_transformed, columns_transformed = transform_parameter_names(df, feature_cols)
            
            # Run the comprehensive analysis
            comprehensive_feature_analysis(df_transformed, columns_transformed, exp_name)
            
            # Optionally, run the analysis excluding certain parameters
            # comprehensive_feature_analysis_exclude_intracel_scaling(df_transformed, columns_transformed, exp_name)
        else:
            print(f"File not found, skipping: {file_path}")

    
if __name__ == "__main__":
    main()

