import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import seaborn as sns
import os

def plot_3d_reduction(data, labels, title, output_path, method='pca'):
    """
    Create 3D plot of dimensionality reduction results
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create color map based on generation number
    norm = plt.Normalize(labels.min(), labels.max())
    cmap = plt.cm.viridis
    
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                        c=labels, 
                        cmap=cmap,
                        alpha=0.6)
    
    plt.colorbar(scatter, label='Generation')
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title(f'3D {method.upper()} of Parameter Space\nColored by Generation')
    
    # Remove top and right spines
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_parameter_space_3d(strategy_name):
    """
    Analyze parameter space using different dimensionality reduction techniques
    """
    # Read the data
    if "CMA" in strategy_name:
        results_df = pd.read_csv(f'results/CMA_summaries/final_summary_{strategy_name}.csv')
    elif "GA" in strategy_name:
        results_df = pd.read_csv(f'results/GA_summaries/final_summary_{strategy_name}.csv')
    else:
        raise ValueError(f"Strategy name {strategy_name} not recognized")

    # Create output directory
    output_dir = f'results/evolutionary_algorithm_analysis/{strategy_name}/dimensionality_reduction'
    os.makedirs(output_dir, exist_ok=True)

    # Separate features and generation numbers
    non_param_columns = ['individual', 'replicate', 'iteration', 'fitness', 'time']
    param_columns = [col for col in results_df.columns if col not in non_param_columns]
    
    # Extract features and scale them
    X = results_df[param_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Get generation labels
    labels = results_df['iteration'].values

    # 1. PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X_scaled)
    plot_3d_reduction(pca_result, labels, 
                     'PCA', 
                     os.path.join(output_dir, 'pca_3d.png'),
                     'PCA')
    
    # Print explained variance
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)

    # 2. t-SNE
    tsne = TSNE(n_components=3, random_state=42)
    tsne_result = tsne.fit_transform(X_scaled)
    plot_3d_reduction(tsne_result, labels,
                     't-SNE',
                     os.path.join(output_dir, 'tsne_3d.png'),
                     't-SNE')

    # 3. UMAP
    reducer = umap.UMAP(n_components=3, random_state=42)
    umap_result = reducer.fit_transform(X_scaled)
    plot_3d_reduction(umap_result, labels,
                     'UMAP',
                     os.path.join(output_dir, 'umap_3d.png'),
                     'UMAP')

    # # Create animation (optional)
    # create_rotation_animation(pca_result, labels,
    #                         os.path.join(output_dir, 'pca_3d_rotation.gif'),
    #                         'PCA')

def create_rotation_animation(data, labels, output_path, method):
    """
    Create rotating animation of 3D plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create color map
    norm = plt.Normalize(labels.min(), labels.max())
    cmap = plt.cm.viridis
    
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                        c=labels, 
                        cmap=cmap,
                        alpha=0.6)
    
    plt.colorbar(scatter, label='Generation')
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title(f'3D {method.upper()} of Parameter Space\nColored by Generation')
    
    # Remove pane fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Create frames for rotation
    angles = np.linspace(0, 360, 180)
    
    def update(frame):
        ax.view_init(elev=20., azim=angles[frame])
        return fig,
    
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=len(angles), interval=50, blit=True)
    anim.save(output_path, writer='pillow')
    plt.close()

def plot_combined_3d_reduction(strategy_names, method='pca'):
    """
    Create combined 3D plot of dimensionality reduction results for multiple experiments
    """
    # Setup markers and colors for drugs and optimization strategies
    markers = {'PI3Ki': 'o', 'MEKi': 's', 'AKTi': '^'}  # circle, square, triangle
    colors = {'CMA': 'tab:blue', 'GA': 'tab:orange'}
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Store all data for combined scaling
    all_data = []
    all_labels = []
    
    # First, collect and preprocess all data
    for strategy_name in strategy_names:
        # Read data
        if "CMA" in strategy_name:
            results_df = pd.read_csv(f'results/CMA_summaries/final_summary_{strategy_name}.csv')
            opt_strategy = "CMA"
        elif "GA" in strategy_name:
            results_df = pd.read_csv(f'results/GA_summaries/final_summary_{strategy_name}.csv')
            opt_strategy = "GA"
            
    
        fitness_column = results_df.columns[-1]
        # Get parameters
        non_param_columns = ['individual', 'replicate', 'iteration', 'time']
        param_columns = [col for col in results_df.columns if col not in non_param_columns]
        
        # Extract and store features
        X = results_df[param_columns].values
        all_data.append(X)
        all_labels.append((results_df[fitness_column].values, opt_strategy))
    
    # Combine and scale all data together
    X_combined = np.vstack(all_data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Apply dimensionality reduction to all data
    if method.lower() == 'pca':
        reducer = PCA(n_components=3)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=3, random_state=42)
    else:  # UMAP
        reducer = umap.UMAP(n_components=3, random_state=42)
    
    reduced_data = reducer.fit_transform(X_scaled)
    
    # Split back into separate experiments
    start_idx = 0
    scatter_handles = []
    
    for i, strategy_name in enumerate(strategy_names):
        drug_name = strategy_name.split('_')[0]  # Extract drug name from strategy
        opt_strategy = all_labels[i][1]  # Get optimization strategy (CMA or GA)
        
        data_size = len(all_labels[i][0])
        experiment_data = reduced_data[start_idx:start_idx + data_size]
        
        # Create scatter plot for this experiment
        scatter = ax.scatter(experiment_data[:, 0],
                           experiment_data[:, 1],
                           experiment_data[:, 2],
                           c=all_labels[i][0],  # Generation number for color
                           marker=markers[drug_name],
                           edgecolors=colors[opt_strategy],
                           linewidth=0.8,
                           label=f"{drug_name} ({opt_strategy})",
                           cmap='viridis',
                           alpha=0.6)
        
        scatter_handles.append(scatter)
        start_idx += data_size
    
    # Add colorbar for generation
    cbar = plt.colorbar(scatter)
    cbar.set_label('Generation', fontsize=12)
    
    # Customize plot
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_zlabel('Component 3', fontsize=12)
    ax.set_title(f'Combined 3D {method.upper()} of Parameter Space\nColored by Generation', fontsize=14)
    
    # Remove pane fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Add legend with both drug type and optimization strategy
    ax.legend(title='Drug Type & Algorithm', title_fontsize=12, fontsize=10, 
              loc='upper right', frameon=True, framealpha=0.9)
    
    # Save plot
    output_dir = 'results/evolutionary_algorithm_analysis/combined_analysis'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'combined_{method.lower()}_3d.png'), 
                dpi=300, bbox_inches='tight')
    
    # Create animation
    create_combined_rotation_animation(reduced_data, all_labels, strategy_names,
                                    os.path.join(output_dir, f'combined_{method.lower()}_3d_rotation.gif'),
                                    method, markers, colors)
    plt.close()

def create_combined_rotation_animation(data, labels_list, strategy_names, output_path, method, markers, colors):
    """
    Create rotating animation of combined 3D plot
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    start_idx = 0
    for i, strategy_name in enumerate(strategy_names):
        drug_name = strategy_name.split('_')[0]
        opt_strategy = labels_list[i][1]  # Get optimization strategy
        
        data_size = len(labels_list[i][0])
        experiment_data = data[start_idx:start_idx + data_size]
        
        scatter = ax.scatter(experiment_data[:, 0],
                           experiment_data[:, 1],
                           experiment_data[:, 2],
                           c=labels_list[i][0],
                           marker=markers[drug_name],
                           edgecolors=colors[opt_strategy],
                           linewidth=0.8,
                           label=f"{drug_name} ({opt_strategy})",
                           cmap='viridis',
                           alpha=0.6)
        
        start_idx += data_size
    
    plt.colorbar(scatter, label='Generation')
    ax.legend(title='Drug Type & Algorithm', loc='upper right')
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title(f'Combined 3D {method.upper()} of Parameter Space\nColored by Generation')
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    angles = np.linspace(0, 360, 180)
    
    def update(frame):
        ax.view_init(elev=20., azim=angles[frame])
        return fig,
    
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=len(angles), interval=50, blit=True)
    anim.save(output_path, writer='pillow')
    plt.close()

# Add new 2D dimensionality reduction functions
def plot_2d_reduction(data, labels, title, output_path, method='pca'):
    """
    Create 2D plot of dimensionality reduction results
    """
    plt.figure(figsize=(10, 8))
    
    # Create color map based on generation number
    scatter = plt.scatter(data[:, 0], data[:, 1],
                         c=labels, 
                         cmap='viridis',
                         alpha=0.6)
    
    plt.colorbar(scatter, label='Generation')
    
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.title(f'2D {method.upper()} of Parameter Space\nColored by Generation', fontsize=14)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_parameter_space_2d(strategy_name):
    """
    Analyze parameter space using different dimensionality reduction techniques (2D)
    """
    # Read the data
    if "CMA" in strategy_name:
        results_df = pd.read_csv(f'results/CMA_summaries/final_summary_{strategy_name}.csv')
    elif "GA" in strategy_name:
        results_df = pd.read_csv(f'results/GA_summaries/final_summary_{strategy_name}.csv')
    else:
        raise ValueError(f"Strategy name {strategy_name} not recognized")

    # Create output directory
    output_dir = f'results/evolutionary_algorithm_analysis/{strategy_name}/dimensionality_reduction'
    os.makedirs(output_dir, exist_ok=True)

    # Separate features and generation numbers
    non_param_columns = ['individual', 'replicate', 'iteration', 'time']
    param_columns = [col for col in results_df.columns if col not in non_param_columns]
    
    # Extract features and scale them
    X = results_df[param_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Get generation labels
    labels = results_df['iteration'].values

    # 1. PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    plot_2d_reduction(pca_result, labels, 
                     'PCA', 
                     os.path.join(output_dir, 'pca_2d.png'),
                     'PCA')
    
    # Print explained variance
    print("PCA explained variance ratio (2D):", pca.explained_variance_ratio_)

    # 2. t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(X_scaled)
    plot_2d_reduction(tsne_result, labels,
                     't-SNE',
                     os.path.join(output_dir, 'tsne_2d.png'),
                     't-SNE')

    # 3. UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_result = reducer.fit_transform(X_scaled)
    plot_2d_reduction(umap_result, labels,
                     'UMAP',
                     os.path.join(output_dir, 'umap_2d.png'),
                     'UMAP')

def plot_synergy_2d_reduction(synergy_experiment, drug1_in_synergy, drug2_in_synergy, method='pca'):
    """
    Create 2D plot comparing a synergy experiment with its two constituent 
    drugs in the synergy parameter space.
    """
    # Setup markers for different experiment types
    markers = {
        'synergy': 'o',          # Circle for synergy
        'drug1_in_synergy': 's', # Square for drug1
        'drug2_in_synergy': '^'  # Triangle for drug2
    }
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Determine synergy type based on input file names
    if "pi3k_mek" in synergy_experiment.lower() or "PI3Ki" in synergy_experiment:
        synergy_type = "pi3kmek"
    elif "akt_mek" in synergy_experiment.lower() or "AKTi" in synergy_experiment:
        synergy_type = "aktmek"
    else:
        # Default case if we can't determine
        synergy_type = "synergy"
    
    # Load data from all three experiments
    all_data = []
    all_rmse = []
    all_types = []  # To track which experiment each point belongs to
    experiment_names = []  # To track experiment names for labels
    
    # Extract drug names from experiment names for better labels
    try:
        if "pi3k_mek" in synergy_experiment.lower():
            synergy_label = "PI3K+MEK Synergy"
            drug1_label = "PI3K in Synergy Space"
            drug2_label = "MEK in Synergy Space"
        elif "akt_mek" in synergy_experiment.lower():
            synergy_label = "AKT+MEK Synergy"
            drug1_label = "AKT in Synergy Space"
            drug2_label = "MEK in Synergy Space"
        else:
            # Extract from filenames as fallback
            synergy_label = synergy_experiment.split("_")[0] + "+" + synergy_experiment.split("_")[1] + " Synergy"
            drug1_label = drug1_in_synergy.split("_")[0] + " in Synergy Space"
            drug2_label = drug2_in_synergy.split("_")[0] + " in Synergy Space"
    except:
        # Generic labels if extraction fails
        synergy_label = "Synergy"
        drug1_label = "Drug 1 in Synergy Space"
        drug2_label = "Drug 2 in Synergy Space"
    
    experiment_names = [synergy_label, drug1_label, drug2_label]
    
    # Load each dataset separately since they might have different feature counts
    datasets = []
    rmse_values = []
    dataset_types = []
    labels = []
    
    # 1. Load synergy experiment
    try:
        synergy_df = pd.read_csv(f'results/sweep_summaries/final_summary_{synergy_experiment}.csv')
        fitness_column = synergy_df.columns[-1]  # Assuming fitness/RMSE is last column
        non_param_cols = ['individual', 'replicate', 'iteration', fitness_column, 'time']
        param_cols = [col for col in synergy_df.columns if col not in non_param_cols]
        
        datasets.append({
            'data': synergy_df[param_cols].values,
            'rmse': synergy_df[fitness_column].values,
            'type': 'synergy',
            'label': synergy_label,
            'columns': param_cols
        })
    except Exception as e:
        print(f"Error loading synergy data: {e}")
    
    # 2. Load drug1 in synergy space
    try:
        # Determine if this is sweep or GA/CMA result
        if "sweep" in drug1_in_synergy:
            drug1_df = pd.read_csv(f'results/sweep_summaries/final_summary_{drug1_in_synergy}.csv')
        else:
            # Try to detect if it's GA or CMA
            for strategy in ["GA", "CMA"]:
                try:
                    drug1_df = pd.read_csv(f'results/{strategy}_summaries/final_summary_{drug1_in_synergy}.csv')
                    break
                except:
                    continue
        
        fitness_column = drug1_df.columns[-1]  # Assuming fitness/RMSE is last column
        non_param_cols = ['individual', 'replicate', 'iteration', fitness_column, 'time']
        param_cols = [col for col in drug1_df.columns if col not in non_param_cols]
        
        datasets.append({
            'data': drug1_df[param_cols].values,
            'rmse': drug1_df[fitness_column].values,
            'type': 'drug1_in_synergy',
            'label': drug1_label,
            'columns': param_cols
        })
    except Exception as e:
        print(f"Error loading drug1 data: {e}")
    
    # 3. Load drug2 in synergy space
    try:
        # Similar pattern to drug1
        if "sweep" in drug2_in_synergy:
            drug2_df = pd.read_csv(f'results/sweep_summaries/final_summary_{drug2_in_synergy}.csv')
        else:
            for strategy in ["GA", "CMA"]:
                try:
                    drug2_df = pd.read_csv(f'results/{strategy}_summaries/final_summary_{drug2_in_synergy}.csv')
                    break
                except:
                    continue
        
        fitness_column = drug2_df.columns[-1]  # Assuming fitness/RMSE is last column
        non_param_cols = ['individual', 'replicate', 'iteration', fitness_column, 'time']
        param_cols = [col for col in drug2_df.columns if col not in non_param_cols]
        
        datasets.append({
            'data': drug2_df[param_cols].values,
            'rmse': drug2_df[fitness_column].values,
            'type': 'drug2_in_synergy',
            'label': drug2_label,
            'columns': param_cols
        })
    except Exception as e:
        print(f"Error loading drug2 data: {e}")
    
    # Check if we have datasets to plot
    if len(datasets) < 2:
        print("Error: Need at least 2 valid datasets for comparison")
        return
    
    # Find datasets with compatible feature counts
    by_feature_count = {}
    for i, dataset in enumerate(datasets):
        count = dataset['data'].shape[1]
        if count not in by_feature_count:
            by_feature_count[count] = []
        by_feature_count[count].append(i)
    
    # Find largest group of datasets with same feature count
    largest_group = max(by_feature_count.values(), key=len)
    if len(largest_group) < 2:
        # If no groups have at least 2 datasets, try finding common features
        common_columns = set(datasets[0]['columns'])
        for dataset in datasets[1:]:
            common_columns.intersection_update(dataset['columns'])
        
        if len(common_columns) == 0:
            print("Error: No common features across datasets")
            return
        
        print(f"Using {len(common_columns)} common features for dimensionality reduction")
        
        # Extract only common columns from all datasets
        common_columns = list(common_columns)
        all_data = []
        all_rmse = []
        all_types = []
        
        for dataset in datasets:
            idx = [dataset['columns'].index(col) for col in common_columns]
            all_data.append(dataset['data'][:, idx])
            all_rmse.append(dataset['rmse'])
            all_types.extend([dataset['type']] * len(dataset['rmse']))
    else:
        # Use datasets with matching feature counts
        compatible_datasets = [datasets[i] for i in largest_group]
        print(f"Using {len(compatible_datasets)} datasets with matching feature counts")
        
        all_data = []
        all_rmse = []
        all_types = []
        
        for dataset in compatible_datasets:
            all_data.append(dataset['data'])
            all_rmse.append(dataset['rmse'])
            all_types.extend([dataset['type']] * len(dataset['rmse']))
    
    # Combine data
    X_combined = np.vstack(all_data)
    rmse_combined = np.concatenate(all_rmse)
    
    # Apply dimensionality reduction
    try:
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        
        # Apply the chosen dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
            title_method = 'PCA'
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            title_method = 't-SNE'
        else:  # UMAP
            reducer = umap.UMAP(n_components=2, random_state=42)
            title_method = 'UMAP'
        
        reduced_data = reducer.fit_transform(X_scaled)
        
        # Normalize RMSE for better color mapping (lower is better)
        rmse_norm = (rmse_combined - np.min(rmse_combined)) / (np.max(rmse_combined) - np.min(rmse_combined))
        
        # Plot each dataset with its marker
        start_idx = 0
        for i, dataset in enumerate(datasets):
            if dataset['type'] in all_types:
                # Count how many points we have for this dataset type
                count = all_types.count(dataset['type'])
                end_idx = start_idx + count
                
                # Plot this dataset
                plt.scatter(
                    reduced_data[start_idx:end_idx, 0],
                    reduced_data[start_idx:end_idx, 1],
                    c=rmse_norm[start_idx:end_idx],
                    marker=markers[dataset['type']],
                    cmap='plasma_r',
                    s=70,
                    alpha=0.7,
                    edgecolor='k',
                    linewidth=0.5,
                    label=dataset['label']
                )
                
                start_idx = end_idx
        
        # Add colorbar and styling
        cbar = plt.colorbar()
        cbar.set_label('Normalized RMSE (lower is better)', fontsize=12)
        
        plt.title(f'{title_method} of Parameter Space: Synergy Comparison', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Component 1', fontsize=14, fontweight='bold')
        plt.ylabel('Component 2', fontsize=14, fontweight='bold')
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.legend(fontsize=12, frameon=True, framealpha=0.9, 
                  loc='best', title='Experiment Type', title_fontsize=13)
        
        # Save plot with synergy type in the filename
        output_dir = 'results/evolutionary_algorithm_analysis/synergy_comparison'
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(os.path.join(output_dir, f'{method.lower()}_{synergy_type}_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{method.lower()}_{synergy_type}_comparison.svg'), 
                   format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"Successfully saved {method} plot for {synergy_type} comparison")
        
    except Exception as e:
        print(f"Error during dimensionality reduction: {e}")
        import traceback
        traceback.print_exc()

# Keep the original plot_combined_2d_reduction but with aesthetic improvements
def plot_combined_2d_reduction(strategy_names, method='pca', strategy_filter='CMA'):
    """
    Create combined 2D plot of dimensionality reduction results for multiple experiments
    Only plots one strategy (CMA or GA) at a time
    
    Args:
        strategy_names: List of strategy names
        method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
        strategy_filter: Which strategy to plot ('CMA' or 'GA')
    """
    # Setup markers for drugs
    markers = {'PI3Ki': 'o', 'MEKi': 's', 'AKTi': '^'}  # circle, square, triangle
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Store all data for combined scaling
    all_data = []
    all_labels = []
    filtered_strategy_names = []
    
    # First, collect and preprocess all data
    for strategy_name in strategy_names:
        # Only process the selected strategy
        if strategy_filter in strategy_name:
            # Read data
            results_df = pd.read_csv(f'results/{strategy_filter}_summaries/final_summary_{strategy_name}.csv')
            
            # Get parameters
            non_param_columns = ['individual', 'replicate', 'iteration', 'fitness', 'time']
            param_columns = [col for col in results_df.columns if col not in non_param_columns]
            
            # Extract and store features
            X = results_df[param_columns].values
            all_data.append(X)
            all_labels.append(results_df['iteration'].values)
            filtered_strategy_names.append(strategy_name)
    
    # Combine and scale all data together
    X_combined = np.vstack(all_data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Apply dimensionality reduction to all data
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:  # UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
    
    reduced_data = reducer.fit_transform(X_scaled)
    
    # Split back into separate experiments
    start_idx = 0
    
    for i, strategy_name in enumerate(filtered_strategy_names):
        drug_name = strategy_name.split('_')[0]  # Extract drug name from strategy
        
        data_size = len(all_labels[i])
        experiment_data = reduced_data[start_idx:start_idx + data_size]
        generation_data = all_labels[i]
        
        # Normalize generation values for colormapping
        gen_norm = (generation_data - generation_data.min()) / (generation_data.max() - generation_data.min())
        
        # Create scatter plot for this experiment
        scatter = plt.scatter(experiment_data[:, 0],
                            experiment_data[:, 1],
                            c=gen_norm,  # Color by normalized generation
                            marker=markers[drug_name],
                            label=drug_name,
                            cmap='plasma',  # Changed from viridis to plasma
                            s=60,          # Slightly larger points
                            alpha=0.7,     # Some transparency
                            edgecolor='k', # Black edge
                            linewidth=0.5) # Thin edge
        
        start_idx += data_size
    
    # Add colorbar for generation
    cbar = plt.colorbar()
    cbar.set_label('Generation (normalized)', fontsize=12)
    
    # Customize plot
    plt.xlabel('Component 1', fontsize=14, fontweight='bold')
    plt.ylabel('Component 2', fontsize=14, fontweight='bold')
    plt.title(f'Combined 2D {method.upper()} of Parameter Space ({strategy_filter})\nColored by Generation', 
              fontsize=16, fontweight='bold')
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend
    plt.legend(title='Drug Type', title_fontsize=13, fontsize=12, 
              loc='best', frameon=True, framealpha=0.9)
    
    # Save plot
    output_dir = 'results/evolutionary_algorithm_analysis/combined_analysis'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'combined_{method.lower()}_2d_{strategy_filter}.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'combined_{method.lower()}_2d_{strategy_filter}.svg'), 
                format='svg', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    drugs = ["PI3Ki", "MEKi", "AKTi"]
    strategies = ["CMA", "GA"]
    strategy_names = [f"{drug}_{strategy}-0704-1815-18p_delayed_transient_rmse_postdrug_25gen" 
                     for drug in drugs for strategy in strategies]
    
    # Individual analysis (2D)
    # for strategy_name in strategy_names:
    #     analyze_parameter_space_2d(strategy_name)
    
    # # Combined analysis (2D) - separate plots for each strategy
    # for method in ['pca', 'tsne', 'umap']:
    #     for strategy in strategies:
    #         plot_combined_2d_reduction(strategy_names, method=method, strategy_filter=strategy)
    
    # Define synergy experiments and their single drug components
    pi3k_mek_synergy = "synergy_sweep-pi3k_mek-1104-2212-18p_transient_delayed_uniform_5k_10p"
    pi3k_in_mek_synergy = "synergy_sweep-pi3k_mek-1104-2212-18p_PI3K_transient_delayed_uniform_5k_10p"
    mek_in_pi3k_synergy = "synergy_sweep-pi3k_mek-1104-2212-18p_MEK_transient_delayed_uniform_5k_10p"
    
    akt_mek_synergy = "synergy_sweep-akt_mek-1204-1639-18p_transient_delayed_uniform_postdrug_RMSE_5k"
    akt_in_mek_synergy = "synergy_sweep-akt_mek-1104-2212-18p_AKT_transient_delayed_uniform_5k_singledrug"
    mek_in_akt_synergy = "synergy_sweep-akt_mek-1104-2212-18p_MEK_transient_delayed_uniform_5k_singledrug"
    
    # Run synergy comparison plots
    for method in ['pca', 'tsne', 'umap']:
        # PI3K-MEK synergy
        plot_synergy_2d_reduction(
            pi3k_mek_synergy,
            pi3k_in_mek_synergy,
            mek_in_pi3k_synergy,
            method=method
        )
        
        # AKT-MEK synergy
        plot_synergy_2d_reduction(
            akt_mek_synergy,
            akt_in_mek_synergy,
            mek_in_akt_synergy,
            method=method
        )