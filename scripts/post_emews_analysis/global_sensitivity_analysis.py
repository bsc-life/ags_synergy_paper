import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import kstest
import os
from SALib.sample import saltelli, morris, fast_sampler
from SALib.analyze import sobol, morris as morris_analyze, fast
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set up publication-quality plotting
plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class GlobalSensitivityAnalysis:
    """
    Global Sensitivity Analysis implementation with multiple methods:
    - Sobol indices (variance-based)
    - Morris method (derivative-based)
    - FAST (Fourier Amplitude Sensitivity Test)
    - Correlation-based methods
    """
    
    def __init__(self, X, y, feature_names=None):
        """
        Initialize GSA with data
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Input features
        y : pd.Series or np.array
            Target variable
        feature_names : list, optional
            Names of features
        """
        self.X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_names)
        self.y = y
        self.feature_names = feature_names or list(self.X.columns)
        self.n_features = len(self.feature_names)
        self.n_samples = len(self.X)
        
        # Define problem for SALib
        self.problem = {
            'num_vars': self.n_features,
            'names': self.feature_names,
            'bounds': [[self.X[col].min(), self.X[col].max()] for col in self.feature_names]
        }
        
        self.model = None # To store surrogate model
        print(f"Initialized GSA with {self.n_features} features and {self.n_samples} samples")
    
    def analyze_parameter_space(self, save_dir, filename):
        """
        Analyzes the distribution and correlation of input parameters from CMA-ES.
        """
        print("Analyzing parameter space coverage...")
        
        analysis_save_dir = os.path.join(save_dir, 'parameter_space_analysis')
        os.makedirs(analysis_save_dir, exist_ok=True)

        # 1. Visualize Parameter Distributions (Histograms) and K-S Test
        n_cols = 4
        n_rows = (self.n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten()
        
        ks_results = {}
        
        for i, feature in enumerate(self.feature_names):
            # Plot histogram of the data
            sns.histplot(self.X[feature], kde=True, ax=axes[i], color='skyblue', label='Sampled Distribution')
            
            # Overlay uniform distribution for comparison
            xmin, xmax = self.X[feature].min(), self.X[feature].max()
            uniform_dist = np.random.uniform(xmin, xmax, self.n_samples)
            sns.histplot(uniform_dist, kde=True, ax=axes[i], color='red', label='Uniform Reference', alpha=0.5, element="step", fill=False)

            axes[i].set_title(f'Distribution of {feature}', fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()

            # 2. Perform Kolmogorov-Smirnov (K-S) test against a uniform distribution
            D, p_value = kstest(self.X[feature].to_numpy(), 'uniform', args=(xmin, xmax))
            ks_results[feature] = {'statistic': D, 'p_value': p_value}
            axes[i].text(0.95, 0.95, f'K-S p-val: {p_value:.3f}', ha='right', va='top', transform=axes[i].transAxes,
                         bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

        # Hide unused subplots
        for i in range(self.n_features, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_save_dir, f'parameter_distributions_{filename}.png'))
        plt.savefig(os.path.join(analysis_save_dir, f'parameter_distributions_{filename}.svg'), format='svg')
        plt.close()
        
        # Save K-S results
        ks_df = pd.DataFrame(ks_results).T
        ks_df.to_csv(os.path.join(analysis_save_dir, f'ks_test_results_{filename}.csv'))
        
        # 3. Visualize Parameter Correlations (Heatmap)
        plt.figure(figsize=(12, 10))
        corr_matrix = self.X.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Input Parameter Correlation Matrix', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_save_dir, f'parameter_correlation_heatmap_{filename}.png'))
        plt.savefig(os.path.join(analysis_save_dir, f'parameter_correlation_heatmap_{filename}.svg'), format='svg')
        plt.close()

        print("Parameter space analysis complete.")
        
    def evaluate_surrogate_model(self, save_dir, filename):
        """
        Trains and evaluates the Random Forest surrogate model using two methods:
        1. Standard random split to test interpolation.
        2. Core vs. Edge split to test extrapolation.
        """
        print("--- Evaluating Surrogate Model ---")
        surrogate_save_path = os.path.join(save_dir, 'surrogate_model_evaluation')
        os.makedirs(surrogate_save_path, exist_ok=True)

        # --- 1. Standard Validation (Testing Interpolation) ---
        print("Performing standard validation (20% random test set)...")
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_interpolation = r2_score(y_test, y_pred)
        print(f"  R-squared on random test set (interpolation): {r2_interpolation:.4f}")
        
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values (Random Test Set)', fontweight='bold')
        plt.ylabel('Predicted Values', fontweight='bold')
        plt.title(f'Surrogate Interpolation Performance (R² = {r2_interpolation:.4f})', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(surrogate_save_path, f'surrogate_interpolation_{filename}.png'))
        plt.savefig(os.path.join(surrogate_save_path, f'surrogate_interpolation_{filename}.svg'), format='svg')
        plt.close()

        # --- 2. Rigorous Validation (Testing Extrapolation) ---
        print("\nPerforming rigorous validation (Core vs. Edge)...")
        lower_bound = self.X.quantile(0.10)
        upper_bound = self.X.quantile(0.90)
        core_mask = (self.X >= lower_bound) & (self.X <= upper_bound)
        core_mask = core_mask.all(axis=1)

        X_core, y_core = self.X[core_mask], self.y[core_mask]
        X_edge, y_edge = self.X[~core_mask], self.y[~core_mask]

        r2_extrapolation = np.nan
        if len(X_core) > 10 and len(X_edge) > 10:
            model_core = RandomForestRegressor(n_estimators=100, random_state=42)
            model_core.fit(X_core, y_core)
            y_pred_edge = model_core.predict(X_edge)
            r2_extrapolation = r2_score(y_edge, y_pred_edge)
            print(f"  Trained on {len(X_core)} core samples, tested on {len(X_edge)} edge samples.")
            print(f"  R-squared on edge data (extrapolation): {r2_extrapolation:.4f}")

            plt.figure(figsize=(8, 8))
            plt.scatter(y_edge, y_pred_edge, alpha=0.6, edgecolors='k', c='red')
            plt.plot([y_edge.min(), y_edge.max()], [y_edge.min(), y_edge.max()], 'r--', lw=2)
            plt.xlabel('Actual Values (Edge Data)', fontweight='bold')
            plt.ylabel('Predicted Values (from Core-trained Model)', fontweight='bold')
            plt.title(f'Surrogate Extrapolation Performance (R² = {r2_extrapolation:.4f})', fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(surrogate_save_path, f'surrogate_extrapolation_{filename}.png'))
            plt.savefig(os.path.join(surrogate_save_path, f'surrogate_extrapolation_{filename}.svg'), format='svg')
            plt.close()
        else:
            print("  Warning: Not enough data for core vs. edge validation. Skipping.")

        # --- 3. Final Model Training and Feature Importance ---
        importances = model.feature_importances_ # Using model from standard split
        importance_df = pd.DataFrame({'Feature': self.feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.xlabel('Feature Importance', fontweight='bold')
        plt.ylabel('Features', fontweight='bold')
        plt.title('Random Forest Feature Importance (on 80% training data)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(surrogate_save_path, f'rf_feature_importance_{filename}.png'))
        plt.savefig(os.path.join(surrogate_save_path, f'rf_feature_importance_{filename}.svg'), format='svg')
        plt.close()

        # Save all metrics
        with open(os.path.join(surrogate_save_path, f'surrogate_metrics_{filename}.txt'), 'w') as f:
            f.write(f"R-squared on random test set (interpolation): {r2_interpolation}\n")
            f.write(f"R-squared on edge data (extrapolation): {r2_extrapolation}\n")
        importance_df.to_csv(os.path.join(surrogate_save_path, f'rf_feature_importance_{filename}.csv'), index=False)
        
        # Retrain on full dataset for GSA and store it
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.X, self.y)
        print("\nSurrogate model trained on full dataset and stored for subsequent GSA.")
    
    def correlation_analysis(self, save_dir, filename):
        """
        Correlation-based sensitivity analysis methods
        """
        print("Performing correlation-based sensitivity analysis...")
        
        # Calculate various correlation coefficients
        correlations = {}
        
        # Pearson correlation
        correlations['pearson'] = np.abs(self.X.corrwith(self.y))
        
        # Spearman correlation (rank-based)
        correlations['spearman'] = np.abs(self.X.corrwith(self.y, method='spearman'))
        
        # Partial correlation coefficients
        partial_corrs = []
        for i, feature in enumerate(self.feature_names):
            other_features = [f for j, f in enumerate(self.feature_names) if j != i]
            if len(other_features) > 0:
                # Calculate partial correlation
                X_others = self.X[other_features]
                X_feature = self.X[feature]
                
                # Residuals from regressing feature on others
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                lr.fit(X_others, X_feature)
                residuals_X = X_feature - lr.predict(X_others)
                
                # Residuals from regressing target on others
                lr.fit(X_others, self.y)
                residuals_y = self.y - lr.predict(X_others)
                
                # Correlation of residuals
                partial_corr = np.abs(np.corrcoef(residuals_X, residuals_y)[0, 1])
                partial_corrs.append(partial_corr)
            else:
                partial_corrs.append(np.abs(correlations['pearson'][i]))
        
        correlations['partial'] = pd.Series(partial_corrs, index=self.feature_names)
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for plotting
        methods = ['pearson', 'spearman', 'partial']
        colors = ['#2E8B57', '#FF8C00', '#4169E1']
        
        x = np.arange(len(self.feature_names))
        width = 0.25
        
        for i, (method, color) in enumerate(zip(methods, colors)):
            values = correlations[method].values
            ax.bar(x + i*width, values, width, label=method.title(), 
                   color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Features', fontweight='bold')
        ax.set_ylabel('Absolute Correlation Coefficient', fontweight='bold')
        ax.set_title('Correlation-Based Sensitivity Analysis', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'correlation_sensitivity_{filename}.png'))
        plt.savefig(os.path.join(save_dir, f'correlation_sensitivity_{filename}.svg'), format='svg')
        plt.close()
        
        # Save results
        corr_df = pd.DataFrame(correlations)
        corr_df.to_csv(os.path.join(save_dir, f'correlation_sensitivity_{filename}.csv'))
        
        return correlations
    
    def sobol_analysis(self, n_samples=1000, save_dir=None, filename=None):
        """
        Sobol indices for variance-based sensitivity analysis
        """
        print(f"Performing Sobol analysis with {n_samples} samples...")
        
        # Generate samples
        param_values = saltelli.sample(self.problem, n_samples)
        
        # Evaluate model using pre-trained surrogate
        if self.model is None:
            raise ValueError("Surrogate model has not been trained. Run evaluate_surrogate_model first.")
        Y = self.model.predict(param_values)
        
        # Analyze results
        sobol_indices = sobol.analyze(self.problem, Y)
        
        # Extract first order and total order indices
        S1 = sobol_indices['S1']  # First order indices
        ST = sobol_indices['ST']  # Total order indices
        S2 = sobol_indices['S2']  # Second order indices
        
        # Create visualization
        if save_dir and filename:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # First order indices
            ax1.bar(range(len(self.feature_names)), S1, 
                   color='#2E8B57', alpha=0.8, edgecolor='black', linewidth=0.5)
            ax1.set_xlabel('Features', fontweight='bold')
            ax1.set_ylabel('First Order Sobol Index', fontweight='bold')
            ax1.set_title('First Order Sobol Indices', fontweight='bold')
            ax1.set_xticks(range(len(self.feature_names)))
            ax1.set_xticklabels(self.feature_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Total order indices
            ax2.bar(range(len(self.feature_names)), ST, 
                   color='#FF8C00', alpha=0.8, edgecolor='black', linewidth=0.5)
            ax2.set_xlabel('Features', fontweight='bold')
            ax2.set_ylabel('Total Order Sobol Index', fontweight='bold')
            ax2.set_title('Total Order Sobol Indices', fontweight='bold')
            ax2.set_xticks(range(len(self.feature_names)))
            ax2.set_xticklabels(self.feature_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'sobol_analysis_{filename}.png'))
            plt.savefig(os.path.join(save_dir, f'sobol_analysis_{filename}.svg'), format='svg')
            plt.close()
            
            # Save results
            sobol_df = pd.DataFrame({
                'Feature': self.feature_names,
                'S1': S1,
                'ST': ST,
                'ST_S1': ST - S1  # Interaction effects
            })
            sobol_df.to_csv(os.path.join(save_dir, f'sobol_analysis_{filename}.csv'), index=False)
        
        return {'S1': S1, 'ST': ST, 'S2': S2}
    
    def morris_analysis(self, n_trajectories=10, n_levels=4, save_dir=None, filename=None):
        """
        Morris method for derivative-based sensitivity analysis
        """
        print(f"Performing Morris analysis with {n_trajectories} trajectories...")
        
        # Generate samples
        param_values = morris.sample(self.problem, N=n_trajectories, num_levels=n_levels)
        
        # Evaluate model using pre-trained surrogate
        if self.model is None:
            raise ValueError("Surrogate model has not been trained. Run evaluate_surrogate_model first.")
        Y = self.model.predict(param_values)
        
        # Analyze results
        morris_results = morris_analyze.analyze(self.problem, param_values, Y, conf_level=0.95, print_to_console=False)
        
        mu = morris_results['mu']  # Mean elementary effects
        mu_star = morris_results['mu_star']  # Mean absolute elementary effects
        sigma = morris_results['sigma']  # Standard deviation of elementary effects
        
        # Create visualization
        if save_dir and filename:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # mu* vs sigma plot (Morris plot)
            ax1.scatter(mu_star, sigma, c='#4169E1', alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
            for i, feature in enumerate(self.feature_names):
                ax1.annotate(feature, (mu_star[i], sigma[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax1.set_xlabel('μ* (Mean Absolute Elementary Effect)', fontweight='bold')
            ax1.set_ylabel('σ (Standard Deviation)', fontweight='bold')
            ax1.set_title('Morris Sensitivity Analysis', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Bar plot of mu*
            ax2.bar(range(len(self.feature_names)), mu_star, 
                   color='#FF8C00', alpha=0.8, edgecolor='black', linewidth=0.5)
            ax2.set_xlabel('Features', fontweight='bold')
            ax2.set_ylabel('μ* (Mean Absolute Elementary Effect)', fontweight='bold')
            ax2.set_title('Morris μ* Indices', fontweight='bold')
            ax2.set_xticks(range(len(self.feature_names)))
            ax2.set_xticklabels(self.feature_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'morris_analysis_{filename}.png'))
            plt.savefig(os.path.join(save_dir, f'morris_analysis_{filename}.svg'), format='svg')
            plt.close()
            
            # Save results
            morris_df = pd.DataFrame({
                'Feature': self.feature_names,
                'mu': mu,
                'mu_star': mu_star,
                'sigma': sigma
            })
            morris_df.to_csv(os.path.join(save_dir, f'morris_analysis_{filename}.csv'), index=False)
        
        return {'mu': mu, 'mu_star': mu_star, 'sigma': sigma}
    
    def fast_analysis(self, n_samples=1000, M=4, save_dir=None, filename=None):
        """
        FAST (Fourier Amplitude Sensitivity Test) analysis
        """
        print(f"Performing FAST analysis with {n_samples} samples...")
        
        # Generate samples
        param_values = fast_sampler.sample(self.problem, n_samples, M=M)
        
        # Evaluate model using pre-trained surrogate
        if self.model is None:
            raise ValueError("Surrogate model has not been trained. Run evaluate_surrogate_model first.")
        Y = self.model.predict(param_values)
        
        # Analyze results
        fast_results = fast.analyze(self.problem, Y, M=M)
        
        S1 = fast_results['S1']  # First order indices
        
        # Create visualization
        if save_dir and filename:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(self.feature_names)), S1, 
                          color='#2E8B57', alpha=0.8, edgecolor='black', linewidth=0.5)
            
            plt.xlabel('Features', fontweight='bold')
            plt.ylabel('FAST First Order Index', fontweight='bold')
            plt.title('FAST Sensitivity Analysis', fontweight='bold')
            plt.xticks(range(len(self.feature_names)), self.feature_names, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'fast_analysis_{filename}.png'))
            plt.savefig(os.path.join(save_dir, f'fast_analysis_{filename}.svg'), format='svg')
            plt.close()
            
            # Save results
            fast_df = pd.DataFrame({
                'Feature': self.feature_names,
                'S1': S1
            })
            fast_df.to_csv(os.path.join(save_dir, f'fast_analysis_{filename}.csv'), index=False)
        
        return {'S1': S1}
    
    def comprehensive_analysis(self, save_dir, filename, n_samples=1000):
        """
        Run all GSA methods and create comparison
        """
        print("Running comprehensive global sensitivity analysis...")
        
        # 1. Analyze the parameter space coverage from CMA-ES
        self.analyze_parameter_space(save_dir, filename)
        
        # 2. Evaluate the surrogate model trained on this data
        self.evaluate_surrogate_model(save_dir, filename)

        # 3. Run all GSA analyses
        corr_results = self.correlation_analysis(save_dir, filename)
        sobol_results = self.sobol_analysis(n_samples, save_dir, filename)
        morris_results = self.morris_analysis(save_dir=save_dir, filename=filename)
        fast_results = self.fast_analysis(n_samples, save_dir=save_dir, filename=filename)
        
        # Create comparison plot
        methods = {
            'Pearson': corr_results['pearson'],
            'Spearman': corr_results['spearman'],
            'Partial': corr_results['partial'],
            'Sobol S1': sobol_results['S1'],
            'Sobol ST': sobol_results['ST'],
            'Morris μ*': morris_results['mu_star'],
            'FAST S1': fast_results['S1']
        }
        
        # Normalize all methods to 0-1 scale for comparison
        normalized_methods = {}
        for method_name, values in methods.items():
            if np.sum(values) > 0:
                normalized_methods[method_name] = values / np.sum(values)
            else:
                normalized_methods[method_name] = values
        
        # Create comparison heatmap
        comparison_df = pd.DataFrame(normalized_methods, index=self.feature_names)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(comparison_df.T, annot=True, cmap='YlOrRd', fmt='.3f', 
                   cbar_kws={'label': 'Normalized Sensitivity Index'})
        plt.title('Comparison of Global Sensitivity Analysis Methods', fontweight='bold')
        plt.xlabel('Features', fontweight='bold')
        plt.ylabel('Methods', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'gsa_comparison_{filename}.png'))
        plt.savefig(os.path.join(save_dir, f'gsa_comparison_{filename}.svg'), format='svg')
        plt.close()
        
        # Save comprehensive results
        comprehensive_df = pd.DataFrame(methods, index=self.feature_names)
        comprehensive_df.to_csv(os.path.join(save_dir, f'comprehensive_gsa_{filename}.csv'))
        
        return {
            'correlation': corr_results,
            'sobol': sobol_results,
            'morris': morris_results,
            'fast': fast_results,
            'comparison': comparison_df
        }


def run_gsa_on_experiment(file_path, save_dir, filename):
    """
    Run global sensitivity analysis on a single experiment file
    """
    print(f"Running GSA on {filename}...")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Exclude specified metadata columns if they exist
    cols_to_drop = ["individual", "replicate", "iteration"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # Assuming the last column is the target variable
    target_col = df.columns[-1]
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Prepare data
    X = df[feature_cols]
    y = df[target_col]
    
    # Initialize GSA
    gsa = GlobalSensitivityAnalysis(X, y, feature_cols)
    
    # Run comprehensive analysis
    results = gsa.comprehensive_analysis(save_dir, filename)
    
    return results


def main():
    """
    Main function to run GSA on specific, predefined experiments.
    """
    # Define specific experiments to analyze
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
    save_base_dir = "./results/global_sensitivity_analysis"
    
    all_experiments = []

    # Prepare CMA experiment paths
    for exp_name in cma_experiments:
        # As requested, use the full summary CSV, which starts with "final_summary_"
        filename = f"final_summary_{exp_name}.csv"
        file_path = os.path.join(cma_base_dir, filename)
        all_experiments.append({'path': file_path, 'name': exp_name, 'type': 'CMA'})

    # Prepare Sweep experiment paths
    for exp_name in sweep_experiments:
        filename = f"final_summary_{exp_name}.csv"
        file_path = os.path.join(sweep_base_dir, filename)
        all_experiments.append({'path': file_path, 'name': exp_name, 'type': 'Sweep'})

    # Process all selected experiments
    for exp in all_experiments:
        print(f"\n--- Processing {exp['type']} Experiment: {exp['name']} ---")
        
        if os.path.exists(exp['path']):
            print(f"Found file: {exp['path']}")
            
            # Create a unique save directory for the experiment
            save_dir = os.path.join(save_base_dir, exp['name'])
            
            # Run GSA
            try:
                run_gsa_on_experiment(exp['path'], save_dir, exp['name'])
                print(f"GSA completed for {exp['name']}")
            except Exception as e:
                print(f"Error processing {exp['name']}: {e}")
        else:
            print(f"File not found, skipping: {exp['path']}")


if __name__ == "__main__":
    main() 