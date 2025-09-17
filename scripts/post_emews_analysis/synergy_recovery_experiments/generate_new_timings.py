#!/usr/bin/env python3
"""
generate_optimized_timing_params.py

This script trains machine learning models on synergy sweep data to learn optimal drug timing
parameters based on individual drug properties rather than ratios. It then generates new
parameter sets for EMEWS exploration, keeping the original distribution of parameters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Paths
data_dir = "results/sweep_summaries/"
output_dir = "data/JSON/sweep/sweep_txt/"
figures_dir = "results/ml_optimization_figures/"

# Create directories if they don't exist
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Load the data
filename = "final_summary_synergy_sweep-akt_mek-2504-0137-6p_3D_drugaddition_drugtiming_layerdepth.csv"
df = pd.read_csv(f"{data_dir}{filename}")

print(f"Loaded dataframe with {df.shape[0]} rows and {df.shape[1]} columns")
print("Columns:", df.columns.tolist())

# Identify parameter columns (adjust these based on your actual column names)
# We'll need to identify which columns contain which parameters
print("\nColumn names from the dataframe:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

# This is where we need to properly identify your parameter columns
# You'll need to replace these with the actual column names from your dataframe
param_names = {
    'drug_X_diffusion': 'user_parameters.drug_X_diffusion_coefficient',    # Example column name
    'drug_Y_diffusion': 'user_parameters.drug_Y_diffusion_coefficient',    # Example column name
    'drug_X_pulse': 'user_parameters.drug_X_pulse_period',                 # Example column name
    'drug_Y_pulse': 'user_parameters.drug_Y_pulse_period',                 # Example column name
    'drug_X_membrane': 'user_parameters.drug_X_membrane_length',           # Example column name
    'drug_Y_membrane': 'user_parameters.drug_Y_membrane_length',           # Example column name
    'target': 'FINAL_NUMBER_OF_ALIVE_CELLS'
}

# Check if these columns exist in our dataframe
missing_cols = [col for col in param_names.values() if col not in df.columns]
if missing_cols:
    print(f"Warning: The following columns are missing: {missing_cols}")
    print("Using column indices instead of names. Please adjust the code with your actual column names.")
    # Use column indices if names don't match
    param_names = {
        'drug_X_diffusion': df.columns[0],
        'drug_Y_diffusion': df.columns[1],
        'drug_X_pulse': df.columns[2],
        'drug_Y_pulse': df.columns[3],
        'drug_X_membrane': df.columns[4],
        'drug_Y_membrane': df.columns[5],
        'target': df.columns[-1] if df.columns[-1] == 'FINAL_NUMBER_OF_ALIVE_CELLS' else df.columns[-1]
    }

print("\nUsing the following parameter mappings:")
for key, value in param_names.items():
    print(f"{key}: {value}")

# Explore the data
print(f"\nSummary statistics for parameters:")
for param, col in param_names.items():
    if col in df.columns:
        print(f"\n{param} ({col}):")
        print(df[col].describe())

# Visualize the distribution of parameters
plt.figure(figsize=(15, 10))
for i, (param, col) in enumerate(param_names.items(), 1):
    if col in df.columns and param != 'target':
        plt.subplot(2, 3, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {param}')
plt.tight_layout()
plt.savefig(f"{figures_dir}parameter_distributions.png")
plt.close()

# Identify the threshold for "good" synergy
# We'll consider the lowest 25% of cell counts as good synergy
target_col = param_names['target']
good_threshold = df[target_col].quantile(0.25)
df['is_good_synergy'] = df[target_col] <= good_threshold
print(f"\nThreshold for good synergy: {good_threshold}")
print(f"Number of samples with good synergy: {df['is_good_synergy'].sum()}")

# Visualize correlations
correlation_cols = [col for param, col in param_names.items() if col in df.columns]
plt.figure(figsize=(12, 10))
corr = df[correlation_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(f"{figures_dir}correlation_matrix.png")
plt.close()

# Prepare data for modeling
# We want to predict drug pulse periods based on other parameters
features = [
    param_names['drug_X_diffusion'], 
    param_names['drug_Y_diffusion'],
    param_names['drug_X_membrane'],
    param_names['drug_Y_membrane']
]

# Filter out any features that don't exist in the dataframe
features = [f for f in features if f in df.columns]

# Prepare X data - all parameters except pulse periods
X = df[features]

# Prepare y data for drug_X_pulse and drug_Y_pulse
y_X_pulse = df[param_names['drug_X_pulse']] if param_names['drug_X_pulse'] in df.columns else None
y_Y_pulse = df[param_names['drug_Y_pulse']] if param_names['drug_Y_pulse'] in df.columns else None

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data
X_train, X_test, y_X_train, y_X_test = train_test_split(
    X_scaled_df, y_X_pulse, test_size=0.2, random_state=42)

X_train, X_test, y_Y_train, y_Y_test = train_test_split(
    X_scaled_df, y_Y_pulse, test_size=0.2, random_state=42)

# Train models to predict pulse periods
print("\nTraining model to predict drug_X_pulse_period...")
model_X_pulse = GradientBoostingRegressor(random_state=42)
model_X_pulse.fit(X_train, y_X_train)

print("\nTraining model to predict drug_Y_pulse_period...")
model_Y_pulse = GradientBoostingRegressor(random_state=42)
model_Y_pulse.fit(X_train, y_Y_train)

# Evaluate the models
y_X_pred = model_X_pulse.predict(X_test)
r2_X = r2_score(y_X_test, y_X_pred)
rmse_X = np.sqrt(mean_squared_error(y_X_test, y_X_pred))
print(f"R² Score for drug_X_pulse prediction: {r2_X:.3f}")
print(f"RMSE for drug_X_pulse prediction: {rmse_X:.3f}")

y_Y_pred = model_Y_pulse.predict(X_test)
r2_Y = r2_score(y_Y_test, y_Y_pred)
rmse_Y = np.sqrt(mean_squared_error(y_Y_test, y_Y_pred))
print(f"R² Score for drug_Y_pulse prediction: {r2_Y:.3f}")
print(f"RMSE for drug_Y_pulse prediction: {rmse_Y:.3f}")

# Save the models
joblib.dump(model_X_pulse, f"{output_dir}drug_X_pulse_prediction_model.joblib")
joblib.dump(model_Y_pulse, f"{output_dir}drug_Y_pulse_prediction_model.joblib")
joblib.dump(scaler, f"{output_dir}feature_scaler.joblib")

# Feature importance
feature_importance_X = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model_X_pulse.feature_importances_
}).sort_values('Importance', ascending=False)

feature_importance_Y = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model_Y_pulse.feature_importances_
}).sort_values('Importance', ascending=False)

# Visualize feature importance
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Importance', y='Feature', data=feature_importance_X)
plt.title('Feature Importance for Drug X Pulse Period')

plt.subplot(1, 2, 2)
sns.barplot(x='Importance', y='Feature', data=feature_importance_Y)
plt.title('Feature Importance for Drug Y Pulse Period')

plt.tight_layout()
plt.savefig(f"{figures_dir}feature_importance.png")
plt.close()

print("\nFeature importance for Drug X pulse period:")
print(feature_importance_X)
print("\nFeature importance for Drug Y pulse period:")
print(feature_importance_Y)

# Visualize the relationship between actual and predicted values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_X_test, y_X_pred, alpha=0.6)
plt.plot([y_X_test.min(), y_X_test.max()], [y_X_test.min(), y_X_test.max()], 'r--')
plt.xlabel('Actual Drug X Pulse Period')
plt.ylabel('Predicted Drug X Pulse Period')
plt.title('Drug X Pulse Period Prediction')

plt.subplot(1, 2, 2)
plt.scatter(y_Y_test, y_Y_pred, alpha=0.6)
plt.plot([y_Y_test.min(), y_Y_test.max()], [y_Y_test.min(), y_Y_test.max()], 'r--')
plt.xlabel('Actual Drug Y Pulse Period')
plt.ylabel('Predicted Drug Y Pulse Period')
plt.title('Drug Y Pulse Period Prediction')

plt.tight_layout()
plt.savefig(f"{figures_dir}prediction_accuracy.png")
plt.close()

# Generate a new parameter space for EMEWS exploration
print("\nGenerating new parameter suggestions...")

# We'll use the distribution of parameters from the original CSV for all params except pulse periods
num_new_params = 2000  # Number of new parameter sets to generate
new_params = []

# Generate new parameter sets
for i in range(num_new_params):
    # Sample one existing row for all parameters except pulse
    sample_idx = np.random.randint(0, len(df))
    sampled_row = df.iloc[sample_idx]
    
    # Create parameter dictionary with parameters from the sampled row (except pulse periods)
    param_set = {}
    for param, col in param_names.items():
        if param not in ['drug_X_pulse', 'drug_Y_pulse', 'target'] and col in df.columns:
            param_set[col] = float(sampled_row[col])
    
    # Prepare features for ML prediction
    features_for_prediction = [sampled_row[col] for col in features]
    features_scaled = scaler.transform([features_for_prediction])
    
    # Predict optimal pulse periods using our trained models
    drug_X_pulse = float(model_X_pulse.predict(features_scaled)[0])
    drug_Y_pulse = float(model_Y_pulse.predict(features_scaled)[0])
    
    # Add predicted pulse periods to parameter set
    param_set[param_names['drug_X_pulse']] = drug_X_pulse
    param_set[param_names['drug_Y_pulse']] = drug_Y_pulse
    
    new_params.append(param_set)
    
    # Print progress
    if i % 10 == 0:
        print(f"Sample {i}: Generated parameters with X_pulse={drug_X_pulse:.2f}, Y_pulse={drug_Y_pulse:.2f}")

# Write the new parameter sets to a file in the EMEWS format
output_file = f"{output_dir}optimized_timing_params.txt"
with open(output_file, 'w') as f:
    for param_set in new_params:
        f.write(json.dumps(param_set) + "\n")

print(f"\nGenerated {num_new_params} new parameter sets and saved to {output_file}")

# Create a dataframe from the new parameters for visualization
new_params_df = pd.DataFrame(new_params)

# Visualize the distribution of parameters in the new parameter space
plt.figure(figsize=(15, 10))
for i, col in enumerate(new_params_df.columns, 1):
    plt.subplot(3, 3, i if i <= 9 else 9)
    sns.histplot(new_params_df[col], kde=True, color='blue', alpha=0.6, label='New')
    if col in df.columns:
        sns.histplot(df[col], kde=True, color='orange', alpha=0.4, label='Original')
    plt.title(f'Distribution of {col}')
    plt.legend()
plt.tight_layout()
plt.savefig(f"{figures_dir}new_parameter_distributions.png")
plt.close()

# Create scatter plots for key parameter combinations
plt.figure(figsize=(15, 10))

# Plot 1: X diffusion vs X pulse
plt.subplot(2, 3, 1)
plt.scatter(new_params_df[param_names['drug_X_diffusion']], 
           new_params_df[param_names['drug_X_pulse']], 
           alpha=0.7, label='New Parameters')
plt.scatter(df[param_names['drug_X_diffusion']], 
           df[param_names['drug_X_pulse']], 
           alpha=0.3, label='Original Parameters')
plt.xlabel('Drug X Diffusion')
plt.ylabel('Drug X Pulse Period')
plt.legend()

# Plot 2: Y diffusion vs Y pulse
plt.subplot(2, 3, 2)
plt.scatter(new_params_df[param_names['drug_Y_diffusion']], 
           new_params_df[param_names['drug_Y_pulse']], 
           alpha=0.7, label='New Parameters')
plt.scatter(df[param_names['drug_Y_diffusion']], 
           df[param_names['drug_Y_pulse']], 
           alpha=0.3, label='Original Parameters')
plt.xlabel('Drug Y Diffusion')
plt.ylabel('Drug Y Pulse Period')
plt.legend()

# Plot 3: X membrane vs X pulse
plt.subplot(2, 3, 3)
plt.scatter(new_params_df[param_names['drug_X_membrane']], 
           new_params_df[param_names['drug_X_pulse']], 
           alpha=0.7, label='New Parameters')
plt.scatter(df[param_names['drug_X_membrane']], 
           df[param_names['drug_X_pulse']], 
           alpha=0.3, label='Original Parameters')
plt.xlabel('Drug X Membrane Length')
plt.ylabel('Drug X Pulse Period')
plt.legend()

# Plot 4: Y membrane vs Y pulse
plt.subplot(2, 3, 4)
plt.scatter(new_params_df[param_names['drug_Y_membrane']], 
           new_params_df[param_names['drug_Y_pulse']], 
           alpha=0.7, label='New Parameters')
plt.scatter(df[param_names['drug_Y_membrane']], 
           df[param_names['drug_Y_pulse']], 
           alpha=0.3, label='Original Parameters')
plt.xlabel('Drug Y Membrane Length')
plt.ylabel('Drug Y Pulse Period')
plt.legend()

# Plot 5: Diff X/Y ratio vs Pulse X/Y ratio
plt.subplot(2, 3, 5)
new_params_df['diff_ratio'] = new_params_df[param_names['drug_X_diffusion']] / new_params_df[param_names['drug_Y_diffusion']]
new_params_df['pulse_ratio'] = new_params_df[param_names['drug_X_pulse']] / new_params_df[param_names['drug_Y_pulse']]
df['diff_ratio'] = df[param_names['drug_X_diffusion']] / df[param_names['drug_Y_diffusion']]
df['pulse_ratio'] = df[param_names['drug_X_pulse']] / df[param_names['drug_Y_pulse']]

plt.scatter(new_params_df['diff_ratio'], new_params_df['pulse_ratio'], 
           alpha=0.7, label='New Parameters')
plt.scatter(df['diff_ratio'], df['pulse_ratio'], 
           alpha=0.3, label='Original Parameters')
plt.xlabel('Diffusion Ratio (X/Y)')
plt.ylabel('Pulse Ratio (X/Y)')
plt.legend()

plt.tight_layout()
plt.savefig(f"{figures_dir}parameter_relationships.png")
plt.close()

print("\nAnalysis complete! New parameter space ready for EMEWS exploration.")