#!/usr/bin/env python3
"""
Comprehensive experimental data processing script for AGS growth curves.

This script processes all experimental curve data following the methodology from 
topN_curves_plot.py and stores the processed data for efficient reuse.

Author: Based on topN_curves_plot.py methodology
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy.stats import pearsonr
import logging
import json
from pathlib import Path
import glob
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_min_max_wt_exp_curve(ctrl_file_path):
    """
    Get min/max values from the control (WT) curve for normalization.
    Based on get_min_max_wt_exp_curve() from topN_curves_plot.py
    """
    logging.info(f"Processing control curve: {ctrl_file_path}")
    
    df_control_csv = pd.read_csv(ctrl_file_path)
    df_control = df_control_csv.reset_index(drop=True)
    
    # Convert time from seconds to minutes
    df_control['Time'] = df_control['Time'] / 60
    
    # Calculate average across replicates (columns 1+)
    df_control['Average_Cell_Index'] = df_control.iloc[:, 1:].mean(axis=1)
    df_control = df_control[["Time", "Average_Cell_Index"]]
    
    # Filter to reasonable time range
    df_control_sliced = df_control.loc[df_control["Time"] < 4200]

    try:
        # Interpolate using cubic splines
        f = interpolate.interp1d(df_control.Time, df_control.Average_Cell_Index, 
                                kind="cubic", fill_value="extrapolate")
        xnew_control_exp = np.arange(0, 4200, 40)
        ynew_control_exp = f(xnew_control_exp)
    except ValueError:
        f = interpolate.interp1d(df_control.Time, df_control.Average_Cell_Index, 
                                kind="cubic", bounds_error=False, fill_value="extrapolate")
        xnew_control_exp = np.arange(0, 4200, 40)
        ynew_control_exp = f(xnew_control_exp)
    
    # Get min/max for normalization
    max_value_wt_exp = float(np.max(ynew_control_exp))
    min_value_wt_exp = float(np.min(ynew_control_exp))

    return min_value_wt_exp, max_value_wt_exp, xnew_control_exp, ynew_control_exp

def process_experimental_curve(exp_file_path, min_val, max_val):
    """
    Process a single experimental curve file.
    Based on normalize_exp_curve() from topN_curves_plot.py
    """
    logging.info(f"Processing experimental curve: {os.path.basename(exp_file_path)}")
    
    # Load the data file
    df_drug_csv = pd.read_csv(exp_file_path)
    df_drug = df_drug_csv.reset_index(drop=True)
    
    # Convert time from seconds to minutes
    df_drug['Time'] = df_drug['Time'] / 60
    
    # Calculate statistics across replicates
    df_drug['Average_Cell_Index'] = df_drug.iloc[:, 1:].mean(axis=1)
    df_drug['Std_Cell_Index'] = df_drug.iloc[:, 1:].std(axis=1)
    df_drug = df_drug[["Time", "Average_Cell_Index", "Std_Cell_Index"]]

    # Filter data - remove middle timepoints to focus on pre/post drug periods
    df_drug = df_drug[(df_drug['Time'] < 400.0) | (df_drug["Time"] > 2500.0)]
    df_drug_sliced = df_drug.loc[df_drug["Time"] < 4200]

    try:
        # Interpolate using cubic splines
        f_mean = interpolate.interp1d(df_drug.Time, df_drug.Average_Cell_Index, 
                                     kind="cubic", fill_value="extrapolate")
        f_std = interpolate.interp1d(df_drug.Time, df_drug.Std_Cell_Index, 
                                    kind="cubic", fill_value="extrapolate")
        xnew_drug_exp = np.arange(0, 4240, 40)
        ynew_drug_exp = f_mean(xnew_drug_exp)
        ynew_drug_std = f_std(xnew_drug_exp)
    except ValueError:
        f_mean = interpolate.interp1d(df_drug.Time, df_drug.Average_Cell_Index, 
                                     kind="cubic", bounds_error=False, fill_value="extrapolate")
        f_std = interpolate.interp1d(df_drug.Time, df_drug.Std_Cell_Index, 
                                    kind="cubic", bounds_error=False, fill_value="extrapolate")
        xnew_drug_exp = np.arange(0, 4240, 40)
        ynew_drug_exp = f_mean(xnew_drug_exp)
        ynew_drug_std = f_std(xnew_drug_exp)

    # Normalize to 0-100 scale using control min/max
    normalized_mean = ((ynew_drug_exp - min_val) / (max_val - min_val)) * 100
    normalized_std = (ynew_drug_std / (max_val - min_val)) * 100
    
    return xnew_drug_exp, normalized_mean, normalized_std

def categorize_drug_type(filename):
    """
    Categorize the drug type based on filename.
    """
    filename = filename.lower()
    
    if 'ctrl' in filename:
        return 'Control'
    elif 'pi103' in filename and 'pd0325901' in filename:
        return 'PI3K_MEK_Combination'
    elif 'akt' in filename and 'mek' in filename:
        return 'AKT_MEK_Combination'
    elif 'pi103' in filename:
        return 'PI3K_Inhibitor'
    elif 'pd0325901' in filename:
        return 'MEK_Inhibitor'
    elif 'akt' in filename:
        return 'AKT_Inhibitor'
    elif '(5z)-7-oxozeaenol' in filename:
        return 'TAK1_Inhibitor'
    elif 'pkf118-310' in filename:
        return 'PKF_Inhibitor'
    else:
        return 'Unknown'

def extract_concentrations(filename):
    """
    Extract drug concentrations from filename.
    """
    concentrations = {}
    
    # PI103 concentrations
    if 'pi103(' in filename.lower():
        import re
        match = re.search(r'pi103\(([^)]+)\)', filename.lower())
        if match:
            concentrations['PI103'] = match.group(1)
    
    # PD0325901 concentrations
    if 'pd0325901(' in filename.lower():
        import re
        match = re.search(r'pd0325901\(([^)]+)\)', filename.lower())
        if match:
            concentrations['PD0325901'] = match.group(1)
    
    # Other drugs can be added similarly
    
    return concentrations

def create_metadata(exp_file_path):
    """
    Create metadata for each experimental file.
    """
    filename = os.path.basename(exp_file_path)
    
    metadata = {
        'filename': filename,
        'file_path': exp_file_path,
        'drug_type': categorize_drug_type(filename),
        'concentrations': extract_concentrations(filename),
        'processing_timestamp': pd.Timestamp.now().isoformat(),
    }
    
    return metadata

def save_processed_data(time_points, normalized_mean, normalized_std, metadata, output_dir):
    """
    Save processed data to files.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create processed data DataFrame
    processed_df = pd.DataFrame({
        'Time_min': time_points,
        'Normalized_Mean': normalized_mean,
        'Normalized_Std': normalized_std
    })
    
    # Generate output filename
    base_name = os.path.splitext(metadata['filename'])[0]
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"{base_name}_processed.csv")
    processed_df.to_csv(csv_path, index=False)
    
    # Save metadata as JSON
    json_path = os.path.join(output_dir, f"{base_name}_metadata.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Saved processed data to {csv_path}")
    
    return csv_path, json_path

def create_summary_plot(time_points, normalized_mean, normalized_std, metadata, output_dir):
    """
    Create a summary plot for the processed curve.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the normalized curve with error bars
    plt.plot(time_points, normalized_mean, 'b-', linewidth=2, label='Normalized Mean')
    plt.fill_between(time_points, 
                     normalized_mean - normalized_std,
                     normalized_mean + normalized_std,
                     alpha=0.3, color='blue', label='±1 SD')
    
    # Add treatment window (around 1280 minutes based on topN script)
    plt.axvspan(1280, 1320, color='red', alpha=0.2, label='Treatment Window')
    
    plt.xlabel('Time (min)')
    plt.ylabel('Normalized Cell Count (%)')
    plt.title(f"Processed Curve: {metadata['filename']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    base_name = os.path.splitext(metadata['filename'])[0]
    plot_path = os.path.join(output_dir, 'plots', f"{base_name}_processed.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved plot to {plot_path}")
    
    return plot_path

def process_all_curves(input_dir, output_dir):
    """
    Process all experimental curve files in the input directory.
    """
    logging.info(f"Processing all curves from {input_dir}")
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    # Find control file for normalization
    ctrl_files = [f for f in csv_files if 'CTRL' in os.path.basename(f) and 'SIM' not in os.path.basename(f)]
    
    if not ctrl_files:
        raise ValueError("No control file (CTRL.csv) found for normalization!")
    
    ctrl_file = ctrl_files[0]  # Use first control file found
    logging.info(f"Using control file: {ctrl_file}")
    
    # Get normalization values from control
    min_val, max_val, ctrl_time, ctrl_normalized = get_min_max_wt_exp_curve(ctrl_file)
    
    # Store overall processing info
    processing_summary = {
        'control_file': ctrl_file,
        'normalization_min': min_val,
        'normalization_max': max_val,
        'processed_files': [],
        'processing_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Process each experimental file
    for csv_file in csv_files:
        # Skip simulation files
        if 'SIM_' in os.path.basename(csv_file):
            continue
            
        try:
            # Create metadata
            metadata = create_metadata(csv_file)
            metadata['normalization_min'] = min_val
            metadata['normalization_max'] = max_val
            metadata['control_file'] = ctrl_file
            
            # Process the curve
            time_points, normalized_mean, normalized_std = process_experimental_curve(
                csv_file, min_val, max_val)
            
            # Save processed data
            csv_path, json_path = save_processed_data(
                time_points, normalized_mean, normalized_std, metadata, output_dir)
            
            # Create plot
            plot_path = create_summary_plot(
                time_points, normalized_mean, normalized_std, metadata, output_dir)
            
            # Add to summary
            processing_summary['processed_files'].append({
                'original_file': csv_file,
                'processed_csv': csv_path,
                'metadata_json': json_path,
                'plot': plot_path,
                'drug_type': metadata['drug_type'],
                'concentrations': metadata['concentrations']
            })
            
        except Exception as e:
            logging.error(f"Error processing {csv_file}: {str(e)}")
            continue
    
    # Save processing summary
    summary_path = os.path.join(output_dir, 'processing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(processing_summary, f, indent=2)
    
    logging.info(f"Processing complete! Summary saved to {summary_path}")
    logging.info(f"Processed {len(processing_summary['processed_files'])} files")
    
    return processing_summary

def create_overview_comparison_plot(processing_summary, output_dir):
    """
    Create an overview plot comparing all processed curves by drug type.
    """
    plt.figure(figsize=(15, 10))
    
    # Group by drug type
    drug_types = {}
    for file_info in processing_summary['processed_files']:
        drug_type = file_info['drug_type']
        if drug_type not in drug_types:
            drug_types[drug_type] = []
        drug_types[drug_type].append(file_info)
    
    # Color scheme for different drug types
    colors = plt.cm.Set1(np.linspace(0, 1, len(drug_types)))
    
    for i, (drug_type, files) in enumerate(drug_types.items()):
        for j, file_info in enumerate(files):
            # Load processed data
            processed_df = pd.read_csv(file_info['processed_csv'])
            
            # Plot with transparency for multiple files of same type
            alpha = 0.7 if len(files) > 1 else 1.0
            label = drug_type if j == 0 else None  # Only label first curve of each type
            
            plt.plot(processed_df['Time_min'], processed_df['Normalized_Mean'],
                    color=colors[i], alpha=alpha, linewidth=1.5, label=label)
    
    # Add treatment window
    plt.axvspan(1280, 1320, color='red', alpha=0.2, label='Treatment Window')
    
    plt.xlabel('Time (min)')
    plt.ylabel('Normalized Cell Count (%)')
    plt.title('Overview: All Processed Experimental Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    overview_path = os.path.join(output_dir, 'overview_all_curves.png')
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Overview plot saved to {overview_path}")
    
    return overview_path

def main():
    """
    Main function to process all experimental curves.
    """
    # Set paths
    input_dir = "/home/oth/BSC/AGS/mn5sync/EMEWS/data/AGS_data/AGS_growth_data/output/csv"
    output_dir = "/home/oth/BSC/AGS/mn5sync/EMEWS/data/AGS_data/AGS_growth_data/output/csv/processed_comprehensive"
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        return
    
    logging.info("Starting comprehensive experimental curve processing...")
    
    # Process all curves
    processing_summary = process_all_curves(input_dir, output_dir)
    
    # Create overview plot
    create_overview_comparison_plot(processing_summary, output_dir)
    
    # Print summary statistics
    drug_type_counts = {}
    for file_info in processing_summary['processed_files']:
        drug_type = file_info['drug_type']
        drug_type_counts[drug_type] = drug_type_counts.get(drug_type, 0) + 1
    
    logging.info("Processing Summary:")
    logging.info(f"Total files processed: {len(processing_summary['processed_files'])}")
    logging.info("Files by drug type:")
    for drug_type, count in drug_type_counts.items():
        logging.info(f"  {drug_type}: {count} files")
    
    logging.info(f"All processed data saved to: {output_dir}")

if __name__ == "__main__":
    main()
