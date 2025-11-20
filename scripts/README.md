# Analysis Scripts Directory

This directory contains all scripts for experimental data processing, simulation summarization, post-EMEWS analysis, and utility functions used throughout the PhysiBoSS-AGS drug synergy modeling project.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [1. Experimental Data Processing](#1-experimental-data-processing)
  - [`get_drug_response_curve.py`](#get_drug_response_curvepy)
  - [`process_all_experimental_curves.py`](#process_all_experimental_curvespy)
- [2. Simulation Summarization](#2-simulation-summarization)
  - [`summarize_ags_pcdl.py`](#summarize_ags_pcdlpy)
- [3. Post-EMEWS Analysis](#3-post-emews-analysis)
  - [3.1 EMEWS Run Analysis](#31-emews-run-analysis)
    - [`summarize_deap_run.py`](#summarize_deap_runpy)
    - [`convergence_plot.py`](#convergence_plotpy)
  - [3.2 Machine Learning Analysis](#32-machine-learning-analysis)
    - [`feature_analysis.py`](#feature_analysispy)
  - [3.3 Synergy Experiment Preparation](#33-synergy-experiment-preparation)
    - [`obtain_param_distro.py`](#obtain_param_distropy)
    - [`compare_top_params.py`](#compare_top_paramspy)
  - [3.4 Top Parameter Analysis](#34-top-parameter-analysis)
    - [`topN_curves_plot.py`](#topn_curves_plotpy)
    - [`topN_sim_details.py`](#topn_sim_detailspy)
- [4. 3D Drug Timing and Diffusion Analysis](#4-3d-drug-timing-and-diffusion-analysis)
  - [Representative Selection and Validation](#representative-selection-and-validation)
    - [`select_all_scenario_representatives.py`](#select_all_scenario_representativespy)
    - [`select_representative_simulations.py`](#select_representative_simulationspy)
    - [`plot_synergy_validation.py`](#plot_synergy_validationpy)
  - [Efficacy and Timing Analysis](#efficacy-and-timing-analysis)
    - [`plot_efficacy_grid.py`](#plot_efficacy_gridpy)
    - [`analyze_synergy_parameters.py`](#analyze_synergy_parameterspy)
    - [`plot_high_dose_scenario.py`](#plot_high_dose_scenariopy)
  - [Dynamics and Mechanistic Analysis](#dynamics-and-mechanistic-analysis)
    - [`plot_aggregated_timing_dynamics.py`](#plot_aggregated_timing_dynamicspy)
    - [`plot_representative_dynamics.py`](#plot_representative_dynamicspy)
    - [`plot_positive_controls_singledrugs.py`](#plot_positive_controls_singledrugspy)
- [5. Utility Scripts](#5-utility-scripts)
  - [`generate_json_sweep.py`](#generate_json_sweeppy)
  - [`physiboss_paraview_viewer.py`](#physiboss_paraview_viewerpy)
  - [`summarize_run.sh`](#summarize_runsh)
- [6. Main Execution Script](#6-main-execution-script)
  - [`growth_model.sh`](#growth_modelsh)
- [Workflow Integration](#workflow-integration)
- [Development Status](#development-status)
- [Additional Resources](#additional-resources)

---

## Directory Structure

```
scripts/
├── exp_data_preprocessing/              # Experimental data processing and curve fitting
│   ├── get_drug_response_curve.py       # Hill equation fitting for dose-response curves
│   └── process_all_experimental_curves.py # Comprehensive experimental data processing
│
├── 2_summary_scripts/                   # Simulation output summarization during EMEWS runs
│   ├── summarize_ags_pcdl.py            # Core EMEWS instance summarization script
│   └── to_delete/                       # Deprecated versions
│
├── 3_post_emews_analysis/               # Post-calibration analysis and visualization
│   ├── emews_run_analysis/              # DEAP optimization run summarization
│   │   ├── summarize_deap_run.py        # Aggregate results into CSV summaries
│   │   └── convergence_plot.py          # Evolutionary algorithm convergence plots
│   │
│   ├── ml_analysis/                     # Machine learning feature importance analysis
│   │   ├── feature_analysis.py          # SHAP, permutation importance, PDP
│   │   └── run_feature_analysis.sh      # Batch execution wrapper
│   │
│   ├── synergy_experiment_preparation/  # Parameter consensus and synergy space generation
│   │   ├── obtain_param_distro.py       # Extract distribution statistics from top-N
│   │   └── compare_top_params.py        # Statistical comparison and consensus generation
│   │
│   ├── top_N_params_analysis/           # Top parameter set visualization
│   │   ├── topN_curves_plot.py          # Growth curve comparisons (sim vs exp)
│   │   ├── topN_sim_details.py          # Mechanistic dynamics and node states
│   │   └── to_delete/                   # Superseded scripts
│   │
│   └── 3D_AGS_simulations_analysis/     # 3D drug timing and diffusion analysis
│       ├── select_all_scenario_representatives.py  # Representative selection
│       ├── select_representative_simulations.py    # Alternative selection approach
│       ├── plot_synergy_validation.py              # Statistical synergy validation
│       ├── plot_efficacy_grid.py                   # Efficacy vs timing grids
│       ├── analyze_synergy_parameters.py           # Top synergistic parameter sets
│       ├── plot_high_dose_scenario.py              # High-dose focused analysis
│       ├── plot_aggregated_timing_dynamics.py      # Aggregated multi-scale dynamics
│       ├── plot_representative_dynamics.py         # Individual representative plots
│       └── plot_positive_controls_singledrugs.py   # Single-drug dose-response validation
│
├── utils/                               # Utility scripts and helper functions
│   ├── generate_json_sweep.py           # Parameter sweep file generation (5 sampling modes)
│   ├── physiboss_paraview_viewer.py     # PhysiBoSS to ParaView VTK converter
│   ├── physiboss_paraview_viewer_README.md  # ParaView visualization guide
│   ├── summarize_run.sh                 # Quick fitness metric extraction
│   └── to_delete/                       # Unused utility scripts
│
├── growth_model.sh                      # Main simulation execution wrapper (called by Swift/T)
└── README.md                            # This file
```

## Prerequisites

**Python Dependencies:**
- Core: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
- Statistical analysis: `scikit-learn`, `statsmodels`
- PhysiBoSS output handling: `pcdl` (PhysiCell Data Loader)
- Parallel processing: `joblib`, `multiprocessing`

**Other Requirements:**
- Access to experimental data files (Flobak et al. dataset)
- PhysiBoSS simulation outputs in standard format
- EMEWS experiment directory structure: an "experiments" folder, within which we have different instances with the PhysiBoSS simulation output.

---

## 1. Experimental Data Processing

**Location:** `exp_data_preprocessing/`

Scripts for processing raw experimental data, extracting dose-response relationships, and preparing reference datasets for model calibration.

### Scripts

#### `get_drug_response_curve.py`
**Purpose:** Processes PDF-extracted dose-response curve data to derive Hill equation parameters for drug-specific transfer functions. These parameters define the mapping between internal drug concentrations and target Boolean network nodes (PI3K, MEK, AKT) in the PhysiBoSS model.

**Key Operations:**
- Loads replicate dose-response measurements from PDF-extracted CSV files
- Averages across replicates and computes standard deviations
- Fits Hill equation to determine EC50, Hill coefficient, minimum and maximum response
- Converts between log and linear concentration scales
- Generates both log-scale and linear-scale dose-response plots

**Inputs:**
- Replicate CSV files from `data/AGS_data/AGS_growth_data/drug_response_curves/{drug_name}/points_from_pdf/`
- Files named as `{drug_name}_rep1.csv`, `{drug_name}_rep2.csv`, etc.
- Each CSV contains columns: `drug_concentration_log`, `cell_index`

**Outputs:**
- `{drug_name}_DR_averaged_linear_uM.csv`: Averaged dose-response data with standard deviations
- `{drug_name}_DR_hill_params.csv`: Fitted Hill equation parameters (EC50, Hill coefficient, min/max response)
- `{drug_name}_DR_fitted_curve.csv`: Smooth fitted curve data points
- `{drug_name}_DR_hill_curve_log.png` and `{drug_name}_DR_hill_curve_linear.png`: Visualization plots

**Usage:**
```bash
# Edit drug_name variable in script, then run:
python get_drug_response_curve.py
```

**Note:** The fitted Hill parameters are used to parameterize the drug effect transfer functions in the PhysiBoSS model's XML configuration files.

---

#### `process_all_experimental_curves.py`
**Purpose:** Comprehensive batch processing of all experimental growth curves from Flobak et al. dataset. Normalizes time-series data, categorizes by drug treatment, and generates summary statistics and comparative visualizations.

**Key Operations:**
- Processes control (WT) curves to establish normalization baseline
- Normalizes all experimental curves to 0-100% scale using control min/max values
- Interpolates curves using cubic splines for consistent time points
- Categorizes experiments by drug type (single drugs, combinations, control)
- Extracts drug concentrations from filenames
- Generates individual curve plots and overview comparison plots

**Inputs:**
- All experimental CSV files from `data/AGS_data/AGS_growth_data/output/csv/`
- Control file (containing 'CTRL' in filename) for normalization reference
- CSV format: Time column (seconds) + replicate measurement columns

**Outputs:**
- `processed_comprehensive/` directory containing:
  - `{experiment}_processed.csv`: Normalized time-series data (Time_min, Normalized_Mean, Normalized_Std)
  - `{experiment}_metadata.json`: Experiment metadata (drug type, concentrations, normalization parameters)
  - `plots/{experiment}_processed.png`: Individual curve visualization
  - `processing_summary.json`: Complete processing log with all file paths and metadata
  - `overview_all_curves.png`: Comparative plot of all curves grouped by drug type

**Usage:**
```bash
# Update input_dir and output_dir in main() function, then run:
python process_all_experimental_curves.py
```

**Processing Details:**
- Time conversion: seconds → minutes
- Normalization: (value - min_control) / (max_control - min_control) × 100
- Time filtering: Removes middle timepoints (400-2500 min) to focus on pre/post-drug periods
- Treatment window visualization: Marked at ~1280 minutes in plots


---

## 2. Simulation Summarization

**Location:** `2_summary_scripts/`

Scripts executed during or immediately after PhysiBoSS simulations to extract key metrics and prepare data for fitness evaluation in EMEWS workflows.

### Scripts

#### `summarize_ags_pcdl.py`
**Purpose:** Core summarization script that processes PhysiBoSS simulation outputs to generate comprehensive analysis reports for EMEWS experiments (both parameter sweeps and evolutionary algorithm optimizations).

**Key Operations:**
- **Data Loading**: Uses pcdl (PhysiCell Data Loader) to read PhysiBoSS output files from the instance directory
- **Data Processing**: Extracts cell population dynamics (alive, apoptotic, necrotic cells), drug concentrations, and Boolean network node states
- **Normalization**: Normalizes simulation growth curves against control experiments using the same min/max scaling as experimental data
- **Visualization Generation**: Creates comprehensive plots depending on experiment type:
  - **Control experiments**: Cell population dynamics and normalized growth curves
  - **Single-drug experiments**: Cell dynamics, drug density over time, Boolean node states, experimental vs simulation comparison
  - **Combination experiments**: All of the above for both drug X and drug Y
- **Data Export**: Saves processed data in compressed CSV format for downstream analysis

**Inputs:**
- `instance_dir`: Path to PhysiBoSS simulation output folder (containing `output/` subdirectory with simulation files)
- `drug_name`: Drug identifier for experimental data comparison (PI3K, MEK, AKT, PI3K_MEK, AKT_MEK, or WT)

**Outputs:**
- `simulation_growth.csv`: Time-series of alive cell counts
- `pcdl_total_info_sim.csv.gz`: Comprehensive compressed cell-level data
- `full_summary_plot.png`: Multi-panel figure with all relevant visualizations
- Individual plot files: `cellgrowth.png`, `densityplot_{drug}.png`, `nodeplot.png`, `curve_comparison_seaborn.png`

**Usage:**
```bash
# Called automatically by Swift/T workflows during EMEWS runs
python summarize_ags_pcdl.py /path/to/instance_folder PI3K
```

**Integration with EMEWS:**
This script is invoked by the Swift/T workflow (`swift_run_sweep.swift` or `swift_run_eqpy.swift`) for each simulation instance. The generated `simulation_growth.csv` is used by subsequent fitness evaluation scripts to compute metrics like RMSE against experimental data, which drives parameter selection in evolutionary algorithms or evaluates parameter sweep results.

**Experiment Type Detection:**
The script automatically detects experiment type by reading the PhysiBoSS XML configuration:
- Checks `drug_X_target` and `drug_Y_target` tags
- If both are "none": Control experiment
- If only drug_Y is "none": Single-drug experiment  
- If both are active: Drug combination experiment

**Note:** Deprecated versions (`summarize_ags_pcdl_debug.py`, `summarize_ags_pcdl_nord4.py`) and unused scripts (`create_table_topN.py`) have been moved to the `to_delete/` subdirectory.

---

## 3. Post-EMEWS Analysis

**Location:** `3_post_emews_analysis/`

Comprehensive analysis scripts for processing completed EMEWS experiments, comparing parameter distributions, visualizing convergence, analyzing parameter importance, and generating publication-quality figures.

### 3.1 EMEWS Run Analysis

**Location:** `3_post_emews_analysis/emews_run_analysis/`

Scripts for summarizing DEAP optimization runs and visualizing evolutionary algorithm convergence.

#### `summarize_deap_run.py`
**Purpose:** Primary script for aggregating and summarizing completed DEAP (Genetic Algorithm or CMA-ES) optimization runs. Collects instance-level JSON results, compiles them into comprehensive CSV summaries, and extracts top-performing parameter sets.

**Key Operations:**
- Scans experiment directory for all instance folders and summary JSON files
- Aggregates individual simulation results into single CSV file per experiment
- Automatically detects experiment type (CMA, GA, or sweep) from naming convention
- Calculates percentile-based and fixed-number top selections (top 1%, 5%, 10%, 25%, and top 10, 20, 50, 100, 200)
- Saves summaries to appropriate results subdirectory (CMA_summaries/, GA_summaries/, or sweep_summaries/)

**Inputs:**
- `--single_experiment` or `-s`: Path to single experiment folder containing `instance_X_Y_Z/` subdirectories
- `--all_experiments_folder` or `-exp`: Path to parent folder containing multiple experiments (default: `/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/experiments`)
- `--run_all_experiments` or `-a`: Flag to process all experiments in the parent folder
- Each instance folder must contain: `{iteration}_{individual}_{replicate}_summary.json`

**Outputs:**
- `results/{algorithm}_summaries/final_summary_{experiment_name}.csv`: Complete aggregated results with all parameters and fitness metrics
- `results/{algorithm}_summaries/final_summary_{experiment_name}/`:
  - `top_10.csv`, `top_20.csv`, `top_50.csv`, `top_100.csv`, `top_200.csv`: Fixed number selections
  - `top_1p.csv`, `top_5p.csv`, `top_10p.csv`, `top_25p.csv`: Percentile-based selections
- Copy of `final_summary_{experiment_name}.csv` in original experiment directory

**Usage:**
```bash
# Process single experiment
python summarize_deap_run.py -s /path/to/experiment_folder

# Process all experiments in directory
python summarize_deap_run.py -exp /path/to/experiments_parent -a
```

**Note:** This script automatically handles both evolutionary algorithm runs (with iteration column) and parameter sweeps (without iteration column). The fitness metric (last column of CSV) is used for ranking.

---

#### `convergence_plot.py`
**Purpose:** Generates convergence plots showing fitness metric evolution across generations for DEAP optimization runs (CMA-ES and Genetic Algorithms). Visualizes population mean and standard deviation to assess algorithm performance.

**Key Operations:**
- Loads aggregated summary CSV files from `summarize_deap_run.py`
- Calculates mean and standard deviation of fitness metric for each generation
- Creates publication-quality line plots with error bars and confidence intervals
- Automatically detects algorithm type (CMA or GA) from filename
- Generates separate plots for each experiment and comparison plots between algorithms
- Uses Cell Systems journal styling (sans-serif fonts, minimalist design)

**Inputs:**
- `results/CMA_summaries/final_summary_{experiment_name}.csv`: CMA-ES run results
- `results/GA_summaries/final_summary_{experiment_name}.csv`: GA run results
- Script automatically discovers experiments in `./experiments/` folder

**Outputs:**
- `results/convergence_plots/{experiment_names}/`:
  - `convergence_plot_{experiment_name}.png`: Individual convergence plot (300 dpi)
  - `convergence_plot_{experiment_name}.svg`: Vector format for publication
  - `convergence_plot_{CMA_name}_{GA_name}.png`: Side-by-side comparison of CMA vs GA

**Usage:**
```bash
python convergence_plot.py
```

**Note:** Script processes all CMA experiments found in `./experiments/` folder and automatically looks for corresponding GA experiments for comparison. For control experiments, x-axis is limited to 24 generations.

---

### 3.2 Machine Learning Analysis

**Location:** `3_post_emews_analysis/ml_analysis/`

Random Forest-based feature importance analysis using SHAP values to identify key parameters influencing model fitness.

#### `feature_analysis.py`
**Purpose:** Performs comprehensive machine learning analysis of calibration results to quantify parameter importance using Random Forest regression and SHAP (SHapley Additive exPlanations) values. Identifies which PhysiBoSS parameters most strongly influence model fitness (RMSE).

**Key Operations:**
- **Random Forest Training**: Fits ensemble regression models to predict fitness from parameter values
- **SHAP Analysis**: Calculates SHAP importance values showing contribution magnitude and direction (positive/negative effect on RMSE)
- **Permutation Importance**: Computes normalized permutation importance as alternative metric
- **Partial Dependence**: Generates partial dependence plots showing non-linear parameter-fitness relationships
- **Feature Interaction Analysis**: Identifies parameter interactions using SHAP interaction values
- **Stability Analysis**: Performs bootstrap resampling (100 iterations) to assess SHAP value robustness across data subsets
- **Parameter Name Transformation**: Converts technical parameter names to LaTeX mathematical notation for publication
- **Optimized Processing**: Uses subsampling (1000 samples), reduced trees (50), and parallel processing for large datasets

**Inputs:**
- `results/CMA_summaries/final_summary_{experiment_name}.csv`: CMA-ES calibration results
- `results/sweep_summaries/final_summary_{experiment_name}.csv`: Parameter sweep results
- Predefined experiment lists in `main()` function:
  - CMA experiments: PI3Ki, MEKi, AKTi single-drug calibrations (25 generations, 18 parameters)
  - Sweep experiments: PI3K-MEK and AKT-MEK synergy sweeps (single-drug and combined)

**Outputs:**
- `results/feature_analysis/shap_importance/`:
  - `shap_importance_{exp_name}_{metric}.png/svg`: Horizontal bar plot showing mean SHAP values (red=increases RMSE, blue=decreases RMSE)
  - `shap_importance_{exp_name}_{metric}.csv`: Quantitative SHAP values for each parameter
- `results/feature_analysis/permutation_importance/`:
  - `permutation_importance_{exp_name}_{metric}.png/svg`: Normalized permutation importance (black bars)
  - `permutation_importance_{exp_name}_{metric}.csv`: Quantitative importance values
- `results/feature_analysis/partial_dependence/`:
  - `pdp_grid_{exp_name}_{metric}.png/svg`: Grid of partial dependence plots showing non-linear relationships
- `results/feature_analysis/interactions/`:
  - `shap_interactions_{exp_name}.png/svg`: Heatmap of parameter interaction strengths
  - `pdp_{param1}_{param2}_{exp_name}.png/svg`: 2D contour plots for top 5 interacting parameter pairs
- `results/feature_analysis/stability_analysis/`:
  - `shap_stability_{exp_name}_{metric}.png/svg`: Box plots showing SHAP value distributions across 100 bootstrap iterations
  - `shap_stability_data_{exp_name}_{metric}.csv`: Raw bootstrap data

**Usage:**
```bash
python feature_analysis.py
```

**Note:** This script uses joblib for parallel processing and LRU caching for performance. It automatically excludes metadata columns (individual, iteration, replicate) and uses the last column as the fitness metric. For control experiments, it detects and handles different parameter sets (cell spacing, pressure Hill parameters).

---

### 3.3 Synergy Experiment Preparation

**Location:** `3_post_emews_analysis/synergy_experiment_preparation/`

Scripts for generating synergy parameter spaces from single-drug calibrations through statistical comparison and consensus parameter identification.

#### `obtain_param_distro.py`
**Purpose:** Extracts parameter distribution statistics (mean, std, min, max, count) from top-N CSV files and saves them as JSON files for downstream analysis and synergy experiment design.

**Key Operations:**
- Recursively scans CMA, GA, and sweep summary directories
- For each top-N CSV file, calculates distribution statistics for all parameters
- Excludes metadata columns (individual, iteration, replicate) and fitness metric
- Uses custom NumPy encoder to handle numpy data types in JSON serialization
- Automatically determines experiment name and top-N value from file path

**Inputs:**
- `results/CMA_summaries/final_summary_{exp_name}/top_{N}.csv`: CMA top-N parameter sets
- `results/GA_summaries/final_summary_{exp_name}/top_{N}.csv`: GA top-N parameter sets
- `results/sweep_summaries/final_summary_{exp_name}/top_{N}.csv`: Sweep top-N parameter sets

**Outputs:**
- `{same_directory_as_input}/{exp_name}_param_distribution_top_{N}.json`: JSON file containing:
  ```json
  {
    "parameter_name": {
      "mean": float,
      "std": float,
      "min": float,
      "max": float,
      "count": int
    },
    ...
  }
  ```

**Usage:**
```bash
python obtain_param_distro.py
```

**Note:** Script automatically processes all subdirectories in CMA_summaries/, GA_summaries/, and sweep_summaries/, skipping base directories. This is typically run after `summarize_deap_run.py` to prepare data for `compare_top_params.py`.

---

#### `compare_top_params.py`
**Purpose:** Performs statistical comparison of parameter distributions between different calibration experiments (PI3K vs MEK, AKT vs MEK) to identify consensus parameters for synergy exploration. Uses Mann-Whitney U tests to determine statistically significant differences and generates consensus parameter ranges.

**Key Operations:**
- **Statistical Testing**: Applies Mann-Whitney U test to compare parameter distributions between drug pairs
- **Consensus Generation**: Creates parameter ranges based on statistical significance:
  - **Significant differences** → Use union of ranges (widest coverage)
  - **No significant differences** → Use intersection of ranges (overlapping region)
- **Drug-Specific Handling**: Separately processes drug-specific parameters (drug_X_*) which are unique to each drug
- **Parameter Renaming**: For synergy sweeps, renames drug_X to drug_Y for the second drug
- **Visualization**: Generates box plots comparing distributions with significance stars (***, **, *, ns)
- **XML Generation**: Creates averaged XML configuration files from top percentile parameters for 3D simulations
- **Multiple Comparison Tables**: Generates comprehensive statistical tables comparing all six conditions (3 single drugs × 2 synergy combinations)

**Inputs:**
- JSON parameter distributions from `obtain_param_distro.py`:
  - PI3K, MEK, AKT single-drug calibration top 10% parameters
  - PI3K-MEK synergy sweep (combined, PI3K single, MEK single)
  - AKT-MEK synergy sweep (combined, AKT single, MEK single)
- `data/JSON/deap/deap_18p_single_drug_exp_v2.json`: Original parameter structure with bounds
- Template XML files for 3D simulation generation

**Outputs:**
- `results/comparing_top_distributions/`:
  - `final_summary_{exp1}_and_{exp2}_top_{N}_comparison.csv`: Mann-Whitney U test results with p-values and significance
  - `sweep_combined_{exp1}_and_{exp2}_top_{N}.json`: Consensus ranges for synergy sweep (both drugs)
  - `{exp1}_single_drug_sweep_combined_{exp1}_{exp2}_top_{N}.json`: Single-drug ranges in synergy space (drug 1 only)
  - `{exp2}_single_drug_sweep_combined_{exp1}_{exp2}_top_{N}.json`: Single-drug ranges in synergy space (drug 2 only)
- `results/comparing_top_distributions/violin_plots_*/`:
  - `combined_dist_{exp1}_{exp2}_A4_third_boxplot.png`: Box plots with significance annotations
- `results/comparing_top_distributions/top_params_table/`:
  - `single_drug_parameter_distributions.csv`: Mean ± std table for all single-drug parameters
  - `synergy_comparison_table.csv`: Effect sizes and overlap percentages between synergy conditions
  - `comprehensive_comparison_table.csv`: Kruskal-Wallis test results across all conditions
  - `synergy_parameter_analysis.csv`: Analysis of synergy-specific parameter differences
  - `six_way_comparison_table.csv`: Full pairwise comparison statistics
  - `six_way_comparison_simplified.csv`: Summary of overlap percentages only
  - `parameter_counts_per_condition.csv`: Count of total, common, and drug-specific parameters
- `data/physiboss_config/dose_curves_experiments/`:
  - `{drug}_CMA-..._top_{N}_averaged.xml`: 3D-ready XML configurations with averaged parameters

**Usage:**
```bash
python compare_top_params.py
```

**Note:** Script contains hardcoded experiment names and paths in `main()`. Significant differences are defined as p < 0.05. For drug-specific parameters, original ranges from each drug are preserved but renamed appropriately for synergy sweeps (drug_X and drug_Y).

---

### 3.4 Top Parameter Analysis

**Location:** `3_post_emews_analysis/top_N_params_analysis/`

Scripts for visualizing and analyzing top-performing parameter sets from calibration experiments, comparing simulation vs experimental growth curves and generating detailed mechanistic plots.

#### `topN_curves_plot.py`
**Purpose:** Generates publication-quality growth curve comparisons between experimental data and simulations from top-N parameter sets. Creates multi-panel figures for single-drug calibrations and synergy validations with statistical correlation analysis.

**Key Operations:**
- **Data Loading and Caching**: Uses LRU caching and file modification time checking to avoid reprocessing
- **Normalization**: Normalizes both experimental and simulation curves to 0-100% scale using control (WT) min/max values
- **Interpolation**: Applies cubic spline interpolation for smooth curves
- **Statistical Analysis**: Calculates Pearson correlation between simulation mean and experimental data
- **Parallel Processing Support**: Includes caching mechanisms for efficient batch processing
- **Multi-Panel Layouts**: Creates 4-panel horizontal figures for calibration summaries and synergy comparisons
- **AKT Data Handling**: Special handling to remove last timepoint from AKT and AKT-MEK data for alignment
- **RMSE Violin Plots**: Generates distribution plots of fitness metrics across top-N parameter sets
- **Color Scheme**: Professional color palette with separate colors for experimental (darker) and simulation (lighter) for each drug

**Inputs:**
- `results/{strategy}_summaries/final_summary_{exp_name}/top_{N}.csv`: Top-N parameter sets with instance identifiers
- `experiments/{exp_name}/instance_{iteration}_{individual}_{replicate}/`:
  - `pcdl_total_info_sim.csv.gz` or `.bz2`: Compressed full simulation data
  - `simulation_growth.csv`: Time-series of alive cell counts
- `data/AGS_data/AGS_growth_data/output/csv/`:
  - `CTRL.csv`: Control experimental data for normalization
  - `PI103(0.70uM).csv`, `PD0325901(35.00nM).csv`, `AKT_2.csv`: Single-drug experimental curves
  - `PD0325901(17.50nM)+PI103(0.35uM).csv`, `AKT_MEK_final_ok.csv`: Combination experimental curves
- `data/AGS_data/AGS_growth_data/output/csv/SIM_CTRL_CMA-1110-1637-5p.csv`: Simulation control reference

**Outputs:**
- `results/{strategy}_summaries/final_summary_{exp_name}/`:
  - `growth_comparison_{drug_name}_top{N}.png/svg`: Individual growth curve plot (1.67" × 2.5", 300 dpi)
  - `growth_comparison_{drug_name}_top{N}_legend.png/svg`: Separate legend figure
  - `pearson_correlation_{drug_name}_top{N}.txt`: Correlation coefficient and p-value
- `results/publication_plots/single_drug_calibration/`:
  - `growth_comparison_{drug}_calibration.png/svg`: Single drug with control
  - `growth_comparison_{drug}_calibration_side_by_side.png/svg`: Experimental vs simulation side-by-side
  - `summary_single_drug_calibrations.png/svg`: 3-panel summary (PI3Ki, MEKi, AKTi)
- `results/publication_plots/synergy_comparison/`:
  - `growth_comparison_{drug1}i_{drug2}i_synergy.png/svg`: Synergy with control and single drugs
  - `summary_synergy_comparisons.png/svg`: 2-panel summary (PI3Ki-MEKi, AKTi-MEKi)

**Usage:**
```bash
# Edit experiment names and paths in script, then run:
python topN_curves_plot.py
```

**Hardcoded Configurations:**
- `best_control_experiment_name = "CTRL_CMA-1110-1637-5p"`
- `best_pi3k_experiment_name = "PI3Ki_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen"`
- `best_mek_experiment_name = "MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen"`
- `best_akt_experiment_name = "AKTi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen"`
- `top_n = "10p"` (top 10%)
- Synergy experiments use 5000-parameter uniform sweeps with 10% top selection

**Note:** Script uses extensive caching (DATA_CACHE dictionary) to speed up reprocessing. Treatment window is marked at 1280-1292 minutes in plots. For sweep experiments, top-N selection is done by sorting the full summary CSV by fitness metric.

---

#### `topN_sim_details.py`
**Purpose:** Generates detailed mechanistic plots for top-N parameter sets showing intracellular dynamics, Boolean network states, cell rates, signals, and phenotypic weights. Creates comparison grids for single-drug calibrations and synergy experiments.

**Key Operations:**
- **Cell Population Dynamics**: Plots alive and apoptotic cell counts over time
- **Boolean Network States**: Visualizes continuous node state values for pro-survival (cMYC, RSK, TCF) and anti-survival (FOXO, Caspase8, Caspase9) nodes
- **Cell Rates**: Shows growth rate and apoptosis rate dynamics
- **Cell Signals**: Displays pro-survival and anti-survival signal trajectories
- **Node Weights**: Generates box plots showing distributions of Boolean node weights across cells
- **Comparison Grids**: Creates multi-panel layouts comparing:
  - Single-drug experiments (PI3Ki, MEKi, AKTi) in 3×6 grids
  - Drug combinations (PI3Ki-MEKi, AKTi-MEKi) in 2×6 grids
  - Top-N parameter sets in comprehensive comparison grids
- **Parallel Processing**: Uses ProcessPoolExecutor for efficient batch processing of multiple experiments

**Inputs:**
- `results/{strategy}_summaries/final_summary_{exp_name}/top_{N}.csv`: Top-N parameter sets
- `experiments/{exp_name}/instance_{iteration}_{individual}_{replicate}/`:
  - `pcdl_total_info_sim.csv.gz` or `.bz2`: Compressed full simulation data with cell-level information
  - `sim_summary.json`: Instance metadata

**Outputs:**
- `results/{strategy}_summaries/final_summary_{exp_name}/`:
  - `alive_apoptotic_plot_{exp_name}.png`: Cell population dynamics (alive and apoptotic)
  - `node_states_plot_full_{exp_name}.png`: All Boolean node states (continuous values)
  - `anti_node_states_plot_{exp_name}.png`: Anti-survival nodes only (FOXO, Caspase8, Caspase9)
  - `cell_rates_plot_{exp_name}.png`: Growth rate and apoptosis rate over time
  - `cell_signals_plot_{exp_name}.png`: Pro-survival and anti-survival signals
  - `prosurvival_weights_boxplot_{exp_name}.png`: Box plots for cMYC, RSK, TCF weights
  - `antisurvival_weights_boxplot_{exp_name}.png`: Box plots for FOXO, Caspase8, Caspase9 weights
- `results/comparison_grids/`:
  - `single_drug_grid_PI3K_MEK_AKT.png`: 3×6 grid comparing single-drug calibrations
  - `combination_comparison_PI3KMEK_AKTMEK.png`: 2×6 grid comparing synergy experiments
  - `topN_comparison_grid_all_experiments.png`: Comprehensive grid with all experiments
  - `survival_nodes_grid_all_experiments.png`: 2×3 grid focusing on pro/anti-survival nodes

**Usage:**
```bash
# Processes predefined experiment list in main(), then run:
python topN_sim_details.py
```

**Hardcoded Configurations:**
- Same experiment names as `topN_curves_plot.py`
- `top_n = 10` (number of instances to process)
- Uses parallel processing with 4 workers for batch processing

**Plot Specifications:**
- Figure sizes optimized for publication (typically 10" × 6" for grids)
- Cell Systems journal styling (sans-serif fonts, minimalist design, 0.7-0.8 linewidths)
- Treatment window marked at 1280-1292 minutes
- Color palette: Blue (#1E88E5) for PI3K, Green (#2E7D32) for MEK, Purple (#7B1FA2) for AKT, Orange/Dark Orange for combinations

**Note:** This script provides cross-scale mechanistic insights from population-level (alive/apoptotic) to molecular-level (node states, signals, rates). Boolean node weights represent the influence of each node on survival/apoptosis decisions in the PhysiBoSS model.

---

## 4. 3D Drug Timing and Diffusion Analysis

**Location:** `post_emews_analysis/3D_AGS_simulations_analysis/`

Scripts for analyzing 3D parameter sweeps exploring drug diffusion coefficients and administration timing effects on synergistic efficacy. These scripts generate the manuscript figures demonstrating timing-dependent and diffusion-dependent synergy.

### Representative Selection and Validation

#### `select_all_scenario_representatives.py`
**Purpose:** Iterates through all diffusion coefficient scenarios in a synergy sweep experiment and selects representative simulations based on efficacy and Bliss synergy scores.

**Key Operations:**
- Loads control and single-drug reference data
- Calculates Bliss synergy scores for all parameter combinations
- For each diffusion scenario (x_diff, y_diff), selects representatives for:
  - Max Efficacy (minimum mean % alive cells)
  - Min Efficacy (maximum mean % alive cells, excluding simultaneous)
  - Best Synergy (minimum mean Bliss score)
  - Worst Synergy (maximum mean Bliss score)
  - Simultaneous administration (Δt=0, best-performing replicate)

**Inputs:**
- `results/sweep_summaries/final_summary_{experiment_name}.csv`: Main experiment summary
- `results/sweep_summaries/final_summary_{control_experiment}.csv`: No-drug control data
- `results/sweep_summaries/final_summary_{single_drug_experiments}.csv`: Single-drug reference data

**Outputs:**
- `scripts/post_emews_analysis/synergy_recovery_experiments/representative_simulations/{exp_name}/representatives_X_{x_diff}_Y_{y_diff}.csv`: One file per scenario containing selected representative instances

#### `select_representative_simulations.py`
**Purpose:** Alternative selection approach categorizing simulations into four predefined scenarios (symmetric low/high dose, asymmetric fast/slow drug dominant).

**Inputs:**
- `scripts/post_emews_analysis/synergy_recovery_experiments/optimal_timings_synergy/combined_synergy_metrics_{exp_name}.csv`: Metrics file
- `results/sweep_summaries/final_summary_{experiment_name}.csv`: Instance identifiers

**Outputs:**
- `scripts/post_emews_analysis/synergy_recovery_experiments/optimal_timings_synergy/representative_simulations_{exp_name}.csv`: Four representative simulations

#### `plot_synergy_validation.py`
**Purpose:** Creates violin plots comparing control, single-drug, and synergy conditions with statistical significance tests (Mann-Whitney U) to validate true synergistic effects.

**Key Operations:**
- Filters out confounding parameter sets where single-drug efficacy exceeds synergy
- Normalizes all data to control mean
- Performs pairwise statistical comparisons
- Generates publication-quality side-by-side violin plots for PI3Ki-MEKi and AKTi-MEKi

**Inputs:**
- `results/sweep_summaries/final_summary_{synergy_experiment}.csv` (for both PI3K-MEK and AKT-MEK)

**Outputs:**
- `scripts/post_emews_analysis/synergy_recovery_experiments/synergy_validation_plots_2D/internal_synergy_validation_publication.png` and `.svg`

### Efficacy and Timing Analysis

#### `plot_efficacy_grid.py`
**Purpose:** Generates comprehensive efficacy grids showing % alive cells versus timing categories across all diffusion coefficient pairs, with statistical comparisons between timing strategies.

**Key Operations:**
- Filters out confounding individuals based on control and single-drug benchmarks
- Aggregates runs by timing categories (Drug X First, Simultaneous, Drug Y First)
- Creates grid plots with each subplot representing a diffusion coefficient pair
- Adds Mann-Whitney U test annotations comparing timing strategies
- Generates detailed plots showing all individual delta_time values with color gradients
- Creates high-dose focused comparison plots (bar plots, violin plots, with jitter)

**Inputs:**
- `results/sweep_summaries/final_summary_{experiment_name}.csv`

**Outputs:**
- `efficacy_grid_plots_{dim_tag}/{exp_name}/`:
  - `efficacy_grid_aggregated_{exp_name}_publication.png/svg`: Main grid with timing categories
  - `efficacy_grid_detailed_{exp_name}_publication.png/svg`: Grid with all delta_time values
  - `efficacy_summary_aggregated_{exp_name}.csv`: Aggregated efficacy data
- `efficacy_grid_plots_{dim_tag}/`:
  - `high_dose_comparison_publication.png/svg`: Side-by-side high-dose comparison
  - `high_dose_comparison_detailed_publication.png/svg`: Detailed timing at high dose
  - `high_dose_comparison_with_jitter_publication.png/svg`: Bar plot with individual data points
  - `high_dose_comparison_violin_publication.png/svg`: Violin plot distributions

#### `analyze_synergy_parameters.py`
**Purpose:** Identifies top-performing parameter sets exhibiting true synergistic effects by comparing combination outcomes to internal proxy controls and single-drug benchmarks.

**Key Operations:**
- Defines internal proxy conditions (control: low diffusion + late addition; single drugs: asymmetric scenarios)
- Filters for combinations outperforming single-drug benchmarks
- Aggregates by timing categories (positive/negative delta_time signs)
- Ranks parameter sets by efficacy and annotates with proxy outcome comparisons

**Inputs:**
- `results/sweep_summaries/final_summary_{experiment_name}.csv`

**Outputs:**
- `scripts/post_emews_analysis/synergy_recovery_experiments/synergy_analysis_results/top_synergistic_params_quantified_{exp_name}.csv`: Top N parameter sets with quantified outcomes

#### `plot_high_dose_scenario.py`
**Purpose:** Creates focused comparison plots for high-dose symmetric scenarios, including efficacy versus detailed timing for all diffusion coefficient pairs.

**Key Operations:**
- Loads combined synergy metrics from processed data
- Generates full efficacy grids for supplementary materials
- Creates focused high-dose comparison plots for main text
- Generates individual scenario comparison plots (line plots and bar plots with error bars)
- Creates aggregated timing bar plots with statistical tests

**Inputs:**
- `scripts/post_emews_analysis/synergy_recovery_experiments/optimal_timings_synergy/combined_synergy_metrics_{exp_name}.csv`
- `results/sweep_summaries/final_summary_{negative_control}.csv`: For normalization
- `results/sweep_summaries/final_summary_{experiment_name}.csv`: For aggregated bar plots

**Outputs:**
- `scripts/post_emews_analysis/synergy_recovery_experiments/optimal_timings_synergy/`:
  - `efficacy_delta_time_plots/efficacy_grid_{exp_name}.png/svg`: Full grid for supplementary
  - `high_dose_symmetric_efficacy_comparison.png/svg`: Side-by-side high-dose comparison
  - `efficacy_comparison_by_scenario/efficacy_comparison_X_{x_diff}_Y_{y_diff}.png/svg`: Individual scenario line plots
  - `efficacy_barplot_by_scenario/efficacy_barplot_X_{x_diff}_Y_{y_diff}.png/svg`: Individual scenario bar plots
  - `efficacy_aggregated_barplot/efficacy_aggregated_barplot_X_{x_diff}_Y_{y_diff}.png/svg`: Aggregated timing categories with statistics

### Dynamics and Mechanistic Analysis

#### `plot_aggregated_timing_dynamics.py`
**Purpose:** Generates multi-scale aggregated dynamics plots showing population-level behavior across timing categories (Drug X First, Simultaneous, Drug Y First) at high dose.

**Key Operations:**
- Uses feather caching for fast reprocessing of aggregated data
- Processes compressed simulation files in parallel to extract time-series data
- Aggregates cell counts, rates, signals, and node activation fractions across replicates
- Creates 4×3 summary grids (rows: alive/apoptotic, cell rates, cell signals, target activation; columns: timing cases)
- Generates 6×6 complete grids including pro/anti-survival node activation dynamics
- Uses internal control (low diffusion + late addition) for normalization

**Inputs:**
- `results/sweep_summaries/final_summary_{exp_name}.csv`: For filtering and grouping
- `experiments/{exp_name}/instance_{...}/pcdl_total_info_sim.csv.gz`: Individual compressed simulation outputs

**Outputs:**
- `results/aggregated_timing_dynamics/{exp_name}/`:
  - `aggregated_dynamics_grid_D{dose}_filtered.png/svg/pdf`: 4×3 main dynamics grid
  - `survival_nodes_grid_plot.png`: 2×3 grid for pro/anti-survival nodes
  - `cache_D{dose}/`: Feather cache files for fast reloading
- `results/aggregated_timing_dynamics/`:
  - `super_summary_grid_{dim_tag}.png/svg/pdf`: 4×6 cross-experiment comparison
  - `super_summary_complete_grid_{dim_tag}.png/svg/pdf`: 6×6 complete multi-scale comparison

#### `plot_representative_dynamics.py`
**Purpose:** Generates detailed multi-panel visualizations for selected representative simulations, showing mechanistic intracellular dynamics including node states, signals, and phenotypic responses.

**Key Operations:**
- Loads representative simulation data from compressed files
- Processes cell-level data to extract:
  - Alive/apoptotic counts with global normalization
  - Cell growth and apoptosis rates
  - Pro-survival and anti-survival signals
  - Boolean network node states (continuous values)
  - Percentage of cells with activated target nodes (PI3K, AKT, MEK, cMYC, RSK, TCF, FOXO, Caspase8, Caspase9)
- Creates 4×3 summary grids comparing Max/Min Efficacy and Simultaneous cases
- Creates 4×4 cross-experiment comparison grids (PI3Ki-MEKi vs AKTi-MEKi)
- Generates individual plots for each representative

**Inputs:**
- `scripts/post_emews_analysis/synergy_recovery_experiments/representative_simulations/{exp_name}/representatives_X_{x_diff}_Y_{y_diff}.csv`: Representative selections
- `experiments/{exp_name}/instance_{...}/pcdl_total_info_sim.csv.gz`: Individual simulation outputs

**Outputs:**
- `results/representative_dynamics/{exp_name}/X_{x_diff}_Y_{y_diff}/`:
  - `{selection_reason}_cell_counts_normalized.png`: Normalized cell populations
  - `{selection_reason}_cell_rates.png`: Growth and apoptosis rates
  - `{selection_reason}_cell_signals.png`: Pro/anti-survival signals
  - `{selection_reason}_target_activation.png`: Drug target node activation
  - `{selection_reason}_pro_survival_nodes.png`: cMYC, RSK, TCF dynamics
  - `{selection_reason}_anti_survival_nodes.png`: FOXO, Caspase8, Caspase9 dynamics
  - `{selection_reason}_pro_survival_nodes_boolean.png`: Boolean activation fractions for pro-survival
  - `{selection_reason}_anti_survival_nodes_boolean.png`: Boolean activation fractions for anti-survival
  - `summary_grid.png`: 4×3 grid comparing key cases
- `results/representative_dynamics/cross_experiment_comparison/`:
  - `{scenario}_comparison_grid.png`: 4×4 cross-experiment comparison grids

#### `plot_positive_controls_singledrugs.py`
**Purpose:** Visualizes growth curves for single-drug positive controls and no-drug controls across diffusion coefficients to validate dose-response behavior in 3D.

**Key Operations:**
- Processes multiple single-drug experiments (PI3K, MEK, AKT inhibitors) and no-drug control
- Groups growth curves by diffusion coefficient
- Calculates mean and standard deviation across replicates
- For combined drug experiments, filters for specific conditions (symmetric high dose, early simultaneous administration)
- Creates individual plots for each drug and a multi-panel publication figure

**Inputs:**
- `experiments/{single_drug_exp_name}/instance_{...}/simulation_growth.csv`: Growth curves
- `experiments/{single_drug_exp_name}/instance_{...}/sim_summary.json`: Diffusion coefficient metadata
- `experiments/{combined_exp_name}/`: For AKT+MEK and PI3K+MEK at D=600

**Outputs:**
- `scripts/post_emews_analysis/3D_AGS_simulations_analysis/positive_controls_singledrugs_results/`:
  - `pi3k_combined_growth_curves.png/svg`: PI3K inhibitor dose-response
  - `mek_combined_growth_curves.png/svg`: MEK inhibitor dose-response
  - `akt_combined_growth_curves.png/svg`: AKT inhibitor dose-response
  - `nodrug_combined_growth_curves.png/svg`: No-drug control
  - `FIG_suppmat_singledrug_effect_diff_coeff.pdf/png/svg`: Multi-panel publication figure (4 panels: Control, PI3K, MEK, AKT)
  - `control_vs_treatment_comparison.png/svg`: Single panel comparing control vs all treatments at D=600
  - `FIG_suppmat_control_vs_treatment_comparison.pdf`: Publication version

**Usage:**
```bash
python plot_positive_controls_singledrugs.py
```

---

## 5. Utility Scripts

**Location:** `utils/`

Helper scripts for parameter file generation, visualization setup, and quick result summarization.

### Scripts

#### `generate_json_sweep.py`
**Purpose:** Converts JSON parameter distribution files (with min/max/mean/std) into `.txt` files containing individual parameter sets for EMEWS parameter sweep execution. Supports multiple sampling strategies optimized for different exploration objectives.

**Key Operations:**
- **Uniform Sampling**: Random uniform sampling within parameter bounds, ideal for unbiased exploration
- **Grid Sampling**: Systematic grid covering parameter space using `np.linspace` and `ParameterGrid`
- **Logscale Sampling**: Logarithmic sampling maintaining consistent leading digits across orders of magnitude (useful for diffusion coefficients)
- **Hybrid Sampling**: Mixed strategy with log-uniform for specific parameters (diffusion, pulse) and standard uniform for others
- **Structured Hybrid Sampling**: Creates a strategic grid for critical parameters (diffusion, timing) and samples remaining parameters randomly at each grid point

**Inputs:**
- `param_json`: JSON file with parameter specifications:
  ```json
  {
    "parameter.path.name": {
      "min": float,
      "max": float,
      "loc": float,  // for normal mode
      "scale": float  // for normal mode
    }
  }
  ```
- Command-line arguments:
  - `--mode`: Sampling strategy (`uniform`, `grid`, `logscale`, `hybrid`, `structured_hybrid`)
  - `--size`: Number of samples (interpretation varies by mode)
  - `--out`: Output path (optional, defaults to stdout)

**Outputs:**
- `.txt` file with one JSON parameter set per line:
  ```json
  {"param1": value1, "param2": value2, ...}
  {"param1": value3, "param2": value4, ...}
  ```
- Files saved to: `data/JSON/sweep/sweep_txt/{input_name}_{mode}_{size}.txt`

**Usage:**
```bash
# Uniform random sampling (5000 parameter sets)
python generate_json_sweep.py sweep_consensus_pi3k_mek.json --mode uniform --size 5000

# Grid sampling (10 values per parameter, creates 10^N combinations)
python generate_json_sweep.py sweep_consensus_pi3k_mek.json --mode grid --size 10

# Structured hybrid (4 log-spaced diffusion values × 20 random samples each = 80 sets)
python generate_json_sweep.py sweep_consensus_pi3k_mek.json --mode structured_hybrid --size 20
```

**Sampling Mode Details:**
- **`uniform`**: Size = total number of parameter sets, each parameter sampled independently
- **`grid`**: Size = number of points per parameter dimension (total sets = size^n_params)
- **`logscale`**: Generates values maintaining leading digit consistency: [min, 10×min, 100×min, ..., max]
- **`hybrid`**: Size = total parameter sets, uses log-uniform for diffusion/pulse keywords, uniform otherwise
- **`structured_hybrid`**: Size = samples per strategic grid point
  - Creates 4×4 log-spaced grid for diffusion coefficients (drug_X_diffusion, drug_Y_diffusion)
  - Samples consensus parameters uniformly at each grid point
  - Total sets = 16 (grid points) × size (samples per point)

**Hardcoded Paths in Script:**
- Input: `results/comparing_top_distributions/` (consensus parameter JSONs from `compare_top_params.py`)
- Output: `./data/JSON/sweep/sweep_txt/`
- Strategic keywords: `["diffusion_coefficient", "pulse_period"]`
- Experiment configurations for PI3Ki-MEKi and AKTi-MEKi synergy sweeps

**Note:** This script is typically run after `compare_top_params.py` generates consensus parameter ranges. The output `.txt` files are used as input to `sweep_battery.sh` in the EMEWS workflow (see `emews_data/JSON/README.md`).

---

#### `physiboss_paraview_viewer.py`
**Purpose:** Converts PhysiBoSS simulation outputs (`.mat` and `.xml` files) into ParaView-compatible VTK formats (`.vtu` and `.pvd`) for interactive 3D visualization and analysis. Preserves cell geometry, positions, and all intracellular state variables.

**Key Operations:**
- **XML Parsing**: Extracts field labels, sizes, and units from PhysiCell output XML files
- **MAT File Processing**: Loads cell position, radius, type, and custom data from MATLAB format
- **Geometry Generation**: Creates VTK representations:
  - Type 0 (default): Spheres with appropriate radius
  - Type 1: Cylinders for rod-shaped cells with orientation
- **Data Preservation**: Converts all PhysiBoSS fields to VTK arrays:
  - Scalar fields: single-component arrays (e.g., cell_ID, current_phase, pressure)
  - Vector fields: multi-component arrays (e.g., velocity, orientation)
  - Boolean network states: continuous node values
  - Custom data: drug concentrations, signals, rates
- **Temporal Integration**: Creates PVD collection file linking all timesteps for time-series animation
- **Data Validation**: Handles NaN/Inf values, missing fields, and malformed data gracefully
- **Compression**: Uses ZLib compression for efficient file storage

**Inputs:**
- Directory containing PhysiBoSS output files:
  - `output{N}.xml`: XML metadata files with field labels and simulation parameters
  - `output{N}_cells.mat`: MATLAB files with cell data arrays
- Command-line arguments:
  - `output_dir`: Path to PhysiBoSS output directory
  - `--clean`: Remove existing VTU/PVD files before processing (optional)
  - `--prefix`: Custom prefix for VTU files (default: "timestep")

**Outputs:**
- `{output_dir}/simulation.pvd`: ParaView Data collection file (main entry point)
- `{output_dir}/{prefix}_{timestep:06d}.vtu`: VTK Unstructured Grid files (one per timestep)
- Files contain:
  - **Point coordinates**: (x, y, z) positions
  - **Cell properties**: cell_type, radius
  - **All PhysiBoSS fields**: Preserved with original names and units
  - **Temporal metadata**: Timestep information for animation

**Usage:**
```bash
# Basic conversion
python physiboss_paraview_viewer.py experiments/my_simulation/instance_0_0_0/output/

# Clean existing files and regenerate
python physiboss_paraview_viewer.py experiments/my_simulation/instance_0_0_0/output/ --clean

# Use custom file prefix
python physiboss_paraview_viewer.py experiments/my_simulation/instance_0_0_0/output/ --prefix my_sim
```

**Visualization in ParaView:**
1. Open ParaView application
2. File → Open → Navigate to `simulation.pvd`
3. Click "Apply" in Properties panel
4. Select visualization style:
   - Point Gaussian for fast rendering
   - Glyph (sphere) for accurate cell representation
5. Color by any field (e.g., current_phase, drug concentration, node states)
6. Use animation controls for time-series playback

**Technical Details:**
- Uses VTK (Visualization Toolkit) Python bindings
- Binary format with ZLib compression for file size optimization
- Automatically sanitizes field names for XML compatibility
- Robust error handling for incomplete or corrupted data
- Progress reporting for batch conversion

**Note:** For detailed ParaView visualization instructions, filters, and rendering options, see `physiboss_paraview_viewer_README.md` in the same directory. Typical output directory structure: `experiments/{exp_name}/instance_{iter}_{ind}_{rep}/output/`.

---

#### `summarize_run.sh`
**Purpose:** Quick bash script for extracting fitness metrics from all instance directories in an EMEWS experiment. Provides rapid command-line summary without full CSV generation.

**Key Operations:**
- Iterates through all `instance_*` folders in current directory
- Extracts final fitness value from each instance's `metrics.txt` file
- Outputs sorted list of instances with their fitness scores

**Inputs:**
- Assumes current directory contains `instance_{iteration}_{individual}_{replicate}/` folders
- Each instance must have `metrics.txt` file with fitness metric in column 2 of last line

**Outputs:**
- `summary.txt`: Space-separated file with format:
  ```
  fitness_value1 instance_0_0_0
  fitness_value2 instance_0_1_0
  fitness_value3 instance_1_0_0
  ...
  ```

**Usage:**
```bash
# Navigate to experiment directory
cd experiments/PI3Ki_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen/

# Run summarization
bash ../../scripts/utils/summarize_run.sh

# View top 10 best instances (assuming lower is better)
sort -n summary.txt | head -10

# Count total instances
wc -l summary.txt
```

**Script Contents:**
```bash
for i in instance_*
do 
    C=$(tail -1 $i/metrics.txt | cut -f2)
    echo $C $i
done > summary.txt
```

**Use Cases:**
- Quick fitness check without full CSV aggregation
- Identifying best/worst performing instances during runs
- Debugging failed instances (missing from summary indicates errors)
- Pre-filtering before running resource-intensive analysis scripts

**Note:** This is a lightweight alternative to `summarize_deap_run.py` for rapid exploration. For comprehensive analysis with all parameters, use the Python script instead. The `metrics.txt` format depends on the Swift/T workflow configuration.

---

## 6. Main Execution Script

### `growth_model.sh`

**Purpose:** Wrapper script for executing PhysiBoSS simulations with parameter modifications. Called by Swift/T workflows during parameter sweeps and optimization runs.

**Usage:**
```bash
bash growth_model.sh <executable> <param_settings_xml> <emews_root> <instance_dir>
```

---

## Workflow Integration

These scripts integrate with the EMEWS workflow at different stages:

1. **Pre-calibration:** `exp_data_preprocessing/` scripts prepare reference data
2. **During EMEWS runs:** `summarize/` scripts extract metrics for fitness evaluation
3. **Post-calibration:** `post_emews_analysis/` scripts analyze results and identify top parameters
4. **Synergy analysis:** `3D_AGS_simulations_analysis/` scripts validate synergistic effects and explore timing and diffusion dependencies
5. **Throughout:** `utils/` scripts provide supporting functionality

---

## Development Status

This README provides a structural overview of the scripts directory. Detailed documentation for individual scripts marked with "[To be documented]" will be added progressively. For immediate usage questions, refer to inline script documentation and docstrings.

---

## Additional Resources

- EMEWS workflow documentation: `emews_data/JSON/README.md`
- Post-EMEWS analysis detailed guide: `post_emews_analysis/README.md`
- ParaView visualization guide: `utils/physiboss_paraview_viewer_README.md`
