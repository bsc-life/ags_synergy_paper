# AGS PhysiCell+PhysiBoSS Model

This directory contains a complete multiscale agent-based model for simulations of AGS gastric adenocarcinoma cells under single-drug and combination drug treatments. The model uses PhysiCell with integrated PhysiBoSS support for Boolean network modeling of intracellular signaling. This README provides detailed step-by-step instructions for users unfamiliar with PhysiCell/PhysiBoSS to get the model up and running.

## Table of Contents

- [AGS PhysiCell+PhysiBoSS Model](#ags-physicellphysiboss-model)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
    - [Required Software](#required-software)
  - [Installation and Setup](#installation-and-setup)
    - [Step 1: Install PhysiCell Dependencies (Ubuntu)](#step-1-install-physicell-dependencies-ubuntu)
    - [Step 2: Download PhysiCell](#step-2-download-physicell)
    - [Step 3: Integrate the AGS Model into Sample Projects](#step-3-integrate-the-ags-model-into-sample-projects)
    - [Step 4: Register the Model in PhysiCell](#step-4-register-the-model-in-physicell)
    - [Step 5: Load and Compile the AGS Model](#step-5-load-and-compile-the-ags-model)
  - [Running Simulations](#running-simulations)
    - [Quick Test Run](#quick-test-run)
    - [Running Different Experiment Types](#running-different-experiment-types)
      - [1. Control Experiment (No Drugs)](#1-control-experiment-no-drugs)
      - [2. Single-Drug Experiments](#2-single-drug-experiments)
      - [3. Combination Drug Experiments](#3-combination-drug-experiments)
      - [4. 3D Drug Timing Experiments](#4-3d-drug-timing-experiments)
  - [Configuration Files Guide](#configuration-files-guide)
    - [Directory Structure](#directory-structure)
    - [Control Experiments](#control-experiments)
    - [Single-Drug Experiments](#single-drug-experiments)
    - [Combination Drug Experiments](#combination-drug-experiments)
    - [3D Drug Timing Experiments](#3d-drug-timing-experiments)
  - [HPC Deployment](#hpc-deployment)
  - [Model Components](#model-components)
    - [Boolean Network](#boolean-network)
    - [Custom C++ Modules](#custom-c-modules)
      - [`boolean_model_interface.cpp/h`](#boolean_model_interfacecpph)
      - [`drug_transport_model.cpp/h`](#drug_transport_modelcpph)
      - [`custom.cpp/h`](#customcpph)
      - [`submodel_data_structures.cpp/h`](#submodel_data_structurescpph)
    - [XML Configuration Structure](#xml-configuration-structure)
  - [Troubleshooting](#troubleshooting)
    - [Compilation Issues](#compilation-issues)
    - [Runtime Issues](#runtime-issues)
    - [Model-Specific Issues](#model-specific-issues)
  - [Additional Resources](#additional-resources)
    - [Related Documentation](#related-documentation)
    - [External Resources](#external-resources)
  - [Citation](#citation)

---

## Overview

**Note:** These installation instructions are written for Ubuntu Linux. PhysiCell can run on other systems (macOS, Windows via WSL2), but you may need to adapt the dependency installation commands. PhysiCell includes integrated PhysiBoSS support with MaBoSS for Boolean network modeling.

This directory contains:

- **`main.cpp`**: Entry point for the PhysiBoSS executable
- **`Makefile`**: Build configuration for compilation
- **`custom_modules/`**: C++ source code implementing:
  - Boolean network interface (`boolean_model_interface.cpp/h`)
  - Drug transport models (`drug_transport_model.cpp/h`)
  - Custom cell behaviors (`custom.cpp/h`)
  - Data structures (`submodel_data_structures.cpp/h`)
- **`config/`**: Configuration files organized by experiment type
  - `boolean_network/`: Boolean network files (`.bnd`, `.cfg`) defining the AGS signaling network
  - `initial_sim_conditions/`: Initial cell position files (`.csv`) for different domain sizes
  - `physiboss_config/`: XML configuration files for different experimental scenarios

---

## Prerequisites

Before starting, ensure you have the following installed:

### Required Software

1. **C++ Compiler**: GCC 7.5.0 or higher
   
2. **Build Tools**: `make`

3. **Required Libraries**:
   - **libxml2**: XML parsing library
   - **OpenMP**: Parallel computing support (usually included with GCC)

---

## Installation and Setup

### Step 1: Install PhysiCell Dependencies (Ubuntu)

```bash
# Update package lists
sudo apt-get update

# Install compilers and build tools
sudo apt-get install -y build-essential git

# Install required libraries
sudo apt-get install -y libxml2-dev libz-dev
```

---

### Step 2: Download PhysiCell

```bash
# Navigate to your desired installation directory
cd ~/software  # or any directory you prefer

# Clone the PhysiCell repository (includes PhysiBoSS integration with MaBoSS)
git clone https://github.com/MathCancer/PhysiCell.git
cd PhysiCell
```

This will create a directory structure like:
```
PhysiCell/
├── BioFVM/                            # Microenvironment modeling
├── core/                              # PhysiCell core functions
├── modules/                           # Optional modules (PhysiBoSS, etc.)
├── custom_modules/                    # Your custom code goes here
├── sample_projects/                   # Example projects
├── sample_projects_intracellular/     # Intracellular model examples
│   └── boolean/                       # Boolean network models (PhysiBoSS)
├── config/                            # Configuration files
└── ...
```

**Note:** PhysiCell now includes integrated PhysiBoSS support with MaBoSS, so no separate installation is needed.

---

### Step 3: Integrate the AGS Model into Sample Projects

We'll add the AGS model as a sample project that can be easily loaded using PhysiCell's built-in project management system.

```bash
# Assuming you're in the PhysiCell directory and the ags_synergy_paper repo is at ~/ags_synergy_paper

# Create the directory for the AGS model in the sample projects
mkdir -p sample_projects_intracellular/boolean/ags_synergy_model

# Copy all model files into the sample project directory
cp -r ~/ags_synergy_paper/ags_physiboss_model/* sample_projects_intracellular/boolean/ags_synergy_model/
```

**Verify the integration:**

```bash
# Check that files are in place
ls -l sample_projects_intracellular/boolean/ags_synergy_model/
```

You should see:
- `main.cpp` and `Makefile`
- `custom_modules/` directory with `.cpp` and `.h` files
- `config/` directory with subdirectories for different experiment types

---

### Step 4: Register the Model in PhysiCell

Now we need to add our model to the PhysiCell project list by editing the `Makefile-default`.

```bash
# Open the Makefile-default in the sample_projects directory
nano sample_projects/Makefile-default
# Or use your preferred text editor: vim, gedit, etc.
```

**Add the following lines** to the `Makefile-default` (you can add them after the other project targets, typically around line 230-250):

```makefile
physiboss-drugs-synergy-model:
	cp ./sample_projects_intracellular/boolean/ags_synergy_model/custom_modules/* ./custom_modules/
	touch main.cpp && cp main.cpp main-backup.cpp
	cp ./sample_projects_intracellular/boolean/ags_synergy_model/main.cpp ./main.cpp 
	cp Makefile Makefile-backup
	cp ./sample_projects_intracellular/boolean/ags_synergy_model/Makefile .
	cp ./config/PhysiCell_settings.xml ./config/PhysiCell_settings-backup.xml 
	cp ./sample_projects_intracellular/boolean/ags_synergy_model/config/* ./config/
```

**⚠️ IMPORTANT:** The indentation before each `cp` command **must be a TAB character**, not spaces. If you copy-paste and it doesn't work, manually replace the spaces with tabs.

**Save and close the file** (in nano: Ctrl+X, then Y, then Enter).

**Verify the model is registered:**

```bash
# List all available projects
make list-projects
```

You should now see `physiboss-drugs-synergy-model` in the list of available projects.

---

### Step 5: Load and Compile the AGS Model

Now we can load the AGS model and compile it.

```bash
# Make sure you're in the PhysiCell root directory
cd ~/software/PhysiCell  # adjust path as needed

# Load the AGS model (this copies files to the right locations and backs up existing files)
make physiboss-drugs-synergy-model
```

**What this command does:**
- Backs up your current `main.cpp`, `Makefile`, and `PhysiCell_settings.xml`
- Copies AGS-specific custom modules to `custom_modules/`
- Copies AGS `main.cpp` and `Makefile` to the PhysiCell root
- Copies all AGS configuration files to `config/`

```bash
# Compile the model
# This will take several minutes on first compilation
make -j4  # use -j followed by number of cores (e.g., -j8 for 8 cores)
```

**Expected output:**
You should see compilation messages ending with something like:
```
g++ -o physiboss-drugs-synergy-model main.cpp custom_modules/*.o ... [libraries]
Compilation successful!
```

**Check the executable:**
```bash
ls -lh physiboss-drugs-synergy-model
```

You should see an executable file called `physiboss-drugs-synergy-model` (approximately 2-5 MB).

**Test the compilation:**
```bash
./physiboss-drugs-synergy-model --version
```

This should display the PhysiCell version information.

**Switching between projects:**

If you want to work with a different PhysiBoSS sample project later and then return to the AGS model:

```bash
# To switch to another project (e.g., spheroid)
make list-projects  # see all available projects
make reset          # restore backups from previous project
make <project-name> # load the new project
make -j4            # compile

# To return to the AGS model
make reset
make physiboss-drugs-synergy-model
make -j4
```

**Note:** The `make reset` command restores the backup files created when you loaded a project. Your original AGS model files in `sample_projects_intracellular/boolean/ags_synergy_model/` remain unchanged.

---

## Running Simulations

### Quick Test Run

Let's run a simple control simulation to verify everything works:

```bash
# Run a short control simulation (no drugs)
./physiboss-drugs-synergy-model config/physiboss_config/control/settings_AGSv2_CONTROL.xml
```

**What happens:**
1. PhysiCell reads the XML configuration file
2. Initializes 100 cells in a 2D disk
3. Simulates for 4200 minutes (~70 hours)
4. Outputs data every 40 minutes
5. Creates an `output/` directory with:
   - `output*.xml`: Simulation metadata
   - `output*_cells.mat`: Cell data (position, state, etc.)
   - `output*_microenvironment*.mat`: Drug concentrations (if applicable)
   - Final summary statistics

**Expected runtime:** 2-5 minutes on a modern laptop

**Check the output:**
```bash
ls -l output/
```

You should see numbered output files (e.g., `output00000000.xml`, `output00000001.xml`, ...).

---

### Running Different Experiment Types

#### 1. Control Experiment (No Drugs)

```bash
./physiboss-drugs-synergy-model config/physiboss_config/control/settings_AGSv2_CONTROL.xml
```

**Use case:** Baseline growth dynamics, model calibration

---

#### 2. Single-Drug Experiments

**PI3K Inhibitor:**
```bash
./physiboss-drugs-synergy-model config/physiboss_config/single_drug_transient/settings_AGSv2_PI3K_GI50_transient_delayed.xml
```

**MEK Inhibitor:**
```bash
./physiboss-drugs-synergy-model config/physiboss_config/single_drug_transient/settings_AGSv2_MEK_GI50_transient_delayed.xml
```

**AKT Inhibitor:**
```bash
./physiboss-drugs-synergy-model config/physiboss_config/single_drug_transient/settings_AGSv2_AKT_GI50_transient_delayed.xml
```

**Use case:** Single-agent dose-response calibration

**Note:** These simulations add drug at t=1280 minutes, simulating delayed drug treatment after initial culture establishment.

---

#### 3. Combination Drug Experiments

**PI3K + MEK Combination:**
```bash
./physiboss-drugs-synergy-model config/physiboss_config/combined_drug_transient/settings_AGSv2_SYN_PI3K_MEK_halfGI50_transient.xml
```

**AKT + MEK Combination:**
```bash
./physiboss-drugs-synergy-model config/physiboss_config/combined_drug_transient/settings_AGSv2_SYN_AKT_MEK_halfGI50_transient.xml
```

**Use case:** Synergy validation, comparing simultaneous vs. sequential drug administration

---

#### 4. 3D Drug Timing Experiments

**3D No-Drug Control:**
```bash
./physiboss-drugs-synergy-model config/physiboss_config/3D_above_drugtreatment/settings_AGSv2_3D_nodrug.xml
```

**3D PI3K+MEK Consensus Parameters:**
```bash
./physiboss-drugs-synergy-model config/physiboss_config/3D_above_drugtreatment/settings_AGSv2_3D_SYN_PI3K_MEK_consensus_top1p.xml
```

**3D AKT+MEK Consensus Parameters:**
```bash
./physiboss-drugs-synergy-model config/physiboss_config/3D_above_drugtreatment/settings_AGSv2_3D_SYN_AKT_MEK_consensus_top1p.xml
```

**Use case:** Drug diffusion and timing experiments with different diffusion coefficients

**Note:** 3D simulations are more computationally intensive and may require HPC resources for large parameter sweeps.

---

## Configuration Files Guide

### Directory Structure

The `config/` directory is organized as follows:

```
config/
├── boolean_network/              # Boolean signaling network
│   ├── AGS_all_nodes_real.bnd    # Network structure and logic
│   └── AGS_all_nodes_real.cfg    # Network parameters (rates, initial conditions)
│
├── initial_sim_conditions/       # Initial cell positions
│   ├── cells_2D_disk_100.csv     # 100 cells (quick tests)
│   ├── cells_2D_disk_2000.csv    # 2000 cells (production)
│   └── cells_2D_disk_*.csv       # Various sizes and geometries
│
└── physiboss_config/             # Simulation configurations
    ├── control/                  # No-drug baseline
    ├── single_drug_transient/    # Single-agent treatments
    ├── combined_drug_transient/  # Combination treatments (2D)
    └── 3D_above_drugtreatment/   # 3D timing/diffusion experiments
```

---

### Control Experiments

**File:** `config/physiboss_config/control/settings_AGSv2_CONTROL.xml`

**Purpose:** Baseline cell growth without drug treatment

**Use for:**
- Calibrating pressure-dependent growth parameters
- Establishing normalization baselines
- Validating cell mechanics

---

### Single-Drug Experiments

**Files in:** `config/physiboss_config/single_drug_transient/`

**Available configurations:**
- `settings_AGSv2_PI3K_GI50_transient_delayed.xml`
- `settings_AGSv2_MEK_GI50_transient_delayed.xml`
- `settings_AGSv2_AKT_GI50_transient_delayed.xml`

**Purpose:** Single-agent drug response calibration

**Modifiable Parameters (for custom experiments):**
```xml
<!-- Drug X (e.g., PI3K inhibitor) -->
<drug_X_target>PI3K</drug_X_target>
<drug_X_max_concentration_for_Hill>0.7</drug_X_max_concentration_for_Hill>  <!-- μM -->
<drug_X_Hill_max_effect>1.0</drug_X_Hill_max_effect>
<drug_X_Hill_EC50>0.35</drug_X_Hill_EC50>  <!-- EC50 from experimental fit -->
<drug_X_Hill_coefficient>2.5</drug_X_Hill_coefficient>
```

---

### Combination Drug Experiments

**Files in:** `config/physiboss_config/combined_drug_transient/`

**Available configurations:**
- `settings_AGSv2_SYN_PI3K_MEK_halfGI50_transient.xml`
- `settings_AGSv2_SYN_AKT_MEK_halfGI50_transient.xml`

**Purpose:** Drug combination synergy validation

**Example (PI3K + MEK):**
```xml
<drug_X_target>PI3K</drug_X_target>
<drug_X_max_concentration_for_Hill>0.35</drug_X_max_concentration_for_Hill>  <!-- 0.5 × GI50 -->

<drug_Y_target>MEK</drug_Y_target>
<drug_Y_max_concentration_for_Hill>17.5</drug_Y_max_concentration_for_Hill>  <!-- 0.5 × GI50 -->
```

---

### 3D Drug Timing Experiments

**Files in:** `config/physiboss_config/3D_above_drugtreatment/`

**Key configurations:**

1. **No-drug control:**
   - `settings_AGSv2_3D_nodrug.xml`

2. **Consensus parameter sweeps:**
   - `settings_AGSv2_3D_SYN_PI3K_MEK_consensus_top1p.xml`
   - `settings_AGSv2_3D_SYN_AKT_MEK_consensus_top1p.xml`

3. **Single-drug in synergy space:**
   - `settings_AGSv2_3D_SYN_PI3K_MEK_drugfromabove_top1p_average_singledrug.xml`
   - `settings_AGSv2_3D_SYN_AKT_MEK_drugfromabove_top1p_average_singledrug.xml`

**Purpose:** Explore timing and diffusion effects on synergy

**Key Parameters for Timing Experiments:**
```xml
<!-- Drug diffusion coefficients (μm²/min) -->
<drug_X_diffusion_coefficient>600</drug_X_diffusion_coefficient>
<drug_Y_diffusion_coefficient>600</drug_Y_diffusion_coefficient>

<!-- Drug administration timing -->
<drug_X_pulse_delay>0</drug_X_pulse_delay>         <!-- Drug X starts at t=1280 min -->
<drug_Y_pulse_delay>120</drug_Y_pulse_delay>       <!-- Drug Y starts 120 min later -->
```

**Parameter Ranges for Sweeps:**
- **Diffusion coefficient**: 6 to 6000 μm²/min (log-scale)
- **Timing offset (Δt)**: -120 to +120 minutes
  - Negative: Drug Y first
  - Zero: Simultaneous
  - Positive: Drug X first

---

## HPC Deployment

For large-scale parameter sweeps and evolutionary optimization, example workflows and batch submission scripts are available in the `emews_data/` directory. See [`emews_data/README.md`](../emews_data/README.md) for detailed documentation on Swift/T integration and automated parameter exploration using the EMEWS framework.

---

## Model Components

### Boolean Network

**Location:** `config/boolean_network/`

**Files:**
- **`AGS_all_nodes_real.bnd`**: Network topology and logic rules
- **`AGS_all_nodes_real.cfg`**: Node parameters (rates, initial states)

**Network Structure:**
- Multiple nodes representing key signaling proteins
- **Drug targets**: PI3K, AKT, MEK
- **Pro-survival nodes**: cMYC, RSK, TCF
- **Anti-survival nodes**: FOXO, Caspase8, Caspase9

**Node States:**
- Continuous values [0, 1] representing protein activation levels
- Updated stochastically using MaBoSS at each timestep
- Influence cell fate decisions (proliferation vs. apoptosis)

---

### Custom C++ Modules

**Location:** `custom_modules/`

#### `boolean_model_interface.cpp/h`
- Interface between PhysiCell agents and MaBoSS Boolean network
- Handles node state updates and drug effects
- Implements Hill equation drug-target transfer functions

#### `drug_transport_model.cpp/h`
- Drug diffusion and decay in the microenvironment
- Simple diffusion PDE solver
- Drug uptake by cells (mass-action kinetics)

#### `custom.cpp/h`
- Cell phenotype definitions (proliferative, quiescent, apoptotic)
- Cell fate decision logic based on Boolean network outputs
- Pressure-dependent growth rate modulation
- Apoptosis and necrosis mechanics

#### `submodel_data_structures.cpp/h`
- Data structures for drug parameters
- Hill function parameter storage
- Timing and pulse configurations

---

### XML Configuration Structure

All XML files follow this general structure:

```xml
<PhysiCell_settings>
  <domain> ... </domain>                  <!-- Simulation domain size -->
  <overall> ... </overall>                <!-- Global settings (duration, timestep) -->
  <parallel> ... </parallel>              <!-- OpenMP threading -->
  <save> ... </save>                      <!-- Output frequency -->
  <user_parameters> ... </user_parameters> <!-- Drug and model parameters -->
  <microenvironment_setup> ... </microenvironment_setup> <!-- Substrates (drugs) -->
  <cell_definitions> ... </cell_definitions> <!-- Cell types and behaviors -->
  <initial_conditions> ... </initial_conditions> <!-- Starting cell positions -->
</PhysiCell_settings>
```

**Key sections to modify:**

1. **Drug parameters** (in `<user_parameters>`):
   - Drug targets, concentrations, Hill parameters
   - Diffusion coefficients, timing offsets

2. **Simulation duration** (in `<overall>`):
   - `<max_time>`: Total simulation time

3. **Output settings** (in `<save>`):
   - `<full_data_interval>`: How often to save cell data
   - `<folder>`: Output directory name

---

## Troubleshooting

### Compilation Issues

**Problem:** `fatal error: libxml/parser.h: No such file or directory`

**Solution:**
```bash
sudo apt-get install libxml2-dev
```

---

**Problem:** `undefined reference to 'omp_get_thread_num'`

**Solution:**
```bash
# Install OpenMP libraries
sudo apt-get install libomp-dev
```

---

**Problem:** Custom modules not found or compilation errors in custom code

**Solution:**
```bash
# Make sure you loaded the project correctly
make physiboss-drugs-synergy-model

# Check that custom modules were copied
ls -l custom_modules/

# You should see: boolean_model_interface.cpp, custom.cpp, drug_transport_model.cpp, etc.
```

---

**Problem:** Model not appearing in `make list-projects`

**Solution:**
```bash
# Verify you edited the correct file
cat sample_projects/Makefile-default | grep "physiboss-drugs-synergy-model"

# Should show the target you added. If not, re-edit the file and ensure proper indentation (use tabs, not spaces)
```

---

### Runtime Issues

**Problem:** "Could not find boolean network file"
**Solution:**
```bash
# Verify the boolean network files exist
ls -l config/boolean_network/

# Check the XML file has correct paths:
# Should be relative to PhysiCell root directory
<bnd_file>config/boolean_network/AGS_all_nodes_real.bnd</bnd_file>
<cfg_file>config/boolean_network/AGS_all_nodes_real.cfg</cfg_file>
```

---

**Problem:** Simulation runs but produces no output files
**Solution:**
```bash
# Check write permissions in current directory
ls -ld .

# Explicitly set output folder in XML
<folder>output</folder>

# Ensure output directory exists or will be created
mkdir -p output
```

---

### Model-Specific Issues

If you encounter any problems specific to the AGS model implementation, please contact: [your_email@example.com](mailto:your_email@example.com)

---

## Additional Resources

### Related Documentation

- **EMEWS Workflow**: See [`emews_data/README.md`](../emews_data/README.md) for large-scale parameter exploration
- **Analysis Scripts**: See [`scripts/README.md`](../scripts/README.md) for data processing and visualization
- **Main Repository**: See [`README.md`](../README.md) for project overview

### External Resources

- **PhysiCell Repository**: [https://github.com/MathCancer/PhysiCell](https://github.com/MathCancer/PhysiCell)
- **PhysiCell Documentation**: [http://physicell.org/documentation](http://physicell.org/documentation)
- **PhysiBoSS Integration Tutorial**: See sample projects in `sample_projects_intracellular/boolean/`
- **MaBoSS Documentation**: [https://maboss.curie.fr](https://maboss.curie.fr)

---

## Citation

If you use this model in your research, please cite:

[Citation information to be added upon publication]

---

**Last Updated:** November 2025