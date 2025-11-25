# Multiscale Modeling of Schedule-Dependent Drug Synergy in Gastric Adenocarcinoma

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17710633.svg)](https://doi.org/10.5281/zenodo.17710633)

## Project Overview

Therapeutic synergy in cancer is an emergent phenomenon, arising from a complex interplay between molecular drug action, intracellular signaling, and tissue-level transport dynamics. Understanding and controlling these interactions requires a multiscale perspective that traditional experimental and computational models often lack. This repository contains a multiscale computational framework that directly integrates these scales to mechanistically deconstruct and predict schedule-dependent synergy in gastric adenocarcinoma (AGS). Our PhysiBoSS-based agent-based model combines Boolean signaling networks capturing PI3K/AKT/MEK pathway dynamics and ERK-IRS1 negative feedback (molecular scale), individual cell fate decisions driven by intracellular signaling (cellular scale), and population-level dynamics influenced by drug pharmacokinetics, diffusion, and administration timing (tissue scale).

Our multiscale model was rigorously validated by demonstrating that, when calibrated only on AGS cell line growth curves from single-agent treatments, it could accurately predict the emergent population-level outcomes of drug combinations without any combination-specific training. This predictive success validated the model's underlying molecular and cellular logic, establishing it as a reliable *in silico* laboratory for generating testable hypotheses. Our cross-scale analysis revealed that the efficacy of combination therapy is governed by a cascade of events that flow across scales: population-level drug pharmacokinetics dictate the sequence of molecular target engagement within individual cells, this sequence toggles the critical ERK-IRS1 negative feedback loop, which ultimately determines the collective cell-fate decision. For the PI3K–MEK combination in the AGS system, our simulations predict that inhibiting the PI3K/AKT axis before MEK disables a pro-survival rebound and locks cells into an apoptotic state. More broadly, this work demonstrates that a calibrated multiscale *in silico* framework can generate actionable, translationally relevant hypotheses by simulating complex spatiotemporal experiments that single-scale approaches, such as Boolean models alone, cannot capture.

<p align="center">
  <img src="images/fig_modeldiagram_v2.png" alt="Multiscale Model Architecture" width="900">
</p>

**Figure**: Multiscale model architecture integrating molecular, cellular, and tissue scales. **(A)** Simulation domain showing drug diffusion and cellular agents with integrated Boolean signaling networks governing cell fate decisions (proliferation vs. apoptosis). **(B)** Pressure-dependent growth rate modulation and 3D visualization of evolving cell populations. **(C)** Drug-target interaction kinetics defining the molecular-scale drug effects that propagate through the system.

---

## Repository Structure

This repository is organized into three main components, each with detailed documentation:

### [`scripts/`](scripts/README.md) - Analysis and Visualization Pipeline

Comprehensive Python scripts for the complete analysis workflow, from experimental data preprocessing to publication-quality figure generation.

**Key Components:**
- **Experimental Data Processing**: Hill equation fitting for dose-response curves, normalization and interpolation of growth curves
- **Simulation Summarization**: Real-time EMEWS instance analysis and fitness metric extraction
- **Post-EMEWS Analysis**: 
  - DEAP optimization run aggregation and convergence analysis
  - Machine learning feature importance (SHAP, Random Forest, permutation importance)
  - Statistical comparison and consensus parameter identification for synergy experiments
  - Top parameter set visualization and mechanistic cross-scale dynamics
- **3D Drug Timing Analysis**: Representative selection, efficacy grids, statistical validation, and multi-scale dynamics for diffusion coefficient and administration timing experiments
- **Utilities**: Parameter sweep file generation (5 sampling modes), ParaView VTK conversion, quick result summarization

→ **See [`scripts/README.md`](scripts/README.md) for complete documentation with 20+ annotated scripts**

---

### [`ags_physiboss_model/`](ags_physiboss_model/README.md) - PhysiBoSS Model Setup and Configuration

Model source files and configuration templates for AGS gastric adenocarcinoma simulations using PhysiCell with integrated MaBoSS Boolean network.

**Contents:**
- **Model source files**: Custom modules, Makefile, and main executable for integration into PhysiCell framework
- **Configuration templates**: XML files for control experiments, single-drug calibration, combination therapy, and 3D drug timing experiments
- **Boolean network**: Signaling network capturing PI3K/AKT/MEK pathway dynamics with drug-target interactions
- **Initial conditions**: CSV files defining 2D disk and 3D spheroid cell configurations

The model integrates intracellular Boolean signaling networks with population-level cellular dynamics to understand how drug combinations reshape temporal signaling profiles and drive transitions from recoverable states to stable apoptotic attractors.

→ **See [`ags_physiboss_model/README.md`](ags_physiboss_model/README.md) for step-by-step installation, compilation instructions, and configuration guide**

---

### [`emews_data/`](emews_data/README.md) - EMEWS Workflow Configuration

Complete EMEWS (Extreme-scale Model Exploration with Swift) framework for large-scale parameter space exploration on HPC systems.

**Key Components:**

#### **Workflow Orchestration:**
- **Swift/T workflows**: `swift_run_sweep.swift` (parameter sweeps), `swift_run_eqpy.swift` (evolutionary optimization)
- **Bash orchestration**: `sweep_battery.sh` (dose-response and synergy sweeps), `eqpy_battery.sh` (CMA-ES and GA calibrations)
- **SLURM integration**: HPC job submission scripts for MareNostrum 5

#### **Parameter Space Exploration:**
- **Method 1 - Parameter Sweeps**: Systematic exploration of consensus parameter ranges derived from single-drug calibrations
  - Dose-response experiments (PI3K, MEK, AKT inhibitors)
  - Synergy validation sweeps (PI3Ki-MEKi, AKTi-MEKi combinations)
  - 3D drug timing and diffusion experiments (structured hybrid sampling)
- **Method 2 - Evolutionary Optimization**: Adaptive search using DEAP (Distributed Evolutionary Algorithms in Python)
  - **CMA-ES**: Covariance Matrix Adaptation Evolution Strategy for continuous optimization
  - **Genetic Algorithms**: Population-based metaheuristic for parameter calibration
  - **EQ-Py Queue System**: Bidirectional Swift/T ↔ Python communication for stateful algorithms

#### **Configuration Files:**
- **`JSON/deap/`**: DEAP algorithm configurations (population size, generations, mutation rates, fitness metrics)
- **`JSON/sweep/`**: Parameter distribution JSON files for consensus-based synergy exploration
- **`JSON/sweep/txt_files/`**: Pre-generated parameter sets organized by experiment type

→ **See [`emews_data/README.md`](emews_data/README.md) for complete workflow documentation, sampling strategies, and HPC configuration**

---

## Getting Started

### Prerequisites

1. **PhysiCell** (latest version with integrated MaBoSS): Multiscale agent-based modeling framework
   - **Download**: Clone from [PhysiCell GitHub repository](https://github.com/MathCancer/PhysiCell)
   - **Note**: PhysiCell includes integrated PhysiBoSS support with MaBoSS for Boolean network modeling
   - **Installation**: See [`ags_physiboss_model/README.md`](ags_physiboss_model/README.md) for detailed step-by-step setup instructions

2. **Python 3.8+** with dependencies:
   - Core: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
   - ML: `scikit-learn`, `shap`
   - PhysiCell: `pcdl` (PhysiCell Data Loader)
   - Install via: `pip install numpy pandas scipy matplotlib seaborn scikit-learn shap pcdl`

3. **EMEWS Framework** (for large-scale parameter exploration on HPC systems):
   - **Swift/T v1.5+**: Implicitly parallel programming language for workflow orchestration
     - **Resources**: [EMEWS Tutorial and Documentation](https://web.cels.anl.gov/projects/emews/tutorial/)
   - **DEAP v1.3+**: Distributed Evolutionary Algorithms in Python
     - **Install**: `pip install deap`
   - **EQ-Py**: EMEWS Queues for Python (bidirectional Swift/T ↔ Python communication)
     - Included in EMEWS framework
   - **Setup**: See [`emews_data/README.md`](emews_data/README.md) for complete EMEWS workflow documentation and HPC configuration

### Quick Start

**Note:** This repository contains the model source code and configuration files, but requires proper integration with PhysiCell before simulations can be run. This is not a standalone executable.

```bash
# Clone repository
git clone <repository-url>
cd ags_synergy_paper/
```

**To run simulations:**
1. **Single simulations**: Follow the detailed setup instructions in [`ags_physiboss_model/README.md`](ags_physiboss_model/README.md) to integrate the model with PhysiCell and compile the executable
2. **Parameter exploration**: See [`emews_data/README.md`](emews_data/README.md) for EMEWS workflow setup and HPC configuration
3. **Analysis and visualization**: See [`scripts/README.md`](scripts/README.md) for the complete analysis pipeline

**For analysis of existing simulation results:**
```bash
cd scripts/
python 3_post_emews_analysis/emews_run_analysis/summarize_deap_run.py -s /path/to/experiment
```

### Detailed Documentation

Each subdirectory contains comprehensive README files:
- **[`scripts/README.md`](scripts/README.md)**: Complete analysis pipeline with 20+ annotated scripts
- **[`emews_data/README.md`](emews_data/README.md)**: EMEWS workflow, Swift/T configuration, and sampling strategies
- **[`ags_physiboss_model/README.md`](ags_physiboss_model/README.md)**: Model architecture and configuration details

---

## Data Availability

### Source Code Archive

The source code for this repository is archived on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17710633.svg)](https://doi.org/10.5281/zenodo.17710633)

**Code Repository DOI:** `10.5281/zenodo.17710633`

This archive includes the complete source code, model configurations, EMEWS workflows, and analysis scripts as of release v0.1.0. For the latest version, see the [GitHub repository](https://github.com/bsc-life/ags_synergy_paper).

### Simulation Results Dataset

The complete dataset of EMEWS simulation results (calibration experiments, parameter sweeps, and drug timing analyses) is available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17702426.svg)](https://doi.org/10.5281/zenodo.17702426)

**Dataset DOI:** `10.5281/zenodo.17702426`

The dataset includes:
- **Calibration results**: CMA-ES and Genetic Algorithm optimization outputs for PI3Ki, MEKi, and AKTi single-drug calibrations
- **Synergy sweep experiments**: Parameter space exploration for PI3K-MEK and AKT-MEK drug combinations
- **Drug timing experiments**: 2D and 3D simulations exploring administration timing and diffusion effects

For detailed information about the dataset structure and contents, see the [dataset documentation](https://doi.org/10.5281/zenodo.17702426).

---

## Citation

If you use this framework in your research, please cite:

[Citation information to be added upon publication]

**Code repository citation:**
```
Hayoun Mya, Othmane, Montagud, Arnau, Ponce de León, Miguel, & Valencia, Alfonso. (2025). 
Multiscale Modeling of Schedule-Dependent Drug Synergy in Gastric Adenocarcinoma - Source Code (v0.1.0). 
Zenodo. https://doi.org/10.5281/zenodo.17710633
```

**Dataset citation:**
```
Hayoun Mya, Othmane, Montagud, Arnau, Ponce de León, Miguel, & Valencia, Alfonso. (2025). 
EMEWS-PhysiBoSS calibration and parameter sweep data for PI3K-MEK and AKT-MEK inhibitor combinations 
in a multiscale AGS cancer cell line model (Version v1) [Data set]. 
Zenodo. https://doi.org/10.5281/zenodo.17702426
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact and Support

For questions, issues, or contributions, please open an issue on the repository or contact the authors.

---
