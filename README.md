# AGS Gastric Adenocarcinoma Multiscale Model

## Overview

This repository contains the computational framework and models for studying drug synergy in gastric adenocarcinoma (AGS) through multiscale agent-based simulations.

## What This Repository Contains

- **`model/`**: PhysiBoSS model files for AGS simulations
- **`scripts/`**: Python scripts for data processing and analysis
- **`emews_data/`**: EMEWS parameter exploration configuration files
  - **`JSON/deap/`**: DEAP evolutionary algorithm configurations for model calibration
  - **`JSON/sweep/`**: Parameter sweep configurations for systematic exploration
  - **`JSON/sweep/sweep_txt/`**: Pre-generated parameter sets and optimization results
- **Documentation**: Model specifications and validation details

## Model Architecture

The model integrates intracellular Boolean signaling networks with population-level cellular dynamics to understand how drug combinations reshape temporal signaling profiles and drive transitions from recoverable states to stable apoptotic attractors.

## Getting Started

1. Install PhysiBoSS v2.2.0
2. Clone this repository
3. Check the `model/` and `scripts/` directories for specific usage instructions
4. For parameter exploration and model calibration, use the configuration files in `emews_data/` with the EMEWS framework


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Status

This repository is under active development. Documentation and tutorials will be added as the project progresses.

---
