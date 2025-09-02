# AGS Gastric Adenocarcinoma Multiscale Model

## Overview

This repository contains the computational framework and models for studying drug synergy in gastric adenocarcinoma (AGS) through multiscale agent-based simulations. The model integrates intracellular Boolean signaling networks with population-level cellular dynamics to understand how drug combinations reshape temporal signaling profiles and drive transitions from recoverable states to stable apoptotic attractors.

## Model Architecture

### Multiscale Framework
The model operates across multiple temporal and spatial scales:
- **Molecular scale**: Boolean network modeling of intracellular signaling pathways
- **Cellular scale**: Agent-based modeling of individual cell behavior and phenotype
- **Population scale**: 3D microenvironment with substrate diffusion and mechanical interactions

### Key Components

#### PhysiBoSS Simulations
- **Platform**: Built on [PhysiBoSS](https://github.com/PhysiBoSS/PhysiBoSS) v2.2.0
- **Domain**: 2D (600×600×50 μm) and 3D (225×225×700 μm) simulation spaces
- **Temporal resolution**: 
  - Substrate diffusion: 0.01 min
  - Mechanical updates: 0.1 min
  - Cell phenotype decisions: 6 min
  - Boolean network updates: 1-40 min (calibrated)

#### AGS Boolean Signaling Network
- **Algorithm**: MaBoSS with multivalued logic
- **Readouts**: 
  - Prosurvival: TCF, RSK, cMYC nodes
  - Antisurvival: Caspase8, Caspase9, FOXO nodes
- **Phenotype coupling**: Links Boolean states to growth and apoptosis rates

#### Drug Pharmacodynamics
- **Internalization**: Fick's law diffusion with calibrated permeability coefficients
- **Target binding**: Mass action kinetics for PI3K, MEK, and AKT inhibitors
- **Network perturbation**: Hill function conversion of drug-target complexes to Boolean node inactivation

## Repository Contents

- **Simulation models**: Complete PhysiBoSS configurations for AGS cells
- **Boolean networks**: Signaling pathway definitions and parameters
- **Calibration data**: EMEWS optimization framework parameters
- **Tutorials**: Step-by-step guides for replicating experiments
- **Documentation**: Detailed model specifications and validation

## Applications

This framework enables:
- **Single-drug response prediction**: Calibrated models for PI3K, MEK, and AKT inhibitors
- **Drug synergy analysis**: Investigation of combination therapy mechanisms
- **Temporal dynamics**: Understanding how drug timing affects therapeutic outcomes
- **3D pharmacodynamic heterogeneity**: Analysis of spatial drug distribution effects

## Getting Started

1. Install PhysiBoSS v2.2.0
2. Clone this repository
3. Follow the tutorial for initial configuration
4. Run single-drug calibrated experiments
5. Explore drug combination synergies
6. Investigate 3D pharmacodynamic scenarios

## Citation

If you use this model in your research, please cite our manuscript: [Manuscript Title, Journal, Year]

## Contributing

We welcome contributions and improvements to the model. Please open an issue or submit a pull request for any bugs, feature requests, or enhancements.

## License


## Contact

For questions about the model or repository, please contact [Your Contact Information].
