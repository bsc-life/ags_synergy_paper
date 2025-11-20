# EMEWS Parameter Exploration Data

## Table of Contents

1. [Overview of EMEWS](#overview-of-emews)
   - [Origin and Key Publications](#origin-and-key-publications)
   - [Swift/T Programming Language](#swiftt-programming-language)
2. [Using PhysiBoSS with Swift/T for Parameter Exploration](#using-physiboss-with-swiftt-for-parameter-exploration)
   - [Method 1: Parameter Sweep](#method-1-parameter-sweep)
     - [Parameter Sweep Workflow](#parameter-sweep-workflow)
     - [Workflow Execution](#workflow-execution)
     - [Synergy Exploration with Consensus Parameters](#synergy-exploration-with-consensus-parameters)
   - [Method 2: Evolutionary Optimization with DEAP](#method-2-evolutionary-optimization-with-deap)
     - [Available Optimization Algorithms](#available-optimization-algorithms)
     - [DEAP Configuration Files](#deap-configuration-files)
     - [DEAP Workflow Execution](#deap-workflow-execution)
   - [The EQ-Py Queue System](#the-eq-py-queue-system-bridging-swiftt-and-python)
3. [Additional Resources](#additional-resources)
4. [Support](#support)

---

## Overview of EMEWS

The Extreme-scale Model Exploration with Swift (EMEWS) framework is a powerful tool designed to facilitate large-scale analyses of computational models, including calibration, optimization, and parameter space exploration. EMEWS leverages high-performance computing (HPC) resources to enable efficient model exploration across heterogeneous computing environments.

### Origin and Key Publications

EMEWS was developed by researchers at Argonne National Laboratory and collaborating institutions. Key publications include:

- **Collier, N., et al.** "Distributed Model Exploration with EMEWS." This paper discusses significant advancements in the EMEWS framework, including improved accessibility through binary installation, a new decoupled architecture (EMEWS DB), and enhanced project creation capabilities. The paper demonstrates how EMEWS DB can connect Python-based Bayesian optimization algorithms to worker pools operating both locally and on remote compute resources.

- **Ozik, J., et al.** Various publications on using EMEWS for agent-based modeling and large-scale simulations across multiple domains including epidemiology, social science, and systems biology.

### Swift/T Programming Language

EMEWS workflows are written in Swift/T, an implicitly parallel programming language specifically designed for composing external functions and command-line executables into massively parallel applications.

#### Key Features of Swift/T:

- **Implicit Parallelism**: Swift/T automatically manages parallel execution of tasks based on data dependencies, eliminating the need for explicit parallel programming constructs.

- **High Scalability**: Designed to handle trillions of tasks efficiently on large-scale HPC systems. Swift/T operates as an MPI-based workflow system capable of executing vast numbers of tasks at high rates.

- **Language Interoperability**: Supports integration with languages like Python, R, and Tcl through embedded scripting interpreters, facilitating the use of existing scripts and libraries in parallel workflows.

- **Data Flow Model**: Swift/T uses a dataflow-driven execution model where tasks execute as soon as their input data becomes available, maximizing parallelism and resource utilization.

- **Concise Syntax**: Provides a high-level language for describing data dependencies and iterations without requiring deep knowledge of parallel programming paradigms.

## Using PhysiBoSS with Swift/T for Parameter Exploration

This project uses the EMEWS framework and Swift/T to perform large-scale parameter space explorations with a compiled PhysiBoSS model. We employ two distinct methodologies for exploring the parameter space, each suited to different optimization objectives.

### Method 1: Parameter Sweep

The parameter sweep approach systematically explores predefined parameter ranges by evaluating the model across a grid or set of specific parameter combinations. This method is particularly useful for:

- **Dose-response curve generation**: Testing specific drug concentrations to characterize the relationship between drug exposure and cellular response
- **Sensitivity analysis**: Understanding how individual parameters affect model outputs
- **Validation studies**: Comparing simulation results against experimental data across a range of conditions

#### Parameter Sweep Workflow

Parameter sweeps require a two-step configuration process:

1. **Define parameter distributions**: JSON files in `JSON/sweep/` specify the parameters to vary along with their minimum and maximum values. These files define the parameter space but do not contain actual parameter values.

2. **Generate simulation parameters**: The `generate_json_sweep.py` utility script (located in `scripts/utils/`) converts the JSON configuration into a `.txt` file where each line represents a complete parameter set for a single simulation. This script supports multiple sampling strategies:
   - `uniform`: Random uniform sampling within specified bounds
   - `grid`: Systematic grid-based sampling with evenly spaced values
   - `logscale`: Logarithmic spacing for parameters spanning multiple orders of magnitude
   - `hybrid`: Mixed approach using log-uniform sampling for diffusion and timing parameters, uniform for others
   - `structured_hybrid`: Combines grid sampling for strategic parameters with random sampling for consensus parameters

Once the `.txt` file is generated, the Swift/T workflow script `swift_run_sweep.swift` orchestrates the parallel execution of simulations for each parameter combination, automatically comparing results to experimental data and computing fitness metrics.

#### Workflow Execution

> **⚠️ HPC System Configuration Warning**
>
> The workflow submission scripts (`run_sweep_ags.sh` and `run_eqpy_ags.sh`) contain system-specific configurations for MareNostrum 5 (MN5) at Barcelona Supercomputing Center. These scripts **must be adapted** to your HPC environment before use. Key modifications required:
>
> - **Job scheduler directives**: Update SLURM-specific settings for your scheduler (PBS, LSF, etc.)
> - **Module loading**: Replace `module load` commands with software modules available on your system
> - **Resource allocation**: Adjust node counts, cores per node, and memory requirements
> - **Queue/partition names**: Update queue and project names to match your system
> - **File paths**: Modify paths to executables, libraries, and Python/R installations
> - **Swift/T installation**: Ensure Swift/T is compiled and configured for your system

Parameter sweeps are executed through a hierarchical bash script workflow that manages experiment configuration, resource allocation, and Swift/T invocation:

**1. Master Experiment Script** (`sweep_battery.sh`): This top-level script defines all parameter sweep experiments. Each experiment is executed by calling the workflow submission script with five arguments:

```bash
bash run_sweep_ags.sh EXPERIMENT_ID PARAMS_TXT_FILE SETTINGS_XML DRUG_NAME METRIC
```

Where:
- `EXPERIMENT_ID`: Unique identifier for the experiment (automatically timestamped)
- `PARAMS_TXT_FILE`: Path to the `.txt` file containing parameter sets (one JSON object per line)
- `SETTINGS_XML`: Path to the PhysiBoSS configuration XML template
- `DRUG_NAME`: Drug identifier for experimental data comparison (e.g., PI3K, MEK, AKT, PI3K_MEK, AKT_MEK)
- `METRIC`: Fitness metric for evaluation (e.g., RMSE_SK_POSTDRUG, FINAL_NUMBER_OF_ALIVE_CELLS)

**2. Workflow Submission Script** (`run_sweep_ags.sh`): This script prepares the execution environment and submits the Swift/T workflow to the HPC scheduler. Key operations include:

- Setting up the experiment directory structure under `experiments/EXPERIMENT_ID/`
- Copying the PhysiBoSS executable, configuration files, and parameter file to the experiment directory
- Configuring HPC resources (number of nodes, cores per node, wall time, job queue)
- Loading required software modules (Swift/T, Python, R, compilers)
- Setting environment variables (`EMEWS_PROJECT_ROOT`, `TURBINE_OUTPUT`, `PYTHONPATH`)
- Constructing command-line arguments for the Swift/T script
- Submitting the job to SLURM with: `swift-t -n PROCS -m slurm swift_run_sweep.swift [ARGS]`

**3. Swift/T Workflow** (`swift_run_sweep.swift`): Receives parameters via command-line arguments (`argv()` function in Swift) and coordinates parallel simulation execution. The workflow:

- Reads the parameter file line by line
- For each parameter set, launches multiple replicate simulations (default: 5 replicates)
- Executes simulations in parallel across allocated compute resources
- Calls Python analysis scripts to extract metrics and compare against experimental data
- Cleans up intermediate output files to manage storage

**HPC Resource Configuration**: The workflow is configured for MareNostrum 5 (MN5) with:
- 112 total MPI processes (8 nodes × 14 processes per node)
- 8 cores per process (112 cores per node / 14 processes = 8 cores/process)
- 24-hour wall time limit
- SLURM job scheduler

These resource settings can be adjusted in `run_sweep_ags.sh` by modifying the `PROCS`, `PPN`, and `WALLTIME` variables to match your HPC environment.

#### Synergy Exploration with Consensus Parameters

A key application of the parameter sweep methodology involves investigating drug combination synergies using consensus parameter ranges derived from single-drug calibrations. The sweep configurations `sweep_consensus_pi3k_mek_top1p.json` and `sweep_consensus_akt_mek_top1p.json` exemplify this approach for exploring PI3K-MEK and AKT-MEK combination therapies, respectively.

These configurations contain both drug-specific and overlapping parameters:

**Drug-Specific Parameters**: Each drug (X and Y) has distinct pharmacokinetic properties including:
- Membrane permeability rates
- Binding (kon) and unbinding (koff) kinetics
- Diffusion coefficients governing spatial drug distribution
- Pulse periods determining temporal dosing schedules

**Overlapping Parameters**: Shared cellular response mechanisms including:
- Hill coefficients and half-maximal concentrations for growth inhibition and apoptosis induction
- Boolean network node weights linking intracellular signaling states to phenotypes
- Response rates and temporal scaling factors
- Maximum apoptosis rate

The parameter ranges in these files represent consensus distributions derived from the top-performing parameter sets identified during single-drug calibration experiments. By sampling from these consensus ranges, the sweep explores whether parameter combinations that successfully reproduce individual drug responses can also capture the emergent population-level dynamics observed in drug combination experiments. This approach tests the model's ability to predict synergistic interactions without requiring additional calibration specific to the combination therapy setting. These datasets form the basis for the synergy analysis presented in the manuscript.

### Method 2: Evolutionary Optimization with DEAP

For more complex parameter estimation problems involving high-dimensional parameter spaces, we integrate the DEAP (Distributed Evolutionary Algorithms in Python) package through the EQ-Py communication framework. This approach employs sophisticated optimization algorithms to efficiently search the parameter space and identify parameter sets that best match experimental observations.

#### Available Optimization Algorithms:

**Genetic Algorithms (GA)**: A population-based metaheuristic inspired by natural selection. The GA maintains a population of candidate solutions that evolve over generations through selection, crossover, and mutation operators. This approach is effective for exploring complex, multimodal fitness landscapes and can escape local optima through its stochastic search strategy.

**CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**: A state-of-the-art evolutionary strategy particularly effective for continuous parameter optimization. CMA-ES adaptively updates a full covariance matrix to guide the search direction, making it highly efficient at navigating high-dimensional parameter spaces with complex correlations between parameters.

#### DEAP Configuration Files

DEAP configurations are specified in JSON files in `JSON/deap/`, where each file defines the parameters to optimize, their bounds, and algorithm-specific settings. The configuration files are organized according to different calibration objectives:

**Control Curve Fitting** (`deap_5p_control.json`): This configuration focuses on calibrating the baseline tumor growth dynamics in the absence of drug treatment. It optimizes 5 fundamental parameters related to cell proliferation and mechanical constraints:
- Pressure-related parameters (pressure threshold and Hill coefficient) that govern contact inhibition
- Basal growth rate determining the intrinsic proliferation rate
- Initial tumor geometry (tumor radius and cell spacing)

This simplified parameter set establishes the foundational growth dynamics that must be accurate before introducing drug-specific parameters.

**Single-Drug Experiment Calibration** (`deap_18p_single_drug_exp_v2.json`): This configuration encompasses a comprehensive 18-parameter space for fitting individual drug response curves. It includes:
- Pharmacokinetic parameters (drug permeability, binding/unbinding rates)
- Dose-response relationships (Hill coefficients and half-maximal concentrations for both growth inhibition and apoptosis induction)
- Boolean network node weights linking intracellular signaling states to cellular phenotypes (pro-growth nodes: cMYC, TCF, RSK; pro-apoptotic nodes: FOXO, Caspase8, Caspase9)
- Temporal dynamics (response rates, intracellular simulation time step, and scaling factors)
- Maximal apoptosis rate governing the upper limit of drug-induced cell death

This detailed parameterization captures the complex mechanistic relationship between drug exposure, intracellular signaling network dynamics, and emergent cellular behavior.

The workflow uses EQ-Py to establish bidirectional communication between the Swift/T workflow manager and Python-based DEAP algorithms, enabling adaptive parameter selection based on simulation results. The execution is managed through the bash script `run_eqpy_ags.sh`, which coordinates the distributed optimization process across HPC resources.

#### DEAP Workflow Execution

> **Note**: The same HPC system configuration warning applies here. See the warning in the Parameter Sweep section above regarding system-specific adaptations required for `run_eqpy_ags.sh`.

Evolutionary optimization experiments follow a similar hierarchical structure but with additional complexity for adaptive algorithm communication:

**1. Master Optimization Script** (`eqpy_battery.sh`): Defines all DEAP-based calibration experiments. Each optimization run is executed with seven arguments:

```bash
bash run_eqpy_ags.sh EXPERIMENT_ID EA_PARAMS_FILE STRATEGY FIT DRUG SWIFT_FILE SETTINGS_XML
```

Where:
- `EXPERIMENT_ID`: Unique identifier for the optimization run
- `EA_PARAMS_FILE`: Path to JSON file defining parameters to optimize (e.g., `deap_18p_single_drug_exp_v2.json`)
- `STRATEGY`: Optimization algorithm (e.g., GA for genetic algorithm, CMA for CMA-ES)
- `FIT`: Fitness function identifier for evaluation
- `DRUG`: Drug identifier for experimental data comparison
- `SWIFT_FILE`: Swift/T workflow script (typically `swift_run_eqpy.swift`)
- `SETTINGS_XML`: Path to PhysiBoSS configuration XML template

**2. Optimization Submission Script** (`run_eqpy_ags.sh`): Prepares the EQ-Py enabled execution environment:

- Configures resident work tasks for persistent Python processes (`TURBINE_RESIDENT_WORK_WORKERS=1`)
- Sets up EQ-Py communication between Swift/T and Python optimizer
- Defines algorithm hyperparameters (population size, generations, mutation rates)
- Allocates typically more resources than parameter sweeps (e.g., 112+ processes for population-based algorithms)
- Submits job with EQ-Py module path: `swift-t -I $EQPY -r $EQPY swift_run_eqpy.swift [ARGS]`

**3. Swift/T with EQ-Py Integration** (`swift_run_eqpy.swift`): Implements the adaptive optimization loop:

- Initializes EQ-Py communication queues using `EQPy_init_package()`
- In each generation:
  - Retrieves candidate parameter sets from Python optimizer via `EQPy_get()`
  - Distributes simulations across compute resources in parallel
  - Collects fitness metrics and returns to optimizer via `EQPy_put()`
- Continues until convergence or maximum generations reached

**4. Python DEAP Algorithm**: Runs in a resident Swift/T task and implements the evolutionary strategy:

- Maintains population of candidate solutions
- Applies genetic operators (selection, crossover, mutation for GA; covariance adaptation for CMA-ES)
- Receives fitness evaluations from Swift/T via input queue
- Generates next generation and places candidates in output queue

This architecture enables sophisticated adaptive exploration strategies that would be impossible with static parameter sweeps, while leveraging Swift/T's parallel task management for efficient HPC utilization.

---

#### The EQ-Py Queue System: Bridging Swift/T and Python

A distinctive and innovative feature of EMEWS is the EQ-Py (EMEWS Queues for Python) communication framework, which enables seamless integration between Swift/T's distributed workflow management and Python-based optimization algorithms. EQ-Py implements a thread-safe, bidirectional queue system that allows the two environments to exchange information asynchronously during runtime.

**Architecture**: EQ-Py operates through two main queues:

- **Output Queue** (`output_q`): The Python algorithm places candidate parameter sets into this queue, which Swift/T retrieves to launch simulation tasks across distributed computing resources.

- **Input Queue** (`input_q`): Swift/T returns simulation results and fitness metrics through this queue, which the Python algorithm consumes to inform the next iteration of the optimization process.

**Key Advantages**:

- **Language Interoperability**: Combines Swift/T's superior parallel task management and HPC integration with Python's rich ecosystem of scientific computing and machine learning libraries.

- **Asynchronous Communication**: The queue-based architecture allows the optimizer and simulator to operate independently, maximizing computational efficiency by preventing idle time.

- **Adaptive Exploration**: Unlike static parameter sweeps, EQ-Py enables algorithms to dynamically adjust search strategies based on accumulated results, focusing computational resources on promising regions of parameter space.

- **Resident Task Model**: EQ-Py leverages Swift/T's resident work tasks to maintain persistent Python processes throughout the workflow, avoiding the overhead of repeated initialization and enabling stateful optimization algorithms.

This queue-based communication pattern is what distinguishes adaptive EMEWS workflows from traditional parameter sweeps, enabling sophisticated optimization strategies that would be impractical with static approaches.

---

### Additional Resources

- [Swift/T Documentation](https://swift-lang.github.io/swift-t/)
- [EMEWS Tutorial](http://www.mcs.anl.gov/~emews/tutorial)
- [PhysiBoSS Documentation](http://physiboss.github.io/)
- [EQ-Py for Python Integration](https://github.com/emews/EQ-Py)

### Support

For questions about this workflow or issues with EMEWS/Swift/T integration, consult the EMEWS documentation or reach out to the EMEWS community through their GitHub repository.

