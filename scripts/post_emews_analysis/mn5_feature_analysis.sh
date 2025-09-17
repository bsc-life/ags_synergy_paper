#!/bin/bash
#SBATCH --job-name=feature_analysis    # Job name
#SBATCH --output=feature_analysis.out  # Standard output and error log
#SBATCH --error=feature_analysis.err   # Error log
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=10             # Number of CPU cores per task
#SBATCH --time=02:00:00                # Time limit hrs:min:sec
#SBATCH --account=bsc08
#SBATCH --qos=gp_bscls

# Load necessary modules
module load hdf5 python  # Load the Python module (adjust version as needed)

# Activate your virtual environment if needed
# source /path/to/your/venv/bin/activate

# Run your Python script
srun python scripts/post_emews_analysis/feature_analysis.py 