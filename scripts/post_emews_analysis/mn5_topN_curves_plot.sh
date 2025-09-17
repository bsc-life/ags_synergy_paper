#!/bin/bash
#SBATCH --job-name=topN_curves_plot    # Job name
#SBATCH --output=topN_curves_plot.out  # Standard output and error log
#SBATCH --error=topN_curves_plot.err   # Error log
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=10             # Number of CPU cores per task
#SBATCH --time=02:00:00                # Time limit hrs:min:sec
#SBATCH --account=bsc08
#SBATCH --qos=gp_bscls

# Load necessary modules
module load python  # Load the Python module (adjust version as needed)

# Activate your virtual environment if needed
# source /path/to/your/venv/bin/activate

# Run your Python script
srun python scripts/post_emews_analysis/topN_curves_plot.py
