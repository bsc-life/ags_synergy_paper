#!/bin/bash
#SBATCH --job-name=feature_analysis
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --qos=gp_bscls
#SBATCH --account=bsc08

srun python scripts/post_emews_analysis/feature_analysis.py