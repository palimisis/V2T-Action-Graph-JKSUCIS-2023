#!/bin/bash
#SBATCH --partition=leia
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00


# Activate Anaconda work environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate action_graph_env


cd feature_extractor
ipython -c "%run clip4clip_theta_2_feature_extraction.ipynb"
