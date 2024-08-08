#!/bin/bash
#SBATCH --partition=luke
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00


# Activate Anaconda work environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate action_graph_env

python feature_extractor/temporal_similarity_graph.py
