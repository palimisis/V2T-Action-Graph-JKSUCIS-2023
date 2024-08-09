#!/bin/bash
#SBATCH --partition=luke
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=200G

# Activate Anaconda work environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate action_graph_env

python feature_extractor/action_spatio_temporal_graph_feature_extractor.py
