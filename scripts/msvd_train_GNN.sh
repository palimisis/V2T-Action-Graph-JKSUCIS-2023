#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

# Activate Anaconda work environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate action_graph_env

# Setup
DATATYPE=msvd
N_GPU=1
N_THREAD=8

# PATH to files
DATA_PATH=./dataset/MSVD
CKPT_ROOT=./ckpts
INIT_MODEL_PATH=./weight/univl.pretrained.bin
FEATURES_PATH=./extracted_features/msvd/GridNodeFeatures.hdf5 # <CLIP FEATURE FOR Theta_2> # Change into the features you extracted from CLIP4Clip
DATA_GEOMETRIC_PATH=./extracted_features/msvd/FinalGraph.pickle # <GRAPH FEATURE for Theta_1># Change into the path to the graph-based features (can be grid or object-based features)
NODE_FEATURES=geometric # please only use geometric for now
# Params
LEARNING_RATE=1e-4

#for lr in "${LEARNING_RATE[@]}"
#do
python -m torch.distributed.launch --nproc_per_node=${N_GPU} \
../main_task_caption_GNN.py --do_train --num_thread_reader=${N_THREAD} \
--epochs=50 --batch_size=128 --n_display=50 --gradient_accumulation_steps 2 \
--data_path ${DATA_PATH} --features_path ${FEATURES_PATH} \
--output_dir ${CKPT_ROOT}/${DATATYPE}_lr${LEARNING_RATE}_gnn \
--bert_model bert-base-uncased --do_lower_case \
--lr ${lr} --max_words 48 --max_frames 20 --batch_size_val 16 \
--visual_num_hidden_layers 2 --decoder_num_hidden_layers 2 \
--datatype ${DATATYPE} --init_model ${INIT_MODEL_PATH} \
--data_geometric_path ${DATA_GEOMETRIC_PATH} \
--node_features ${NODE_FEATURES} --node_feat_dim 512 --d_model 512 --video_dim 512 --edge_dim 1024 \
--tradeoff_theta_2 4 --tradeoff_distill 1 --gnn_model_type transformer \
--custom_input_dim 1812
#done
