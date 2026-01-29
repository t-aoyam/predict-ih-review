#!/bin/bash
# run these commands from the root (icml-2026-32900/)

# Experiment 1
# train on natural data
# 50M GPT2 (2 layers, 8 heads) with batch size of 4 and context size of 64
python -m src.training.train --tokenized_data data/cc100_1b_tokens.pkl --config_fp data/configs/matching_steps.json --n_layer 2 --n_head 8 --t_block mlp --ctx_size 64 --batch_size 4 --gacc 4 --use_wandb

# Experiment 2
# train on semi-synthetic data (see data.sh for data generation)
# 50M GPT2 (2 layers, 8 heads) with batch size of 4 and context size of 64
python -m src.training.train_synthetic --segmented_data data/v10000_c64_pa90_pb90_s1_100M_natural.jsonl --validation_data data/v9999_c64_pa90_pb90_s1_val_100K_natural.jsonl --config_fp data/configs/matching_steps_synthetic.json --n_layer 2 --n_head 8 --t_block mlp --ctx_size 64 --batch_size 4 --gacc 1 --seed 1 --use_wandb

# Experiment 3
# train on synthetic data (see data.sh for data generation)
# 50M GPT2 (2 layers, 8 heads) with batch size of 4 and context size of 64
python -m src.training.train_synthetic --segmented_data data/v10000_c64_pa10_pb30_ld_highcat_uniform_s1_train_100M.jsonl --validation_data data/v10000_c64_pa10_pb30_ld_highcat_uniform_s1_val_100K.jsonl --config_fp data/configs/matching_steps_synthetic.json --n_layer 2 --n_head 8 --t_block mlp --ctx_size 64 --batch_size 4 --gacc 1 --seed 1 --use_wandb
