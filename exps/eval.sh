#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=2
NODE_RANK=0
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
BATCH_SIZE=8
LOAD_PATH=checkpoints/molnextr_best.pth
SAVE_PATH=predict_output
mkdir -p ${SAVE_PATH}

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    main.py \
    --data_path data \
    --test_file real/acs.csv\
    --vocab_file MolNexTR/vocab/vocab_chars.json \
    --formats chartok_coords,edges \
    --coord_bins 64 --sep_xy \
    --input_size 384 \
    --load_path $LOAD_PATH \
    --save_path $SAVE_PATH \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE)) \
    --use_checkpoint \
    --print_freq 200 \
    --do_test \
    --fp16 --backend gloo 2>&1
