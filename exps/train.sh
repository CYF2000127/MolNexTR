#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=2
NODE_RANK=0

BATCH_SIZE=64
ACCUM_STEP=1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

DATESTR=$(date +"%m-%d-%H-%M")
SAVE_PATH=/
mkdir -p ${SAVE_PATH}

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    train.py \
    --data_path data \
    --train_file pubchem/train_1m.csv \
    --aux_file uspto_mol/train_680k.csv --coords_file aux_file \
    --valid_file real/USPTO.csv \
    --test_file real/CLEF.csv,real/UOB.csv,real/USPTO.csv,real/staker.csv,real/acs.csv,synthetic/indigo.csv,synthetic/chemdraw.csv \
    --vocab_file MolNexTR/vocab/vocab_chars.json \
    --formats chartok_coords,edges \
    --dynamic_indigo --augment --mol_augment \
    --include_condensed \
    --coord_bins 64 --sep_xy \
    --input_size 384 \
    --encoder convnext \
    --decoder transformer \
    --encoder_lr 4e-4 \
    --decoder_lr 4e-4 \
    --save_path $SAVE_PATH --save_mode all \
    --label_smoothing 0.1 \
    --epochs 30 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --warmup 0.02 \
    --print_freq 200 \
    --do_train --do_valid --do_test \
    --fp16 --backend gloo 2>&1

