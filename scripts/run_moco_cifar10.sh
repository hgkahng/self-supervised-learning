#!/bin/bash

GPUS="1"

WANDB_PROJECT="UNCL(CIFAR10)"
WANDB_GROUP="moco"
WANDB_JOB_TYPE="NEG"

for NUM_NEGATIVES in 4096 3072 2048 1024 512; do
    python run_moco.py @txt/moco_cifar10_pretrain.txt \
        --gpus=${GPUS} \
        --num_negatives=${NUM_NEGATIVES} \
        --wandb_project=${WANDB_PROJECT} \
        --wandb_job_type=${WANDB_JOB_TYPE}
done
