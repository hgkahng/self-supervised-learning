#!/bin/bash

GPUS="0"

ENSEMBLE_NUM_ESTIMATORS=128
ENSEMBLE_DROPOUT_RATE=0.2
UNCERTAINTY_THRESHOLD=0.5

WANDB_PROJECT="UNCL(CIFAR10)"
WANDB_GROUP="uncl"
WANDB_JOB_TYPE="FN-revised"

for NUM_FALSE_NEGATIVES in 1 2 4 8 16 32 64 128 256 512 1024; do
    python run_uncl.py @txt/uncl_cifar10_pretrain.txt \
        --gpus=${GPUS} \
        --ensemble_num_estimators=${ENSEMBLE_NUM_ESTIMATORS} \
        --ensemble_dropout_rate=${ENSEMBLE_DROPOUT_RATE} \
        --uncertainty_threshold=${UNCERTAINTY_THRESHOLD} \
        --num_false_negatives=${NUM_FALSE_NEGATIVES} \
        --wandb_project=${WANDB_PROJECT} \
        --wandb_job_type=${WANDB_JOB_TYPE}
done
