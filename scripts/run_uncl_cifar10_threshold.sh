#!/bin/bash

GPUS="1"

ENSEMBLE_NUM_ESTIMATORS=128
ENSEMBLE_DROPOUT_RATE=0.2
NUM_FALSE_NEGATIVES=128

WANDB_PROJECT="UNCL(CIFAR10)"
WANDB_GROUP="uncl"
WANDB_JOB_TYPE="THRESHOLD"

for UNCERTAINTY_THRESHOLD in 0.6 0.7 0.8 0.9; do
    python run_uncl.py @txt/uncl_cifar10_pretrain.txt \
        --gpus=${GPUS} \
        --ensemble_num_estimators=${ENSEMBLE_NUM_ESTIMATORS} \
        --ensemble_dropout_rate=${ENSEMBLE_DROPOUT_RATE} \
        --uncertainty_threshold=${UNCERTAINTY_THRESHOLD} \
        --num_false_negatives=${NUM_FALSE_NEGATIVES} \
        --wandb_project=${WANDB_PROJECT} \
        --wandb_job_type=${WANDB_JOB_TYPE}
done
