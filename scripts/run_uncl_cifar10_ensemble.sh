#!/bin/bash

GPUS="2"

ENSEMBLE_NUM_ESTIMATORS=128
ENSEMBLE_DROPOUT_RATE=0.2
NUM_FALSE_NEGATIVES=128
UNCERTAINTY_THRESHOLD=0.5

WANDB_PROJECT="UNCL(CIFAR10)"
WANDB_GROUP="uncl"
WANDB_JOB_TYPE="ENSEMBLE"

for ENSEMBLE_NUM_ESTIMATORS in 1 8 64 128 256 512; do
    python run_uncl.py @txt/uncl_cifar10_pretrain.txt \
        --gpus=${GPUS} \
        --ensemble_num_estimators=${ENSEMBLE_NUM_ESTIMATORS} \
        --ensemble_dropout_rate=${ENSEMBLE_DROPOUT_RATE} \
        --uncertainty_threshold=${UNCERTAINTY_THRESHOLD} \
        --num_false_negatives=${NUM_FALSE_NEGATIVES} \
        --wandb_project=${WANDB_PROJECT} \
        --wandb_job_type=${WANDB_JOB_TYPE}
done
