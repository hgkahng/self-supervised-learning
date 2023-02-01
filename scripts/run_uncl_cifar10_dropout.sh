#!/bin/bash

GPUS="3"

ENSEMBLE_NUM_ESTIMATORS=128
UNCERTAINTY_THRESHOLD=0.5
NUM_FALSE_NEGATIVES=128

WANDB_PROJECT="UNCL(CIFAR10)"
WANDB_GROUP="uncl"
WANDB_JOB_TYPE="DROPOUT"

for ENSEMBLE_DROPOUT_RATE in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
    python run_uncl.py @txt/uncl_cifar10_pretrain.txt \
        --gpus=${GPUS} \
        --ensemble_num_estimators=${ENSEMBLE_NUM_ESTIMATORS} \
        --ensemble_dropout_rate=${ENSEMBLE_DROPOUT_RATE} \
        --uncertainty_threshold=${UNCERTAINTY_THRESHOLD} \
        --num_false_negatives=${NUM_FALSE_NEGATIVES} \
        --wandb_project=${WANDB_PROJECT} \
        --wandb_job_type=${WANDB_JOB_TYPE}
done
