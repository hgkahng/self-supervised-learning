#!/bin/bash

ENSEMBLE_NUM_ESTIMATORS=128
ENSEMBLE_DROPOUT_RATE=0.2
UNCERTAINTY_THRESHOLD=0.9

WANDB_PROJECT="UNCL(ImageNet)"
WANDB_GROUP="uncl"
WANDB_JOB_TYPE="fn"

for NUM_FALSE_NEGATIVES in 32; do
    python run_uncl.py @txt/uncl_imagenet_pretrain.txt \
        --ensemble_num_estimators=${ENSEMBLE_NUM_ESTIMATORS} \
        --ensemble_dropout_rate=${ENSEMBLE_DROPOUT_RATE} \
        --uncertainty_threshold=${UNCERTAINTY_THRESHOLD} \
        --num_false_negatives=${NUM_FALSE_NEGATIVES} \
        --wandb_project=${WANDB_PROJECT} \
        --wandb_job_type=${WANDB_JOB_TYPE}
done
