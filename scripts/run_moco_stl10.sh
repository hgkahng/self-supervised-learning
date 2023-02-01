#!/bin/bash

WANDB_PROJECT="UNCL(STL10)"
WANDB_GROUP="moco"
WANDB_JOB_TYPE="baseline"

for NUM_NEGATIVES in 4096 8192; do
    python run_moco.py @txt/moco_stl10_pretrain.txt \
        --num_negatives=${NUM_NEGATIVES} \
        --wandb_project=${WANDB_PROJECT} \
        --wandb_job_type=${WANDB_JOB_TYPE}
done
