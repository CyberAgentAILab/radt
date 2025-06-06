#!/usr/bin/env bash

set -x

TASK_NAME=$1
TRAIN_COMMAND=${@:2}

python3 experiment.py --multirun task_name=${TASK_NAME} \
    seed=0,1,2 \
    ${TRAIN_COMMAND}
