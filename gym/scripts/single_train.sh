#!/usr/bin/env bash

set -x

TASK_NAME=$1
TRAIN_COMMAND=${@:2}

python3 ${RUNFILE_DIR}/experiment.py task_name=${TASK_NAME} \
    ${TRAIN_COMMAND}
