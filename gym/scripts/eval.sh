#!/usr/bin/env bash

set -x

LOGFILE_DIR=$1
EVAL_COMMAND=${@:2}

for SEED in 0 1 2
do
    python3 ./evaluate.py --path ${LOGFILE_DIR}/${SEED} ${EVAL_COMMAND}
done
