TASK_NAME=$1
TRAIN_COMMAND=${@:2}

python3 ./run_dt_atari.py \
        suffix=$TASK_NAME $TRAIN_COMMAND
