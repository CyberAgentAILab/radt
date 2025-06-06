LOGFILE_DIR=$1
EVAL_COMMAND=${@:2}

for seed in 123 231 312
do
    python3 ./eval_dt_atari.py --path ${LOGFILE_DIR}/${seed} ${EVAL_COMMAND}
done
