docker run --gpus all \
    -it --rm \
    -v "$(pwd)/logs:/radt_atari/logs" \
    -v "$(pwd)/dqn_replay:/radt_atari/dqn_replay" --shm-size=1g \
    radt_atari