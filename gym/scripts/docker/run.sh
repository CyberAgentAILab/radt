docker run --gpus all \
    -it --rm \
    -v "$(pwd)/logs:/radt_gym/logs" \
    -v "$(pwd)/data-gym:/radt_gym/data-gym" \
    radt_gym