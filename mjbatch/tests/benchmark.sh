#!/bin/bash

WIDTH=1024
HEIGHT=1024
STEPS=500
COLD_START=20

ENVS=(8)

for NUM in "${ENVS[@]}"
do
    echo "--------------------------------------------------------"
    echo "Running async benchmark | ENVS: $NUM | RES: ${WIDTH}x${HEIGHT}"
    echo "--------------------------------------------------------"

    python test_readback.py \
        --num_envs "$NUM" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --steps "$STEPS" \
        --cold_start "$COLD_START" \
        --plot

done

echo "All benchmarks completed."