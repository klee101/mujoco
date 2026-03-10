#!/bin/bash

WIDTH=1024
HEIGHT=1024
STEPS=50
COLD_START=5

ENVS=(8)

LOG_DIR="debug_logs"
mkdir -p "$LOG_DIR"

for NUM in "${ENVS[@]}"
do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    DEBUG_LOG="${LOG_DIR}/debug_env${NUM}_${WIDTH}x${HEIGHT}_${TIMESTAMP}.log"

    echo "--------------------------------------------------------"
    echo "Running async benchmark | ENVS: $NUM | RES: ${WIDTH}x${HEIGHT}"
    echo "Output -> $DEBUG_LOG"
    echo "--------------------------------------------------------"

    # stdout + stderr 同时写文件，同时显示在终端（tee）
    python test_readback.py \
        --num_envs "$NUM" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --steps "$STEPS" \
        --cold_start "$COLD_START" \
        --plot \
        2>&1 | tee "$DEBUG_LOG"

    echo "[Done] Log saved to $DEBUG_LOG"
done

echo "All benchmarks completed."