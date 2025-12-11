#!/bin/bash

set -e

MODE=$1   # "nohup" or empty

LOG_DIR="./logs_paddle_bench"

if [[ "$MODE" == "nohup" ]]; then
    echo "[MODE] Running with nohup (logs saved to $LOG_DIR)"
    mkdir -p $LOG_DIR
else
    echo "[MODE] Running in terminal (no logs saved)"
fi

for file in benchmark/test_paddle*; do
    if [[ -f "$file" ]]; then
        echo "Running $file ..."
        
        if [[ "$MODE" == "nohup" ]]; then
            # 后台运行 + 日志
            log_file="$LOG_DIR/$(basename ${file}).log"
            nohup pytest -xsv "$file" > "$log_file" 2>&1 &
            echo "  → Log: $log_file"
        else
            # 前台运行
            pytest -xsvv "$file"
        fi
    fi
done

if [[ "$MODE" == "nohup" ]]; then
    echo "All tests launched in background with nohup."
    echo "Logs saved in: $LOG_DIR"
fi