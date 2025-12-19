#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../" && pwd)

echo "=================================================="
echo "Project Root: $PROJECT_ROOT"
echo "Starting Experiment: run_thunder_PII all"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

accelerate launch --num_processes=2 "$PROJECT_ROOT/run_thunder_PII.py" \
    --mode test \
    --dataset_path /mnt/data3/Korean_abstraction/python/coreference/datasets/pii_ner_3merge_SNU_v3 \
    > "$PROJECT_ROOT/logs/thunder_all.txt" 2>&1

echo "=================================================="
echo "Finished Experiment: run_thunder_PII all"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo ""