#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../" && pwd)

echo "=================================================="
echo "Project Root: $PROJECT_ROOT"
echo "Starting Experiment: run_KlueBERT_PII all"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

accelerate launch --num_processes=2 "$PROJECT_ROOT/run_KlueBERT_PII.py" \
    --mode test \
    --dataset_path /mnt/data3/Korean_abstraction/python/coreference/datasets/pii_ner_3dataset \
    > "$PROJECT_ROOT/logs/baseline_all.txt" 2>&1

echo "=================================================="
echo "Finished Experiment: run_KlueBERT_PII all"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo ""