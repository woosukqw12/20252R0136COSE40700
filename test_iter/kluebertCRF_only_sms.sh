#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../" && pwd)

echo "=================================================="
echo "Project Root: $PROJECT_ROOT"
echo "Starting Experiment: run_crf sms"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

accelerate launch --num_processes=2 "$PROJECT_ROOT/run_crf.py" \
    --mode test \
    --dataset_path /mnt/data3/Korean_abstraction/python/coreference/datasets/pii_ner_only_sms_kluebert \
    > "$PROJECT_ROOT/logs/crf_only_sms.txt" 2>&1

echo "=================================================="
echo "Finished Experiment: run_crf sms"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo ""