#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../" && pwd)

echo "=================================================="
echo "Project Root: $PROJECT_ROOT"
echo "Starting Experiment: crf_only_law_with_2ds"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

accelerate launch --num_processes=2 "$PROJECT_ROOT/run_crf.py" \
    --mode test \
    --dataset_path /mnt/data3/Korean_abstraction/python/coreference/datasets/pii_ner_only_law_251119 \
    --ckpt_path /mnt/data3/Korean_abstraction/python/coreference/results/PII_klueBERT/CRF_train_thunder_and_sns \
    > "$PROJECT_ROOT/logs/crf_only_law_with_2ds.txt" 2>&1

echo "=================================================="
echo "Finished Experiment: crf_only_law_with_2ds"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo ""