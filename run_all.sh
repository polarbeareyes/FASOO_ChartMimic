#!/bin/bash
set -e  # 에러 시 즉시 종료

export PROJECT_PATH=/root/SOJUNG_STUFF/ChartMimic_non_changed/ChartMimic

echo "Step 1: Get Model Response"
bash scripts/direct_mimic/run_generation.sh

echo "Step 2: Run the Code in the Response"
bash scripts/direct_mimic/run_code.sh

echo "Step 3: Get Lowlevel Score"
bash scripts/direct_mimic/run_evaluation_lowlevel.sh

echo "Done for Gemma with max_tokens 4096"