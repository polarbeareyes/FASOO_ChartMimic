#!/bin/bash
set -e  # 에러 시 즉시 종료

export PROJECT_PATH=/root/SOJUNG_STUFF/ChartMimic

echo "Step 1: Get Model Response"
python chart2code/main.py \
--cfg_path eval_configs/direct_generation.yaml \
--tasks chart2code \
--model qwenvl

echo "Step 2: Run the Code in the Response"
python chart2code/utils/post_process/code_checker.py \
--input_file ${PROJECT_PATH}/results/direct/chart2code_qwenvl_DirectAgent_results.json \
--template_type direct

python chart2code/utils/post_process/code_interpreter.py \
--input_file ${PROJECT_PATH}/results/direct/chart2code_qwenvl_DirectAgent_results.json \
--template_type direct


echo "Step 3: Get Lowlevel Score"
python chart2code/main.py --cfg eval_configs/direct/code4evaluation.yaml --tasks code4evaluation --model qwenvl

echo "Done for Gemma with max_tokens 4096"
