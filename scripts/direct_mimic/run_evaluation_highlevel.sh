evaluation_dir=/root/SOJUNG_STUFF/ChartMimic/results/direct/chart2code_deepseek-vl-7b-chat_DirectAgent_results/direct

python chart2code/main.py --cfg_path eval_configs/evaluation_direct.yaml --tasks gpt4evaluation --evaluation_dir ${evaluation_dir}

