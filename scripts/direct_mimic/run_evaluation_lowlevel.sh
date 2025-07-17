models=(
  "deepseek-vl-7b-chat"
)

for model in "${models[@]}"; do
  python chart2code/main.py --cfg eval_configs/direct/code4evaluation.yaml --tasks code4evaluation --model "${model}"
done