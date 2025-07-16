model=gemma
#export PYTHONPATH=$(pwd):$PYTHONPATH
#export PROJECT_PATH=/root/SOJUNG_STUFF/ChartMimic
python chart2code/main.py \
--cfg_path eval_configs/direct_generation.yaml \
--tasks chart2code \
--model ${model}