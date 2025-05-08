# 激活llama_factory环境

source ~/anaconda3/bin/activate llama_factory

for task_id in {0..5}
do
  echo "开始训练任务 task_${task_id}"
  python ./multi_pre_training.py --config_path ./config/multi_pre_training.json --task_name task_"${task_id}"
  echo "任务 task_${task_id} 训练完成"
done