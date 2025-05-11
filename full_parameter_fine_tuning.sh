source ~/anaconda3/bin/activate llama_factory

for data_id in {0..5};
do
  for ((data_id=0; data_id<=task_id; data_id++))
  do
    python multi_full_parameter_fine_tuning.py --config_path ./config/multi_fine_tuning.json --run_config_dict_key "0"_"${data_id}" --run_name "0"_"${data_id}"
  done
done

