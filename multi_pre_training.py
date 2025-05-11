

import argparse
from dotenv import load_dotenv
import json
from typing import Dict
from transformers import TrainingArguments, AutoTokenizer, AutoConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from transformers.trainer import Trainer
import os
import wandb
from callback import FirstTokenAccuracyCallback, PreTrainingShuffleBiographyCallBack
from data_module import construct_pre_training_data_module, construct_pre_training_first_token_accuracy_data_module
from utils import AdditionalTrainingArguments, AttentionMaskType, DataArguments, FirstTokenAccuracyCalculationStrategy, train_and_save_model
def train(args):
    
    # region read config
    pre_training_config = json.load(open(args.config_path, 'r'))
    wandb_config = pre_training_config['wandb']
    wandb_config['run_name'] = args.task_name
    shared_config = pre_training_config['shared']  # config that is shared by all tasks
    task_config= pre_training_config['task'][args.task_name]  # config that is different for each task
    del pre_training_config  # avoid misuse
    # endregion

    # region set trainer arguments
    output_dir = task_config['output_dir']
    max_steps = task_config['max_steps']  # When reach this step, the training will stop
    warmup_steps = task_config['warmup_steps']  # When reach this step, the learning rate will start to decrease
    training_args = TrainingArguments(
        output_dir=output_dir, 
        optim='adamw_torch',  
        per_device_train_batch_size=96, 
        eval_strategy='no', 
        gradient_accumulation_steps=1,  
        max_steps=max_steps,
        weight_decay=0.1, 
        adam_epsilon=1e-6,  
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine_with_min_lr",  
        lr_scheduler_kwargs={"min_lr": 0.0001},  
        learning_rate=0.001,  
        save_strategy='steps',  
        save_steps=max_steps//10,  
        bf16=True,  
        logging_steps=1,  
        report_to=['wandb'],  
        # deepspeed='./ds_configs/stage2.json',
    )
    additional_training_args = AdditionalTrainingArguments(
        attention_mask_type=AttentionMaskType.ALL_TRUE,
        first_token_accuracy_calculation_strategy=FirstTokenAccuracyCalculationStrategy.EPOCH,
        first_token_accuracy_calculation_interval=5,
        pre_training_person_index_info_list=task_config['person_index_info_list']
    )
    # endregion
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # region set wandb
    wandb.login(key=os.environ.get('WANDB_API'), relogin=True)  
    wandb.init(
        project=wandb_config['project'],
        name=wandb_config['run_name'],
        config={**wandb_config, 'all_config': {'shared': shared_config, 'task': task_config, 'wandb': wandb_config}},
    )
    # endregion

    # region construct tokenizer and model
    model_path = shared_config['model_path']
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=512, 
        padding_side="right",
        use_fast=True,
    )
    assert tokenizer.pad_token is None
    tokenizer.pad_token = tokenizer.eos_token
    if "gpt-neox" in model_path:
        if task_config['previous_output_dir'] == '':
            config = AutoConfig.from_pretrained(
                model_path,
                vocab_size=len(tokenizer),
                max_position_embeddings=tokenizer.model_max_length,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            model = GPTNeoXForCausalLM(config)
        else:
            model = GPTNeoXForCausalLM.from_pretrained(os.path.join(task_config['previous_output_dir'], 'final_model'))
    else:
        raise ValueError(f"Unsupported model: {model_path}")
    
    # region construct data module and trainer
    data_args = DataArguments(
        biography_data_path=shared_config['biography_data_path']
    )
    # no eval_dataset
    train_dataset = construct_pre_training_data_module(tokenizer, data_args, additional_training_args)
    first_token_accuracy_dataset = construct_pre_training_first_token_accuracy_data_module(tokenizer, data_args, additional_training_args)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[
            FirstTokenAccuracyCallback(first_token_accuracy_dataset,
                                       additional_training_args.first_token_accuracy_calculation_strategy,
                                       additional_training_args.first_token_accuracy_calculation_interval),
            PreTrainingShuffleBiographyCallBack(
                train_dataset
            )
        ],
    )
    # endregion

    # region train and save model
    train_and_save_model(trainer, training_args, remove_all_checkpoint=False)
    # endregion

if __name__ == "__main__":
    load_dotenv()
    args = argparse.ArgumentParser()
    args.add_argument("--config_path", type=str, default="config/multi_fine_tuning.json")
    args.add_argument("--task_name", type=str, default="task_0")
    args = args.parse_args()
    train(args)