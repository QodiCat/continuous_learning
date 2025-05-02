
from typing import Union, List, Optional

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments, PreTrainedTokenizer
from data_module import QAFirstTokenAccuracyDataset, attribute_list, StepIntervalList, construct_selected_step_set
from utils import FirstTokenAccuracyCalculationStrategy, first_token_accuracy_calculation_interval_when_only_end
from data_module import PreTrainingFirstTokenAccuracyDataset
import torch
import wandb
from tqdm import trange
import os
import json

FirstTokenAccuracyDatasetType = Union[PreTrainingFirstTokenAccuracyDataset, QAFirstTokenAccuracyDataset]


class FirstTokenAccuracyCallback(TrainerCallback):
    def __init__(self,
                 first_token_accuracy_dataset: FirstTokenAccuracyDatasetType,
                 calculation_strategy: FirstTokenAccuracyCalculationStrategy,
                 calculation_interval: int,
                 log_prefix: str = '',
                 additional_step_interval_list: Optional[StepIntervalList] = None):
        self.dataset = first_token_accuracy_dataset
        self.history = {}
        self.log_prefix = log_prefix
        self.calculation_strategy = calculation_strategy
        self.calculation_interval = calculation_interval
        if self.calculation_strategy == FirstTokenAccuracyCalculationStrategy.ONLY_END:
            assert self.calculation_interval == first_token_accuracy_calculation_interval_when_only_end
        if additional_step_interval_list is None:
            additional_step_interval_list = []
        self.additional_step_set = construct_selected_step_set(additional_step_interval_list)

    def calculate_first_token_accuracy(
            self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        model.eval()
        attribute_to_count = {}
        for attribute in attribute_list:
            attribute_to_count[attribute] = {
                'total': 0,
                'hard_correct': 0,
                'soft_correct': 0
            }
        for i in trange(0, len(self.dataset), args.train_batch_size, desc='Calculating first token accuracy'):
            batch = self.dataset[i: i + args.train_batch_size]
            inputs_ids = torch.cat([ids.unsqueeze(0) for ids in batch['input_ids']], dim=0)
            inputs_ids = inputs_ids.to(model.device)
            with torch.no_grad():
                logits = model(input_ids=inputs_ids).logits  # [bs, seq_len, vocab_size]
            shift_logits = logits[..., :-1, :].contiguous()  # [bs, seq_len-1, vocab_size]
            shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))  # [bs * (seq_len-1), vocab_size]
            for attribute in attribute_list:
                index = torch.tensor([], dtype=torch.bool)
                first_token_list = []
                for item in batch['token_position']:
                    item_first_token_list = item[attribute]['first_token_list']
                    item_index = item[attribute]['index']
                    if len(item_first_token_list) > 0 and item_index[0] == torch.tensor(True):
                        item_first_token_list = item_first_token_list[1:]
                    first_token_list.extend(item_first_token_list)
                    item_index = item_index[1:]
                    index = torch.cat([index, item_index])
                first_token = torch.tensor(first_token_list, dtype=torch.int64)
                selected_logits = shift_logits[index]
                # hard correct
                prediction = torch.argmax(selected_logits, dim=-1)
                hard_correct = torch.sum(prediction.cpu() == first_token).item()
                # soft correct
                prediction_prob = torch.softmax(selected_logits, dim=-1)
                soft_correct = torch.sum(
                    torch.gather(prediction_prob.cpu(), dim=-1, index=first_token.unsqueeze(-1))).item()
                # record_result
                attribute_to_count[attribute]['total'] += first_token.shape[0]
                attribute_to_count[attribute]['hard_correct'] += hard_correct
                attribute_to_count[attribute]['soft_correct'] += soft_correct
        # log to wandb
        for attribute, count in attribute_to_count.items():
            wandb.log({
                f'first_token_accuracy/hard/{self.log_prefix}{attribute}': count['hard_correct'] / count['total'],
                f'first_token_accuracy/soft/{self.log_prefix}{attribute}': count['soft_correct'] / count['total'],
            })
        # record to history
        match self.calculation_strategy:
            case FirstTokenAccuracyCalculationStrategy.EPOCH:
                history_key = state.epoch
            case FirstTokenAccuracyCalculationStrategy.STEP | FirstTokenAccuracyCalculationStrategy.ONLY_END:
                history_key = state.global_step
            case _:
                raise ValueError(f'Invalid calculation strategy: {self.calculation_strategy}')
        self.history[history_key] = attribute_to_count
        output_path = os.path.join(args.output_dir, f'{self.log_prefix}first_token_accuracy_history.json')
        json.dump(self.history, open(output_path, 'w'), indent=4)
        model.train()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (self.calculation_strategy == FirstTokenAccuracyCalculationStrategy.STEP and
                (state.global_step % self.calculation_interval == 0 or state.global_step in self.additional_step_set)):
            self.calculate_first_token_accuracy(args, state, control, **kwargs)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (self.calculation_strategy == FirstTokenAccuracyCalculationStrategy.EPOCH and
                int(state.epoch) % self.calculation_interval == 0):
            self.calculate_first_token_accuracy(args, state, control, **kwargs)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.calculate_first_token_accuracy(args, state, control, **kwargs)

class PreTrainingShuffleBiographyCallBack(TrainerCallback):
    def __init__(self,dataset):
        self.dataset=dataset
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.dataset.construct_dataset()
        #kwargs['train_dataloader'].dataset.construct_dataset()
