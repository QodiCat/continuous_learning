import json
from typing import Dict, Any,List,Tuple
import transformers
from utils import DataArguments, AdditionalTrainingArguments, AttentionMaskType, FineTuningTrainPersonIndexInfoList, FirstTokenAccuracyCalculationStrategy, PreTrainingPersonIndexInfoList, construct_selected_person_index_set, attribute_list
from utils import first_token_accuracy_calculation_interval_when_only_end
from tqdm import tqdm
import torch
import random
from tqdm import trange
from torch.utils.data import Dataset
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast

from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
QARawData = Dict[str, Dict[str, Dict[str, str]]]

class BiographyDataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 raw_data: Dict[str, str],
                 additional_training_args: AdditionalTrainingArguments):
        self.tokenizer = tokenizer
        self.additional_training_args = additional_training_args
        self.input_ids_list, self.label_list, self.attention_mask_list = [], [], []
        self.biography_index_to_token_ids = {}
        # the code is written under the assumption that the tokenizer is GPT2Tokenizer
        assert type(tokenizer) in (GPT2Tokenizer, GPT2TokenizerFast, GPTNeoXTokenizerFast)
        # concatenate the biography data
        for biography_index, biography in tqdm(raw_data.items(), desc='Biography data tokenizing'):
            assert biography[0] == ' ', ('Each biography should start with a space, so that the tokenizer result is correct.')
            self.biography_index_to_token_ids[biography_index] = tokenizer(biography + tokenizer.eos_token)['input_ids']
        # this dataset will not be used in the training process, it is set to pass to validation of huggingface Trainer
        self.construct_dataset()

    def construct_dataset(self) -> None:
        self.input_ids_list, self.label_list, self.attention_mask_list = [], [], []
        biography_key_list = list(self.biography_index_to_token_ids.keys())
        random.shuffle(biography_key_list)
        biography_ids = []
        for biography_key in biography_key_list:
            biography_ids.extend(self.biography_index_to_token_ids[biography_key])
        for i in trange(0, len(biography_ids), self.tokenizer.model_max_length, desc='Biography data chunking'):
            ids_chunk = biography_ids[i:i + self.tokenizer.model_max_length]
            if len(ids_chunk) == self.tokenizer.model_max_length:
                input_ids = torch.tensor(ids_chunk, dtype=torch.int64)
                target = input_ids.clone()
            else:
                input_ids = torch.tensor(
                    ids_chunk + [self.tokenizer.pad_token_id] * (self.tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
                target = torch.tensor(
                    ids_chunk + [IGNORE_TOKEN_ID] * (self.tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
            target_mask = torch.cat(
                (torch.tensor([True], dtype=torch.bool), input_ids.ne(self.tokenizer.pad_token_id)), dim=0)[:-1]
            target = target.masked_fill(~target_mask, IGNORE_TOKEN_ID)

            # region construct attention mask
            match self.additional_training_args.attention_mask_type:
                case AttentionMaskType.MASK_EOS:
                    # attention_mask 1: only mask the eos token
                    attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
                case AttentionMaskType.ALL_TRUE:
                    # attention_mask 2: no token will be masked
                    attention_mask = torch.tensor([True] * len(input_ids), dtype=torch.bool)
                case _:
                    raise ValueError('Invalid attention mask type.')
                    # attention_mask 3: mask previous biography entries
            # endregion
            self.input_ids_list.append(input_ids)
            self.label_list.append(target)
            self.attention_mask_list.append(attention_mask)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'labels': self.label_list[idx],
            'attention_mask': self.attention_mask_list[idx],
        }




class QADataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 raw_data: QARawData,
                 max_length: int):
        self.input_ids_list, self.label_list, self.attention_mask_list = [], [], []
        assert type(tokenizer) in (GPT2Tokenizer, GPT2TokenizerFast, GPTNeoXTokenizerFast)
        for person_index in tqdm(raw_data, desc='Constructing QADataset'):
            # if len(self.input_ids_list) > 5000: break
            for attribute in raw_data[person_index]:
                prompt = raw_data[person_index][attribute]['prompt']
                answer = raw_data[person_index][attribute]['answer']
                assert prompt[-1] == ':'
                prompt_ids = tokenizer(prompt)['input_ids']
                answer_ids = tokenizer(answer)['input_ids']
                assert len(prompt_ids) + len(answer_ids) + 1 <= max_length  # +1 for <eos>
                assert len(prompt_ids) + len(answer_ids) == len(tokenizer(prompt + answer)['input_ids'])
                pad_length = max_length - (len(prompt_ids) + len(answer_ids) + 1)
                assert pad_length >= 0
                input_ids = torch.tensor(
                    prompt_ids + answer_ids + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * pad_length,
                    dtype=torch.int64
                )
                target_ids = torch.tensor(
                    [IGNORE_TOKEN_ID] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id] +
                    [IGNORE_TOKEN_ID] * pad_length,
                    dtype=torch.int64
                )
                # although the token indicating <eos> will also be masked, it is not a problem
                attention_mask = input_ids.ne(tokenizer.pad_token_id)
                self.input_ids_list.append(input_ids)
                self.label_list.append(target_ids)
                self.attention_mask_list.append(attention_mask)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'labels': self.label_list[idx],
            'attention_mask': self.attention_mask_list[idx],
        }

class PreTrainingFirstTokenAccuracyDataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 raw_data: Dict,
                 additional_training_args: AdditionalTrainingArguments):
        self.input_ids_list, self.token_position_list, self.attention_mask_list = [], [], []
        assert isinstance(tokenizer, GPTNeoXTokenizerFast)
        biography_ids = []
        attribute_first_token_position = []
        raw_data_list = list(raw_data.values())
        random.shuffle(raw_data_list)
        for info_dict in tqdm(raw_data_list, desc='Biography data tokenizing'):
            biography = info_dict['biography'] + tokenizer.eos_token
            token_info = info_dict['token_info']
            single_person_biography_ids = tokenizer(biography)['input_ids']
            single_person_attribute_first_token_position: List[None | Tuple[str, int]] = (
                    [None] * len(single_person_biography_ids))
            for attribute, first_token_position_info in token_info.items():
                single_person_attribute_first_token_position[first_token_position_info['first_token_position']] = \
                    (attribute, first_token_position_info['first_token'])
            biography_ids.extend(single_person_biography_ids)
            attribute_first_token_position.extend(single_person_attribute_first_token_position)
        for i in trange(0, len(biography_ids), tokenizer.model_max_length, desc='Biography data chunking'):
            ids_chunk = biography_ids[i:i + tokenizer.model_max_length]
            first_token_position_chunk = attribute_first_token_position[i:i + tokenizer.model_max_length]
            if len(ids_chunk) == tokenizer.model_max_length:
                input_ids = torch.tensor(ids_chunk, dtype=torch.int64)
            else:
                input_ids = torch.tensor(
                    ids_chunk + [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(ids_chunk)),
                    dtype=torch.int64)
            token_position = {}
            for attribute in attribute_list:
                token_position[attribute] = {
                    'index': torch.tensor([False] * len(input_ids), dtype=torch.bool),
                    'first_token_list': []
                }
            for info_id in range(len(first_token_position_chunk)):
                info = first_token_position_chunk[info_id]
                if info is None:
                    continue
                attribute, first_token = info
                assert input_ids[info_id] == first_token
                token_position[attribute]['index'][info_id] = True
                token_position[attribute]['first_token_list'].append(first_token)
            # region construct attention mask
            match additional_training_args.attention_mask_type:
                case AttentionMaskType.ALL_TRUE:
                    attention_mask = torch.tensor([True] * len(input_ids), dtype=torch.bool)
                case _:
                    raise ValueError('Invalid attention mask type.')
            # endregion
            self.input_ids_list.append(input_ids)
            self.token_position_list.append(token_position)
            self.attention_mask_list.append(attention_mask)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'token_position': self.token_position_list[idx],
            'attention_mask': self.attention_mask_list[idx],
        }

class QAFirstTokenAccuracyDataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 raw_data: QARawData,
                 max_length: int):
        self.input_ids_list, self.token_position_list, self.attention_mask_list = [], [], []
        assert isinstance(tokenizer, GPTNeoXTokenizerFast)
        for person_index in tqdm(raw_data, desc='Constructing QAFirstTokenAccuracyDataset'):
            # if len(self.input_ids_list) > 5000: break
            for attribute in raw_data[person_index]:
                # the validation of the dataset is completed by QADataset
                prompt_ids = tokenizer(raw_data[person_index][attribute]['prompt'])['input_ids']
                answer_ids = tokenizer(raw_data[person_index][attribute]['answer'])['input_ids']
                pad_length = max_length - (len(prompt_ids) + len(answer_ids) + 1)
                input_ids = torch.tensor(
                    prompt_ids + answer_ids + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * pad_length,
                    dtype=torch.int64
                )
                token_position = {}
                for _attribute in attribute_list:
                    token_position[_attribute] = {
                        'index': torch.tensor([False] * len(input_ids), dtype=torch.bool),
                        'first_token_list': []
                    }
                token_position[attribute]['index'][len(prompt_ids)] = True
                token_position[attribute]['first_token_list'].append(answer_ids[0])
                attention_mask = input_ids.ne(tokenizer.pad_token_id)
                self.input_ids_list.append(input_ids)
                self.token_position_list.append(token_position)
                self.attention_mask_list.append(attention_mask)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'token_position': self.token_position_list[idx],
            'attention_mask': self.attention_mask_list[idx],
        }

class QAExactMatchDataset(Dataset):
    def __init__(self,
                 raw_data: QARawData, ):
        self.prompt_list, self.answer_list, self.attribute_list = [], [], []
        for person_index in tqdm(raw_data, desc='Constructing QAExactMatchDataset'):
            for attribute in raw_data[person_index]:
                # the tokenization of the dataset is left in the callback
                self.prompt_list.append(raw_data[person_index][attribute]['prompt'])
                self.answer_list.append(raw_data[person_index][attribute]['answer'])
                self.attribute_list.append(attribute)

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        return {
            'prompt': self.prompt_list[idx],
            'answer': self.answer_list[idx],
            'attribute': self.attribute_list[idx],
        }


def filter_biography_data_with_token_info(data_with_token_info: Dict[str, Any],
                                          person_index_info_list: PreTrainingPersonIndexInfoList) -> Dict[str, Any]:
    selected_person_index_set = construct_selected_person_index_set(person_index_info_list)
    result = {}
    for biography_index in data_with_token_info:
        person_index, _ = biography_index.split('_')
        if int(person_index) in selected_person_index_set:
            result[biography_index] = data_with_token_info[biography_index]
    return result

def filter_qa_data_with_token_info(raw_data: QARawData,
                                   person_index_info_list: FineTuningTrainPersonIndexInfoList) -> QARawData:
    selected_person_index_set = construct_selected_person_index_set(person_index_info_list)
    result = {}
    for person_index in raw_data:
        if int(person_index) in selected_person_index_set:
            result[person_index] = raw_data[person_index]
    return result

def construct_pre_training_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        additional_training_args: AdditionalTrainingArguments) -> BiographyDataset:
    data_with_token_info = json.load(open(data_args.biography_data_path))
    data_with_token_info = filter_biography_data_with_token_info(
        data_with_token_info, additional_training_args.pre_training_person_index_info_list)
    raw_data: Dict[str, Any] = {}
    for biography_index in data_with_token_info:
        assert tokenizer.__class__.__name__ == data_with_token_info[biography_index]['tokenizer']
        raw_data[biography_index] = data_with_token_info[biography_index]['biography']
    return BiographyDataset(tokenizer, raw_data, additional_training_args)

def construct_pre_training_first_token_accuracy_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        additional_training_args: AdditionalTrainingArguments) -> PreTrainingFirstTokenAccuracyDataset:
    data_with_token_info = json.load(open(data_args.biography_data_path))
    data_with_token_info = filter_biography_data_with_token_info(
        data_with_token_info, additional_training_args.pre_training_person_index_info_list)
    assert tokenizer.__class__.__name__ == list(data_with_token_info.values())[0]['tokenizer']
    return PreTrainingFirstTokenAccuracyDataset(tokenizer, data_with_token_info, additional_training_args)

def construct_qa_fine_tuning_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        additional_training_args: AdditionalTrainingArguments,
        max_length: int) -> QADataset:
    raw_data: QARawData = json.load(open(data_args.all_qa_data_path))
    raw_data = filter_qa_data_with_token_info(raw_data,
                                              additional_training_args.fine_tuning_training_person_index_info_list)
    return QADataset(tokenizer, raw_data, max_length)


def construct_qa_first_token_accuracy_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        additional_training_args: AdditionalTrainingArguments,
        max_length: int
) -> List[Dict[str, QAFirstTokenAccuracyDataset | str | FirstTokenAccuracyCalculationStrategy | int]]:
    raw_data: QARawData = json.load(open(data_args.all_qa_data_path))
    dataset_info_list = []
    log_prefix_set = set()
    for info_dict, calculation_strategy, calculation_interval in [
        (additional_training_args.fine_tuning_validation_person_index_info_dict,
         additional_training_args.first_token_accuracy_calculation_strategy,
         additional_training_args.first_token_accuracy_calculation_interval),
        (additional_training_args.fine_tuning_test_person_index_info_dict,
         FirstTokenAccuracyCalculationStrategy.ONLY_END,
         first_token_accuracy_calculation_interval_when_only_end)
    ]:
        for key, value in info_dict.items():
            _raw_data = filter_qa_data_with_token_info(raw_data, value)
            # avoid duplicate log prefix
            match calculation_strategy:
                case FirstTokenAccuracyCalculationStrategy.EPOCH:
                    log_prefix = f'{key}__epoch__{calculation_interval}'
                case FirstTokenAccuracyCalculationStrategy.STEP:
                    log_prefix = f'{key}__step__{calculation_interval}'
                case FirstTokenAccuracyCalculationStrategy.ONLY_END:
                    log_prefix = f'{key}__only_end'
                case _:
                    raise ValueError('Invalid calculation strategy.')
            assert log_prefix not in log_prefix_set
            log_prefix_set.add(log_prefix)
            # construct dataset
            dataset_info_list.append({
                'dataset': QAFirstTokenAccuracyDataset(tokenizer, _raw_data, max_length),
                'log_prefix': f'{log_prefix}__',
                'calculation_strategy': calculation_strategy,
                'calculation_interval': calculation_interval
            })
    return dataset_info_list

def construct_qa_exact_match_data_module(
        data_args: DataArguments,
        additional_training_args: AdditionalTrainingArguments
) -> List[Dict[str, QAExactMatchDataset | str]]:
    raw_data: QARawData = json.load(open(data_args.all_qa_data_path))
    dataset_info_list = []
    for key, value in additional_training_args.fine_tuning_test_person_index_info_dict.items():
        test_raw_data = filter_qa_data_with_token_info(raw_data, value)
        dataset_info_list.append({
            'dataset': QAExactMatchDataset(test_raw_data),
            'log_prefix': f'{key}__'
        })
    return dataset_info_list