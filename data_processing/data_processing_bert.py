from peft import (
    PeftModel
)

from transformers import GPTNeoXForCausalLM, AutoTokenizer, GPTNeoXConfig

import torch

from typing import List

import json
import io

import random
import numpy as np

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')

list_data_dict = jload('GEC_RM_data.json')#[0:1000]

sources = []
targets = []

for data_dict in list_data_dict:
    source = data_dict['source']
    target = data_dict['ground_truth']

    sources.append(source)
    targets.append(target)

file = open('ranking_dataset_correct', 'w', encoding='utf-8')

check = 0

max_length = 64

count = 0
input_ids = np.zeros(shape=[len(sources), max_length], dtype=np.int32)
attention_mask = np.zeros(shape=[len(sources), max_length], dtype=np.int32)
label_ids = np.full(shape=[len(sources), max_length], dtype=np.int32, fill_value=-100)
answers = np.zeros(shape=[len(sources)], dtype='<U200')


for i in range(len(sources)):
    source_text = sources[i]
    ground_truth = targets[i]

    try:
        source_tks = source_text.split(' ')
        target_tks = ground_truth.split(' ')
    except:
        continue

    if len(source_tks) != len(target_tks):
        continue

    input_tokens = ['[CLS]']
    label_tokens = ['[CLS]']

    source_tokens = tokenizer.tokenize(source_text)
    input_tokens.extend(source_tokens)
    label_tokens.extend(source_tokens)

    input_tokens.append('[SEP]')
    label_tokens.append('[SEP]')

    try:
        for t in range(len(source_tks)):
            s_tokens = tokenizer.tokenize(source_tks[t])
            t_tokens = tokenizer.tokenize(target_tks[t])
            t_tokens_copy = tokenizer.tokenize(target_tks[t])

            diff_indices = []
            for k in range(len(t_tokens)):
                if t_tokens[k] not in s_tokens:
                    for l in range(k, len(t_tokens)):
                        t_tokens[l] = '[MASK]'
                    # print()
                    break

            input_tokens.extend(t_tokens)
            label_tokens.extend(t_tokens_copy)
    except:
        continue

    ids = tokenizer.convert_tokens_to_ids(tokens=input_tokens)
    ids2 = tokenizer.convert_tokens_to_ids(tokens=label_tokens)

    if len(ids) > max_length:
        continue

    for j in range(len(ids)):
        input_ids[count, j] = ids[j]
        attention_mask[count, j] = 1

        if ids[j] == 4:
            label_ids[count, j] = ids2[j]
        else:
            label_ids[count, j] = -100
    answers[count] = ground_truth
    count += 1

    if count % 100 == 0:
        print(count, i, '/', len(sources))

    """
    print(i, '/', len(sources))
    print(source_text, tokenizer.tokenize(source_text))
    print(input_tokens)
    print(label_tokens, '\n')
    """

np.save('input_ids_bert', input_ids[:count])
np.save('attention_mask_bert', attention_mask[:count])
np.save('label_ids_bert', label_ids[:count])
np.save('answer_texts', answers[:count])

