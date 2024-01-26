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


tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")

list_data_dict = jload('GEC_RM_data.json')#[0:1000]

sources = []
targets = []

for data_dict in list_data_dict:
    source = data_dict['source']
    target = data_dict['ground_truth']

    sources.append(source)
    targets.append(target)

check = 0

max_length = 48

count = 0
input_ids = np.zeros(shape=[len(sources), max_length], dtype=np.int32)
attention_mask = np.zeros(shape=[len(sources), max_length], dtype=np.int32)
label_ids = np.full(shape=[len(sources), max_length], dtype=np.int32, fill_value=-100)


for i in range(len(sources)):
    source_text = sources[i]
    ground_truth = targets[i]

    try:
        inputs = tokenizer(source_text, return_tensors="np")
        ids = list(inputs['input_ids'][0, :])
        length = min(len(ids), max_length)
        input_ids[count, 0: length] = ids[0: length]

        inputs = tokenizer(ground_truth, return_tensors="np")
        ids = list(inputs['input_ids'][0, :])
        length = min(len(ids), max_length)
        label_ids[count, 0: length] = ids[0: length]
    except:
        continue

    count += 1

    if count % 100 == 0:
        print(count, i, '/', len(sources))

np.save('input_ids_bart', input_ids[:count])
np.save('attention_mask_bart', attention_mask[:count])
np.save('label_ids_bart', label_ids[:count])
