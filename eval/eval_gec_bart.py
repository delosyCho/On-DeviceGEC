from __future__ import absolute_import, division, print_function

import logging
import random

import numpy as np
import torch

from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForMaskedLM, AutoTokenizer

import DataHolder_BERT_GEC as Dataholder
from collections import OrderedDict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() and not False else "cpu")
    n_gpu = torch.cuda.device_count()

    # device = torch.device("cuda:0" if torch.cuda.is_available() and not False else "cpu")
    # n_gpu = 1 #torch.cuda.device_count()

    tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")

    print('n gpu:', n_gpu)

    seed = 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Prepare model
    epoch = 10
    batch_size = 196
    model = AutoModelForMaskedLM.from_pretrained('klue/roberta-large')
    model.to(device)
    model.train()

    loaded_state_dict = torch.load('roberta-large-gec')
    new_state_dict = OrderedDict()
    for n, v in loaded_state_dict.items():
        name = n.replace("module.", "")  # .module이 중간에 포함된 형태라면 (".module","")로 치환
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)

    data_holder = Dataholder.Dataholder()
    data_holder.batch_size = batch_size

    num_step = int(data_holder.input_ids.shape[0] / data_holder.batch_size) * epoch

    model.eval()

    cor = 0
    count = 0

    # num_step = int(data_holder.input_ids.shape[0] / data_holder.batch_size) * 5

    for step in range(num_step):
        batch = data_holder.test_batch()
        input_ids, attention_mask, labels = batch

        if n_gpu == 1:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        outputs = model.generate(input_ids, attention_mask=attention_mask)

        prediction = tokenizer.decode(outputs[0])
        if prediction == labels:
            cor += 1
        count += 1

        print(cor / count, cor, '/,', count)


if __name__ == "__main__":
    main()