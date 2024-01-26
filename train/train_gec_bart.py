from __future__ import absolute_import, division, print_function

import logging
import random

import numpy as np
import torch

from transformers import AdamW, get_linear_schedule_with_warmup, BartForConditionalGeneration

import DataHolder_BERT_GEC as Dataholder


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
    model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')
    model.to(device)
    model.train()

    # # bart-base_synth
    # loaded_state_dict = torch.load('bart-base_synth')
    # new_state_dict = OrderedDict()
    # for n, v in loaded_state_dict.items():
    #     name = n.replace("module.", "")  # .module이 중간에 포함된 형태라면 (".module","")로 치환
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict, strict=True)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)
    # input()
    # Prepare optimizer
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = None
    scheduler = None

    data_holder = Dataholder.Dataholder()
    data_holder.batch_size = batch_size

    num_step = int(data_holder.input_ids.shape[0] / data_holder.batch_size) * epoch

    if optimizer is None:
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-6)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(num_step * 0.1), num_training_steps=num_step
        )

    model.train()
    epoch = 0

    max_grad_norm = 1.0

    total_loss = 0
    tr_step = 0

    # num_step = int(data_holder.input_ids.shape[0] / data_holder.batch_size) * 5

    for step in range(num_step):
        batch = data_holder.next_batch()

        if n_gpu == 1:
            batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scheduler.step()
        optimizer.step()
        optimizer.zero_grad()

        tr_step += 1
        total_loss += loss.item()
        mean_loss = total_loss / tr_step
        print(step, loss.item(), mean_loss, '/', num_step)
        print('-----------------------------')
    epoch += 1
    logger.info("** ** * Saving file * ** **")

    output_model_file = "roberta-large-gec"
    torch.save(model.state_dict(), output_model_file)


if __name__ == "__main__":
    main()