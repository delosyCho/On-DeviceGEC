import random
import torch
import numpy as np
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("roberta-large")


class Dataholder:
    def __init__(self, model_name='roberta'):
        # eng wiki pre-training data
        self.input_ids = np.load('input_ids_bart.npy')
        self.attention_mask = np.load('attention_mask_bart.npy')
        self.labels = np.load('label_ids_bart.npy')

        total_num = 70000

        self.b_ix = 0
        self.b_ix2 = total_num

        self.r_ix = np.array(
            range(total_num), dtype=np.int32
        )
        np.random.shuffle(self.r_ix)
        self.batch_size = 16

    def next_batch(self):
        total_num = 70000

        indexes = []

        length = self.input_ids.shape[1]
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        attention_mask = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        labels = np.zeros(shape=[self.batch_size, self.labels.shape[1]], dtype=np.int32)

        for i in range(self.batch_size):
            if self.b_ix + self.batch_size >= total_num:
                self.b_ix = 0
                np.random.shuffle(self.r_ix)

            ix = self.r_ix[self.b_ix]
            self.b_ix += 1

            indexes.append(ix)

            input_ids[i] = self.input_ids[ix]
            attention_mask[i] = self.attention_mask[ix]
            labels[i] = self.labels[ix]

        return torch.tensor(np.array(input_ids), dtype=torch.long), \
               torch.tensor(np.array(attention_mask), dtype=torch.long), \
               torch.tensor(np.array(labels), dtype=torch.long)

    def test_batch(self):
        self.batch_size = 1
        length = self.input_ids.shape[1]
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        attention_mask = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        indices = []
        labels = []

        for i in range(self.batch_size):
            if self.b_ix2 + self.batch_size >= self.input_ids[0]:
                exit(0)

            ix = self.b_ix2
            self.b_ix2 += 1

            input_ids[i] = self.input_ids[ix]
            attention_mask[i] = self.attention_mask[ix]

            for j in range(64):
                if input_ids[i, j] == 4:
                    indices.append(j)
                    labels.append(self.labels[ix, j])

        return torch.tensor(np.array(input_ids), dtype=torch.long), \
               torch.tensor(np.array(attention_mask), dtype=torch.long), \
               labels, indices