import numpy as np
import torch
import preprocessor
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import GPT2Config, GPT2DoubleHeadsModel, GPT2Tokenizer, \
                         OpenAIGPTConfig, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer
import tensorflow as tf

import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import pandas as pd
import torch

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pathlib import Path

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


class Model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('DialoGPT-small')
        self.device = torch.device('cpu')
        self.model = AutoModelWithLMHead.from_pretrained('DialoGPT-small', from_tf=False)
        self.model.to(self.device)

    def train(self, dataset: preprocessor.ConversationData) -> (int, float):
        # print(dataset.tokenized_convos)
        num_epochs = 3
        sampler = RandomSampler(dataset)

        def collate(tokens: [torch.Tensor]):
            return torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)

        dataloader = DataLoader(dataset, sampler=sampler, batch_size=4, collate_fn=collate, drop_last=True)
        print(len(dataset))
        num_rows = len(dataset) * 3
        self.model.resize_token_embeddings(len(self.tokenizer))

        optimizer = AdamW([{
            "params": [param for name, param in self.model.named_parameters()
                       if not ("bias" in name or "LayerNorm.weight" in name)],
            "weight_decay": 0
        }, {
            "params": [param for name, param in self.model.named_parameters()
                       if ("bias" in name or "LayerNorm.weight" in name)],
            "weight_decay": 0
        }])

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_rows)

        global_step = current_epoch = 0
        total_loss = 0.0
        self.model.zero_grad()

        # This has the nice effect of printing progress bar.
        for _ in trange(current_epoch, num_epochs, desc="Epoch"):
            for step, batch in enumerate(tqdm(dataloader, desc="Step")):
                if batch.shape[1] > 1024:
                    continue

                inputs = batch.to(self.device)
                labels = batch.to(self.device)
                self.model.train()
                out = self.model(inputs, labels=labels)
                loss = out[0]
                loss.backward()
                total_loss += loss.item()

                torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                global_step += 1

                if global_step % 2000 == 0:
                    print('Saving model')
                    output = f'./checkpt-{global_step}'
                    self.save_model(output)
                    torch.save(optimizer.state_dict(), f'{output}/optimizer.pt')
                    torch.save(scheduler.state_dict(), f'{output}/scheduler.pt')

        self.save_model(f'./final-model')
        return global_step, total_loss / global_step

    def save_model(self, output_dir: str):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def evaluate(self, dataset: preprocessor.ConversationData):
        def collate(tokens: [torch.Tensor]):
            return torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=4, collate_fn=collate, drop_last=True)

        loss = 0.0
        eval_steps = 0
        self.model.eval()

        for batch in tqdm(dataloader, desc="Evaluation steps."):
            inputs = batch.to(self.device)
            labels = batch.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs, labels=labels)
                batch_loss = outputs[0]
                if __name__ == '__main__':
                    loss += batch_loss.mean().item()

            eval_steps += 1

        loss = loss / eval_steps
        # Return perplexity.
        return torch.exp(torch.tensor(loss))

if __name__ == '__main__':
    model = Model()
    model.train(preprocessor.ConversationData(json_path="./training/training.json", cutoff=10000))

