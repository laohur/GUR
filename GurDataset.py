# !/usr/bin/env python3
# -*- coding: utf-8 -*-


import collections
import math
import os
import random
import subprocess
import time
import unicodedata

import numpy as np
import torch
from logzero import logger
from torch.utils.data import DataLoader, Dataset
from transformers import (BertTokenizer, BertTokenizerFast,
                          MT5ForConditionalGeneration, RobertaTokenizer,
                          T5Tokenizer)
from transformers.models.bert.tokenization_bert import BasicTokenizer

from SpanMasker import SpanMasker

# https://github.com/joeljang/Pretraining_T5_custom_dataset/blob/7dfbee9963197f2cda37c8a14085b78ed7c0bd54/pretrain.py#L472
sentinels = [f"<extra_id_{i}>" for i in range(100)]
# sentinels = [f'<extra{i}>' for i in range(100)]

# never_split=["</s>","<unk>","<pad>"]
# basicTokenizer = BasicTokenizer(do_lower_case=False,never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "<S>", "<T>", "<s>", "</s>"] + sentinels,)
basicTokenizer = BasicTokenizer(do_lower_case=False)


def randomcrop(x, ratio_min=0.5, ratio_max=1, min_len=10, max_len=128, start=-1):
    ratio = ratio_min + (ratio_max - ratio_min) * random.random()
    length = int(len(x) * ratio)
    length = max(length, min_len)
    length = min(length, max_len)
    if length >= len(x):
        return x
    if start < 0:
        start = random.randint(0, len(x) - length)
    crop = x[start: start + length]
    return crop


class GurDataset(Dataset):
    """
    custom  Dataset
    """

    def __init__(
        self,
        path="",
        doc=[],
        max_length=128,
        tgt_max_length=32,
        tokenizer=None,
        shuffle=False,
        n_lines=-1,
        task_name="pair",
    ):
        """init"""
        super(GurDataset, self).__init__()
        self.max_length = max_length
        self.tgt_max_length = tgt_max_length
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        assert task_name in ["pair", "d2t", "answer", "commentary"]
        self.task_name = task_name
        self.maskers = [SpanMasker(p/100, sentinels=sentinels)
                        for p in range(10, 30)]
        self.path = path
        if doc:
            doc = self.parse_doc(doc)
            n_lines = len(doc)
            self.samples = collections.deque(doc)
        else:
            logger.info(self.path)
            if n_lines > 0:
                if task_name == "pair":
                    n_lines *= 2
            else:
                n_lines = self.get_lines()
            # self.reader =None
            self.reader = subprocess.Popen(
                self.path, shell=True, stdout=subprocess.PIPE, errors="ignore"
            )
            n_lines = int(n_lines * 0.95)
            self.samples = collections.deque()
        self.n_lines = n_lines
        logger.info(f"{path} load lines:{self.n_lines} ")

    def get_lines(self):
        sampled = 0
        reader = subprocess.Popen(
            self.path, shell=True, stdout=subprocess.PIPE, errors="ignore"
        )
        doc = reader.stdout.read(1024 * 1024 * 32).splitlines()
        total = 0
        while doc:
            total += len(doc)
            doc1 = self.parse_doc(doc)
            sampled += len(doc1)
            logger.info((total, sampled, sampled / total))
            doc = reader.stdout.read(1024 * 1024 * 32).splitlines()
        logger.info(f"{self.path} total:{total} --> get lines {sampled}")
        return sampled

    def parse_doc(self, doc):
        """parse_doc"""
        doc = [l.strip("...").strip() for l in doc]
        doc = [x.split("\t") for x in doc if len(x) >= 2]
        doc = [x for x in doc if len(x) == 2 and len(
            x[0]) >= 2 and len(x[1]) >= 2]
        if self.shuffle:
            random.shuffle(doc)
        if self.task_name == "pair":
            doc = [x for y in doc for x in y]
        return doc

    def mask_line(self, line, style, max_length):
        tokens = list(line[:128])
        if not style:
            src = tgt = tokens
        else:
            masker = random.choice(self.maskers)
            src, tgt = masker.mask_tokens(tokens, style=style)
        src = "".join(src)
        tgt = "".join(tgt)
        return src, tgt

    def __len__(self):
        """len"""
        return self.n_lines

    def __getitem__(self, idx):
        """item"""
        if len(self.samples) < 1024:
            if not self.samples and not self.reader:
                self.reader = subprocess.Popen(
                    self.path, shell=True, stdout=subprocess.PIPE, errors="ignore"
                )
            doc = self.reader.stdout.read(1024 * 1024 * 8).splitlines()
            doc = self.parse_doc(doc)
            self.samples.extend(doc)
        line = self.samples.popleft()
        if self.task_name == "pair":
            (src, tgt) = self.mask_line(
                line, style="t5", max_length=int(self.max_length * 1.5)
            )
        return (src, tgt, line)

    def collate_fn(self, doc):
        """collate_fn"""
        tokenizer = self.tokenizer
        srcs_doc = [x[0] for x in doc]
        tgts_doc = [x[1] for x in doc]
        srcs = tokenizer(
            srcs_doc,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        tgts = tokenizer(
            tgts_doc,
            padding="max_length",
            truncation=True,
            max_length=self.tgt_max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = srcs.input_ids
        attention_mask = srcs.attention_mask
        labels = tgts.input_ids
        # labels[labels == tokenizer.pad_token_id] = -100
        decoder_attention_mask = tgts.attention_mask

        row = [input_ids, attention_mask, labels, decoder_attention_mask, doc]
        return row

def fetch_data(
    tokenizer,
    train_path,
    max_length=32,
    batch_size=32,
    n_lines=-1,
    task_name="pair",
    shuffle=False,
):
    n_lines = -1
    # n_lines -= max(10000, n_lines // 100)
    batch_size //= 8
    lcs = 1
    if lcs:
        path = train_path.replace("lcs0", "lcs1")
    else:
        path = train_path.replace("lcs1", "lcs0")
    data_set = GurDataset(
        path=path,
        max_length=max_length,
        tokenizer=tokenizer,
        shuffle=shuffle,
        n_lines=n_lines,
        task_name=task_name,
    )
    dataloader = DataLoader(
        data_set,
        collate_fn=data_set.collate_fn,
        batch_size=batch_size,
        num_workers=1,
        drop_last=True,
        # pin_memory=True
    )
    return dataloader


if __name__ == "__main__":

    special_tokens = ["<extra_id_{}>".format(i) for i in range(100)]

    pretrained_model = "t5_small"
    from transformers import T5Tokenizer, T5TokenizerFast
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model,)  # slow

    line = "<extra_id_0>、黑森<extra_id_1>酒 店 ， <extra_id_2> 星"
    line += "<extra0>登录<extra1>按复<extra2>查<extra3>中心"
    # <extra_id_0> 办法 <extra_id_1> 以 <extra_id_2> 要拍打</s>
    line = "<extra_id_0>办法<extra_id_1>以<extra_id_2>要拍打"
    logger.info(tokenizer.tokenize(line))
    logger.info(basicTokenizer.tokenize(line))
    b = tokenizer.decode(tokenizer(line, max_length=32)["input_ids"])
    logger.info(b)
    # exit()

    afs = "cat   /corpus/sents2pair-r2/"
    train_file = afs + "csl_pairs-lcs1-title1-repeat2.tsv.uniq"
    data_set = GurDataset(
        path=train_file,
        max_length=32,
        tgt_max_length=12,
        tokenizer=tokenizer,
        shuffle=False,
        n_lines=532280,
        task_name="pair",
    )
    dataloader = DataLoader(
        data_set,
        collate_fn=data_set.collate_fn,
        batch_size=1024,
        num_workers=2,
        drop_last=True,
    )

    for step, batch in enumerate(dataloader):
        if step % 100 != 0:
            continue
        (input_ids, attention_mask, labels, decoder_attention_mask, doc) = batch
        logger.info((step, doc[0]))
        # labels[labels ==-100 ] = tokenizer.pad_token_id
        logger.info("".join(tokenizer.tokenize(doc[0][0])))
        logger.info(tokenizer.decode(input_ids[0]))
        logger.info("".join(tokenizer.tokenize(doc[0][1])))
        logger.info(tokenizer.decode(labels[0]))
        d = 0
