# !/usr/bin/env python3
# -*- coding: utf-8 -*-


import collections
import os
import random
import shutil
import time
from pathlib import Path

import logzero
import numpy as np
import torch
from logzero import logger
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertTokenizerFast, get_scheduler

from GurDataset import GurDataset as TaskDataset
from GurTask import GurTask


def fetch_data(
    trin_file,
    tokenizer,
    max_length=32,
    tgt_max_length=128,
    batch_size=32,
    n_lines=-1,
    task_name="pair",
    shuffle=True,
    num_workers=2,
):
    if args.lcs:
        train_path = trin_file.replace("lcs0", "lcs1")
    else:
        train_path = trin_file.replace("lcs1", "lcs0")

    # n_lines -= max(10000, n_lines // 100)
    batch_size = int(batch_size * args.batch_scale)
    batch_size = 600

    data_set = TaskDataset(
        path=train_path,
        max_length=max_length,
        tgt_max_length=tgt_max_length,
        tokenizer=tokenizer,
        shuffle=shuffle,
        n_lines=n_lines,
        task_name=task_name,
    )
    dataloader = DataLoader(
        data_set,
        collate_fn=data_set.collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=shuffle,
    )
    return dataloader


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES']='6'
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default=""
    )
    parser.add_argument("--save_dir", type=str, default="gur-demo-model")
    parser.add_argument("--lcs", type=int, default=1)
    parser.add_argument("--train_pair", type=int, default=1)
    parser.add_argument("--train_lm", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--task_name", type=str, default="pair")
    parser.add_argument(
        "--trin_file",
        type=str,
        default="cat pairs-lcs1-repeat5-title1.tsv",
    )
    parser.add_argument("--n_lines", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--tgt_max_length", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_scale", type=float, default=1)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    logzero.logfile(f"{args.save_dir}/pretrain.log", mode="w")
    logger.info(args)
    task = GurTask(
        model_path=args.model_path,
        save_dir=args.save_dir,
        train_lm=args.train_lm,
        train_pair=args.train_pair,
    )
    trin_file = args.trin_file

    logger.info(vars(task))
    afs = "cat  /data/gur/sents2pair-repeat2/"
    loaders = [
        fetch_data(
            afs + "wikisource_pairs-lcs1-title1-repeat2.tsv.uniq",
            task.tokenizer,
            max_length=32,
            tgt_max_length=12,
            batch_size=1024,
            n_lines=-1,
            task_name=args.task_name,
            shuffle=args.shuffle,
        ),
        fetch_data(
            afs + "wiki_pairs-lcs1-title1-repeat2.tsv.uniq",
            task.tokenizer,
            max_length=32,
            tgt_max_length=12,
            batch_size=1024,
            n_lines=12117916,
            task_name=args.task_name,
            shuffle=args.shuffle,
        ),
        fetch_data(
            "cat  /home/entropy/data/gur/product_zhidao.tsv",
            task.tokenizer,
            max_length=32,
            tgt_max_length=12,
            batch_size=1024,
            n_lines=-1,
            task_name=args.task_name,
            shuffle=args.shuffle,
        ),
        fetch_data(
            afs + "csl_pairs-lcs1-title1-repeat2.tsv.uniq",
            task.tokenizer,
            max_length=32,
            tgt_max_length=12,
            batch_size=1024,
            n_lines=-1,
            task_name=args.task_name,
            shuffle=args.shuffle,
        ),
        fetch_data(
            afs + "sku_pairs-lcs1-title1-repeat2.tsv.uniq",
            task.tokenizer,
            max_length=32,
            tgt_max_length=12,
            batch_size=1024,
            n_lines=-1,
            task_name=args.task_name,
            shuffle=args.shuffle,
        ),
    ]
    for i, x in enumerate(loaders):
        logger.info((i, vars(x)))
    task.train(loaders, args.learning_rate, args.gradient_accumulation_steps)
