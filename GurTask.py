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
from accelerate import Accelerator
from logzero import logger
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (BertTokenizer, BertTokenizerFast,
                          MT5ForConditionalGeneration, T5Tokenizer,
                          get_scheduler)

from GurDataset import GurDataset as TaskDataset
from GurDataset import sentinels
from modeling_gurt5 import GurT5ForPretraining

accelerator = Accelerator()
device = accelerator.device


def answer(
    model,
    tokenizer,
    doc=[
        "四驱<extra_id_0>车和普通叉车有什么区别?",
        "<extra_id_0>是满足用户对蒸汽不同压力要求的理想设备",
        "<extra_id_0>机是满足用户对蒸汽不同压力要求的理想设备",
    ],
):
    model = model.to(device)
    encoding = tokenizer(
        text=doc, truncation=True, padding=True, max_length=32, return_tensors="pt"
    )
    out = model.generate(
        input_ids=encoding.input_ids.to(device),
        attention_mask=encoding.attention_mask.to(device),
        return_dict_in_generate=True,
        output_scores=False,
        max_length=128,
        num_beams=4,
        length_penalty=0.6,
    )
    out_text = tokenizer.batch_decode(
        out["sequences"], skip_special_tokens=True)
    return out_text


def compute_clip_loss(image_features, text_features, temperature=0.1):
    """  
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py
    """
    logits_per_image = image_features  @ text_features.T / temperature
    logits_per_text = text_features @ image_features.T / temperature
    labels = torch.arange(len(logits_per_image),
                          device=logits_per_image.device)
    total_loss = (nn.functional.cross_entropy(logits_per_image, labels) +
                  nn.functional.cross_entropy(logits_per_text, labels)) / 2
    return total_loss


class GurTask:
    """
    class to execute for model
    """

    def __init__(
        self,
        model_path="",
        save_dir="save_dir",
        train_lm=True,
        train_pair=True,
    ):
        """ init """
        self.save_dir = save_dir
        self.train_lm = train_lm
        self.train_pair = train_pair
        # self.device = torch.device("cuda")
        self.device = device
        # tokenizer = BertTokenizerFast.from_pretrained(model_path,do_lower_case=True,truncation=True,)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = GurT5ForPretraining.from_pretrained(model_path)
        logger.info(
            f" model loaded from {model_path} to gpu{os.environ.get('CUDA_VISIBLE_DEVICES',-1)}"
        )
        model.to(self.device)
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = 0.1

    def comput_metrix(self, logits, labels):
        pred = torch.argmax(logits, dim=-1)
        y_pred = pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float()) / y_true.shape[0]
        return acc, pred

    def valid_batch(self, batch, task_name):
        self.model.eval()
        with torch.no_grad():
            logger.info(("valid", answer(self.model, self.tokenizer)))
            (input_ids, attention_mask, labels,
             decoder_attention_mask, doc) = batch
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
                output_hidden_states=True
            )
        lm_logits = outputs.logits
        lm_loss = outputs.loss
        vector_logits = self.model.get_sentence_vector(
            outputs.encoder_last_hidden_state, input_ids)
        if task_name == "pair":
            s = vector_logits[::2]
            t = vector_logits[1::2]
        elif task_name in ['answer', "d2t"]:
            s = vector_logits
            t = self.model.get_sentence_vector(
                outputs.decoder_hidden_states[-1], labels)
        else:
            s = vector_logits
            t = self.model.get_sentence_vector(
                outputs.encoder_last_hidden_state, input_ids)
        cl_loss = compute_clip_loss(s, t, self.temperature)

        total_loss = 0
        loss_lm = 0
        loss_cl = 0
        acc = 0
        if self.train_lm:
            lm_logits = outputs.logits
            lm_loss = outputs.loss
            total_loss += lm_loss
            loss_lm = lm_loss.item()

            true = labels.clone()
            true[true == -100] = self.model.config.pad_token_id
            acc, y_pred = self.comput_metrix(lm_logits, true)

            label = self.tokenizer.decode(true[0])
            pred = self.tokenizer.decode(y_pred[0])
            logger.info(f"src:{ doc[0][0]} tgt:{doc[0][1]}  line:{doc[0][2]}")
            logger.info(f" 标签:{ label[:100]}  ")
            logger.info(f" 预测:{pred[:100]} ")
        if self.train_pair and task_name == "pair":
            # vector_logits=self.model.get_sentence_vector(outputs.encoder_last_hidden_state,input_ids)
            vector_logits = outputs.vector_logits
            # cl_loss = simcse_unsup_loss(vector_logits, self.temperature)
            s = vector_logits[::2]
            t = vector_logits[1::2]
            cl_loss = compute_clip_loss(s, t, self.temperature)
            total_loss += cl_loss
            loss_cl = cl_loss.item()
        loss_total = total_loss.item()

        line = f" acc:{acc:.3f} total:{loss_total:.3f} loss_lm:{loss_lm:.3f} cl:{loss_cl:.3f} "
        return line

    def train_batch(self, model, batch, optimizer, lr_scheduler, step, task_name, gradient_accumulation_steps):
        model.train()
        (input_ids, attention_mask, labels, decoder_attention_mask, doc) = batch
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            # output_hidden_states=True
        )

        total_loss = 0
        loss_lm = 0
        loss_cl = 0
        if self.train_lm:
            lm_loss = outputs.loss
            total_loss += lm_loss
            loss_lm = lm_loss.item()
        if self.train_pair and task_name == "pair":
            # vector_logits=model.get_sentence_vector(outputs.encoder_last_hidden_state,input_ids)
            vector_logits = outputs.vector_logits
            # cl_loss = simcse_unsup_loss(vector_logits, self.temperature)
            s = vector_logits[::2]
            t = vector_logits[1::2]
            cl_loss = compute_clip_loss(s, t, self.temperature)
            total_loss += cl_loss
            loss_cl = cl_loss.item()
        loss_total = total_loss.item()
        total_loss /= gradient_accumulation_steps
        # total_loss.backward()
        accelerator.backward(total_loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        line = f"total:{loss_total:.3f} loss_lm:{loss_lm:.3f} cl:{loss_cl:.3f} "
        return line

    def train(self, dataloaders, learning_rate, gradient_accumulation_steps):
        """        train        """
        model = self.model
        model.train()
        num_training_steps = sum([len(dataloader)
                                 for dataloader in dataloaders])
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps,
        )
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer,  lr_scheduler)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        logger.info(f"  num_training_steps {num_training_steps} ")
        step = 0
        folder = self.save_dir+f"/init"
        self.save(folder)
        progress_bar = tqdm(range(num_training_steps), ncols=110,
                            disable=not accelerator.is_local_main_process)
        for epoch, dataloader in enumerate(dataloaders):
            logger.info(f"epoch:{epoch} steps:{len(dataloader)}")
            dataloader = accelerator.prepare(dataloader)
            for batch in dataloader:
                step += 1
                for i in range(len(batch) - 1):
                    batch[i] = batch[i].to(accelerator.device)
                if step % 1000 == 0:
                    line = self.valid_batch(
                        batch, task_name=dataloader.dataset.task_name)
                    logger.info(f"step:{step} 验证" + line)
                line = self.train_batch(
                    model,
                    batch,
                    optimizer,
                    lr_scheduler,
                    step,
                    task_name=dataloader.dataset.task_name,
                    gradient_accumulation_steps=gradient_accumulation_steps)
                if accelerator.is_local_main_process:
                    progress_bar.update(1)
                    progress_bar.set_description(line)
                if step % 1000 == 0:
                    lr = lr_scheduler.get_last_lr()[0]
                    shape = list(batch[0].size())
                    logger.info(
                        f"epoch:{epoch} step:{step} shape:{shape} lr:{lr} "+line)
                if accelerator.is_local_main_process and step % (100000) == 0:
                    total, used, free = shutil.disk_usage(self.save_dir)
                    folder = self.save_dir+f"/step-{step}"
                    if free < 2 ** 30:
                        logger.error(f"{folder} disk full, continue")
                        continue
                    self.save(folder)
            folder = self.save_dir+f"/epoch-{epoch}"
            FULL = True
            while FULL:
                total, used, free = shutil.disk_usage(self.save_dir)
                if free >= 2 ** 30:
                    FULL = False
                else:
                    if random.random() < 0.001:
                        logger.error(f"{folder} disk full, continue")
                    time.sleep(100)
            self.save(folder)
        logger.info(f"trained")

    def save(self, saved_dir):
        accelerator.wait_for_everyone()
        if not accelerator.is_local_main_process:
            return
        try:
            tokenizer = self.tokenizer
            if isinstance(self.model, nn.DataParallel):
                model = self.model.module
            else:
                model = self.model
            Path(saved_dir).mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(saved_dir)
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(saved_dir)
            logger.info(f"   saved -> {saved_dir}")
        except Exception as e:
            logger.error(e)

    def fetch_data(
            self, train_path, tokenizer, max_length=32, tgt_max_length=128, batch_size=32, n_lines=-1, task_name="pair", shuffle=False, num_workers=2
    ):
        # n_lines=1024
        # n_lines -= max(10000, n_lines // 100)
        # batch_size=4
        if args.lcs:
            path = train_path.replace('lcs0', 'lcs1')
        else:
            path = train_path.replace('lcs1', 'lcs0')
        data_set = TaskDataset(
            path=path,
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
            # pin_memory=True
        )
        return dataloader


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES']='6'
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="/home/entropy/work/gurt5b/t5_small"
    )
    parser.add_argument("--save_dir", type=str, default="gur-demo-model")
    parser.add_argument("--lcs", type=int, default=1)
    parser.add_argument("--train_pair", type=int, default=1)
    parser.add_argument("--train_lm", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--task_name", type=str, default="")
    parser.add_argument("--trin_file", type=str, default="")
    parser.add_argument("--n_lines", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--tgt_max_length", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
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
    logger.info(vars(task))
    tokenizer = task.tokenizer
    loaders = [
        task.fetch_data(f"{args.trin_file}", tokenizer, max_length=args.max_length, tgt_max_length=args.tgt_max_length, batch_size=args.batch_size, n_lines=args.n_lines, task_name=args.task_name, shuffle=args.shuffle) for i in range(args.epoch)
    ]
    task.train(loaders, args.learning_rate, args.gradient_accumulation_steps)

"""
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=true python GurTask.py 
tensorboard --host 0.0.0.0 --port 8109  --logdir="${model_dir}/summary"
"""
