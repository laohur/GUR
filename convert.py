# !/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
import shutil
from pathlib import Path

import logzero
import numpy as np
import torch
import torch.nn.functional as F
from logzero import logger
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (AdamW, BertForMaskedLM, BertTokenizer,
                          BertTokenizerFast, MT5Config,
                          MT5ForConditionalGeneration, T5Tokenizer,
                          Text2TextGenerationPipeline, get_scheduler)


def save(model, tokenizer, saved_dir):
    Path(saved_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(saved_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(saved_dir)
    logger.info("saved --> "+saved_dir)


def answer(model, tokenizer, doc=["四驱<extra_id_0>车和普通叉车有什么区别?"]):
    model = model.to('cuda')
    encoding = tokenizer(text=doc, truncation=True,
                         padding=True, max_length=32, return_tensors="pt")
    out = model.generate(input_ids=encoding.input_ids.to('cuda'), attention_mask=encoding.attention_mask.to(
        "cuda"), return_dict_in_generate=True, output_scores=False, max_length=128, num_beams=4, length_penalty=0.6)
    out_text = tokenizer.batch_decode(out["sequences"])
    return out_text



def init_e12d4(pretrained_model,save_dir):
    os.system(f"mkdir {save_dir}")
    logzero.logfile(save_dir + '/init.log', mode="w")
    from transformers import (MT5Config, MT5ForConditionalGeneration, T5Config,
                              T5ForConditionalGeneration, T5Tokenizer,
                              Text2TextGenerationPipeline)

    # logger.info((tokenizer.config))
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
    logger.info((tokenizer))

    config = T5Config.from_pretrained(pretrained_model)
    # config.num_decoder_layers = 2
    # model = T5ForConditionalGeneration.from_pretrained(pretrained_model,config=config)
    model = T5ForConditionalGeneration(config=config)
    # ('auto cut', ['<pad> <extra_id_44> </s>'])
    logger.info(("auto cut", answer(model, tokenizer)))

    # model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
    # block = model.decoder.block
    # model.decoder.block= nn.ModuleList( block[i] for i in [0,11] )

    save(model, tokenizer, save_dir)


def init_t5_small(pretrained_model,save_dir):
    os.system(f"mkdir {save_dir}")
    logzero.logfile(save_dir + '/init.log', mode="w")

    from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

    special_tokens = ["<extra_id_{}>".format(i) for i in range(100)]
    tokenizer = T5Tokenizer.from_pretrained(
        pretrained_model,
        do_lower_case=True,
        truncation=True,
        additional_special_tokens=special_tokens,
    )
    config = T5Config.from_pretrained(pretrained_model)
    config.d_vector = 128
    model = T5ForConditionalGeneration.from_pretrained(
        pretrained_model, config=config)
    logger.info((model))

    model.resize_token_embeddings(len(tokenizer))

    logger.info((tokenizer))
    logger.info((model))
    save(model, tokenizer, save_dir)
    answer(model, tokenizer)


def demo_t5_base(pretrained_model,save_dir):
    os.system(f"mkdir {save_dir}")
    logzero.logfile(save_dir + '/init.log', mode="w")
    from transformers import (MT5Config, MT5ForConditionalGeneration, T5Config,
                              T5ForConditionalGeneration, T5Tokenizer,
                              Text2TextGenerationPipeline)

    # logger.info((tokenizer.config))

    tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
    logger.info((tokenizer))
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
    # logger.info((model))
    logger.info(("origin ", answer(model, tokenizer)))

    config = T5Config.from_pretrained(pretrained_model)
    config.num_decoder_layers = 2
    model = T5ForConditionalGeneration.from_pretrained(
        pretrained_model, config=config)
    logger.info(("auto cut", answer(model, tokenizer)))

    model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
    block = model.decoder.block
    model.decoder.block = nn.ModuleList(block[i] for i in [0, 11])
    # for i in range(10,0,-1):
    #     del block[i]
    # logger.info((model))
    # model.decoder.block = block
    # model.config.d_vector = 128
    # model.config.num_decoder_layers = 2
    logger.info(("bound", answer(model, tokenizer)))

    model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
    block = model.decoder.block
    model.decoder.block = nn.ModuleList(block[i] for i in [10, 11])
    logger.info(("last", answer(model, tokenizer)))

    model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
    block = model.decoder.block
    model.decoder.block = nn.ModuleList(block[i] for i in [0, 1])
    logger.info(("first", answer(model, tokenizer)))

    # save(model, tokenizer, save_dir)


init_e12d4()
# init_t5_small()

# init_t5_base()
# init_t5_large()

