# #!/bin/bash
# -*- coding: utf-8 -*-


CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=true python train_click.py \
    --model_path="gur-sp-small-e12d4" \
    --save_dir="gur-sp-e12d4-click" \
    --train_lm=1 \
    --train_pair=1 \
    --lcs=1


CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true python train_files.py \
    --model_path="gur-sp-small-e12d4" \
    --save_dir="gur-sp-e12d4-full" \
    --train_lm=1 \
    --train_pair=1 \
    --lcs=1

CUDA_VISIBLE_DEVICES=5 TOKENIZERS_PARALLELISM=true python train_files.py \
    --model_path="gur-sp-small-e12d4" \
    --save_dir="gur-sp-e12d4-lcs" \
    --train_lm=1 \
    --train_pair=1 \
    --lcs=0


CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=true python train_files.py \
    --model_path="gur-sp-small-e12d4" \
    --save_dir="gur-sp-e12d4-cl" \
    --train_lm=1 \
    --train_pair=0 \
    --lcs=1

CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true python train_files.py \
    --model_path="gur-sp-small-e12d4" \
    --save_dir="gur-sp-e12d4-lm" \
    --train_lm=0 \
    --train_pair=1 \
    --lcs=1

