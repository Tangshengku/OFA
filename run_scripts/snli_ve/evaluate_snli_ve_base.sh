#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7091
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# dev or test
split=test

data=/data/tsk/snli_ve/snli_ve_${split}.tsv
# path=../../checkpoints/snli_ve_base_best.pt
path=/data/tsk/checkpoints/ofa/{10,}_{5e-5,}/checkpoint.best_snli_score_0.8870.pt
result_path=../../results/snli_ve
selected_cols=0,2,3,4,5

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} \
     ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=snli_ve \
    --batch-size=1 \
    --log-format=simple --log-interval=100 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"