#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7090

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# dev or test
split=test

data=../../alldata/snli_ve/snli_ve_${split}.tsv
path=/home/dongk/dkgroup/tsk/projects/OFA/run_scripts/snli_ve/checkpoints/ofa_snli_ve/encoder_14_25_36_cos_loss_detach_decoder_layerwise_cosloss_no_detach_{1e-4,}/checkpoint.best_snli_score_0.8850.pt
result_path=../../results/snli_ve
selected_cols=0,2,3,4,5

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=snli_ve \
    --batch-size=1 \
    --log-format=simple --log-interval=500 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --num-workers=0 \
    --fp16\
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"