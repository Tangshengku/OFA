#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6091
export CUDA_VISIBLE_DEVICES=4
export GPUS_PER_NODE=1


########################## Evaluate Refcoco ##########################
user_dir=../../ofa_module
bpe_dir=../../utils/BPE
selected_cols=0,4,2,3

data=/data/tsk/refcoco/refcoco_val.tsv
path=/data/tsk/checkpoints/ofa_refcoco/finetune_decompose_{5e-5,}_{512,}/checkpoint_5_2500.pt
result_path=../../results/refcoco
split='refcoco_val'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=1 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

data=/data/tsk/refcoco/refcoco_testA.tsv
path=/data/tsk/checkpoints/ofa_refcoco/finetune_decompose_{5e-5,}_{512,}/checkpoint_5_2500.pt
result_path=../../results/refcoco
split='refcoco_testA'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=1 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

data=/data/tsk/refcoco/refcoco_testB.tsv
path=/data/tsk/checkpoints/ofa_refcoco/finetune_decompose_{5e-5,}_{512,}/checkpoint_5_2500.pt
result_path=../../results/refcoco
split='refcoco_testB'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=1 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"



######################### Evaluate Refcocoplus ##########################
# data=../../dataset/refcocoplus_data/refcocoplus_val.tsv
# path=../../checkpoints/refcocoplus_base_best.pt
# result_path=../../results/refcocoplus
# split='refcocoplus_val'
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#     ${data} \
#     --path=${path} \
#     --user-dir=${user_dir} \
#     --task=refcoco \
#     --batch-size=16 \
#     --log-format=simple --log-interval=10 \
#     --seed=7 \
#     --gen-subset=${split} \
#     --results-path=${result_path} \
#     --beam=5 \
#     --min-len=4 \
#     --max-len-a=0 \
#     --max-len-b=4 \
#     --no-repeat-ngram-size=3 \
#     --fp16 \
#     --num-workers=0 \
#     --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

# data=../../dataset/refcocoplus_data/refcocoplus_testA.tsv
# path=../../checkpoints/refcocoplus_base_best.pt
# result_path=../../results/refcocoplus
# split='refcocoplus_testA'
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#     ${data} \
#     --path=${path} \
#     --user-dir=${user_dir} \
#     --task=refcoco \
#     --batch-size=16 \
#     --log-format=simple --log-interval=10 \
#     --seed=7 \
#     --gen-subset=${split} \
#     --results-path=${result_path} \
#     --beam=5 \
#     --min-len=4 \
#     --max-len-a=0 \
#     --max-len-b=4 \
#     --no-repeat-ngram-size=3 \
#     --fp16 \
#     --num-workers=0 \
#     --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

# data=../../dataset/refcocoplus_data/refcocoplus_testB.tsv
# path=../../checkpoints/refcocoplus_base_best.pt
# result_path=../../results/refcocoplus
# split='refcocoplus_testB'
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#     ${data} \
#     --path=${path} \
#     --user-dir=${user_dir} \
#     --task=refcoco \
#     --batch-size=16 \
#     --log-format=simple --log-interval=10 \
#     --seed=7 \
#     --gen-subset=${split} \
#     --results-path=${result_path} \
#     --beam=5 \
#     --min-len=4 \
#     --max-len-a=0 \
#     --max-len-b=4 \
#     --no-repeat-ngram-size=3 \
#     --fp16 \
#     --num-workers=0 \
#     --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"



# ########################## Evaluate Refcocog ##########################
# data=../../dataset/refcocog_data/refcocog_val.tsv
# path=../../checkpoints/refcocog_base_best.pt
# result_path=../../results/refcocog
# split='refcocog_val'
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#     ${data} \
#     --path=${path} \
#     --user-dir=${user_dir} \
#     --task=refcoco \
#     --batch-size=16 \
#     --log-format=simple --log-interval=10 \
#     --seed=7 \
#     --gen-subset=${split} \
#     --results-path=${result_path} \
#     --beam=5 \
#     --min-len=4 \
#     --max-len-a=0 \
#     --max-len-b=4 \
#     --no-repeat-ngram-size=3 \
#     --fp16 \
#     --num-workers=0 \
#     --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

# data=../../dataset/refcocog_data/refcocog_test.tsv
# path=../../checkpoints/refcocog_base_best.pt
# result_path=../../results/refcocog
# split='refcocog_test'
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#     ${data} \
#     --path=${path} \
#     --user-dir=${user_dir} \
#     --task=refcoco \
#     --batch-size=16 \
#     --log-format=simple --log-interval=10 \
#     --seed=7 \
#     --gen-subset=${split} \
#     --results-path=${result_path} \
#     --beam=5 \
#     --min-len=4 \
#     --max-len-a=0 \
#     --max-len-b=4 \
#     --no-repeat-ngram-size=3 \
#     --fp16 \
#     --num-workers=0 \
#     --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
