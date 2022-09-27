#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1092

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data=../../alldata/caption_data/caption_test.tsv
path=/home/dongk/dkgroup/tsk/projects/OFA/run_scripts/caption/checkpoints/stage2_checkpoints/shallow_deep_freeze_stage2_{3,}/checkpoint.best_cider_1.4380.pt
result_path=../../results/caption
selected_cols=1,4,2
split='test'

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=caption \
    --batch-size=1 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=16 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"
python coco_eval.py ../../results/caption/test_predict.json ../../alldata/caption_data/test_caption_coco_format.json
