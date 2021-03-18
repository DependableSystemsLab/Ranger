#!/bin/sh
export CUDA_VISIBLE_DEVICES=  0
export LD_PRELOAD="/usr/lib/libtcmalloc.so"
checkpoint="./checkpoint/model.ckpt"

test_dataset="./val.txt" 
 
test_image_root="./fiImg/"


output_file="./"

python fi-eval.py --checkpoint $checkpoint \
    --test_dataset $test_dataset \
    --test_image_root $test_image_root \
    --output_file $output_file \
    --batch_size 1 \
    --test_iter 5 \
    --ngroups1 1 \
    --ngroups2 1 \
    --gpu_fraction 0.3 \
    --display 10 \
