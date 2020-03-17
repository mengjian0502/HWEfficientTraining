#!/bin/bash
block_size=${1:-8}
gamma=${2:-0.75}
alpha=${3:-1.0}

log_name="./logs/resnet32_FP8_TD_${block_size}_${gamma}_${alpha}_eval.log" 
save_file_name="resnet32_FP8_TD_${block_size}_${gamma}_${alpha}_eval.pth" 

eval_path="./checkpoint/resnet32_FP8_TD_${block_size}_0.0_0.0_${gamma}_0.99.pth"

python train.py --dataset CIFAR10 \
                --data_path ./data \
                --model ResNet32LP_TD \
                --log_file $log_name \
                --save_file $save_file_name \
                --block_size $block_size \
                --TD_gamma $gamma \
                --TD_alpha $alpha \
                --epochs=200 \
                --lr_init=0.05 \
                --wd=5e-4 \
                --weight-man 2 \
                --evaluate $eval_path \
                --grad-man 2 \
                --momentum-man 9 \
                --activate-man 2 \
                --error-man 2 \
                --acc-man 9 \
                --weight-rounding nearest \
                --grad-rounding nearest \
                --momentum-rounding stochastic \
                --activate-rounding nearest \
                --error-rounding nearest \
                --acc-rounding stochastic \
                --weight-exp 5 \
                --grad-exp 5 \
                --momentum-exp 6 \
                --activate-exp 5 \
                --error-exp 5 \
                --acc-exp 6 \
                --batch_size 128;
