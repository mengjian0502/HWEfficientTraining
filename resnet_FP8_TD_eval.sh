#!/bin/bash
block_size=${1:-4}
gamma=${2:-0.9375}
alpha=${3:-1.0}

log_name="./logs/resnet20_FP8_TD_${block_size}_${gamma}_${alpha}_${gamma_final}_${alpha_final}_sparse_eval.log" 
save_file_name="resnet20_FP8_TD_${block_size}_${gamma}_${alpha}_${gamma_final}_${alpha_final}_sparse_eval.pth" 

eval_path="./checkpoint/sparsity_analysis/resnet20_FP8_TD_4_0.0_0.0_0.9375_0.99_5.0_0.0_0_0.pth"

python train1.py --dataset CIFAR10 \
                --data_path ./data \
                --model ResNet20LP_TD \
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
