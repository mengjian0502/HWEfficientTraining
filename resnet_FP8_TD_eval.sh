#!/bin/bash
block_size=${1:-8}
gamma=${2:-0.75}
alpha=${3:-1.0}
ramping_power=${6:--5.0}
lambda_BN=${7:-1e-4}
init_BN_bias=${8:-0}
gradient_gamma=${9:-0}

log_name="./logs/resnet20_FP8_TD_${block_size}_${gamma}_${alpha}_${gamma_final}_${alpha_final}_${ramping_power}_${lambda_BN}_${init_BN_bias}_${gradient_gamma}_eval.log" 
save_file_name="resnet20_FP8_TD_${block_size}_${gamma}_${alpha}_${gamma_final}_${alpha_final}_${ramping_power}_${lambda_BN}_${init_BN_bias}_${gradient_gamma}_eval.pth" 

eval_path="./checkpoint/resnet20_FP8_TD_${block_size}_${gamma}_${alpha}_${gamma_final}_${alpha_final}_${ramping_power}_${lambda_BN}_${init_BN_bias}_${gradient_gamma}.pth"

python train.py --dataset CIFAR10 \
                --data_path ./data \
                --model ResNet20LP_TD \
                --log_file $log_name \
                --save_file $save_file_name \
                --block_size $block_size \
                --TD_gamma $gamma \
                --TD_alpha $alpha \
                --ramping_power $ramping_power \
                --lambda_BN $lambda_BN \
                --init_BN_bias $init_BN_bias \
                --gradient_gamma $gradient_gamma \
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
