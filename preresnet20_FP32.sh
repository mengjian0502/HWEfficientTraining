#!/bin/bash

log_name="./logs/preresnet20_FP32_bl.log" 
save_file_name="preresnet20_FP32_TD_bl.pth" 

python3 train.py --dataset CIFAR10 \
                --data_path ./data \
                --model PreResNet20_TD \
                --log_file $log_name \
                --save_file $save_file_name \
                --epochs=200 \
                --lr_init=0.1 \
                --wd=5e-4 \
                --weight-man 2 \
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
