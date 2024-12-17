#!/bin/bash

Llama3_8B='/home/riyasatohib_cohere_com/repos/models/meta-llama/Meta-Llama-3-8B'
HIST_PATH='/home/riyasatohib_cohere_com/repos/teal_clone/regularization/reg_hist'

python act_reg.py \
    --output_dir ./out_dir \
    --reg_types l1 hoyer \
    --reg_weights 0.01 0.01 \
    --batch_size 4