MODEL_PATH='/home/riyasatohib_cohere_com/repos/models/meta-llama/Meta-Llama-3-8B'

# Run on multiple GPUs
# With tensor parallelism (no need for accelerate or torchrun)
python finetune.py \
    --model_path $MODEL_PATH \
    --output_dir "./output" \
    --batch_size 1 \
    --gradient_accumulation_steps 16
# python train.py \
#     --model_path /path/to/model \
#     --output_dir /path/to/save \
#     --batch_size 4 \
#     --use_regularization \
#     --reg_weight 1e-8