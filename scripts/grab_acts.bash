# Specify output path to store activations and histograms


# CUDA_VISIBLE_DEVICES=0 python teal/grab_acts.py --model_name meta-llama/Llama-2-7b-hf --output_path $OUTPUT_PATH

# LLAMA='/home/riyasatohib_cohere_com/repos/models/meta-llama/Meta-Llama-3-8B'
# COMMAND_R="/home/riyasatohib_cohere_com/repos/models/command-r"
# OUTPUT_PATH="/home/riyasatohib_cohere_com/repos/teal_clone/models/test_del"

REFRESH="/home/riyasatohib_cohere_com/repos/models/command-r-refresh"
OUTPUT_PATH="/home/riyasatohib_cohere_com/repos/teal_clone/models/command-r-refresh"

MODEL=$REFRESH

python ../teal/grab_acts.py --model_name $MODEL --output_path $OUTPUT_PATH
