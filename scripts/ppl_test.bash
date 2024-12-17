# TEAL_PATH is the output path specified in grab_acts.py

# if Llama - 2
# MODEL_PATH='/home/riyasatohib_cohere_com/repos/models/meta-llama/Llama-2-7b-hf'
# HIST_PATH='/home/riyasatohib_cohere_com/repos/teal_clone/models/Llama-2-7B/histograms'
# LOOKUP='/home/riyasatohib_cohere_com/repos/teal_clone/models/Llama-2-7B/lookup'

# if Llama - 3
MODEL_PATH='/home/riyasatohib_cohere_com/repos/models/meta-llama/Meta-Llama-3-8B'
HIST_PATH='/home/riyasatohib_cohere_com/repos/teal_clone/models/Llama-3-8B/histograms'
LOOKUP='/home/riyasatohib_cohere_com/repos/teal_clone/models/Llama-3-8B/'
SAVE='/home/riyasatohib_cohere_com/repos/models/llama-3-8B-greedy/'

RUN_FILE='/home/riyasatohib_cohere_com/repos/teal_clone/teal/ppl_test.py'
python $RUN_FILE --model_name $MODEL_PATH --teal_path $LOOKUP --hist_path $HIST_PATH --sparsity 0.5 --greedy_flag --save_path $SAVE

# MODEL_PATH='/home/riyasatohib_cohere_com/repos/models/command-r'
# HIST_PATH='/home/riyasatohib_cohere_com/repos/teal_clone/models/command-r/histograms'
# LOOKUP='/home/riyasatohib_cohere_com/repos/teal_clone/models/command-r'
# SAVE='/home/riyasatohib_cohere_com/repos/models/command-r-sparse/'
# RUN_FILE='/home/riyasatohib_cohere_com/repos/teal_clone/teal/ppl_test.py'

# # python $RUN_FILE --model_name $MODEL_PATH --teal_path $LOOKUP --hist_path $HIST_PATH --sparsity 0.5 --greedy_flag --save_path $SAVE
# python $RUN_FILE --model_name $MODEL_PATH --teal_path $LOOKUP --hist_path $HIST_PATH --save_path $SAVE
