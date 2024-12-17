# if Llama - 3
MODEL_PATH='/home/riyasatohib_cohere_com/repos/models/command-r'
HIST_PATH='/home/riyasatohib_cohere_com/repos/teal_clone/models/command-r-refresh/histograms'
LOOKUP='/home/riyasatohib_cohere_com/repos/teal_clone/models/Llama-3-8B/'
SAVE='/home/riyasatohib_cohere_com/repos/models/command_r_ppl/'

RUN_FILE='/home/riyasatohib_cohere_com/repos/teal_clone/teal/ppl_test.py'
python $RUN_FILE --model_name $MODEL_PATH --teal_path $LOOKUP --hist_path $HIST_PATH --sparsity 0.5 --greedy_flag --save_path $SAVE

