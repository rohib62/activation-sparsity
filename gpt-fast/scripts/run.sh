export CUDA_VISIBLE_DEVICES=4,5,6,7 
# MODEL_PATH='/home/riyasatohib_cohere_com/repos/models/meta-llama/Llama-2-7b-hf/model.pth'
MODEL_PATH='/home/riyasatohib_cohere_com/repos/models/meta-llama/Meta-Llama-3-8B/model.pth'
HIST_PATH='../models/Llama-3-8B/histograms'

echo "Llama-3-8B"
for sparsity in 0.5 0.6 0.7 0.8 0.9; do
  for i in {1..3}; do
    python generate.py --compile --checkpoint_path $MODEL_PATH --hist_path $HIST_PATH --sparsity $sparsity # --interactive
  done    
done

