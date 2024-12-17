import sys
import time
from pathlib import Path
import torch
from typing import Optional, List, Tuple  # Add this line
import lm_eval
from lm_eval.api.model import LM
from lm_eval import evaluator
import lm_eval.tasks  # Changed this line
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tokenizer import get_tokenizer

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _load_model(checkpoint_path, device, precision, use_tp, hist_path, sparsity):
    use_cuda = 'cuda' in device
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)

    import os
    from distribution import Distribution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # hack import parent dir
    from kernels.sparse_gemv import SparseGEMV, SparseQKVGEMV, DenseGEMV
    from utils.utils import get_layer_greedy_sparsities
    sparse_levels = [sparsity]*len(model.layers)
    projs = ['q', 'k', 'v', 'o', 'up', 'gate', 'down']
    
    sparsities = {proj: sparse_levels for proj in projs}

    def monkeypatch_layer(layer_idx, layer, sparsity, hist_path, device):
        mlp_path = os.path.join(hist_path, f"layer-{layer_idx}", "mlp")
        attn_path = os.path.join(hist_path, f"layer-{layer_idx}", "self_attn")

        distrs = {}

        distrs["mlp_h1"] = Distribution(mlp_path, "h1")
        distrs["mlp_h2"] = Distribution(mlp_path, "h2")
        distrs["attn_h1"] = Distribution(attn_path, "h1")
        distrs["attn_h2"] = Distribution(attn_path, "h2")

        sparses = {}
        for proj in projs:
            val = 0.5 + 0.5*sparsities[proj][layer_idx]
            if proj in ['q', 'k', 'v']:
                sparses[proj] = distrs["attn_h1"].icdf(val).item()
            elif proj in ['o']:
                sparses[proj] = distrs["attn_h2"].icdf(val).item()
            elif proj in ['up', 'gate']:
                sparses[proj] = distrs["mlp_h1"].icdf(val).item()
            elif proj in ['down']:
                sparses[proj] = distrs["mlp_h2"].icdf(val).item()

        is_sparse = True

        layer.feed_forward.gemv1_kernel = SparseGEMV.initialize("sparse_gemv", device) if is_sparse else DenseGEMV.initialize("dense_gemv", device)
        layer.feed_forward.gemv1 = layer.feed_forward.gemv1_kernel.operator(True)
        layer.feed_forward.thresh_up = sparses["up"]
        layer.feed_forward.thresh_gate = sparses["gate"]
        layer.feed_forward.sparsity_bin = 0
        layer.feed_forward.w1.weight.data = layer.feed_forward.w1.weight.data.T.contiguous().T # column major
        layer.feed_forward.w3.weight.data = layer.feed_forward.w3.weight.data.T.contiguous().T # column major

        layer.feed_forward.gemv2_kernel = SparseGEMV.initialize("sparse_gemv", device) if is_sparse else DenseGEMV.initialize("dense_gemv", device)
        layer.feed_forward.gemv2 = layer.feed_forward.gemv2_kernel.operator(True)
        layer.feed_forward.thresh_down = sparses["down"]
        layer.feed_forward.sparsity_bin = 0
        layer.feed_forward.w2.weight.data = layer.feed_forward.w2.weight.data.T.contiguous().T # column major

        layer.attention.gemv1_kernel = SparseQKVGEMV.initialize("sparse_qkv_gemv", device) if is_sparse else DenseGEMV.initialize("dense_gemv", device)
        layer.attention.gemv1 = layer.attention.gemv1_kernel.operator(True)
        layer.attention.thresh_q = sparses["q"]
        layer.attention.thresh_k = sparses["k"]
        layer.attention.thresh_v = sparses["v"]
        layer.attention.sparsity_bin = 0
        layer.attention.wqkv.weight.data = layer.attention.wqkv.weight.data.T.contiguous().T # column major

        layer.attention.gemv2_kernel = SparseGEMV.initialize("sparse_gemv", device) if is_sparse else DenseGEMV.initialize("dense_gemv", device)
        layer.attention.gemv2 = layer.attention.gemv2_kernel.operator(True)
        layer.attention.thresh_o = sparses["o"]
        layer.attention.sparsity_bin = 0
        layer.attention.wo.weight.data = layer.attention.wo.weight.data.T.contiguous().T # column major

        # replace forwards
        layer.feed_forward.apply_monkeypatch()
        layer.attention.apply_monkeypatch()

        torch.cuda.empty_cache()

    print("Monkeypatching with activation sparsity...")
    print(model.layers[0].feed_forward.w1.weight.data.shape)
    from tp import _get_rank
    if hist_path is not None:
        device = model.layers[0].feed_forward.w1.weight.device.type
        for layer_idx, layer in enumerate(model.layers):
            monkeypatch_layer(layer_idx, layer, sparsity, hist_path, device)     
            
            
    return model.eval()

class CustomEvalWrapper(LM):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
        # Get device directly from model parameters
        self._device = next(model.parameters()).device
        print(f"Model device: {self._device}")
        
        # Setup initial caches
        print(f"Setting up caches...")
        self.model.setup_caches(max_batch_size=1, max_seq_length=self.max_length)
        
        # Ensure causal mask stays on CPU
        if hasattr(self.model, 'causal_mask'):
            self.model.causal_mask = self.model.causal_mask.cpu()
            print("Moved causal_mask to CPU")
        
        print("Initialization complete")

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_id()
    
    @property
    def max_length(self):
        return self.model.config.block_size
    
    @property
    def max_gen_toks(self):
        return 256
    
    @property
    def batch_size(self):
        return 1
        
    @property
    def device(self):
        return self._device

    def _prepare_inputs(self, input_ids):
        """Prepare inputs with correct device placement"""
        # Move input_ids to GPU
        input_ids = input_ids.to(device=self._device)
        # Keep position indices on CPU for causal mask indexing
        input_pos = torch.arange(len(input_ids), device='cpu')
        return input_ids, input_pos

    def loglikelihood(self, requests):
        results = []
        for request in requests:
            context, continuation = request.args
            
            # Encode both context and continuation
            if context == "":
                context_enc = []
                context_enc_tensor = None
            else:
                context_enc = self.tokenizer.encode(context)
                context_enc_tensor = torch.tensor(context_enc, dtype=torch.long)

            continuation_enc = self.tokenizer.encode(continuation)
            continuation_enc_tensor = torch.tensor(continuation_enc, dtype=torch.long)

            # Get logits for the continuation
            with torch.no_grad():
                if context_enc_tensor is not None:
                    input_ids = torch.cat([context_enc_tensor, continuation_enc_tensor])
                else:
                    input_ids = continuation_enc_tensor
                
                # Prepare inputs - keeping input_pos on CPU
                input_ids, input_pos = self._prepare_inputs(input_ids)
                
                logits = self.model(input_ids.unsqueeze(0), input_pos)
                
                # Calculate loss for the continuation tokens
                log_probs = torch.log_softmax(logits[0], dim=-1)
                continuation_log_probs = torch.gather(
                    log_probs[len(context_enc):],
                    dim=1,
                    index=continuation_enc_tensor.to(self._device)[None].T
                )
                
                total_log_prob = continuation_log_probs.sum().item()
                results.append((total_log_prob, True))
        
        return results

    def loglikelihood_rolling(self, requests):
        results = []
        for request in requests:
            context, = request.args
            tokens = self.tokenizer.encode(context)
            total_log_prob = 0.0
            
            for i in range(0, len(tokens), self.max_length):
                chunk_tokens = tokens[i:i + self.max_length]
                input_ids = torch.tensor(chunk_tokens, dtype=torch.long)
                input_ids, input_pos = self._prepare_inputs(input_ids)
                
                with torch.no_grad():
                    logits = self.model(input_ids.unsqueeze(0), input_pos)
                    log_probs = torch.log_softmax(logits[0], dim=-1)
                    
                    for pos in range(len(chunk_tokens) - 1):
                        next_token = chunk_tokens[pos + 1]
                        total_log_prob += log_probs[pos, next_token].item()
            
            results.append(total_log_prob)
        
        return results

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        results = []
        for request in requests:
            context, gen_kwargs = request.args
            max_tokens = gen_kwargs.get('max_tokens', 64)
            temperature = gen_kwargs.get('temperature', 0.8)
            stop_sequences = gen_kwargs.get('until', [])

            context_tokens = torch.tensor(self.tokenizer.encode(context), dtype=torch.long)
            current_tokens, position_ids = self._prepare_inputs(context_tokens)
            
            generated = []

            for _ in range(max_tokens):
                with torch.no_grad():
                    logits = self.model(current_tokens.unsqueeze(0), position_ids)
                    
                    if temperature > 0:
                        probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                    else:
                        next_token = torch.argmax(logits[0, -1]).item()
                    
                    generated.append(next_token)
                    
                    current_text = self.tokenizer.decode(generated)
                    if any(stop_seq in current_text for stop_seq in stop_sequences):
                        first_stop_idx = len(current_text)
                        for stop_seq in stop_sequences:
                            if stop_seq in current_text:
                                idx = current_text.index(stop_seq)
                                first_stop_idx = min(first_stop_idx, idx)
                        current_text = current_text[:first_stop_idx]
                        break

                    next_token_tensor = torch.tensor([next_token], dtype=torch.long, device=self._device)
                    current_tokens = torch.cat([current_tokens, next_token_tensor])
                    current_tokens, position_ids = self._prepare_inputs(current_tokens)

            results.append(self.tokenizer.decode(generated))
        
        return results
    
def main(
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    hist_path: Optional[str] = None,
    sparsity: float = 0.0,
    device: str = default_device,
    task: str = "hellaswag",
    num_fewshot: int = 0
) -> None:
    """Evaluates a pre-trained Transformer model on specified tasks using lm-evaluation-harness.
    """
    assert checkpoint_path.is_file(), checkpoint_path
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    print(f"Using device={device}")
    precision = torch.float16

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, False, hist_path, sparsity)
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)
    
    # Create evaluation wrapper
    model_wrapper = CustomEvalWrapper(model, tokenizer)
    
    # Run evaluation - modified this part
    # import ipdb
    # ipdb.set_trace()
    results = evaluator.simple_evaluate(
        model=model_wrapper,
        tasks=[task],  # Changed from task_dict to list
        num_fewshot=num_fewshot,
        batch_size=model_wrapper.batch_size,
    )
        
        
        
    # Print results
    print(f"\nResults for {task}:")
    print(evaluator.make_table(results))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate model on benchmark tasks.')
    
    parser.add_argument('--checkpoint_path', type=Path, 
                       default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
                       help='Model checkpoint path.')
    parser.add_argument('--hist_path', type=str, default=None, help='Histogram path.')
    parser.add_argument('--sparsity', type=float, default=0, help='Sparsity level.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument('--task', type=str, default="hellaswag", 
                       help='Task to evaluate on (e.g., hellaswag, arc_easy, arc_challenge)')
    parser.add_argument('--num_fewshot', type=int, default=0,
                       help='Number of examples to use for few-shot evaluation')
    
    args = parser.parse_args()
    main(
        args.checkpoint_path,
        args.hist_path,
        args.sparsity,
        args.device,
        args.task,
        args.num_fewshot
    )