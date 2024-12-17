import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import wandb
from datetime import datetime
import argparse
from tqdm import tqdm
from typing import Dict, Optional

def is_main_process():
    """Check if this is the main process"""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return local_rank == 0

class WandBLogger:
    def __init__(self, project_name: str, run_name: str, config: Dict):
        # Only initialize wandb on the main process
        if is_main_process():
            self.run = wandb.init(
                project=project_name,
                name=run_name,
                config=config
            )
        self.is_main = is_main_process()

    def log_training_step(self, step: int, epoch: int, loss: float, lr: Optional[float] = None):
        if not self.is_main:
            return
            
        metrics = {
            "train/loss": loss,
            "train/epoch": epoch,
            "train/step": step,
        }
        if lr is not None:
            metrics["train/learning_rate"] = lr
        wandb.log(metrics, step=step)

def get_model_and_tokenizer(args):
    print(f"Loading model from {args.model_path}...")
    
    # Load model with tensor parallelism
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def prepare_batch(batch, tokenizer, max_length=512):
    texts = []
    for item in batch:
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item["output"]
        
        if input_text:
            full_text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
        else:
            full_text = f"Instruction: {instruction}\nOutput: {output}"
        texts.append(full_text)
    
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": encodings["input_ids"].clone()
    }

def train(args, model, train_dataloader, optimizer, scheduler, wandb_logger=None):
    model.train()
    total_steps = 0
    if is_main_process():
        progress_bar = tqdm(range(args.num_epochs * len(train_dataloader)))
    
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Check for NaN loss
            if torch.isnan(loss).any():
                if is_main_process():
                    print(f"Warning: NaN loss detected at step {total_steps}")
                continue
            
            # Backward pass
            (loss / args.gradient_accumulation_steps).backward()
            
            # Optimizer step with gradient accumulation
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            if wandb_logger and step % args.log_every == 0:
                wandb_logger.log_training_step(
                    step=total_steps,
                    epoch=epoch,
                    loss=loss.item(),
                    lr=scheduler.get_last_lr()[0]
                )
            
            if is_main_process():
                progress_bar.update(1)
            total_steps += 1
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)  # Reduced batch size
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)  # Increased gradient accumulation
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--log_every', type=int, default=10)
    
    # Wandb parameters
    parser.add_argument('--wandb_project', type=str, default="llama-finetuning")
    parser.add_argument('--disable_wandb', action='store_true')
    
    args = parser.parse_args()
    
    # Set memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

    # Initialize wandb only on main process
    wandb_logger = None
    if not args.disable_wandb and is_main_process():
        run_name = f"llama-ft-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_logger = WandBLogger(
            project_name=args.wandb_project,
            run_name=run_name,
            config=vars(args)
        )

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args)

    # Load dataset
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    # Create dataloader
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: prepare_batch(batch, tokenizer, args.max_length)
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_dataloader) * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Train
    model = train(
        args=args,
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        wandb_logger=wandb_logger
    )

    # Save model only on main process
    if is_main_process():
        print("Saving model...")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()