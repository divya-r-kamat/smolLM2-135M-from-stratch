import os
import time
import argparse
import math
import torch
from transformers import AutoTokenizer
from model import SmolLM2, SmolLM2Config


# -----------------------------------------------------------------------------
# Simple DataLoader
# -----------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T, input_file="input.txt"):
        self.B = B
        self.T = T

        with open(input_file, "r") as f:
            text = f.read()

        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        print(f"Loaded {len(self.tokens)} tokens")
        print(f"~{len(self.tokens) // (B*T)} batches per full pass")

        self.position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.position: self.position + (B * T + 1)]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.position += B * T
        if self.position + (B * T + 1) > len(self.tokens):
            self.position = 0

        return x, y


# -----------------------------------------------------------------------------
# Learning Rate Schedule
# -----------------------------------------------------------------------------
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine decay with warmup"""
    # Warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # Cosine decay
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------
def save_checkpoint(model, optimizer, step, loss, filepath, lr_config=None):
    ckpt = {
        "step": step,
        "loss": loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.config,
        "lr_config": lr_config,  # Save LR schedule config
    }
    torch.save(ckpt, filepath)
    print(f"\nSaved checkpoint: {filepath}")


def load_checkpoint(filepath, device, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Load model weights
    model.load_state_dict(checkpoint['model_state'])

    # Load optimizer state
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        print("Optimizer state entries:", len(optimizer.state_dict()['state']))

    step = checkpoint['step']
    loss = checkpoint['loss']
    lr_config = checkpoint.get('lr_config', None)  # Get saved LR config

    print(f"\nCheckpoint loaded from {filepath}")
    print(f"Resuming from step {step} | loss {loss:.4f}")
    if lr_config:
        print(f"Original LR schedule: max_steps={lr_config['original_max_steps']}")

    return step, loss, lr_config


# -----------------------------------------------------------------------------
# Training Loop (STEPS-BASED)
# -----------------------------------------------------------------------------
def train(total_steps, ckpt_path, save_path, use_lr_schedule=True, log_interval=100):
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.set_float32_matmul_precision("high")

    # Model config
    config = SmolLM2Config(
        block_size=1024,
        vocab_size=49152,
        n_layer=30,
        n_head=9,
        n_kv_head=3,
        n_embd=576,
        intermediate_size=1536,
        head_dim=64,
    )

    model = SmolLM2(config).to(device)
    
    # LR schedule params
    max_lr = 3e-4
    min_lr = max_lr * 0.1
    warmup_steps = 100
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Data loader
    loader = DataLoaderLite(B=4, T=256)

    # Load checkpoint if provided
    start_step = 0
    original_max_steps = total_steps
    loaded_lr_config = None
    
    if ckpt_path and os.path.exists(ckpt_path):
        start_step, _, loaded_lr_config = load_checkpoint(ckpt_path, device, model, optimizer)
        
        # Use the original schedule's max_steps if available
        if loaded_lr_config and 'original_max_steps' in loaded_lr_config:
            original_max_steps = loaded_lr_config['original_max_steps']
            print(f"Continuing with original LR schedule (original max_steps: {original_max_steps})")

    # Store LR config for saving
    lr_config = {
        'original_max_steps': original_max_steps,
        'max_lr': max_lr,
        'min_lr': min_lr,
        'warmup_steps': warmup_steps,
    }

    print(f"\nTraining from step {start_step} → {total_steps}")
    print(f"LR schedule: {'enabled' if use_lr_schedule else 'disabled'} (max={max_lr}, min={min_lr})")
    print(f"Using max_steps={original_max_steps} for LR calculation\n")

    # Main training loop
    for step in range(start_step, total_steps):
        t0 = time.time()

        # Update learning rate using ORIGINAL max_steps
        if use_lr_schedule:
            lr = get_lr(step, warmup_steps, original_max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = optimizer.param_groups[0]['lr']

        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        tok_per_sec = (loader.B * loader.T) / (t1 - t0)

        if step % log_interval == 0:
            print(f"step {step} | loss {loss.item():.4f} | lr {lr:.6f} | tok/s {tok_per_sec:8.1f}")
        
        # Save checkpoint every 1000 steps
        if step % 1000 == 0 and step > 0:
            ckpt_name = f"checkpoint_step_{step}.pt"
            save_checkpoint(model, optimizer, step, loss.item(), ckpt_name, lr_config)

    print(f"\nFinal loss: {loss.item():.4f}")
    save_checkpoint(model, optimizer, total_steps, loss.item(), save_path, lr_config)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmolLM2 (steps-based)")

    parser.add_argument("--steps", type=int, required=True,
                        help="Total steps to train to (not incremental)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint to resume from")
    parser.add_argument("--save", type=str, default="checkpoint.pt",
                        help="Where to save final checkpoint")
    parser.add_argument("--no-lr-schedule", action="store_true",
                        help="Disable learning rate schedule (use fixed LR)")
    parser.add_argument("--log-interval", type=int, default=100,
                    help="How often to print training logs")

    args = parser.parse_args()

    train(
        total_steps=args.steps,
        ckpt_path=args.ckpt,
        save_path=args.save,
        use_lr_schedule=not args.no_lr_schedule,
        log_interval=args.log_interval
    )
