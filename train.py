import os
import time
import argparse
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
# Checkpoints
# -----------------------------------------------------------------------------
def save_checkpoint(model, optimizer, step, loss, filepath):
    ckpt = {
        "step": step,
        "loss": loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.config,
    }
    torch.save(ckpt, filepath)
    print(f"\nSaved checkpoint: {filepath}")


import os
import time
import argparse
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
# Checkpoints
# -----------------------------------------------------------------------------
def save_checkpoint(model, optimizer, step, loss, filepath):
    ckpt = {
        "step": step,
        "loss": loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.config,
    }
    torch.save(ckpt, filepath)
    print(f"\nSaved checkpoint: {filepath}")


def load_checkpoint(filepath, device, model,  optimizer=None):
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

    print(f"\nCheckpoint loaded from {filepath}")
    print(f"Resuming from loss {loss:.4f}\n")

    return step, loss


# -----------------------------------------------------------------------------
# Training Loop (STEPS-BASED)
# -----------------------------------------------------------------------------
def train(total_steps, ckpt_path, save_path):
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Data loader
    loader = DataLoaderLite(B=4, T=256)

    # Load checkpoint if provided
    start_step = 0
    if ckpt_path and os.path.exists(ckpt_path):
        start_step, _ = load_checkpoint(ckpt_path, device, model, optimizer)

    print(f"Training from step {start_step} → {total_steps}\n")

    # Main training loop
    for step in range(start_step, total_steps):
        t0 = time.time()

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

        if step % 100 == 0:
            print(f"step {step} | loss {loss.item():.4f} | tok/s {tok_per_sec:8.1f}")

    print(f"\nFinal loss: {loss.item():.4f}")
    save_checkpoint(model, optimizer, total_steps, loss.item(), save_path)


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

    args = parser.parse_args()

    train(
        total_steps=args.steps,
        ckpt_path=args.ckpt,
        save_path=args.save
    )



# -----------------------------------------------------------------------------
# Training Loop (STEPS-BASED)
# -----------------------------------------------------------------------------
def train(total_steps, ckpt_path, save_path):
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Data loader
    loader = DataLoaderLite(B=4, T=256)

    # Load checkpoint if provided
    start_step = 0
    if ckpt_path and os.path.exists(ckpt_path):
        start_step, _ = load_checkpoint(ckpt_path, model, optimizer)

    print(f"Training from step {start_step} → {total_steps}\n")

    # Main training loop
    for step in range(start_step, total_steps):
        t0 = time.time()

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

        if step % 100 == 0:
            print(f"step {step} | loss {loss.item():.4f} | tok/s {tok_per_sec:8.1f}")

    print(f"\nFinal loss: {loss.item():.4f}")
    save_checkpoint(model, optimizer, total_steps, loss.item(), save_path)


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

    args = parser.parse_args()

    train(
        total_steps=args.steps,
        ckpt_path=args.ckpt,
        save_path=args.save
    )
