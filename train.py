import os
import time
import argparse
import torch
from transformers import AutoTokenizer
from model import SmolLM2, SmolLM2Config   # keep your existing model import


# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T, input_file='input.txt'):
        self.B = B
        self.T = T

        with open(input_file, 'r') as f:
            text = f.read()

        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y


# -----------------------------------------------------------------------------
# Checkpoint Handling
# -----------------------------------------------------------------------------
def save_checkpoint(model, optimizer, step, loss, filepath):
    checkpoint = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.config,
    }
    torch.save(checkpoint, filepath)
    print(f"\nCheckpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer):
    ckpt = torch.load(filepath, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    print(f"\nCheckpoint loaded: {filepath}")
    return ckpt["step"], ckpt["loss"]


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def train_model(epochs, ckpt_path=None, save_path="checkpoint.pt"):
    # device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    torch.set_float32_matmul_precision("high")

    # model config
    config = SmolLM2Config(
        block_size=1024,
        vocab_size=49152,
        n_layer=30,
        n_head=9,
        n_kv_head=3,
        n_embd=576,
        intermediate_size=1536,
        head_dim=64
    )

    # model + optimizer
    model = SmolLM2(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # data loader
    train_loader = DataLoaderLite(B=4, T=256)
    steps_per_epoch = len(train_loader.tokens) // (train_loader.B * train_loader.T)

    # load checkpoint if provided
    start_step = 0
    last_loss = None

    if ckpt_path and os.path.exists(ckpt_path):
        start_step, last_loss = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resuming from step {start_step}, loss={last_loss}")

    print(f"\nTraining for {epochs} epochs ({epochs * steps_per_epoch} steps total)\n")

    # training
    global_step = start_step
    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} ---")

        for _ in range(steps_per_epoch):
            t0 = time.time()

            x, y = train_loader.next_batch()
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
            tok_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)

            if global_step % 100 == 0:
                print(f"step {global_step} | loss {loss.item():.4f} | tok/sec {tok_per_sec:.1f}")

            global_step += 1

    print(f"\nFinal loss: {loss.item():.4f}")
    save_checkpoint(model, optimizer, global_step, loss.item(), save_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmolLM2 model")

    parser.add_argument("--epochs", type=int, required=True,
                        help="Number of epochs to train")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save", type=str, default="checkpoint.pt",
                        help="Where to save final checkpoint")

    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        ckpt_path=args.ckpt,
        save_path=args.save
    )
