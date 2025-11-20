import os
import time
import torch
import tiktoken
#from model import SmolLM2, SmolLM2Config
from transformers import AutoTokenizer, AutoModelForCausalLM


# Clear any existing CUDA errors
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Set matmul precision for faster training on modern GPUs
torch.set_float32_matmul_precision('high')

# ============================================================================
# DataLoader
# ============================================================================
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


# ============================================================================
# Checkpoint utilities
# ============================================================================

def save_checkpoint(model, optimizer, step, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.config,
    }
    torch.save(checkpoint, filepath)
    print(f"\nCheckpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    loss = checkpoint['loss']
    print(f"\nCheckpoint loaded from {filepath}")
    print(f"Resuming from step {step}, loss {loss:.4f}\n")
    return step, loss


# ============================================================================
# Setup
# ============================================================================

# Device setup
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

# Model config
config = SmolLM2Config(
    block_size=1024,
    vocab_size=49152,
    n_layer=30,  # Smaller for faster training
    n_head=9,
    n_kv_head=3,
    n_embd=576,
    intermediate_size=1536,
    head_dim=64,
)

# Initialize model
model = SmolLM2(config)
model.to(device)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params/1e6:.2f}M\n")

# Initialize dataloader
train_loader = DataLoaderLite(B=4, T=256)

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

# ============================================================================
# PHASE 1: Train for 5000 steps
# ============================================================================

print("=" * 80)
print("PHASE 1: Training for 5000 steps")
print("=" * 80)

max_steps = 5000
checkpoint_path = 'smollm2_checkpoint_5000.pt'

for step in range(max_steps):
    t0 = time.time()
    
    # Get batch
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    
    # Forward pass with autocast for mixed precision
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Optimizer step
    optimizer.step()
    
    # Synchronize for accurate timing
    if device == 'cuda':
        torch.cuda.synchronize()
    
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    
    # Print progress
    if step % 100 == 0 or step == max_steps - 1:
        print(f'step {step:4d} | loss: {loss.item():.4f} | dt: {dt:6.2f}ms | tok/sec: {tokens_per_sec:8.2f}')

print(f'\nFinal loss: {loss.item():.4f}')

# Save checkpoint
save_checkpoint(model, optimizer, max_steps, loss.item(), checkpoint_path)

print("=" * 80)
print("PHASE 1 Complete!")
print("=" * 80)
