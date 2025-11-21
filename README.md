# SmolLM2-135M from Scratch 
This repository contains a full, reverse-engineered implementation of the SmolLM2-135M language model, re-built from scratch using pure PyTorch.
The model is trained on the complete works of Shakespeare to achieve basic text-generation capability and closely reproduces the architecture of the official HuggingFace model: [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

### Overview
This project implements a modern Transformer architecture with key improvements used in current large language models (LLMs):

- Grouped Query Attention (GQA) — reduces KV head count for faster inference
- Rotary Position Embeddings (RoPE) — injects relative positions directly into Q/K vectors
- SwiGLU activation — improves MLP expressiveness while keeping stability
- RMSNorm — scale-based normalization replacing LayerNorm
- Shared input/output embeddings — ties word embeddings and final LM head
- 134.5M total parameters — matching the official SmolLM2 architecture

The model contains 134.5M parameters and was trained entirely from scratch on Shakespeare's text, without any pretrained weights.

## Dataset
The Shakespeare Dataset : We train on Shakespeare’s complete works (input.txt ≈ 1.1MB), including all plays, sonnets, and poems..

    File size: ~1.1 MB
    
      class DataLoaderLite:
          def __init__(self, B, T):
              self.B = B  # Batch size
              self.T = T  # Sequence length
              
              with open('input.txt', 'r') as f:
                  text = f.read()
              
              tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
              tokens = tokenizer.encode(text)
              self.tokens = torch.tensor(tokens, dtype=torch.long)
              
## Dataset Statistics

    ======================================================================
    DATASET DIAGNOSTICS
    ======================================================================
    Total characters: 1,115,394
    Total tokens: 338,025
    Unique tokens in dataset: 11,706 / 50257 total vocab

     First 300 characters of data:
    ----------------------------------------------------------------------
    First Citizen:
    Before we proceed any further, hear me speak.
    
    All:
    Speak, speak.
    
    First Citizen:
    You are all resolved rather to die than to famish?
    
    All:
    Resolved. resolved.
    
    First Citizen:
    First, you know Caius Marcius is chief enemy to the people.
    
    All:
    We know't, we know't.
    
    First Citizen:
    Let us
    ----------------------------------------------------------------------
    ======================================================================

## Model Architecture

    SmolLM2(
      (model): ModuleDict(
        (embed_tokens): Embedding(49152, 576)
        (layers): ModuleList(
          (0-29): 30 x Block(
            (input_layernorm): RMSNorm()
            (self_attn): GroupedQueryAttention(
              (q_proj): Linear(in_features=576, out_features=576, bias=False)
              (k_proj): Linear(in_features=576, out_features=192, bias=False)
              (v_proj): Linear(in_features=576, out_features=192, bias=False)
              (o_proj): Linear(in_features=576, out_features=576, bias=False)
              (attn_dropout): Dropout(p=0.0, inplace=False)
              (rope): RotaryPositionalEmbedding()
            )
            (post_attention_layernorm): RMSNorm()
            (mlp): SwiGLU(
              (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
              (up_proj): Linear(in_features=576, out_features=1536, bias=False)
              (down_proj): Linear(in_features=1536, out_features=576, bias=False)
            )
          )
        )
        (norm): RMSNorm()
      )
    )

## Model Configurations

    block_size: 1024          # Context window
    vocab_size: 49152         # Tokenizer vocabulary
    n_layer: 30               # Transformer blocks
    n_head: 9                 # Query heads
    n_kv_head: 3              # Key-value heads (3:1 GQA ratio)
    n_embd: 576               # Embedding dimension
    intermediate_size: 1536   # MLP hidden dimension
    head_dim: 64              # Dimension per attention head

## Parameter Breakdown

    ================================================================================
    Parameter Breakdown
    ================================================================================
    Token Embeddings                    28,311,552      21.05%
    Attention Q                          9,953,280       7.40%
    Attention K                          3,317,760       2.47%
    Attention V                          3,317,760       2.47%
    Attention O                          9,953,280       7.40%
    MLP Gate                            26,542,080      19.73%
    MLP Up                              26,542,080      19.73%
    MLP Down                            26,542,080      19.73%
    RMSNorm                                 35,136       0.03%
    Output (shared)                              0       0.00%
    
    Total Parameters                   134,515,008
    Total (Millions)                        134.52M

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/24b64a3f-3e9a-47e4-a2a6-89fdedc6cb2f" />

##  Training Results

### Performance Optimizations
Two key optimizations enable efficient training:

#### High-precision matrix multiplication

    torch.set_float32_matmul_precision("high")

#### Mixed precision training

BF16 automatic mixed precision for faster computation

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
           logits, loss = model(x, y)

These optimizations provide ~2-3x speedup while maintaining training stability.

### Initial Training (5000 steps)

    !python train.py --steps 5000

Trained from random initialization on Shakespeare's complete works (341,094 tokens):

- Dataset: Shakespeare's complete works
- Training steps: 5,000
- Batch size: 4
- Sequence length: 256 tokens
- Initial loss: 10.98 → Final loss: 2.09
- Training speed: ~2,180 tokens/sec
- Learning rate: Cosine decay (3e-4 → 3e-5) with 100-step warmup

      Using device: cuda
      tokenizer_config.json: 3.66kB [00:00, 2.42MB/s]
      vocab.json: 801kB [00:00, 11.5MB/s]
      merges.txt: 466kB [00:00, 49.8MB/s]
      tokenizer.json: 2.10MB [00:00, 71.2MB/s]
      special_tokens_map.json: 100% 831/831 [00:00<00:00, 7.83MB/s]
      Token indices sequence length is longer than the specified maximum sequence length for this model (341094 > 8192). Running this sequence through the model will result in indexing errors
      Loaded 341094 tokens
      ~333 batches per full pass
      
      Training from step 0 → 5000
      LR schedule: enabled (max=0.0003, min=2.9999999999999997e-05)
      Using max_steps=5000 for LR calculation
      
      step 0 | loss 10.9773 | lr 0.000003 | tok/s    750.4
      step 100 | loss 6.4851 | lr 0.000300 | tok/s   2254.1
      step 200 | loss 5.4612 | lr 0.000300 | tok/s   2125.8
      step 300 | loss 5.0673 | lr 0.000299 | tok/s   2197.1
      step 400 | loss 4.9364 | lr 0.000298 | tok/s   2192.0
      step 500 | loss 5.1199 | lr 0.000296 | tok/s   2184.1
      step 600 | loss 5.0958 | lr 0.000293 | tok/s   2187.4
      step 700 | loss 4.9131 | lr 0.000290 | tok/s   2168.6
      step 800 | loss 5.1765 | lr 0.000287 | tok/s   2164.6
      step 900 | loss 5.0306 | lr 0.000283 | tok/s   2189.3
      step 1000 | loss 4.7559 | lr 0.000278 | tok/s   2171.2
      
      Saved checkpoint: checkpoint_step_1000.pt
      step 1100 | loss 4.3848 | lr 0.000273 | tok/s   2205.7
      step 1200 | loss 4.0042 | lr 0.000268 | tok/s   2141.3
      step 1300 | loss 4.0868 | lr 0.000262 | tok/s   2203.6
      step 1400 | loss 3.3788 | lr 0.000256 | tok/s   2183.6
      step 1500 | loss 4.4099 | lr 0.000249 | tok/s   2175.0
      step 1600 | loss 4.2747 | lr 0.000242 | tok/s   2181.3
      step 1700 | loss 3.5355 | lr 0.000235 | tok/s   2161.6
      step 1800 | loss 4.0896 | lr 0.000227 | tok/s   2178.4
      step 1900 | loss 3.9698 | lr 0.000220 | tok/s   2183.1
      step 2000 | loss 4.3216 | lr 0.000212 | tok/s   2175.2
      
      Saved checkpoint: checkpoint_step_2000.pt
      step 2100 | loss 3.7819 | lr 0.000203 | tok/s   2219.8
      step 2200 | loss 3.4133 | lr 0.000195 | tok/s   2119.5
      step 2300 | loss 3.6641 | lr 0.000187 | tok/s   2208.2
      step 2400 | loss 3.2863 | lr 0.000178 | tok/s   2189.6
      step 2500 | loss 3.3747 | lr 0.000169 | tok/s   2189.3
      step 2600 | loss 3.1533 | lr 0.000161 | tok/s   2187.1
      step 2700 | loss 3.2520 | lr 0.000152 | tok/s   2188.3
      step 2800 | loss 3.2419 | lr 0.000143 | tok/s   2193.7
      step 2900 | loss 3.0943 | lr 0.000135 | tok/s   2174.8
      step 3000 | loss 3.2204 | lr 0.000127 | tok/s   2159.7
      
      Saved checkpoint: checkpoint_step_3000.pt
      step 3100 | loss 3.7422 | lr 0.000118 | tok/s   2219.2
      step 3200 | loss 2.8041 | lr 0.000110 | tok/s   2132.0
      step 3300 | loss 3.0445 | lr 0.000103 | tok/s   2195.3
      step 3400 | loss 2.5220 | lr 0.000095 | tok/s   2180.8
      step 3500 | loss 2.6858 | lr 0.000088 | tok/s   2189.1
      step 3600 | loss 2.2979 | lr 0.000081 | tok/s   2167.2
      step 3700 | loss 2.4174 | lr 0.000074 | tok/s   2190.7
      step 3800 | loss 3.0252 | lr 0.000068 | tok/s   2195.6
      step 3900 | loss 2.5828 | lr 0.000062 | tok/s   2189.7
      step 4000 | loss 2.6368 | lr 0.000057 | tok/s   2176.3
      
      Saved checkpoint: checkpoint_step_4000.pt
      step 4100 | loss 2.3251 | lr 0.000052 | tok/s   2216.6
      step 4200 | loss 1.8115 | lr 0.000047 | tok/s   2132.4
      step 4300 | loss 1.8660 | lr 0.000043 | tok/s   2186.4
      step 4400 | loss 1.7019 | lr 0.000040 | tok/s   2177.4
      step 4500 | loss 1.9442 | lr 0.000037 | tok/s   2177.1
      step 4600 | loss 2.0697 | lr 0.000034 | tok/s   2170.6
      step 4700 | loss 1.4722 | lr 0.000032 | tok/s   2182.9
      step 4800 | loss 2.0765 | lr 0.000031 | tok/s   2166.9
      step 4900 | loss 1.9064 | lr 0.000030 | tok/s   2181.7
      step 5000 | loss 1.9470 | lr 0.000030 | tok/s   2181.7
      
      Saved checkpoint: checkpoint.pt


### Extended Training (+50 steps)

Clear the cache and Resumed from checkpoint for 50 more steps:

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


- Continued from: Step 5001
- Additional steps: 50
- Final loss: 1.53
- Maintained LR schedule: 3e-5 (min LR from original schedule)

      !python train.py --steps 5050 --ckpt checkpoint.pt --save checkpoint_5050.pt --log-interval 5


      Using device: cuda
      Token indices sequence length is longer than the specified maximum sequence length for this model (341094 > 8192). Running this sequence through the model will result in indexing errors
      Loaded 341094 tokens
      ~333 batches per full pass
      Optimizer state entries: 272
      
      Checkpoint loaded from checkpoint.pt
      Resuming from step 5001 | loss 2.0870
      Original LR schedule: max_steps=5000
      Continuing with original LR schedule (original max_steps: 5000)
      
      Training from step 5001 → 5050
      LR schedule: enabled (max=0.0003, min=2.9999999999999997e-05)
      Using max_steps=5000 for LR calculation
      
      step 5001 | loss 1.9662 | lr 0.000030 | tok/s    301.9
      
      Saved checkpoint: checkpoint_step_5001.pt
      step 5005 | loss 1.7517 | lr 0.000030 | tok/s   2142.9
      step 5010 | loss 1.6168 | lr 0.000030 | tok/s   2150.3
      step 5015 | loss 1.7878 | lr 0.000030 | tok/s   2114.4
      step 5020 | loss 1.3299 | lr 0.000030 | tok/s   2104.3
      step 5025 | loss 1.3462 | lr 0.000030 | tok/s   2079.6
      step 5030 | loss 1.2582 | lr 0.000030 | tok/s   2052.7
      step 5035 | loss 1.4290 | lr 0.000030 | tok/s   2048.8
      step 5040 | loss 1.7178 | lr 0.000030 | tok/s   1693.5
      step 5045 | loss 1.5445 | lr 0.000030 | tok/s   1964.6
      
      Final loss: 1.5272
      
      Saved checkpoint: checkpoint_5050.pt

### Technical Details

- Precision: BFloat16 for forward pass, FP32 for optimizer
- Hardware: Requires CUDA GPU with BF16 support (recommended: Ampere or newer)
- Memory: ~4-5GB GPU memory for batch_size=4, seq_len=256
- Training time: ~40 minutes for 5,000 steps on modern GPU

### Estimate the number of FLOPs using the method in the PaLM paper by Chowdhery, et al.

FLOPs = 6NT + 12LHdT²

Where:

6NT: Linear operations (embeddings, projections, MLP) 12LHdT²: Quadratic attention operations

Total Flops: 220949544960


## Demo
You can try the trained model interactively on Hugging Face Spaces: 👉 https://huggingface.co/spaces/dkamat/gpt2-from-stratch

<img width="1794" height="860" alt="image" src="https://github.com/user-attachments/assets/59c20a46-4eab-4942-8939-74cd38c76c11" />
