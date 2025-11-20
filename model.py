import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    Unlike LayerNorm which normalizes using mean and variance, RMSNorm only uses
    the root mean square. This is simpler, faster, and works just as well in practice.
    
    Formula: RMSNorm(x) = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    
    Args:
        dim: The dimension of the input features
        eps: Small constant for numerical stability (prevents division by zero)
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (one per feature dimension)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS (Root Mean Square) across the last dimension
        # x shape: (batch, seq_len, dim)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize by dividing by RMS
        x_norm = x / rms
        
        # Scale by learnable weight parameter
        return self.weight * x_norm


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    RoPE encodes position information by rotating the query and key vectors in the
    attention mechanism. Unlike absolute position embeddings, RoPE naturally encodes
    relative positions and can extrapolate to longer sequences.
    
    How it works:
    1. Split each head dimension into pairs of values (treat as 2D vectors)
    2. Rotate each pair by an angle that depends on the position in the sequence
    3. Different dimensions rotate at different frequencies (like Fourier series)
    
    This creates a position-dependent transformation that makes nearby positions
    have similar representations and distant positions have different ones.
    
    Args:
        dim: Dimension per attention head (must be even)
        max_seq_len: Maximum sequence length to precompute frequencies for
        theta: Base value for frequency calculation (higher = lower frequencies)
    """
    def __init__(self, dim, max_seq_len=8192, theta=100000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute the rotation frequencies
        # These are stored as a buffer (not trained, but moved with the model)
        freqs_cis = self._precompute_freqs_cis()
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
    
    def _precompute_freqs_cis(self):
        """
        Precompute the complex exponential frequencies for RoPE
        
        Returns:
            freqs_cis: Complex-valued frequencies of shape (max_seq_len, dim//2)
        """
        # Calculate frequencies for each dimension pair
        # Lower dimensions rotate faster, higher dimensions rotate slower
        # Formula: freq_i = 1 / (theta ^ (2i / dim)) for i in [0, dim/2)
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # Create position indices [0, 1, 2, ..., max_seq_len-1]
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        
        # Compute outer product: each position gets rotated by each frequency
        # Shape: (max_seq_len, dim//2)
        freqs = torch.outer(t, freqs)
        
        # Convert to complex exponentials: e^(i*theta) = cos(theta) + i*sin(theta)
        # Using torch.polar(magnitude, angle) creates complex numbers
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        return freqs_cis
    
    def forward(self, x, start_pos=0):
        """
        Apply rotary embeddings to input tensor
        
        Args:
            x: Input tensor of shape (batch, n_heads, seq_len, head_dim)
            start_pos: Starting position (used for caching in inference)
        
        Returns:
            Rotated tensor of same shape as input
        """
        batch, n_heads, seq_len, head_dim = x.shape
        
        # Get the frequencies for this sequence length
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        
        # Reshape input to treat consecutive pairs as complex numbers
        # (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim//2, 2)
        x_reshaped = x.float().reshape(batch, n_heads, seq_len, -1, 2)
        
        # View pairs as complex numbers: [a, b] -> a + bi
        # Shape: (B, n_heads, T, head_dim//2)
        x_complex = torch.view_as_complex(x_reshaped)
        
        # Reshape freqs_cis to broadcast correctly
        # (T, head_dim//2) -> (1, 1, T, head_dim//2)
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation by complex multiplication
        # Multiplying by e^(i*theta) rotates by angle theta
        x_rotated = x_complex * freqs_cis
        
        # Convert back to real representation
        # (B, n_heads, T, head_dim//2) -> (B, n_heads, T, head_dim//2, 2)
        x_out = torch.view_as_real(x_rotated)
        
        # Flatten last two dimensions back to head_dim
        # (B, n_heads, T, head_dim//2, 2) -> (B, n_heads, T, head_dim)
        x_out = x_out.reshape(batch, n_heads, seq_len, head_dim)
        
        # Return in original dtype
        return x_out.type_as(x)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    
    Standard multi-head attention has separate key and value heads for each query head,
    which is expensive in memory. Multi-Query Attention (MQA) uses just one KV head for
    all query heads, which is fast but may hurt quality.
    
    GQA is a middle ground: multiple query heads share each KV head in groups.
    For example, with 9 query heads and 3 KV heads, each KV head serves 3 query heads.
    
    Benefits:
    - Reduces KV cache size (important for long sequences)
    - Maintains most of the quality of full multi-head attention
    - Speeds up inference with longer contexts
    
    Args:
        config: Model configuration containing dimensions and head counts
    """
    def __init__(self, config):
        super().__init__()
        # Ensure embedding dimension is divisible by number of heads
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head  # Number of query heads (e.g., 9)
        self.n_kv_head = config.n_kv_head  # Number of key-value heads (e.g., 3)
        self.head_dim = config.head_dim  # Dimension per head (e.g., 64)
        self.n_embd = config.n_embd  # Total embedding dimension (e.g., 576)
        
        # Calculate how many query heads share each KV head
        self.n_rep = self.n_head // self.n_kv_head  # e.g., 9 // 3 = 3
        
        # Linear projections for queries, keys, and values
        # Q has one head per query head, K and V have fewer heads (grouped)
        self.q_proj = nn.Linear(
            config.n_embd, 
            config.n_head * config.head_dim, 
            bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.n_embd, 
            config.n_kv_head * config.head_dim, 
            bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.n_embd, 
            config.n_kv_head * config.head_dim, 
            bias=config.attention_bias
        )
        
        # Output projection to combine all heads back to embedding dimension
        self.o_proj = nn.Linear(
            config.n_head * config.head_dim, 
            config.n_embd, 
            bias=config.attention_bias
        )
        
        # Dropout for regularization
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        
        # Create causal mask (lower triangular matrix)
        # This ensures tokens can only attend to previous tokens (autoregressive)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )
        
        # Initialize RoPE for position encoding
        self.rope = RotaryPositionalEmbedding(
            config.head_dim,
            config.block_size,
            config.rope_theta
        )
        
    def forward(self, x):
        """
        Forward pass for grouped query attention
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)
        
        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimension
        
        # Project input to queries, keys, and values
        q = self.q_proj(x)  # (B, T, n_head * head_dim)
        k = self.k_proj(x)  # (B, T, n_kv_head * head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_head * head_dim)
        
        # Reshape to separate heads
        # (B, T, n_head * head_dim) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        # Apply rotary position embeddings to queries and keys
        # This encodes position information without adding extra parameters
        q = self.rope(q)
        k = self.rope(k)
        
        # Expand K and V to match number of query heads (GQA mechanism)
        # Each KV head is replicated n_rep times to serve multiple query heads
        if self.n_rep > 1:
            # Add a new dimension and expand
            # (B, n_kv_head, T, head_dim) -> (B, n_kv_head, n_rep, T, head_dim)
            k = k.unsqueeze(2).expand(B, self.n_kv_head, self.n_rep, T, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.n_kv_head, self.n_rep, T, self.head_dim)
            
            # Reshape to (B, n_head, T, head_dim)
            k = k.reshape(B, self.n_head, T, self.head_dim)
            v = v.reshape(B, self.n_head, T, self.head_dim)
        
        # Compute attention scores
        # Q @ K^T gives us the similarity between each query and key
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask: set future positions to -inf so they get 0 probability
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax to get attention probabilities
        att = F.softmax(att, dim=-1)
        
        # Apply dropout for regularization
        att = self.attn_dropout(att)
        
        # Apply attention weights to values
        # This gives us a weighted combination of values based on attention scores
        y = att @ v  # (B, n_head, T, head_dim)
        
        # Concatenate all heads back together
        # (B, n_head, T, head_dim) -> (B, T, n_head, head_dim) -> (B, T, n_head * head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        
        # Final output projection
        y = self.o_proj(y)
        
        return y


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) Feedforward Network
    
    This is a variant of the feedforward network used in transformers.
    Instead of the standard FFN: FFN(x) = W2 * activation(W1 * x)
    
    SwiGLU uses a gating mechanism:
    SwiGLU(x) = (Swish(W_gate * x) ⊙ (W_up * x)) * W_down
    
    Where:
    - Swish(x) = x * sigmoid(x) (also called SiLU)
    - ⊙ represents element-wise multiplication (gating)
    
    This is more expressive and often performs better than standard FFN.
    
    Args:
        config: Model configuration containing dimensions
    """
    def __init__(self, config):
        super().__init__()
        # Three linear projections:
        # 1. gate_proj: used with swish activation for gating
        # 2. up_proj: parallel projection that gets gated
        # 3. down_proj: projects back to model dimension
        self.gate_proj = nn.Linear(
            config.n_embd, 
            config.intermediate_size, 
            bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.n_embd, 
            config.intermediate_size, 
            bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, 
            config.n_embd, 
            bias=config.mlp_bias
        )

    def forward(self, x):
        """
        Forward pass through SwiGLU
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)
        
        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        # Apply swish (SiLU) activation to gate projection
        gate = F.silu(self.gate_proj(x))
        
        # Compute parallel up projection
        up = self.up_proj(x)
        
        # Element-wise multiplication (gating) and project back down
        return self.down_proj(gate * up)


class Block(nn.Module):
    """
    Transformer Block with Pre-Normalization
    
    This is one layer of the transformer. It consists of:
    1. Multi-head attention (with RMSNorm before)
    2. Feedforward network (with RMSNorm before)
    
    Both use residual connections (add input to output).
    
    Pre-norm architecture (used here):
    x = x + Attention(Norm(x))
    x = x + FFN(Norm(x))
    
    This is more stable than post-norm and is standard in modern LLMs.
    
    Args:
        config: Model configuration
    """
    def __init__(self, config):
        super().__init__()
        # Normalization before attention
        self.input_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        
        # Grouped query attention mechanism
        self.self_attn = GroupedQueryAttention(config)
        
        # Normalization before feedforward
        self.post_attention_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        
        # SwiGLU feedforward network
        self.mlp = SwiGLU(config)

    def forward(self, x):
        """
        Forward pass through transformer block
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)
        
        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        # Attention block with residual connection
        # Normalize -> Attend -> Add to input
        x = x + self.self_attn(self.input_layernorm(x))
        
        # Feedforward block with residual connection
        # Normalize -> FFN -> Add to input
        x = x + self.mlp(self.post_attention_layernorm(x))
        
        return x


@dataclass
class SmolLM2Config:
    """
    Configuration for SmolLM2 model
    
    This defines all the hyperparameters needed to construct the model.
    Default values are for SmolLM2-135M variant.
    
    Key parameters:
    - block_size: Maximum sequence length the model can handle
    - vocab_size: Number of unique tokens in vocabulary
    - n_layer: Number of transformer blocks (depth of model)
    - n_head: Number of query heads in attention
    - n_kv_head: Number of key-value heads (for grouped query attention)
    - n_embd: Dimension of embeddings and hidden states
    - intermediate_size: Hidden dimension in feedforward network (usually 2-4x n_embd)
    """
    block_size: int = 8192  # max sequence length
    vocab_size: int = 49152  # number of tokens
    n_layer: int = 30  # number of transformer layers
    n_head: int = 9  # number of query heads
    n_kv_head: int = 3  # number of key-value heads (GQA)
    n_embd: int = 576  # embedding dimension
    intermediate_size: int = 1536  # MLP hidden dimension
    head_dim: int = 64  # dimension per attention head
    rms_norm_eps: float = 1e-5  # epsilon for numerical stability in RMSNorm
    rope_theta: float = 100000.0  # base for RoPE frequency calculation
    attention_bias: bool = False  # whether to use bias in attention projections
    mlp_bias: bool = False  # whether to use bias in MLP layers
    attention_dropout: float = 0.0  # dropout rate in attention
    tie_word_embeddings: bool = True  # share weights between input and output embeddings


class SmolLM2(nn.Module):
    """
    SmolLM2 Language Model
    
    This is the main model class that puts everything together:
    1. Token embeddings: Convert token IDs to vectors
    2. Transformer layers: Process sequences with attention and feedforward
    3. Output projection: Convert final hidden states to vocabulary logits
    
    The model is trained to predict the next token in a sequence (language modeling).
    
    Args:
        config: SmolLM2Config containing all hyperparameters
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Main model components
        self.model = nn.ModuleDict(dict(
            # Token embedding: maps token IDs to vectors
            embed_tokens = nn.Embedding(config.vocab_size, config.n_embd),
            
            # Stack of transformer blocks
            layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            
            # Final normalization layer
            norm = RMSNorm(config.n_embd, eps=config.rms_norm_eps),
        ))
        
        # Output projection to vocabulary
        # If tie_word_embeddings is True, we reuse the input embedding weights
        # This reduces parameters and often works just as well
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight in forward()
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize all weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize model weights
        
        Uses careful initialization to help with training stability:
        - Normal distribution with small standard deviation
        - Special scaling for residual connections (helps with deep networks)
        """
        if isinstance(module, nn.Linear):
            # Standard deviation for weight initialization
            std = 0.02
            
            # Scale down weights in residual paths for stability
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            
            # Initialize weights from normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            
            # Initialize biases to zero if they exist
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights from normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        Forward pass through the model
        
        Args:
            idx: Input token IDs of shape (batch, seq_len)
            targets: Target token IDs for computing loss (batch, seq_len)
                    If None, only returns logits without computing loss
        
        Returns:
            logits: Predicted logits for each token (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.size()
        
        # Ensure sequence length doesn't exceed model capacity
        assert T <= self.config.block_size, \
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # Get token embeddings
        # (B, T) -> (B, T, n_embd)
        x = self.model.embed_tokens(idx)
        
        # Pass through all transformer blocks
        # Each block applies attention and feedforward with residual connections
        for block in self.model.layers:
            x = block(x)
        
        # Apply final layer normalization
        x = self.model.norm(x)
        
        # Project to vocabulary to get logits
        if self.config.tie_word_embeddings:
            # Reuse input embedding weights (transposed)
            logits = F.linear(x, self.model.embed_tokens.weight)
        else:
            # Use separate output projection
            logits = self.lm_head(x)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten batch and sequence dimensions for cross_entropy
            # (B, T, vocab_size) -> (B*T, vocab_size)
            # (B, T) -> (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
        
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type='HuggingFaceTB/SmolLM2-135M'):
        """
        Load pretrained SmolLM2 model weights from HuggingFace
        
        This allows you to use pre-trained models without training from scratch.
        
        Args:
            model_type: HuggingFace model identifier
        
        Returns:
            Loaded model with pretrained weights
        """
        from transformers import AutoModelForCausalLM
        print(f"Loading weights from pretrained model: {model_type}")
        
        # Load model from HuggingFace
        model_hf = AutoModelForCausalLM.from_pretrained(model_type)
        
        # Create our config from HuggingFace config
        config = SmolLM2Config(
            vocab_size=model_hf.config.vocab_size,
            block_size=model_hf.config.max_position_embeddings,
            n_layer=model_hf.config.num_hidden_layers,
            n_head=model_hf.config.num_attention_heads,
            n_kv_head=model_hf.config.num_key_value_heads,
            n_embd=model_hf.config.hidden_size,
            intermediate_size=model_hf.config.intermediate_size,
            head_dim=model_hf.config.head_dim,
            rms_norm_eps=model_hf.config.rms_norm_eps,
            rope_theta=model_hf.config.rope_theta,
            attention_bias=model_hf.config.attention_bias,
            mlp_bias=model_hf.config.mlp_bias,
            tie_word_embeddings=model_hf.config.tie_word_embeddings,
        )
        
        # Create our model
        model = cls(config)
        
        # Copy weights from HuggingFace model to our model
        sd = model.state_dict()
        sd_hf = model_hf.state_dict()

        skip_keys = ['freqs_cis', 'bias']
        
        # Copy each weight tensor
        for key in sd.keys():
            if any(skip_key in key for skip_key in skip_keys):
                continue
            if key in sd_hf:
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key])
            else:
                print(f"Warning: {key} not found in pretrained model")
        
        print("Model loaded successfully!")
        return model
