"""
Advanced Neural Network Layers for Stock Prediction
State-of-the-art components for maximum accuracy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    Better than standard positional encoding for sequences
    Used in latest LLMs like LLaMA
    """
    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute rotation matrices
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Build cos/sin cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[2]
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to queries and keys."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FlashAttention(nn.Module):
    """
    Memory-efficient attention with flash attention pattern
    Faster and uses less memory than standard attention
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Rotary embeddings
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply rotary embeddings
        cos, sin = self.rotary(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


class GatedMLP(nn.Module):
    """
    Gated MLP with SwiGLU activation
    Better than standard FFN - used in modern transformers
    """
    def __init__(self, dim: int, hidden_mult: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(dim * hidden_mult * 2 / 3)  # SwiGLU uses 2/3 hidden
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: w2(SiLU(w1(x)) * w3(x))
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts Layer
    Routes inputs to different expert networks for specialized processing
    Increases model capacity without proportional compute increase
    """
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
        # Experts (each is a GatedMLP)
        self.experts = nn.ModuleList([
            GatedMLP(dim, dropout=dropout) for _ in range(num_experts)
        ])
        
        # Load balancing loss coefficient
        self.aux_loss_coef = 0.01
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        
        # Compute routing probabilities
        router_logits = self.gate(x)  # (B, N, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = router_probs.topk(self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        output = torch.zeros_like(x)
        
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]  # (B, N)
            expert_prob = top_k_probs[:, :, k:k+1]  # (B, N, 1)
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_prob[mask].squeeze(-1).unsqueeze(-1) * expert_output
        
        # Auxiliary load balancing loss
        aux_loss = self._compute_aux_loss(router_probs)
        
        return output, aux_loss
    
    def _compute_aux_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Compute load balancing auxiliary loss"""
        # Fraction of tokens routed to each expert
        tokens_per_expert = router_probs.mean(dim=[0, 1])
        # Ideal uniform distribution
        uniform = torch.ones_like(tokens_per_expert) / self.num_experts
        # Load balancing loss
        return self.aux_loss_coef * F.mse_loss(tokens_per_expert, uniform)


class TemporalFusionDecoder(nn.Module):
    """
    Temporal Fusion Transformer Decoder
    State-of-the-art for time series - combines multiple temporal patterns
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention for temporal patterns
        self.temporal_attention = FlashAttention(dim, num_heads, dropout)
        
        # Gated residual network
        self.grn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim * 2)
        )
        
        # Layer norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Static context
        self.static_context = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor, static: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Temporal attention
        attn_out = self.temporal_attention(self.norm1(x))
        x = x + attn_out
        
        # Gated residual
        grn_out = self.grn(self.norm2(x))
        gate, hidden = grn_out.chunk(2, dim=-1)
        x = x + torch.sigmoid(gate) * hidden
        
        # Add static context if provided
        if static is not None:
            x = x + self.static_context(static).unsqueeze(1)
        
        return x


class UncertaintyHead(nn.Module):
    """
    Predicts both mean and uncertainty (epistemic + aleatoric)
    Critical for knowing when NOT to trade
    """
    def __init__(self, dim: int, num_classes: int = 3):
        super().__init__()
        
        # Mean prediction
        self.mean_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, num_classes)
        )
        
        # Log variance prediction (aleatoric uncertainty)
        self.var_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_classes)
        )
        
        # Confidence prediction (epistemic uncertainty)
        self.conf_head = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.mean_head(x)
        log_var = self.var_head(x)
        confidence = self.conf_head(x)
        
        return mean, log_var, confidence
    
    def sample(self, x: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """Monte Carlo sampling for uncertainty estimation"""
        mean, log_var, _ = self.forward(x)
        std = torch.exp(0.5 * log_var)
        
        samples = []
        for _ in range(num_samples):
            eps = torch.randn_like(std)
            sample = mean + eps * std
            samples.append(F.softmax(sample, dim=-1))
        
        return torch.stack(samples).mean(dim=0)


class AdaptiveComputationTime(nn.Module):
    """
    Adaptive Computation Time (ACT)
    Allows model to 'think' more for difficult predictions
    """
    def __init__(self, dim: int, max_steps: int = 5, threshold: float = 0.99):
        super().__init__()
        self.max_steps = max_steps
        self.threshold = threshold
        
        # Halting probability network
        self.halt_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Processing network
        self.process = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        device = x.device
        
        # Initialize
        halted = torch.zeros(batch_size, 1, device=device)
        remainders = torch.zeros(batch_size, 1, device=device)
        n_updates = torch.zeros(batch_size, 1, device=device)
        
        state = x
        outputs = torch.zeros_like(x)
        
        for step in range(self.max_steps):
            # Compute halting probability
            p = self.halt_net(state)
            
            # Determine which samples to halt
            still_running = (halted < 1.0).float()
            new_halted = (halted + p * still_running >= self.threshold).float()
            
            # Compute remainders
            remainders += (1 - new_halted) * p * still_running
            
            # Update states
            delta = self.process(state)
            update_weights = p * still_running * (1 - new_halted) + remainders * new_halted
            outputs = outputs + update_weights * (state + delta)
            
            n_updates += still_running
            halted = halted + p * still_running
            state = state + delta
            
            if (halted >= self.threshold).all():
                break
        
        # Ponder cost for regularization
        ponder_cost = n_updates.mean() + remainders.mean()
        
        return outputs, ponder_cost