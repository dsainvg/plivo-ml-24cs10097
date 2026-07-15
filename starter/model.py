"""A small GPT in plain PyTorch with Mixture of Experts (MoE) feed-forward blocks.
2 language experts (one learns English, one learns Hindi patterns) are gated
per-token using a learned routing network. The auxiliary load-balancing loss
prevents expert collapse and encourages balanced routing.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 2048      # BPE vocab default
    block_size = 128       # Context length
    n_layer = 5            # 5 layers (best config — Run 3)
    n_head = 4             # 4 heads (head dim = 36)
    n_embd = 144           # Embedding dimension
    n_ff = 368             # SwiGLU hidden dim per expert (2.56 * 144 ≈ 368)
    n_experts = 2          # Number of MoE experts
    moe_layers = [2, 3]    # Layers 2 and 3 are MoE; layers 0, 1, 4 are dense (incl. last)
    dropout = 0.0
    tie_weights = True
    aux_loss_weight = 0.01 # Load-balancing auxiliary loss weight


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        T = seq_len if seq_len is not None else x.shape[2]
        return self.cos_cached[:, :, :T, :], self.sin_cached[:, :, :T, :]


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj._is_residual = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos=None, sin=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if cos is not None and sin is not None:
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class SwiGLUExpert(nn.Module):
    def __init__(self, d_embd, d_ff, dropout=0.0):
        super().__init__()
        self.w12 = nn.Linear(d_embd, 2 * d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_embd, bias=False)
        self.w3._is_residual = True
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.w3(F.silu(x1) * x2))


class MoELayer(nn.Module):
    """Top-1 gated Mixture of Experts with load-balancing auxiliary loss.
    
    Each token is routed to exactly 1 expert using a learned linear gate.
    The auxiliary loss encourages balanced routing across experts to prevent
    one expert from capturing all tokens (expert collapse).
    """
    def __init__(self, cfg):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.gate = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(cfg.n_embd, cfg.n_ff, cfg.dropout)
            for _ in range(cfg.n_experts)
        ])

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(B * T, C)

        # Compute routing probabilities: (B*T, n_experts)
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-1 routing: select the expert with highest probability
        expert_idx = gate_probs.argmax(dim=-1)  # (B*T,)

        # Load balancing auxiliary loss:
        # Encourages uniform distribution of tokens across experts.
        # Loss = n_experts * sum(fraction_routed_i * mean_gate_prob_i)
        # This is the Switch Transformer style auxiliary loss.
        fraction_routed = torch.zeros(self.n_experts, device=x.device)
        for i in range(self.n_experts):
            fraction_routed[i] = (expert_idx == i).float().mean()
        mean_gate_prob = gate_probs.mean(dim=0)
        aux_loss = self.n_experts * (fraction_routed * mean_gate_prob).sum()

        # Route each token to its selected expert
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (expert_idx == i)
            if mask.any():
                output[mask] = expert(x_flat[mask])

        # Scale output by the gate probability (importance weighting)
        output = output * gate_probs.gather(1, expert_idx.unsqueeze(1))
        return output.view(B, T, C), aux_loss


class Block(nn.Module):
    def __init__(self, cfg, is_moe=False):
        super().__init__()
        self.is_moe = is_moe
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        if is_moe:
            self.moe = MoELayer(cfg)
        else:
            self.mlp = SwiGLUExpert(cfg.n_embd, cfg.n_ff, cfg.dropout)

    def forward(self, x, cos=None, sin=None):
        x = x + self.attn(self.ln1(x), cos, sin)
        if self.is_moe:
            moe_out, aux_loss = self.moe(self.ln2(x))
            x = x + moe_out
        else:
            x = x + self.mlp(self.ln2(x))
            aux_loss = 0.0
        return x, aux_loss


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(cfg.n_embd // cfg.n_head, max_seq_len=cfg.block_size * 2)
        self.blocks = nn.ModuleList(
            Block(cfg, is_moe=(i in cfg.moe_layers))
            for i in range(cfg.n_layer)
        )
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, "_is_residual", False):
                std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos, sin = self.rope(x, seq_len=T)
        total_aux_loss = 0.0
        for blk in self.blocks:
            x, aux_loss = blk(x, cos, sin)
            total_aux_loss = total_aux_loss + aux_loss
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                      targets.reshape(-1))
            loss = ce_loss + self.cfg.aux_loss_weight * (total_aux_loss / self.cfg.n_layer)
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
