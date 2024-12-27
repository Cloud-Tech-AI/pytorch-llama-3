import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from .config import ModelArgs


class RMSNorm(nn.Module):
    def __init__(self, dim: int, norm_eps: float = 1e-8):
        super().__init__()
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(dim))

    def rms_norm(self, x: torch.Tensor):
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.norm_eps)

    def forward(self, x: torch.Tensor):
        # x: [batch, seq, dim]
        # weight(gamma): [dim]
        # output: [batch, seq, dim]
        output = self.rms_norm(x.float()).type_as(x)
        return output * self.weight


class Rope:
    def __init__(self, head_dim: int, max_seq_len: int, rope_theta: float = 10000):
        super().__init__()
        self.freqs_cis = Rope.pre_compute_freq_cis(head_dim, max_seq_len, rope_theta)

    @staticmethod
    def pre_compute_freq_cis(head_dim: int, max_seq_len: int, rope_theta: float):
        new_seq_len = 2 * max_seq_len
        assert head_dim % 2 == 0, "Dimension must be even"
        # theta: [head_dim // 2]
        theta = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # m: [new_seq_len]
        m = torch.arange(new_seq_len).float()

        # freqs: [new_seq_len, head_dim // 2]
        freqs = torch.outer(m, theta)
        # complex_freqs: [new_seq_len, head_dim // 2]
        complex_freqs = torch.polar(torch.ones_like(freqs), freqs)

        return complex_freqs

    def reshape_for_broadcast(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        # x: [batch, seq, num_heads, head_dim // 2]
        # freqs_cis: [seq, head_dim // 2]

        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

        # freqs_cis: [1, seq, 1, head_dim // 2]
        return freqs_cis.view(*shape)

    def apply_rotary_emb(
        self, xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
    ):
        # xq: [batch, seq, num_heads, head_dim]
        # xk: [batch, seq, num_heads, head_dim]
        # output: [batch, seq, num_heads, head_dim]

        # xq_: [batch, seq, num_heads, head_dim // 2]
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        # xk_: [batch, seq, num_heads, head_dim // 2]
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

        freqs_cis = self.reshape_for_broadcast(xq_, freqs_cis)

        # xq_out: [batch, seq, num_heads, head_dim]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        # xk_out: [batch, seq, num_heads, head_dim]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

        return xq_out.type_as(xq), xk_out.type_as(xk)

    def clip_freq_cis(self, start_pos: int, seq_len: int):
        # freqs_cis: [max_seq_len*2, head_dim // 2]
        # output: [seq_len, head_dim // 2]
        return self.freqs_cis[start_pos : start_pos + seq_len]


class GQA:
    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int):
        """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        B, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(B, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(B, seq_len, n_kv_heads * n_rep, head_dim)
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 / 3 * hidden_dim)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SelfAttention(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.n_heads = params.n_heads
        self.n_kv_heads = params.n_kv_heads
        self.head_dim = params.dim // params.n_heads
        self.n_rep = self.n_heads // params.n_kv_heads

        self.wq = nn.Linear(params.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(params.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(params.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, params.dim, bias=False)

        self.cache_k = torch.zeros(
            (params.max_batch_size, params.max_seq_len, self.n_kv_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (params.max_batch_size, params.max_seq_len, self.n_kv_heads, self.head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        rope: Rope,
        mask: Optional[torch.Tensor] = None,
    ):
        # x: [batch, seq, dim]

        B, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(B, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(B, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, seq_len, self.n_kv_heads, self.head_dim)

        freqs_cis = rope.clip_freq_cis(start_pos, seq_len)
        xq, xk = rope.apply_rotary_emb(xq, xk, freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:B, start_pos : start_pos + seq_len] = xk
        self.cache_v[:B, start_pos : start_pos + seq_len] = xv

        keys = self.cache_k[:B, : start_pos + seq_len]
        values = self.cache_v[:B, : start_pos + seq_len]

        keys = GQA.repeat_kv(keys, self.n_rep)
        values = GQA.repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.wo(output)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: str, params: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        # self.n_heads = params.n_heads
        # self.dim = params.dim
        # self.head_dim = params.dim // params.n_heads

        self.attention = SelfAttention(params)
        self.attention_norm = RMSNorm(params.dim, norm_eps=params.norm_eps)
        self.feed_forward = FeedForward(
            params.dim, params.dim * 4, params.multiple_of, params.ffn_dim_multiplier
        )
        self.ffn_norm = RMSNorm(params.dim, norm_eps=params.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        rope: Rope,
        mask: Optional[torch.Tensor] = None,
    ):
        # x: [batch, seq, dim]
        # output: [batch, seq, dim]

        h = x + self.attention(self.attention_norm(x), start_pos, rope, mask)
        out = h + self.feed_forward(self.ffn_norm(h))

        # Output also accounts for the residual connection
        return out
