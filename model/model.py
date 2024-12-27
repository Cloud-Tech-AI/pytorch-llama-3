import torch
from torch import nn

from .config import ModelArgs
from .base import RMSNorm, Rope, TransformerBlock


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, norm_eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.rotary_pe = Rope(
            params.dim // params.n_heads, params.max_seq_len, params.rope_theta
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            # The attention scores are calculated only for the current step, covering both cached and new tokens.
            # The score matrix has dimensions (seq_len, cache_len + seq_len), where:
            # - Rows (i): Correspond to each new token being processed.
            # - Columns (j): Cover all cached tokens (prompt) followed by new tokens.
            # Masking ensures each new token (row i) can only attend to:
            # 1. All cached tokens (columns 0 to cache_len - 1).
            # 2. Tokens up to its position in the new sequence (columns cache_len to cache_len + i).
            # This setup prevents any token from attending to future tokens, maintaining causal attention.
            mask = torch.hstack(
                [torch.zeros((seq_len, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, self.rotary_pe, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
