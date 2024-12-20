from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    dim: int = 4096
    n_layers: 32
    n_heads: int = 32
    n_kv_heads: int = 8
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 4
    max_seq_len: int = 128