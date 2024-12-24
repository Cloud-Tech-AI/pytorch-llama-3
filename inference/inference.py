from typing import Optional
import torch
import time
from pathlib import Path
import json
from tqdm import tqdm

from model.config import ModelArgs
from model.model import Transformer
from tokenizer.tokenizer import Tokenizer


class LLaMA3:
    def __init__(self, model: Transformer, tokenizer: Tokenizer, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(
        ckpts_dir: str,
        tokenizer_path: str,
        load_model: bool,
        max_seq_len: int,
        max_batch_size: int,
        device: str,
    ):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(ckpts_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {ckpts_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        with open(Path(ckpts_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params,
        )

        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

        return LLaMA3(model, tokenizer, model_args)
