import torch

from inference import LLaMA3

torch.manual_seed(0)

allow_cuda = False
device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

prompts = [
    "What is the capital of France?",
]

model = LLaMA3.build(
    ckpts_dir="model_store/Meta-Llama-3-8B/",
    tokenizer_path="model_store/Meta-Llama-3-8B/tokenizer.model",
    load_model=True,
    max_seq_len=128,
    max_batch_size=len(prompts),
    device=device,
)
