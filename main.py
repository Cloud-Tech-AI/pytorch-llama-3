import torch

from inference import LlaMA3

torch.manual_seed(0)

allow_cuda = False
device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

prompts = ["Hello llama, how are you ?"]

model = LlaMA3.build(
    ckpts_dir="model_store/Llama3.2-1B",
    tokenizer_path="model_store/Llama3.2-1B/tokenizer.model",
    load_model=True,
    max_seq_len=128,
    max_batch_size=len(prompts),
    device=device,
)

generation = model.text_completion(
    prompts,
    max_gen_len=20,
    temperature=0.7,
    top_p=0.9,
)

print(generation)
