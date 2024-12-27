import os
import torch

from inference import LlaMA3


dir_path = "./model_store/Meta-Llama-3.2-1B"

if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    prompts = ["Hi, what is the capital of India ?"]

    model = LlaMA3.build(
        ckpts_dir=dir_path,
        tokenizer_path=os.path.join(dir_path, "tokenizer.model"),
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
