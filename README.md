# pytorch-llama-3

Unofficial PyTorch implementation of the LLAMA 3 models. 

This project contains implementations of the following components.
- RMSNorm
- Rotary PE
- FeedForward Network
- KV Cache
- Self Attention (GQA)

You can run the following llama models on your CPU
- llama3.2-1B --> 4Gb RAM --> 16bit precision
- llama3.1-8B --> 8Gb RAM --> 8bit precision

## Installation
```poetry install```

## Downloading the models
```bash download_models.sh```<br>
OR<br>
```poetry install --only dev```<br>
```llama model download --source meta --model-id <your-model-id> --meta-url <meta-email-url>```

## Usage
```python main.py```

## References

- [The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783)
- [meta-llama/llama3](https://github.com/meta-llama/llama3)
- [hkproj/pytorch-llama](https://github.com/hkproj/pytorch-llama)