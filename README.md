# LM From Scratch

This repository contains my implementation for Assignment 1 of CS336.

This project consists of:
- a BPE tokenizer
- a Transformer language model, including a Linear module, an Embedding module, and a Pre-Norm Transformer Block (with RoPE, Causal Multi-Head Self-Attention)
- a training loop featuring Cross-entropy loss, SGD and AdamW optimizers, learning rate scheduling, and gradient clipping

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

## Setup

### Environment

This project uses `uv` for environment management.

You can run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests

```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
I connected my implementation to the tests, by completing the functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

