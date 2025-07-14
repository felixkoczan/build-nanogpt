# GPT-2 From Scratch (NanoGPT)

This repository contains a minimal reimplementation of GPT-2, inspired by Andrej Karpathy’s “Build NanoGPT” YouTube video and the [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) code.

## Table of Contents

- Overview
- Features
- Prerequisites
- Installation
- Usage
- Project Structure
- Configuration
- Training Loop
- Contributing
- License

## Overview

This project implements the core components of GPT-2 in pure PyTorch, including:

- Causal self-attention with a causal mask  
- Transformer blocks with pre-layer normalization  
- Feed-forward MLP with GELU activation  
- Token and positional embeddings  
- Language modeling head with tied weights  
- Custom weight initialization  
- Simple `DataLoaderLite` for streaming text datasets  

It follows the pedagogical style of Karpathy’s NanoGPT: minimal dependencies, straightforward code, and clear explanations.

## Features

- `CausalSelfAttention`: multi-head attention with PyTorch’s `scaled_dot_product_attention` and causal masking  
- `MLP` & `Block`: 4× hidden expansion, GELU activation, residual connections  
- `GPT`: embedding layers, stacking of Transformer blocks, final layer norm, and LM head  
- Custom Initialization: GPT-2–style weight init including scaled std for deep layers  
- Lightweight Data Loader: tokenizes raw text using `tiktoken` and streams mini-batches  
- Flexible Optimizer: AdamW with separate weight decay groups and optional fused CUDA support  

## Prerequisites

- Python 3.9+  
- PyTorch (1.14+ recommended)  
- `tiktoken` (for GPT-2 BPE tokenizer)  

### Optional (for GPU):

- CUDA- or MPS-enabled PyTorch build  

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/gpt2-from-scratch.git
    cd gpt2-from-scratch
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate        # Linux/macOS
    venv\Scripts\activate           # Windows
    ```

3. Install dependencies:

    ```bash
    pip install torch tiktoken
    ```

## Usage

1. Prepare your training data as plain text in `input.txt`.  
2. Modify hyperparameters (batch size, learning rate, etc.) at the bottom of `gpt2_reimplementation.py`.  
3. Run the training script:

    ```bash
    python gpt2_reimplementation.py
    ```

Training logs will print step, loss, learning rate, gradient norm, and tokens processed per second.

## Project Structure

```text
├── train_gpt2.py              # Main implementation and training loop  
├── play.ipynb                 # Testing and playing around with Hugging Face model  
├── input.txt                  # Plain-text training corpus  
├── README.md                  # This file  
└── requirements.txt           # (Optional) pinned dependencies  
```

## Configuration

At the bottom of gpt2_reimplementation.py, you’ll find constants for:
- B, T: micro-batch size and sequence length
- max_steps, warmup_steps, learning rate schedule
- Device selection (CPU, CUDA, or MPS)

Feel free to extract these into a CLI or config file.

## Training Loop

The script uses gradient accumulation to achieve large effective batch sizes. Key steps:
1. Zero optimizer gradients
2.	Loop over micro-steps: forward → compute loss → backward
3.	Clip gradients to norm 1.0
4.	Step optimizer and update learning rate

## Contributing

Contributions and improvements are welcome! Please submit issues or pull requests for:
- Adding dropout layers
- Implementing LR schedulers
- Checkpointing and validation
- CLI support
- Mixed-precision enhancements

## License

This project is licensed under the MIT License. See LICENSE for details.

⸻

### Based on Andrej Karpathy’s NanoGPT tutorial
YouTube: https://www.youtube.com/watch?v=l8pRSuU81PU
GitHub: https://github.com/karpathy/build-nanogpt