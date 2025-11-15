# Mini GPT-Style Transformer (PyTorch)

A minimal, educational implementation of a causal Transformer built from scratch.

---

## Overview

This project implements a small GPT-like Transformer architecture completely from scratch in PyTorch. The goal is to understand how attention, positional embeddings, normalization, and residual connections work by building each component manually.

The model learns next-token prediction on a small text corpus (for example, Shakespeare or a custom dataset). The code is intentionally simple and well-structured so the architecture is easy to follow, extend, and experiment with.

---

## Project Structure

```
mini-transformer/
│
├── mini_transformer.py       # Transformer architecture
├── train.py                  # Training loop and dataset handling
├── demo_generation.py        # Text generation using trained weights
├── tokenizer.py              # Simple character tokenizer
├── utils.py                  # Masking, seeding, and plotting helpers
│
├── notebooks/
│   └── MiniTransformer.ipynb   # Walkthrough with visualizations
│
├── assets/
│   ├── attention_head_0.png
│   ├── loss_curve.png
│   └── sample_generation.txt
│
└── README.md
```

---

## Model Architecture

The model follows the basic structure of GPT-small:

* Token and positional embeddings
* Multi-head self-attention
* Scaled dot-product attention
* Causal masking
* Feed-forward network (two linear layers with GELU)
* LayerNorm (pre-norm)
* Residual connections
* Final linear projection to vocabulary size

This covers the core mechanisms behind modern LLMs.

---

## Training Objective

The model is trained on a simple next-token prediction task.
Given a context sequence, it learns to predict the next character.

Targets are shifted versions of inputs:

```
input:   "hell"
target:  "ello"
```

Loss is standard cross-entropy over all time steps.

---

## How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Add your dataset

Place a plain text file named `input.txt` in the project root.
A few hundred kilobytes is enough for demonstration.

### 3. Train the model

```
python train.py --epochs 10 --batch_size 32 --seq_len 64
```

Training produces:

* `model_checkpoint.pt`
* `loss_curve.png`
* Attention visualizations for one batch

### 4. Generate text

```
python demo_generation.py --prompt "The meaning of life"
```

---

## Visualizations

### Attention Heatmaps

The notebook produces attention maps for each head and layer.
They help illustrate how the model distributes focus across tokens.

Files are stored in `assets/`.

### Loss Curve

Training loss over epochs to monitor convergence.

### Generated Text

Samples at various temperatures (stored in `assets/sample_generation.txt`).

---

## Key Features

* Multi-head attention and causal masking
* Transformer blocks implemented by hand
* Next-token training on small datasets
* Basic sampling for text generation
* Clean, readable PyTorch code
* Visualizations for attention and training dynamics
* Designed for learning and experimentation

---

## Recommended Experiments

The notebook includes several ablations:

* Removing LayerNorm
* Removing residual connections
* Using a single attention head
* Swapping Q and K projections
* Training with no positional embeddings
* Changing embedding or hidden sizes

Each experiment highlights why certain architectural choices matter.

---

## Configuration

Common hyperparameters (editable in `train.py`):

* `n_layers`
* `num_heads`
* `embed_dim`
* `ff_mult`
* `seq_len`
* `vocab_size`
* `learning_rate`
* `batch_size`

Small models train quickly on Colab or a modest GPU.

---

## References

* Vaswani et al., “Attention Is All You Need”, 2017
* The Annotated Transformer (Harvard NLP)
* PyTorch documentation
* Karpathy’s nanoGPT notes and lectures

---

## License

MIT License.


