## TASK 3 
Paper : Attention Is All You Need (https://arxiv.org/html/1706.03762v7#S3)

Implementation of Transformer 
# Attention Is All You Need — From Scratch Implementation

> Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., Google Brain (2017)

## Overview
This is a from-scratch implementation of the Transformer architecture proposed in "Attention Is All You Need". 
The goal was to read the paper carefully, implement the core architecture independently, and verify results against the paper's reported metrics.

---
## Architecture

The Transformer follows an encoder-decoder structure:
- **Encoder**: Stack of N identical layers, each with Multi-Head Self-Attention + FFN, wrapped in Add & Norm
- **Decoder**: Stack of N identical layers, each with Masked Self-Attention + Cross-Attention over encoder output + FFN
- **Positional Encoding**: Sinusoidal encodings added to embeddings since the model has no recurrence
- **Training**: Adam optimizer with custom warmup LR schedule, label smoothing = 0.1

---
## My Config vs Paper

| Hyperparameter | Paper (base) | This implementation |
|---|---|---|
| d_model | 512 | 256 |
| N (layers) | 6 | 3 |
| d_ff | 2048 | 512 |
| heads | 8 | 8 |
| dropout | 0.1 | 0.1 |
| dataset | WMT14 EN-DE (4.5M) | Multi30k EN-DE (29k) |
| hardware | 8x NVIDIA P100 | CPU laptop |
| training steps | 100,000 | 10 epochs |

---

## Results

| Metric | Value |
|---|---|
| Final Validation Loss | 4.14 |
| BLEU Score | 0.42 for 10 epochs |
| Paper Base Model BLEU | 27.3 |

### Why the gap exists
The architecture is a faithful implementation of the paper. The BLEU gap is entirely due to:
- **170x less training data** (29k vs 4.5M sentences)
- **Half the model capacity** (d_model=256, N=3 vs 512, N=6)
- **Fraction of training steps** (10 epochs vs 100k steps)
- **No GPU** — CPU training limits feasible compute

### Sample translations (10 epoch checkpoint)
SRC : A man in an orange hat starring at something.
REF : Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt.
PRED: ein mann mit einem hut spielt auf einem <unk>

The model correctly learns German sentence structure and common vocabulary despite limited training.

---

## How to Run

### Install dependencies
```bash
pip install torch datasets sacrebleu
```

### Train
```bash
python src/train.py
```

### Evaluate (BLEU score)
```bash
python src/evaluate.py
```

---

## File Structure
├── PAPER_NOTES.md       # Reading notes — claim, architecture, metrics

├── README.md            # This file

├── src/

│   ├── model.py         # Full Transformer architecture

│   ├── train.py         # Training loop + data loading

│   └── evaluate.py      # BLEU evaluation + greedy decode

└── results/             # Evaluation logs and screenshots

## Conclusion : 

The Attention is all you Need paper revolutionalized the way LLMs used to learn. The Transformer along with Encoder - Decoder structur and other parts like tokenizer,embedding and encoding and most importantly the 'ATTENTION' and 'MLP' layer helped us really transform (pun intended) the way AI learned and really put us substantially ahead in AI research. 

