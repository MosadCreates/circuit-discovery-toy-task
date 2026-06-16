
<h1 align="center">
  Circuit Discovery on a Toy Task
</h1>

<p align="center">
  <em>Reverse-engineering a grokked transformer on modular addition:<br>
  Fourier mechanisms, activation patching, and minimal circuit identification.</em>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/MosadCreates/circuit-discovery-toy-task/blob/main/notebooks/colab_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>
<a href="https://github.com/MosadCreates/circuit-discovery-toy-task/actions">
  <img src="https://github.com/MosadCreates/circuit-discovery-toy-task/actions/workflows/ci.yml/badge.svg" alt="CI">
</a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

<p align="center">
  <strong><a href="https://mosadcreates.github.io/circuit-discovery-toy-task/">View the Live Writeup</a></strong>
</p>

---

## Overview

We train a single-layer, four-head transformer from scratch on **modular addition** $(a+b) \bmod 113$ and fully reverse-engineer its internal algorithm using mechanistic interpretability techniques. The model discovers a Fourier-based computation:

- Embeddings are sinusoidal functions of the input tokens
- Attention heads route information to the output position
- The MLP implements trigonometric identities

A **minimal circuit of 2 attention heads and 15 MLP neurons** (fewer than 5% of all components) achieves >95% of the full model's accuracy.

> **All figures are generated programmatically.** See the [Live Writeup](https://mosadcreates.github.io/circuit-discovery-toy-task/) for 9 publication-quality figures including Fourier spectra, activation patching heatmaps, and the minimal circuit ablation curve.

---

## Key Results

| Finding | Evidence |
|---------|----------|
| **Fourier features** | Top-5 frequencies explain >80% of embedding variance |
| **Head specialisation** | 2 of 4 heads route a and b to the output position |
| **Trigonometric MLP** | Neuron 2D Fourier spectra show diagonal (k,k) structure |
| **Minimal circuit** | 2 heads + 15 neurons achieve >95% accuracy |

---

## Table of Contents

- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [License](#license)

---

## Quick Start

```bash
# Clone
git clone https://github.com/mosadcreates/circuit-discovery-toy-task
cd circuit-discovery-toy-task

# Setup (choose one)
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-cpu.txt   # CPU
bash setup.sh                                                                                # CUDA

# Train model (~30 min on T4)
python src/training/train.py --config configs/training/default.yaml

# Run full analysis pipeline
bash scripts/run_full_pipeline.sh

# Build the static site
make report && open site/index.html
```

### Individual Steps

| Command | Description |
|---------|-------------|
| `make train` | Train the model (p=113, 50k steps) |
| `make visualize` | Generate all figures (attention, Fourier, DLA) |
| `make patch` | Run activation patching experiments |
| `make report` | Build the MkDocs site |
| `make test` | Run all unit tests |
| `make serve-docs` | Live preview at localhost:8000 |
| `make clean` | Remove build artifacts |

---

## Repository Structure

```
├── configs/             YAML configurations (training, analysis, patching)
├── docs/                MkDocs static site (the research writeup)
├── notebooks/           Self-contained Colab demo notebook
├── results/             Generated figures, checkpoints, and metrics
├── scripts/             Shell scripts for the full pipeline
├── src/
│   ├── analysis/        Fourier analysis, attention patterns, DLA, weight analysis
│   ├── circuit/         Minimal circuit identification and ablation
│   ├── data/            ModularAdditionDataset, DataModule, train/val split
│   ├── patching/        ActivationPatcher, head/MLP/path patching experiments
│   ├── training/        Training loop, checkpoint manager, grokking dynamics
│   └── utils/           Config loader, seed setting, plotting utilities
├── tests/               Unit tests for data pipeline and config
├── docker/              Dockerfile for containerised reproduction
├── setup.sh             Linux/macOS setup script
├── setup.ps1            Windows PowerShell setup script
├── Makefile             Build automation
├── pyproject.toml       Project metadata and tool configuration
├── requirements.txt     CUDA dependencies (pinned)
└── requirements-cpu.txt CPU-only dependencies (pinned)
```

---

## Methodology

1. **Fourier Analysis** — Project weights and activations onto the $p$-dimensional orthonormal Fourier basis over $\mathbb{Z}/p\mathbb{Z}$ to measure frequency concentration.

2. **Attention Pattern Visualisation** — Extract per-head attention matrices via TransformerLens hooks and classify each head's role (a-attend, b-attend, self, uniform).

3. **Activation Patching** — Replace activations from a corrupted forward pass with clean activations to measure each component's causal necessity using the recovery score metric.

4. **Direct Logit Attribution** — Decompose the final logit into per-component contributions using the linearity of the residual stream.

5. **Minimal Circuit Ablation** — Zero-out non-circuit components to identify the smallest set of heads and neurons sufficient for >95% of full model accuracy.

---

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0
- TransformerLens 3.4.0
- Einops, NumPy, SciPy
- Matplotlib, Seaborn
- PyYAML, WandB
- MkDocs + Material theme

Full pinned versions in [`requirements.txt`](requirements.txt) (CUDA) and [`requirements-cpu.txt`](requirements-cpu.txt) (CPU).

---

## Citation

```bibtex
@misc{circuit-discovery-toy-task,
    title={Circuit Discovery on a Toy Task:
           Reverse-Engineering a Grokked Transformer on Modular Addition},
    author={Mosad Creates},
    year={2026},
    howpublished={\url{https://github.com/mosadcreates/circuit-discovery-toy-task}}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
