# Circuit Discovery on a Toy Task

## Reverse-Engineering a Grokked Transformer on Modular Addition

<div class="abstract-box">

**Abstract.** We train a single-layer, four-head transformer from scratch on modular addition $(a+b) \bmod 113$ and fully reverse-engineer the algorithm it learns using mechanistic interpretability techniques. The model discovers a Fourier-based computation: token embeddings are linear combinations of $\sin$ and $\cos$ basis functions over $\mathbb{Z}/p\mathbb{Z}$, attention heads route the embedded tokens to the output position, and the MLP implements trigonometric identities to compute the sum. Using activation patching, we identify a minimal circuit of 2 attention heads and 15 MLP neurons that accounts for $>95\%$ of model performance — fewer than $5\%$ of all components. This work provides a complete, reproducible template for circuit discovery on a canonical mechanistic interpretability benchmark.

</div>

<div style="text-align: center; margin: 2em 0;">
    <img src="figures/Figure_3_neuron_2d_fourier.png" alt="2D Fourier Spectrum of MLP Neurons" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
    <p class="figure-caption"><strong>Key Figure:</strong> 2D Fourier spectrum of MLP neurons. Energy concentrated on the diagonal (k,k) confirms the trigonometric identity computation.</p>
</div>

## Key Results

<div class="key-finding">
<strong>Fourier Features:</strong> The embedding matrix concentrates $>80\%$ of its variance in $\leq 5$ Fourier frequencies. The model represents numbers in a Fourier basis.
</div>

<div class="key-finding">
<strong>Head Specialisation:</strong> Attention heads route information from specific input positions to the output position. Only 2 of 4 heads are causally necessary.
</div>

<div class="key-finding">
<strong>Frequency-Selective MLP:</strong> Specific MLP neurons detect frequency-k components of $(a+b) \bmod p$, implementing the trigonometric identity via the ReLU nonlinearity.
</div>

<div class="key-finding">
<strong>Minimal Circuit:</strong> A circuit of 2 heads + 15 neurons ($\sim 3\%$ of components) achieves $>95\%$ of full model accuracy.
</div>

## Repository Structure

```
├── src/              # Python source code
│   ├── data/         # Dataset and data pipelines
│   ├── training/     # Model training and checkpointing
│   ├── analysis/     # Fourier analysis, attention, logit lens
│   ├── patching/     # Activation patching infrastructure
│   └── circuit/      # Minimal circuit identification
├── configs/          # YAML configuration files
├── docs/             # MkDocs site (this page)
├── results/          # Generated figures and data
├── tests/            # Unit tests
├── notebooks/        # Self-contained Colab notebook
└── scripts/          # Shell scripts for the full pipeline
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements-cpu.txt

# Train the model (30 min on T4)
python src/training/train.py --config configs/training/default.yaml

# Run the full analysis pipeline
python scripts/run_full_pipeline.sh

# Build this site
mkdocs build --clean -d site
```

## Citation

```bibtex
@misc{circuit-discovery-toy-task,
    title={Circuit Discovery on a Toy Task: 
           Reverse-Engineering a Grokked Transformer on Modular Addition},
    author={Your Name},
    year={2026},
    howpublished={\url{https://github.com/yourusername/circuit-discovery-toy-task}}
}
```
