import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.attention_patterns import load_model_from_checkpoint
from src.analysis.fourier import fourier_basis, fourier_transform, plot_fourier_spectrum
from src.utils.config import load_yaml


def analyse_embedding_fourier(
    model, p: int, output_dir: str = "results/fourier"
):
    """Analyse the Fourier spectrum of the token embedding matrix."""
    W_E = model.embed.W_E.detach()  # [d_vocab, d_model]
    d_vocab, d_model = W_E.shape

    # Extract the token embeddings for tokens 0..p-1 (skip = token at index p)
    token_embeds = W_E[:p]  # [p, d_model]

    # Build Fourier basis
    basis, freqs = fourier_basis(p, device=W_E.device)

    # For each embedding dimension, compute Fourier spectrum
    # token_embeds: [p, d_model] — each column is the embedding of that token
    # We transpose to get [d_model, p] — each row is embeddings across tokens
    embed_by_dimension = token_embeds.T  # [d_model, p]

    # Fourier transform each dimension
    fourier_coeffs = fourier_transform(embed_by_dimension, basis)  # [d_model, p]

    # Plot 2D heatmap: x=frequency, y=embedding dimension
    fig, ax = plt.subplots(figsize=(12, 6))
    magnitudes = fourier_coeffs.abs().cpu().numpy()  # [d_model, p]

    im = ax.imshow(magnitudes.T, aspect="auto", cmap="viridis",
                   extent=[0, d_model, p - 0.5, -0.5])
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Fourier frequency k")
    ax.set_title("Embedding Fourier Spectrum: Magnitude per Dimension")
    plt.colorbar(im, ax=ax, fraction=0.046, label="|Coefficient|")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "embedding_fourier.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved embedding Fourier spectrum to {save_path}")
    plt.close()

    # Compute fraction of variance in top-K frequencies
    total_var = (fourier_coeffs ** 2).sum(dim=1)  # [d_model] variance per dim
    for K in [1, 3, 5, 10]:
        # Find top-K frequencies per dimension
        topk_vals = fourier_coeffs.abs().topk(K, dim=1).values
        topk_var = (topk_vals ** 2).sum(dim=1)
        frac = (topk_var / total_var.clamp(min=1e-10)).mean().item()
        print(f"  Top-{K} Fourier frequencies explain {frac:.2%} of embedding variance")

    # Identify dominant frequencies across all dimensions
    energy_per_freq = (fourier_coeffs ** 2).sum(dim=0)  # [p] total energy per freq
    sorted_freqs = energy_per_freq.argsort(descending=True)
    top_freqs = freqs[sorted_freqs[:10]].long().cpu().tolist()
    print(f"  Top-10 dominant frequencies: {top_freqs}")

    return fourier_coeffs, freqs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/checkpoints/best_val_acc.pt")
    parser.add_argument("--config", type=str,
                        default="configs/analysis/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results/fourier")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model_from_checkpoint(args.checkpoint, device=device)
    p = int(cfg.get("p", 113))

    print("Analysing embedding Fourier spectrum...")
    analyse_embedding_fourier(model, p, output_dir=args.output_dir)
    print("Embedding Fourier analysis complete.")
