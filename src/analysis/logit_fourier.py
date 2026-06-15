import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.attention_patterns import load_model_from_checkpoint
from src.analysis.fourier import fourier_basis, fourier_transform
from src.utils.config import load_yaml


def analyse_unembedding_fourier(
    model, p: int, output_dir: str = "results/fourier"
):
    """
    Analyse the Fourier spectrum of the unembedding matrix W_U.

    W_U has shape [d_model, d_vocab]. We extract the first p columns
    (the actual answer tokens) and compute the Fourier spectrum of
    each row of W_U^T (i.e. each output token's logit weights).
    """
    W_U = model.unembed.W_U.detach()  # [d_model, d_vocab]
    # W_U maps residual stream to logits: logits = resid @ W_U
    # We want the part for tokens 0..p-1
    W_U_tokens = W_U[:, :p]  # [d_model, p]

    # Transpose so rows correspond to output tokens
    W_U_by_token = W_U_tokens.T  # [p, d_model]

    # Build Fourier basis
    basis, freqs = fourier_basis(p, device=W_U.device)

    # For each output token, compute the Fourier spectrum of its
    # logit weights across the d_model dimensions.
    # Actually, we want the Fourier spectrum over token space:
    # each row of W_U (d_model dim) maps to all p output tokens.
    # The Fourier transform of each row tells us which frequencies
    # each d_model dimension contributes to.
    W_U_rows = W_U_tokens  # [d_model, p]
    fourier_coeffs = fourier_transform(W_U_rows, basis)  # [d_model, p]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    magnitudes = fourier_coeffs.abs().cpu().numpy().T  # [p, d_model]

    im = ax.imshow(magnitudes, aspect="auto", cmap="viridis",
                   extent=[0, model.cfg.d_model, p - 0.5, -0.5])
    ax.set_xlabel("d_model dimension")
    ax.set_ylabel("Fourier frequency k")
    ax.set_title("Unembedding Fourier Spectrum: Magnitude per d_model dimension")
    plt.colorbar(im, ax=ax, fraction=0.046, label="|Coefficient|")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "logit_fourier.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved unembedding Fourier spectrum to {save_path}")
    plt.close()

    # Compute energy per frequency
    energy_per_freq = (fourier_coeffs ** 2).sum(dim=0)  # [p]
    total_energy = energy_per_freq.sum()
    for K in [1, 3, 5, 10]:
        topk_energy = energy_per_freq.topk(K).values.sum()
        print(f"  Top-{K} frequencies explain "
              f"{topk_energy.item() / total_energy.item():.2%} of unembedding variance")

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

    print("Analysing unembedding Fourier spectrum...")
    analyse_unembedding_fourier(model, p, output_dir=args.output_dir)
    print("Logit Fourier analysis complete.")
