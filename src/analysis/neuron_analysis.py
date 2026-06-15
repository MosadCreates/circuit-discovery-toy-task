import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.attention_patterns import load_model_from_checkpoint
from src.analysis.fourier import fourier_basis, plot_2d_fourier_spectrum
from src.utils.config import load_yaml


def compute_all_mlp_activations(model, p: int, device: str):
    """
    Run all p^2 inputs through the model and extract MLP activations
    at the = position (position 2).

    Returns:
        activations: [p, p, d_mlp] — activation for each (a, b, =neuron)
    """
    d_mlp = model.cfg.d_mlp
    activations = torch.zeros(p, p, d_mlp)

    # Generate all inputs as a batch
    a_vals = torch.arange(p, device=device)
    b_vals = torch.arange(p, device=device)
    grid_a, grid_b = torch.meshgrid(a_vals, b_vals, indexing="ij")
    eq_vals = torch.full_like(grid_a.flatten(), p)

    tokens = torch.stack([
        grid_a.flatten(),
        grid_b.flatten(),
        eq_vals,
    ], dim=1)  # [p^2, 3]

    # Run in batches to avoid OOM (can handle full batch for p=113)
    batch_size = p * p
    _, cache = model.run_with_cache(
        tokens, return_cache_object=True,
    )

    # MLP post-activation at the = position (index 2)
    mlp_out = cache["blocks.0.mlp.hook_post"]  # [p^2, 3, d_mlp]
    mlp_at_eq = mlp_out[:, 2, :]  # [p^2, d_mlp]

    # Reshape to [p, p, d_mlp]
    activations = mlp_at_eq.reshape(p, p, d_mlp).detach().cpu()
    return activations


def compute_neuron_2d_fourier(
    activations: torch.Tensor, p: int, basis: torch.Tensor
):
    """
    Compute the 2D Fourier spectrum of each MLP neuron's activation
    over the (a,b) input grid.

    For each neuron (index n), we have a [p, p] activation grid.
    The 2D DFT is: F[k_a, k_b] = basis[k_a]^T @ activations[:,:,n] @ basis[k_b]

    Returns:
        spectra: [d_mlp, p, p] — 2D Fourier coefficients per neuron
    """
    d_mlp = activations.shape[-1]
    spectra = torch.zeros(d_mlp, p, p)

    for n in range(d_mlp):
        grid = activations[:, :, n]  # [p, p]
        # 2D Fourier transform
        coeffs = basis.T @ grid @ basis  # [p, p]
        spectra[n] = coeffs

    return spectra


def compute_fourier_concentration(spectra: torch.Tensor):
    """Compute how concentrated each neuron's spectrum is on the diagonal (k,k)."""
    p = spectra.shape[-1]
    d_mlp = spectra.shape[0]

    total_energy = spectra.abs().pow(2).sum(dim=[1, 2])  # [d_mlp]
    diag_energy = torch.zeros(d_mlp)
    for k in range(p):
        diag_energy += spectra[:, k, k].abs().pow(2)

    concentration = (diag_energy / total_energy.clamp(min=1e-10))
    return concentration


def analyse_mlp_neurons(
    model, p: int, device: str, n_top: int = 20,
    output_dir: str = "results/fourier"
):
    """Main analysis: compute MLP neuron Fourier spectra and identify key neurons."""
    print("Computing MLP activations on all inputs...")
    activations = compute_all_mlp_activations(model, p, device)
    print(f"  Activations shape: {activations.shape}")

    basis, freqs = fourier_basis(p, device="cpu")

    print("Computing 2D Fourier spectra per neuron...")
    spectra = compute_neuron_2d_fourier(activations, p, basis)
    print(f"  Spectra shape: {spectra.shape}")

    # Compute Fourier concentration on diagonal (k, k)
    concentration = compute_fourier_concentration(spectra)
    top_idx = concentration.argsort(descending=True)

    print(f"\nTop-{n_top} neurons by Fourier concentration on diagonal (k,k):")
    for rank, idx in enumerate(top_idx[:n_top].tolist()):
        conc = concentration[idx].item()
        spec = spectra[idx]
        # Find dominant frequency
        energy = spec.abs().pow(2)
        max_pos = energy.argmax().item()
        k_a = max_pos // p
        k_b = max_pos % p
        print(f"  #{rank+1}: Neuron {idx} | concentration={conc:.4f} | "
              f"dominant freq=({k_a},{k_b})")

    # Plot top-20 most Fourier-structured neurons
    os.makedirs(output_dir, exist_ok=True)
    n_cols = 5
    n_rows = (n_top + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for rank, idx in enumerate(top_idx[:n_top].tolist()):
        ax = axes[rank]
        spec = spectra[idx].abs().cpu().numpy()
        im = ax.imshow(spec, cmap="viridis", aspect="equal",
                       extent=[-0.5, p - 0.5, p - 0.5, -0.5])
        ax.set_title(f"Neuron {idx}\nconc={concentration[idx]:.3f}")
        ax.set_xlabel("k_a")
        ax.set_ylabel("k_b")
        plt.colorbar(im, ax=ax, fraction=0.046)

    for i in range(n_top, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"2D Fourier Spectrum of Top-{n_top} MLP Neurons (p={p})",
                 fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "neuron_2d_fourier_top20.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved top-{n_top} neuron spectra to {save_path}")
    plt.close()

    # Summary plot: concentration histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(concentration.cpu().numpy(), bins=50, color="steelblue", edgecolor="white")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.7,
               label="50% concentration threshold")
    ax.set_xlabel("Diagonal Fourier concentration")
    ax.set_ylabel("Number of neurons")
    ax.set_title("Distribution of Fourier Concentration Across MLP Neurons")
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(output_dir, "neuron_fourier_concentration.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved concentration histogram to {save_path}")
    plt.close()

    return spectra, concentration, top_idx


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
    p = int(cfg.get("p", 113))
    n_top = int(cfg.get("n_top_neurons", 20))

    model = load_model_from_checkpoint(args.checkpoint, device=device)

    analyse_mlp_neurons(model, p, device, n_top=n_top, output_dir=args.output_dir)
    print("MLP neuron Fourier analysis complete.")
