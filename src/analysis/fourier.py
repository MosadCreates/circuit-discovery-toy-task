import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.config import load_yaml


def fourier_basis(p: int, device="cpu"):
    """
    Constructs the p-dimensional orthonormal Fourier basis for Z/pZ.

    Basis vectors (each of length p):
    - f_0: constant vector 1/sqrt(p) * [1, 1, ..., 1]
    - For k = 1, ..., (p-1)//2:
        cos_k[n] = sqrt(2/p) * cos(2*pi*k*n/p) for n = 0, ..., p-1
        sin_k[n] = sqrt(2/p) * sin(2*pi*k*n/p) for n = 0, ..., p-1
    - If p is even (not used here): additional sin vector for k = p/2

    Returns:
        basis: [p, p] tensor where basis[k] is the k-th basis vector
        freqs: [p] tensor of frequencies corresponding to each basis vector
    """
    n = torch.arange(p, dtype=torch.float32, device=device)
    k_idx = torch.arange(p, dtype=torch.float32, device=device)

    basis = []
    freqs = []

    # Constant vector (k=0)
    basis.append(torch.ones(p, device=device) / np.sqrt(p))
    freqs.append(0.0)

    # Cos and sin for k = 1, ..., (p-1)//2
    n_half = (p - 1) // 2
    for k in range(1, n_half + 1):
        cos_k = torch.cos(2 * np.pi * k * n / p) * np.sqrt(2 / p)
        sin_k = torch.sin(2 * np.pi * k * n / p) * np.sqrt(2 / p)
        basis.append(cos_k)
        basis.append(sin_k)
        freqs.append(float(k))
        freqs.append(float(k))

    # For even p, add the Nyquist frequency vector
    if p % 2 == 0:
        nyq = torch.cos(2 * np.pi * (p // 2) * n / p) / np.sqrt(p)
        basis.append(nyq)
        freqs.append(float(p // 2))

    basis = torch.stack(basis, dim=0)  # [p, p]
    freqs = torch.tensor(freqs, device=device)

    # Verify orthonormality
    dot = basis @ basis.T
    err = (dot - torch.eye(p, device=device)).abs().max().item()
    assert err < 1e-5, f"Fourier basis not orthonormal: max error = {err:.2e}"

    return basis, freqs


def fourier_transform(x: torch.Tensor, basis: torch.Tensor):
    """
    Project vector(s) x onto the Fourier basis.

    Args:
        x: [p] or [N, p] tensor — values indexed by token ID (0..p-1)
        basis: [p, p] Fourier basis matrix (basis[k] is k-th basis vec)

    Returns:
        coeffs: [p] or [N, p] Fourier coefficients
    """
    return x @ basis.T  # project onto basis


def plot_fourier_spectrum(
    coeffs: torch.Tensor,
    freqs: torch.Tensor,
    title: str = "Fourier Spectrum",
    save_path: str = None,
    xlabel: str = "Frequency k",
    ylabel: str = "Magnitude",
):
    """Plot Fourier coefficient magnitudes as a bar chart."""
    coeffs_np = coeffs.detach().cpu().numpy()
    freqs_np = freqs.detach().cpu().numpy()

    if coeffs_np.ndim == 1:
        coeffs_np = coeffs_np[None, :]

    fig, ax = plt.subplots(figsize=(12, 4))
    magnitudes = np.abs(coeffs_np).mean(axis=0)
    ax.bar(freqs_np, magnitudes, width=0.8, color="steelblue", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-0.5, freqs_np.max() + 0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved spectrum to {save_path}")
    plt.close()
    return fig


def plot_2d_fourier_spectrum(
    spectrum_2d: torch.Tensor,
    p: int,
    title: str = "2D Fourier Spectrum",
    save_path: str = None,
):
    """Plot 2D Fourier magnitude spectrum as a heatmap."""
    spec = spectrum_2d.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.abs(spec), cmap="viridis", aspect="equal",
                   extent=[-0.5, p - 0.5, p - 0.5, -0.5])
    ax.set_xlabel("Frequency k_a")
    ax.set_ylabel("Frequency k_b")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Magnitude")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved 2D spectrum to {save_path}")
    plt.close()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--output_dir", type=str, default="results/fourier")
    args = parser.parse_args()

    basis, freqs = fourier_basis(args.p)
    print(f"Fourier basis constructed: {basis.shape}")
    print(f"Orthonormality check passed.")

    # Verify with a simple test signal: delta function at n=5
    signal = torch.zeros(args.p)
    signal[5] = 1.0
    coeffs = fourier_transform(signal, basis)
    reconstructed = coeffs @ basis
    err = (reconstructed - signal).abs().max().item()
    print(f"Forward + inverse reconstruction error: {err:.2e}")

    save_path = os.path.join(args.output_dir, "fourier_basis_check.png")
    plot_fourier_spectrum(coeffs, freqs,
                          title=f"Fourier Spectrum of Delta(5) (p={args.p})",
                          save_path=save_path)
    print("Fourier module test complete.")
