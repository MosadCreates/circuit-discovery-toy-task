import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.attention_patterns import load_model_from_checkpoint
from src.analysis.fourier import fourier_basis
from src.utils.config import load_yaml


def compute_ov_matrix(model, head_idx: int):
    W_V = model.blocks[0].attn.W_V[head_idx]  # [d_model, d_head]
    W_O = model.blocks[0].attn.W_O[head_idx]  # [d_head, d_model]
    return W_V @ W_O  # [d_model, d_model]


def compute_qk_matrix(model, head_idx: int):
    W_Q = model.blocks[0].attn.W_Q[head_idx]  # [d_model, d_head]
    W_K = model.blocks[0].attn.W_K[head_idx]  # [d_model, d_head]
    return W_Q @ W_K.T  # [d_model, d_model]


@torch.no_grad()
def analyse_weight_fourier(model, p: int, output_dir: str = "results"):
    """
    Analyse W_OV and W_QK matrices and their relationship to the
    embedding/unembedding Fourier spectrum.

    For each head, we compute:
    1. W_OV Fourier action: how does W_OV transform Fourier features?
       We project W_E @ W_OV @ W_U[:, :p] into the Fourier basis.
    2. W_QK Fourier structure: what frequencies in the input
       produce high attention scores?
    """
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model

    W_E = model.embed.W_E.detach()[:p]  # [p, d_model] — token embeddings
    W_U = model.unembed.W_U.detach()[:, :p]  # [d_model, p] — unembed for answers
    basis, freqs = fourier_basis(p, device=W_E.device)

    os.makedirs(output_dir, exist_ok=True)

    for h in range(n_heads):
        W_OV = compute_ov_matrix(model, h)  # [d_model, d_model]

        # Full input-output map through the head in token space:
        # input -> W_E -> W_OV -> W_U^T -> logits
        # head_output[a, :] = W_E[a, :] @ W_OV
        # logit_contribution[a, t] = W_E[a, :] @ W_OV @ W_U[:, t]
        full_map = W_E @ W_OV @ W_U  # [p, p]

        # Project into Fourier basis: input side and output side
        fourier_map = basis @ full_map @ basis.T  # [p, p]

        # W_QK analysis
        W_QK = compute_qk_matrix(model, h)  # [d_model, d_model]
        # QK score for (a, b): W_E[a, :] @ W_QK @ W_E[b, :].T
        qk_map = W_E @ W_QK @ W_E.T  # [p, p]
        qk_fourier = basis @ qk_map @ basis.T  # [p, p]

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        titles = [
            f"W_OV Head {h}: Token-space effective map",
            f"W_QK Head {h}: Token-space attention scores",
        ]

        for ax, mat, title in [
            (axes[0], fourier_map, titles[0]),
            (axes[1], qk_fourier, titles[1]),
        ]:
            data = mat.cpu().numpy()
            vmax = max(abs(data.min()), abs(data.max()), 1e-8)
            im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           aspect="equal",
                           extent=[-0.5, p - 0.5, p - 0.5, -0.5])
            ax.set_xlabel("Input frequency k_in")
            ax.set_ylabel("Output frequency k_out")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        save_path = os.path.join(output_dir,
                                 f"weight_fourier_head_{h}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved weight Fourier analysis for head {h} "
              f"to {save_path}")
        plt.close()

        # Diagonal analysis
        ov_diag = torch.diag(fourier_map).abs()
        qk_diag = torch.diag(qk_fourier).abs()
        print(f"  Head {h}: W_OV top-5 frequencies: "
              f"{ov_diag.argsort(descending=True)[:5].tolist()}")
        print(f"  Head {h}: W_QK top-5 frequencies: "
              f"{qk_diag.argsort(descending=True)[:5].tolist()}")

    # Summary: diagonal Fourier magnitude per head
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax_idx, (name, getter) in enumerate([
        ("W_OV", compute_ov_matrix),
        ("W_QK", compute_qk_matrix),
    ]):
        ax = axes[ax_idx]
        data = np.zeros((n_heads, p))
        for h in range(n_heads):
            mat = getter(model, h)
            full_map = W_E @ mat @ W_U
            mat_fourier = (basis @ full_map @ basis.T).cpu().numpy()
            data[h] = np.abs(np.diag(mat_fourier))

        im = ax.imshow(data, cmap="viridis", aspect="auto",
                       extent=[-0.5, p - 0.5, n_heads - 0.5, -0.5])
        ax.set_xlabel("Frequency k")
        ax.set_ylabel("Head")
        ax.set_title(f"{name}: Diagonal Fourier Magnitude per Head")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "weight_fourier_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved weight Fourier summary to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/checkpoints/best_val_acc.pt")
    parser.add_argument("--config", type=str,
                        default="configs/analysis/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model_from_checkpoint(args.checkpoint, device=device)
    p = int(cfg.get("p", 113))

    analyse_weight_fourier(model, p, output_dir=args.output_dir)
    print("Weight analysis complete.")
