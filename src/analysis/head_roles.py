import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.attention_patterns import load_model_from_checkpoint, get_attention_patterns
from src.utils.config import load_yaml


def compute_average_attention(
    model, p: int, n_samples: int, device: str, seed: int = 42
):
    """Compute average attention pattern across n_samples random (a,b) pairs.

    Returns:
        avg_attn: [n_layers, n_heads, seq_len, seq_len] averaged over samples
        per_position: [n_layers, n_heads, seq_len] attention from pos = to each key pos
    """
    rng = torch.Generator().manual_seed(seed)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = 3

    sum_attn = torch.zeros(n_layers, n_heads, seq_len, seq_len)

    for _ in range(n_samples):
        a = torch.randint(0, p, (1,), generator=rng)
        b = torch.randint(0, p, (1,), generator=rng)
        eq = torch.tensor([p])
        tokens = torch.stack([a, b, eq], dim=1).to(device)

        attn = get_attention_patterns(model, tokens)
        sum_attn += attn.cpu()

    avg_attn = sum_attn / n_samples
    # Attention from position = (index 2) to each key position
    per_position = avg_attn[:, :, 2, :]  # [n_layers, n_heads, seq_len]
    return avg_attn, per_position


def classify_head_role(avg_per_position: torch.Tensor, threshold: float = 0.4):
    """Classify each head based on what position it attends to from the = position.

    Returns:
        labels: [n_layers, n_heads] list of strings
    """
    n_layers, n_heads, _ = avg_per_position.shape
    labels = []
    for l in range(n_layers):
        row = []
        for h in range(n_heads):
            weights = avg_per_position[l, h]
            attend_a = weights[0].item()
            attend_b = weights[1].item()
            attend_self = weights[2].item()

            if attend_self > threshold and attend_self > attend_a and attend_self > attend_b:
                row.append("self")
            elif attend_a > attend_b and attend_a > threshold:
                row.append("a-attend")
            elif attend_b > attend_a and attend_b > threshold:
                row.append("b-attend")
            elif attend_a > threshold and attend_b > threshold:
                row.append("a+b-attend")
            else:
                row.append("uniform")
        labels.append(row)
    return labels


def plot_attention_summary(
    avg_per_position: torch.Tensor,
    labels,
    save_path: str = "results/attention_summary.png",
):
    """Plot a heatmap of average attention from = position to each key position."""
    n_layers, n_heads, seq_len = avg_per_position.shape
    position_names = ["pos a", "pos b", "pos ="]

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4), squeeze=False)
    if n_layers == 1:
        axes = axes[0]

    for l in range(n_layers):
        data = avg_per_position[l].cpu().numpy()  # [n_heads, seq_len]
        ax = axes[l] if n_layers > 1 else axes[l]

        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(position_names)
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels([f"H{h}" for h in range(n_heads)])
        ax.set_title(f"Layer {l} — Attention from = position")

        # Annotate cell values
        for h in range(n_heads):
            for pos in range(seq_len):
                ax.text(pos, h, f"{data[h, pos]:.2f}",
                        ha="center", va="center",
                        color="white" if data[h, pos] > 0.5 else "black", fontsize=9)

        # Add role labels on the right
        for h in range(n_heads):
            ax.text(seq_len + 0.3, h, labels[l][h],
                    va="center", fontsize=10, style="italic")

        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Average Attention Weights from Position '='", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved attention summary to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/checkpoints/best_val_acc.pt")
    parser.add_argument("--config", type=str,
                        default="configs/analysis/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results/attention")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = int(cfg.get("p", 113))
    n_samples = int(cfg.get("n_attention_samples", 50))

    model = load_model_from_checkpoint(args.checkpoint, device=device)
    print(f"Loaded model from {args.checkpoint}")
    print(f"Computing average attention over {n_samples} samples...")

    avg_attn, per_position = compute_average_attention(
        model, p, n_samples, device, seed=int(cfg.get("seed", 42))
    )

    labels = classify_head_role(per_position)
    for l in range(per_position.shape[0]):
        for h in range(per_position.shape[1]):
            print(f"  Layer {l} Head {h}: attends to a={per_position[l,h,0]:.3f}, "
                  f"b={per_position[l,h,1]:.3f}, self={per_position[l,h,2]:.3f} "
                  f"-> {labels[l][h]}")

    save_path = os.path.join(args.output_dir, "attention_summary.png")
    plot_attention_summary(per_position, labels, save_path=save_path)
    print("Head role analysis complete.")
