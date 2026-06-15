import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.attention_patterns import load_model_from_checkpoint, get_attention_patterns
from src.utils.config import load_yaml


def attention_vs_input(
    model, p: int, device: str,
    fixed_a: int = None, fixed_b: int = None,
    seed: int = 42,
):
    """
    For fixed_a, vary b from 0 to p-1 (or vice versa) and track attention
    from the = position to each position.

    Returns:
        attn_to_a: [p, n_heads] — attention weight from = to position a
        attn_to_b: [p, n_heads] — attention weight from = to position b
        attn_to_self: [p, n_heads] — attention weight from = to itself
    """
    n_heads = model.cfg.n_heads
    attn_to_a = torch.zeros(p, n_heads)
    attn_to_b = torch.zeros(p, n_heads)
    attn_to_self = torch.zeros(p, n_heads)

    for val in range(p):
        if fixed_b is not None:
            a_val, b_val = val, fixed_b
        elif fixed_a is not None:
            a_val, b_val = fixed_a, val
        else:
            raise ValueError("Either fixed_a or fixed_b must be provided")

        a = torch.tensor([a_val])
        b = torch.tensor([b_val])
        eq = torch.tensor([p])
        tokens = torch.stack([a, b, eq], dim=1).to(device)

        attn = get_attention_patterns(model, tokens)  # [1, n_heads, 3, 3]
        attn_from_eq = attn[0, :, 2, :]  # [n_heads, 3]

        attn_to_a[val] = attn_from_eq[:, 0]
        attn_to_b[val] = attn_from_eq[:, 1]
        attn_to_self[val] = attn_from_eq[:, 2]

    return attn_to_a, attn_to_b, attn_to_self


def plot_attention_variation(
    attn_to_a, attn_to_b, attn_to_self,
    p: int, fixed_label: str, fixed_val: int,
    save_path: str = None,
):
    """Plot how attention from = varies with the varying input."""
    n_heads = attn_to_a.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    x = np.arange(p)
    labels = ["to pos a", "to pos b", "to pos ="]
    data_list = [attn_to_a.cpu().numpy(), attn_to_b.cpu().numpy(), attn_to_self.cpu().numpy()]

    for idx, (data, label) in enumerate(zip(data_list, labels)):
        ax = axes[idx]
        for h in range(n_heads):
            ax.plot(x, data[:, h], label=f"Head {h}", alpha=0.8)
        ax.set_xlabel(f"Varying input ({fixed_label}={fixed_val})")
        ax.set_ylabel(f"Attention weight {label}")
        ax.set_title(f"Attention from = {label} vs varying input")
        ax.legend()
        ax.grid(alpha=0.3)

    # Fourth subplot: summary heatmap of which position each head attends to most
    ax = axes[3]
    dominant = np.argmax(np.stack([
        attn_to_a.mean(0).numpy(),
        attn_to_b.mean(0).numpy(),
        attn_to_self.mean(0).numpy(),
    ]), axis=0)
    ax.bar(range(n_heads), dominant, tick_label=[f"H{h}" for h in range(n_heads)])
    ax.set_ylabel("Dominant position (0=a, 1=b, 2=self)")
    ax.set_title("Dominant attention target per head")
    ax.set_ylim(-0.5, 2.5)

    plt.suptitle(f"Attention Variation ({fixed_label}={fixed_val}, varying the other)",
                 fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved attention variation plot to {save_path}")
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

    model = load_model_from_checkpoint(args.checkpoint, device=device)
    print(f"Loaded model from {args.checkpoint}")

    # Experiment 1: Fix a=0, vary b
    print("Running: fixed a=0, varying b...")
    attn_a0_b, attn_b_b, attn_self_b = attention_vs_input(
        model, p, device, fixed_a=0, seed=int(cfg.get("seed", 42))
    )
    plot_attention_variation(
        attn_a0_b, attn_b_b, attn_self_b, p,
        fixed_label="a", fixed_val=0,
        save_path=os.path.join(args.output_dir, "attention_vary_b_fixed_a0.png"),
    )

    # Experiment 2: Fix b=0, vary a
    print("Running: fixed b=0, varying a...")
    attn_a_a, attn_b_a, attn_self_a = attention_vs_input(
        model, p, device, fixed_b=0, seed=int(cfg.get("seed", 42))
    )
    plot_attention_variation(
        attn_a_a, attn_b_a, attn_self_a, p,
        fixed_label="b", fixed_val=0,
        save_path=os.path.join(args.output_dir, "attention_vary_a_fixed_b0.png"),
    )

    print("Attention variation analysis complete.")
