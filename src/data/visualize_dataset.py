import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch

# Ensure project root is on PATH for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.modular_addition import ModularAdditionDataset


def visualize_dataset(p: int = 113, output_dir: str = "results"):
    dataset = ModularAdditionDataset(p=p)
    labels = dataset.labels.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of answers
    axes[0].hist(labels, bins=p, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[0].set_title(f"Answer Distribution (p={p})")
    axes[0].set_xlabel("Answer token (a + b) mod p")
    axes[0].set_ylabel("Count")
    axes[0].axhline(y=len(labels) / p, color="red", linestyle="--", alpha=0.7,
                    label=f"Expected uniform = {len(labels)/p:.1f}")
    axes[0].legend()

    # Check uniformity
    counts = torch.bincount(torch.tensor(labels), minlength=p)
    axes[1].bar(range(p), counts, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[1].axhline(y=len(labels) / p, color="red", linestyle="--", alpha=0.7)
    axes[1].set_title("Deviation from Uniform")
    axes[1].set_xlabel("Answer token")
    axes[1].set_ylabel("Count minus expected")
    axes[1].set_ylim(counts.min().item() - 10, counts.max().item() + 10)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "dataset_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved dataset distribution plot to {path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    visualize_dataset(p=args.p, output_dir=args.output_dir)
