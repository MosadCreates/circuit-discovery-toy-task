import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.attention_patterns import load_model_from_checkpoint
from src.patching.patcher import ActivationPatcher, recovery_score
from src.utils.config import load_yaml


def patch_mlp_full(
    patcher: ActivationPatcher,
    strategy: str = "random",
    n_samples: int = 100,
):
    """
    Patch the full MLP output (hook_mlp_out) from clean to corrupted.

    Returns:
        avg_recovery: float
    """
    p = patcher.p
    hook_name = "blocks.0.hook_mlp_out"
    total_rec = 0.0
    count = 0

    rng = torch.Generator().manual_seed(42)

    for _ in range(n_samples):
        a = torch.randint(0, p, (1,), generator=rng).item()
        b = torch.randint(0, p, (1,), generator=rng).item()

        _, corrupted_tokens, correct_token = (
            patcher.run_clean_and_corrupted(a, b, strategy=strategy)
        )
        m = patcher._current_metrics
        denom = m["clean_diff"] - m["corrupted_diff"]
        if abs(denom) < 1e-8:
            continue

        diff, acc, _ = patcher.patch_and_run(
            corrupted_tokens, correct_token,
            hook_name=hook_name,
            patch_positions=[2],  # patch at = position only
        )
        rec = recovery_score(diff, m["clean_diff"], m["corrupted_diff"])
        total_rec += rec
        count += 1

    return total_rec / max(count, 1)


def patch_mlp_neurons(
    patcher: ActivationPatcher,
    strategy: str = "random",
    n_samples: int = 100,
):
    """
    Patch individual MLP neurons (hook_pre or hook_post dimensions).

    For each neuron index, patch just that neuron's activation from clean
    to corrupted and measure recovery.

    Returns:
        neuron_recovery: [d_mlp] average recovery per neuron
    """
    p = patcher.p
    d_mlp = patcher.model.cfg.d_mlp
    hook_name = "blocks.0.mlp.hook_post"
    accum = torch.zeros(d_mlp)
    counts = torch.zeros(d_mlp)

    rng = torch.Generator().manual_seed(42)

    for _ in range(n_samples):
        a = torch.randint(0, p, (1,), generator=rng).item()
        b = torch.randint(0, p, (1,), generator=rng).item()

        _, corrupted_tokens, correct_token = (
            patcher.run_clean_and_corrupted(a, b, strategy=strategy)
        )
        m = patcher._current_metrics
        denom = m["clean_diff"] - m["corrupted_diff"]
        if abs(denom) < 1e-8:
            continue

        for n in range(d_mlp):
            def make_neuron_fn(clean_cache, neuron_idx):
                def fn(activations, hook):
                    clean_act = clean_cache[hook.name]
                    activations[:, 2, neuron_idx] = clean_act[:, 2, neuron_idx]
                    return activations
                return fn

            diff, acc, _ = patcher.patch_and_run(
                corrupted_tokens, correct_token,
                hook_name=hook_name,
                patch_fn=make_neuron_fn(patcher.clean_cache, n),
            )
            rec = recovery_score(diff, m["clean_diff"], m["corrupted_diff"])
            accum[n] += rec
            counts[n] += 1

    neuron_recovery = accum / counts.clamp(min=1)
    return neuron_recovery


def plot_neuron_patching(
    neuron_recovery: torch.Tensor,
    top_k: int = 20,
    save_path: str = "results/patching/neuron_patch.png",
):
    """Histogram of neuron-level patching effects."""
    data = neuron_recovery.cpu().numpy() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(data, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Recovery %")
    ax.set_ylabel("Number of neurons")
    ax.set_title("Distribution of Neuron-Level Patching Effects")
    ax.axvline(x=data.mean(), color="green", linestyle="--", alpha=0.7,
               label=f"Mean: {data.mean():.2f}%")
    ax.legend()

    # Top-k neurons
    ax = axes[1]
    top_idx = np.argsort(-data)[:top_k]
    ax.barh(range(top_k), data[top_idx], color="coral", edgecolor="white")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"N{idx}" for idx in top_idx])
    ax.set_xlabel("Recovery %")
    ax.set_title(f"Top-{top_k} Most Causally Important Neurons")
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved neuron patching plot to {save_path}")
    plt.close()

    return top_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/checkpoints/best_val_acc.pt")
    parser.add_argument("--config", type=str,
                        default="configs/patching/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results/patching")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--strategy", type=str, default="random")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = int(cfg.get("p", 113))

    model = load_model_from_checkpoint(args.checkpoint, device=device)
    patcher = ActivationPatcher(model, p, device=device)

    print("Patching full MLP output...")
    full_recovery = patch_mlp_full(patcher, strategy=args.strategy,
                                   n_samples=args.n_samples)
    print(f"  Full MLP patch recovery: {full_recovery:.2%}")

    print(f"Patching individual MLP neurons ({args.n_samples} samples)...")
    neuron_recovery = patch_mlp_neurons(patcher, strategy=args.strategy,
                                        n_samples=args.n_samples)
    top_idx = plot_neuron_patching(neuron_recovery, top_k=args.top_k,
                                    save_path=os.path.join(args.output_dir,
                                                            "neuron_patch.png"))
    print(f"  Top-{args.top_k} neurons by causal importance:")
    data = neuron_recovery.cpu().numpy() * 100
    for rank, idx in enumerate(top_idx[:args.top_k]):
        print(f"    #{rank+1}: Neuron {idx} — recovery {data[idx]:.2f}%")

    print("MLP patching complete.")
