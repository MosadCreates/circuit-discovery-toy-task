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


def patch_attention_heads(
    patcher: ActivationPatcher,
    a: int,
    b: int,
    strategy: str = "random",
    n_samples: int = 100,
):
    """
    For each attention head, patch its output (hook_z) from clean to corrupted,
    averaged over n_samples random inputs.

    Returns:
        head_recovery: [n_layers, n_heads] average recovery per head
    """
    p = patcher.p
    n_layers = patcher.model.cfg.n_layers
    n_heads = patcher.model.cfg.n_heads
    accum = torch.zeros(n_layers, n_heads)
    counts = torch.zeros(n_layers, n_heads)

    rng = torch.Generator().manual_seed(42)

    for _ in range(n_samples):
        a = torch.randint(0, p, (1,), generator=rng).item()
        b = torch.randint(0, p, (1,), generator=rng).item()

        clean_tokens, corrupted_tokens, correct_token = (
            patcher.run_clean_and_corrupted(a, b, strategy=strategy)
        )
        m = patcher._current_metrics
        denom = m["clean_diff"] - m["corrupted_diff"]
        if abs(denom) < 1e-8:
            continue

        for l in range(n_layers):
            for h in range(n_heads):
                # Patch hook_z (value * attention weights, before O_proj)
                hook_name = f"blocks.{l}.attn.hook_z"

                def make_patch_fn(clean_cache, head_idx):
                    def fn(activations, hook):
                        clean_act = clean_cache[hook.name]
                        activations[:, :, head_idx:head_idx+1, :] = \
                            clean_act[:, :, head_idx:head_idx+1, :]
                        return activations
                    return fn

                diff, acc, _ = patcher.patch_and_run(
                    corrupted_tokens, correct_token,
                    hook_name=hook_name,
                    patch_fn=make_patch_fn(patcher.clean_cache, h),
                )
                rec = recovery_score(diff, m["clean_diff"], m["corrupted_diff"])
                accum[l, h] += rec
                counts[l, h] += 1

    head_recovery = accum / counts.clamp(min=1)
    return head_recovery


def plot_head_patching(
    head_recovery: torch.Tensor,
    save_path: str = "results/patching/head_patch.png",
):
    """Bar chart of head-level patching recovery."""
    n_layers, n_heads = head_recovery.shape
    data = head_recovery.cpu().numpy() * 100  # to percent

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(n_heads)
    width = 0.6

    colors = ["steelblue", "coral", "seagreen", "gold"]
    for l in range(n_layers):
        offset = 0
        ax.bar(x + offset * width, data[l], width,
               label=f"Layer {l}", alpha=0.8)

    ax.bar(x, data.flatten(), width, color=colors[:n_heads],
           tick_label=[f"H{h}" for h in range(n_heads)])
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Recovery %")
    ax.set_title("Head-Level Activation Patching: Recovery by Head")

    for i, val in enumerate(data.flatten()):
        ax.text(i, val + 1, f"{val:.1f}%", ha="center", va="bottom",
                fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved head patching plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/checkpoints/best_val_acc.pt")
    parser.add_argument("--config", type=str,
                        default="configs/patching/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results/patching")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--strategy", type=str, default="random")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = int(cfg.get("p", 113))

    model = load_model_from_checkpoint(args.checkpoint, device=device)
    patcher = ActivationPatcher(model, p, device=device)

    print(f"Running head-level patching ({args.n_samples} samples)...")
    head_recovery = patch_attention_heads(
        patcher, None, None, strategy=args.strategy, n_samples=args.n_samples
    )

    for l in range(head_recovery.shape[0]):
        for h in range(head_recovery.shape[1]):
            print(f"  Layer {l} Head {h}: recovery = {head_recovery[l, h]:.2%}")

    plot_head_patching(head_recovery,
                       save_path=os.path.join(args.output_dir, "head_patch.png"))
    print("Head patching complete.")
