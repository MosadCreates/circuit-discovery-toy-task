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


def path_patch_attention_to_mlp(
    patcher: ActivationPatcher,
    strategy: str = "random",
    n_samples: int = 100,
):
    """
    Path patching: patch the output of attention h into the MLP.

    We patch hook_resid_mid (after attention, before MLP) at the
    = position to measure combined attention-to-MLP flow.

    Then for each specific head h, we patch only that head's
    hook_z output at the = position to measure head-specific
    contribution to the MLP.
    """
    p = patcher.p
    n_heads = patcher.model.cfg.n_heads
    n_layers = patcher.model.cfg.n_layers

    # Result containers
    attn_to_mlp_recovery = 0.0
    head_to_mlp_recovery = torch.zeros(n_layers, n_heads)
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

        # Patch resid_mid at = position — this is the combined attention
        # output flowing into the MLP
        diff, acc, _ = patcher.patch_and_run(
            corrupted_tokens, correct_token,
            hook_name="blocks.0.hook_resid_mid",
            patch_positions=[2],
        )
        attn_to_mlp_recovery += recovery_score(
            diff, m["clean_diff"], m["corrupted_diff"]
        )

        # For each head, patch its z-output at the = position specifically
        for l in range(n_layers):
            for h in range(n_heads):
                def make_head_path_fn(clean_cache, layer_idx, head_idx):
                    def fn(activations, hook):
                        clean_act = clean_cache[hook.name]
                        # Patch only the = position for this specific head
                        activations[:, 2, head_idx:head_idx+1, :] = \
                            clean_act[:, 2, head_idx:head_idx+1, :]
                        return activations
                    return fn

                diff_h, _, _ = patcher.patch_and_run(
                    corrupted_tokens, correct_token,
                    hook_name=f"blocks.{l}.attn.hook_z",
                    patch_fn=make_head_path_fn(patcher.clean_cache, l, h),
                )
                head_to_mlp_recovery[l, h] += recovery_score(
                    diff_h, m["clean_diff"], m["corrupted_diff"]
                )

        count += 1

    attn_to_mlp_recovery /= max(count, 1)
    head_to_mlp_recovery /= max(count, 1)

    return attn_to_mlp_recovery, head_to_mlp_recovery


def plot_path_patching(
    attn_recovery: float,
    head_recovery: torch.Tensor,
    save_path: str = "results/patching/path_patch.png",
):
    """Plot path patching results."""
    n_layers, n_heads = head_recovery.shape
    data = head_recovery.cpu().numpy() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar 1: Combined attention -> MLP
    ax = axes[0]
    ax.bar(["Attention -> MLP\n(combined)"], [attn_recovery * 100],
           color="steelblue", width=0.4)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_ylabel("Recovery %")
    ax.set_title("Path: All Attention -> MLP at = position")
    ax.set_ylim(min(attn_recovery * 100 - 10, -5),
                max(attn_recovery * 100 + 10, 10))

    # Bar 2: Per-head path patching
    ax = axes[1]
    x = np.arange(n_heads * n_layers)
    labels = []
    for l in range(n_layers):
        for h in range(n_heads):
            labels.append(f"L{l}H{h}")
    colors = ["steelblue", "coral", "seagreen", "gold"][:len(labels)]
    ax.bar(x, data.flatten(), color=colors, tick_label=labels)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Head")
    ax.set_ylabel("Recovery %")
    ax.set_title("Path: Individual Head -> MLP at = position")

    for i, val in enumerate(data.flatten()):
        ax.text(i, val + 0.5, f"{val:.1f}%", ha="center", va="bottom",
                fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved path patching plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/checkpoints/best_val_acc.pt")
    parser.add_argument("--config", type=str,
                        default="configs/patching/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results/patching")
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--strategy", type=str, default="random")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = int(cfg.get("p", 113))

    model = load_model_from_checkpoint(args.checkpoint, device=device)
    patcher = ActivationPatcher(model, p, device=device)

    print(f"Running path patching ({args.n_samples} samples)...")
    attn_recovery, head_recovery = path_patch_attention_to_mlp(
        patcher, strategy=args.strategy, n_samples=args.n_samples
    )

    print(f"  Attention -> MLP path: {attn_recovery:.2%} recovery")
    for l in range(head_recovery.shape[0]):
        for h in range(head_recovery.shape[1]):
            print(f"  Head L{l}H{h} -> MLP: {head_recovery[l, h]:.2%}")

    plot_path_patching(attn_recovery, head_recovery,
                       save_path=os.path.join(args.output_dir, "path_patch.png"))
    print("Path patching complete.")
