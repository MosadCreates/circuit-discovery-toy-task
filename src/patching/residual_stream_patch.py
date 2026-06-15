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


def patch_residual_stream(
    patcher: ActivationPatcher,
    a: int,
    b: int,
    strategy: str = "random",
    n_samples: int = 50,
):
    """
    Patch the residual stream at every hook point and token position,
    averaged over n_samples random inputs.

    Returns:
        results: dict of {hook_name: {position: recovery}}
    """
    p = patcher.p
    hook_names = [
        "blocks.0.hook_resid_pre",
        "blocks.0.hook_resid_mid",
        "blocks.0.hook_resid_post",
    ]
    positions = [0, 1, 2]  # a, b, =

    rng = torch.Generator().manual_seed(42)
    accum = {h: {pos: 0.0 for pos in positions} for h in hook_names}
    counts = {h: {pos: 0 for pos in positions} for h in hook_names}

    for _ in range(n_samples):
        a = torch.randint(0, p, (1,), generator=rng).item()
        b = torch.randint(0, p, (1,), generator=rng).item()

        clean_tokens, corrupted_tokens, correct_token = (
            patcher.run_clean_and_corrupted(a, b, strategy=strategy)
        )
        m = patcher._current_metrics
        clean_diff = m["clean_diff"]
        corr_diff = m["corrupted_diff"]
        denom = clean_diff - corr_diff
        if abs(denom) < 1e-8:
            continue

        for hook_name in hook_names:
            for pos in positions:
                diff, acc, _ = patcher.patch_and_run(
                    corrupted_tokens, correct_token,
                    hook_name=hook_name, patch_positions=[pos],
                )
                rec = recovery_score(diff, clean_diff, corr_diff)
                accum[hook_name][pos] += rec
                counts[hook_name][pos] += 1

    # Average
    results = {}
    for hook_name in hook_names:
        results[hook_name] = {}
        for pos in positions:
            n = counts[hook_name][pos]
            results[hook_name][pos] = accum[hook_name][pos] / max(n, 1)

    return results


def plot_residual_stream_patch(
    results: dict,
    save_path: str = "results/patching/residual_stream_patch.png",
):
    """Plot the residual stream patching heatmap."""
    hook_names = list(results.keys())
    positions = list(results[hook_names[0]].keys())
    pos_labels = ["pos a (0)", "pos b (1)", "pos = (2)"]

    data = np.zeros((len(hook_names), len(positions)))
    for i, h in enumerate(hook_names):
        for j, pos in enumerate(positions):
            data[i, j] = results[h][pos] * 100  # to percent

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=-20, vmax=100, aspect="auto")

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(pos_labels)
    ax.set_yticks(range(len(hook_names)))
    ax.set_yticklabels([h.replace("blocks.0.", "") for h in hook_names])
    ax.set_title("Residual Stream Patching: Recovery %")

    # Annotate cells
    for i in range(len(hook_names)):
        for j in range(len(positions)):
            val = data[i, j]
            color = "white" if abs(val) > 50 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color,
                    fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, label="Recovery %")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved residual stream patch heatmap to {save_path}")
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

    print(f"Running residual stream patching ({args.n_samples} samples, "
          f"strategy={args.strategy})...")
    results = patch_residual_stream(
        patcher, None, None, strategy=args.strategy, n_samples=args.n_samples
    )

    for hook_name, pos_data in results.items():
        print(f"  {hook_name}:")
        for pos, rec in pos_data.items():
            print(f"    pos {pos}: {rec:.2%}")

    plot_residual_stream_patch(results,
                               save_path=os.path.join(args.output_dir,
                                                       "residual_stream_patch.png"))
    print("Residual stream patching complete.")
