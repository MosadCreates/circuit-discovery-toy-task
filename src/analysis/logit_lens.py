import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.attention_patterns import load_model_from_checkpoint
from src.utils.config import load_yaml


@torch.no_grad()
def logit_lens(
    model, tokens: torch.Tensor, p: int
):
    """
    Apply the logit lens at each layer position.

    Takes the residual stream at each hook point, applies LayerNorm
    (if used), and multiplies by W_U to see the predicted distribution
    at each stage.

    Returns:
        logits_by_layer: [n_layers+1, d_vocab] (pre-attn, post-attn, post-mlp)
        hook_names: list of hook names used
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, return_cache_object=True)

    hook_names = []
    logits_list = []

    # Pre-block (after embedding)
    for hook_name in ["blocks.0.hook_resid_pre",
                       "blocks.0.hook_resid_mid",
                       "blocks.0.hook_resid_post"]:
        resid = cache[hook_name][0, 2, :]  # [d_model] at = position
        # Apply final LayerNorm
        resid_ln = model.ln_final(resid.unsqueeze(0).unsqueeze(0)).squeeze()
        logits = resid_ln @ model.unembed.W_U  # [d_vocab]
        logits_list.append(logits)
        hook_names.append(hook_name)

    logits_by_layer = torch.stack(logits_list, dim=0)  # [n_layers+1, d_vocab]
    return logits_by_layer, hook_names


def plot_logit_lens(
    logits_by_layer: torch.Tensor,
    hook_names: list,
    correct_token: int,
    p: int,
    save_path: str = "results/logit_lens.png",
    top_k: int = 5,
):
    """Plot logit lens evolution through layers."""
    n_layers = len(hook_names)
    layer_labels = [h.replace("blocks.0.", "") for h in hook_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Plot 1: Probability of correct vs incorrect tokens per layer
    ax = axes[0]
    probs_by_layer = torch.softmax(logits_by_layer, dim=-1)
    correct_probs = probs_by_layer[:, correct_token].cpu().numpy()

    # Top-3 incorrect answer tokens at final layer (exclude = token)
    final_probs = probs_by_layer[-1, :p]  # only answer tokens
    incorrect_mask = torch.ones(p, dtype=torch.bool)
    incorrect_mask[correct_token] = False
    top_incorrect = final_probs[incorrect_mask].topk(3).indices

    ax.plot(layer_labels, correct_probs, "o-", color="green",
            label=f"Correct (token {correct_token})", linewidth=2,
            markersize=8)
    for i, idx in enumerate(top_incorrect.tolist()):
        incorrect_probs = probs_by_layer[:, idx].cpu().numpy()
        ax.plot(layer_labels, incorrect_probs, "s--", alpha=0.5,
                label=f"Incorrect (token {idx})", linewidth=1)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Softmax Probability")
    ax.set_title("Logit Lens: Prediction Evolution Through Layers")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Heatmap of logits per layer
    ax = axes[1]
    logit_data = logits_by_layer[:, :p].cpu().numpy()  # [n_layers, p]

    im = ax.imshow(logit_data, cmap="RdBu_r", aspect="auto",
                   vmin=-logit_data.max(), vmax=logit_data.max())
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_labels)
    ax.set_xlabel("Token ID")
    ax.set_ylabel("Layer")
    ax.set_title(f"Logits by Layer (correct token = {correct_token})")
    plt.colorbar(im, ax=ax, fraction=0.046, label="Logit")

    # Mark correct token column
    ax.axvline(x=correct_token - 0.5, color="red", linestyle="--",
               alpha=0.5, linewidth=1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved logit lens plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/checkpoints/best_val_acc.pt")
    parser.add_argument("--config", type=str,
                        default="configs/analysis/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = int(cfg.get("p", 113))

    model = load_model_from_checkpoint(args.checkpoint, device=device)

    rng = torch.Generator().manual_seed(cfg.get("seed", 42))
    for i in range(args.n_samples):
        a = torch.randint(0, p, (1,), generator=rng).item()
        b = torch.randint(0, p, (1,), generator=rng).item()
        correct = (a + b) % p
        tokens = torch.tensor([[a, b, p]], device=device)

        logits_by_layer, hook_names = logit_lens(model, tokens, p)

        save_path = os.path.join(args.output_dir,
                                 f"logit_lens_sample_{i}.png")
        plot_logit_lens(
            logits_by_layer, hook_names, correct, p, save_path=save_path
        )
        print(f"Sample {i}: a={a}, b={b}, correct={correct} — "
              f"final prob correct = "
              f"{torch.softmax(logits_by_layer[-1], dim=-1)[correct]:.3f}")

    print("Logit lens analysis complete.")
