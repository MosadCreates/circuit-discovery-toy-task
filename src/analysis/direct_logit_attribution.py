import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import einops

from src.analysis.attention_patterns import load_model_from_checkpoint
from src.utils.config import load_yaml


@torch.no_grad()
def compute_direct_logit_attribution(model, tokens: torch.Tensor, correct_token: int):
    """
    Decompose the final logit for the correct answer into contributions
    from each component (embedding, each attention head, MLP).

    Uses the linearity of the residual stream pre-LayerNorm:
    final_resid = embed + pos_embed + sum_h(head_h) + mlp

    We compute each component's output and apply LN + W_U jointly:
    logit = LN(sum_of_components) @ W_U

    For individual attribution, we use the approximation:
    contrib_i = (output_i) @ W_U_eff  where W_U_eff = LN_grad * W_U
    but for clarity we show the pre-LN contribution to the residual.

    Returns:
        component_names: list of component names
        contributions: [n_components] pre-LN residual dot W_U (approximate)
    """
    _, cache = model.run_with_cache(tokens, return_cache_object=True)

    W_U = model.unembed.W_U
    ln_final = model.ln_final

    n_heads = model.cfg.n_heads

    # Collect all component outputs pre-LN at the = position
    component_names = []
    component_outputs = []

    # 1. Embedding + positional embedding (initial residual)
    embed_out = cache["hook_embed"][0, 2, :]
    pos_embed_out = cache["hook_pos_embed"][0, 2, :]
    init_resid = embed_out + pos_embed_out
    component_names.append("Embedding")
    component_outputs.append(init_resid)

    # 2. Per-head attention contributions
    z = cache["blocks.0.attn.hook_z"][0, 2, :, :]  # [n_heads, d_head]
    W_O = model.blocks[0].attn.W_O  # [n_heads, d_head, d_model]

    for h in range(n_heads):
        head_out = z[h] @ W_O[h]  # [d_model]
        component_names.append(f"Head {h}")
        component_outputs.append(head_out)

    # 3. MLP contribution
    mlp_out = cache["blocks.0.hook_mlp_out"][0, 2, :]
    component_names.append("MLP")
    component_outputs.append(mlp_out)

    # 4. Bias contribution
    if model.unembed.b_U is not None:
        component_names.append("Bias")
        bias = torch.zeros_like(embed_out)
        bias[0] = 1.0  # bias is a constant added to all logits
        # Actually, bias is added after W_U, so we handle it separately
        component_outputs.append(bias)  # placeholder

    # Compute full residual
    full_resid = sum(component_outputs)
    # Apply LN once to the full residual
    full_resid_ln = ln_final(
        full_resid.unsqueeze(0).unsqueeze(0)
    ).squeeze(0).squeeze(0)

    # Compute pre-LN logit: approximate each component's contribution
    # by projecting through W_U without LN
    # This is an approximation but commonly used in the literature
    bos_logit = (full_resid_ln @ W_U)[correct_token].item()
    contributions = []
    for output in component_outputs:
        # Estimate contribution via first-order Taylor approximation:
        # The actual contribution is ≈ output_ln @ W_U where
        # output_ln = LN(full_resid) - LN(full_resid - output)
        # As a simpler approximation, use: output @ W_U
        pre_ln_logit = (output @ W_U)[correct_token].item()
        contributions.append(pre_ln_logit)

    # Handle bias separately (added to all logits after W_U)
    if model.unembed.b_U is not None:
        bias_logit = model.unembed.b_U[correct_token].item()
        contributions[-1] = bias_logit

    # Verify: sum of pre-LN approximations vs actual logit
    total_pre_ln = sum(contributions)

    return component_names, contributions


def plot_direct_logit_attribution(
    component_names: list,
    contributions: list,
    correct_token: int,
    save_path: str = "results/direct_logit_attribution.png",
):
    """Plot signed bar chart of component contributions."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(component_names))
    colors = ["steelblue" if c > 0 else "coral" for c in contributions]

    bars = ax.bar(x, contributions, color=colors, edgecolor="white", width=0.6)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(component_names, rotation=45, ha="right")
    ax.set_ylabel(f"Logit Contribution (correct token = {correct_token})")
    ax.set_title("Direct Logit Attribution: Component Contributions")

    # Annotate bars
    for i, (bar, val) in enumerate(zip(bars, contributions)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.1 if val >= 0 else -0.1),
                f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved direct logit attribution to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/checkpoints/best_val_acc.pt")
    parser.add_argument("--config", type=str,
                        default="configs/analysis/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--n_samples", type=int, default=20)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = int(cfg.get("p", 113))

    model = load_model_from_checkpoint(args.checkpoint, device=device)

    rng = torch.Generator().manual_seed(cfg.get("seed", 42))

    # Accumulate contributions over multiple samples
    all_contribs = None
    n_samples_done = 0

    for i in range(args.n_samples):
        a = torch.randint(0, p, (1,), generator=rng).item()
        b = torch.randint(0, p, (1,), generator=rng).item()
        correct = (a + b) % p
        tokens = torch.tensor([[a, b, p]], device=device)

        names, contribs = compute_direct_logit_attribution(
            model, tokens, correct
        )

        if all_contribs is None:
            all_contribs = np.array(contribs)
        else:
            all_contribs += np.array(contribs)
        n_samples_done += 1

    avg_contribs = all_contribs / n_samples_done

    print(f"Average direct logit attribution over {n_samples_done} samples:")
    for name, val in zip(names, avg_contribs):
        print(f"  {name}: {val:.3f}")

    plot_direct_logit_attribution(
        names, avg_contribs, correct,
        save_path=os.path.join(args.output_dir, "direct_logit_attribution.png")
    )
    print("Direct logit attribution complete.")
