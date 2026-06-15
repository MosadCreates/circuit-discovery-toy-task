import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from transformer_lens import HookedTransformer, HookedTransformerConfig

from src.utils.config import load_yaml


def get_attention_patterns(model: HookedTransformer, tokens: torch.Tensor):
    """
    Returns the full attention pattern tensor using TransformerLens cache.
    Shape: [n_layers, n_heads, seq_len, seq_len]
    """
    _, cache = model.run_with_cache(tokens, return_cache_object=True)
    n_layers = model.cfg.n_layers
    patterns = []
    for l in range(n_layers):
        key = f"blocks.{l}.attn.hook_pattern"
        pat = cache[key]  # [batch, heads, seq_len, seq_len]
        patterns.append(pat.squeeze(0))  # remove batch dim -> [heads, seq_len, seq_len]
    return torch.stack(patterns, dim=0)  # [n_layers, n_heads, seq_len, seq_len]


def visualize_attention_patterns(
    attn_patterns: torch.Tensor,
    tokens: torch.Tensor,
    title: str = "Attention Patterns",
    save_path: str = None,
):
    """
    Produces a grid of attention heatmaps, one per head.
    attn_patterns: [n_layers, n_heads, seq_len, seq_len]
    tokens: [batch, seq_len] (single example)
    """
    n_layers, n_heads, seq_len, _ = attn_patterns.shape
    token_labels = [str(t.item()) for t in tokens[0]]

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads, 4 * n_layers),
                             squeeze=False)
    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l][h]
            pattern = attn_patterns[l, h].detach().cpu().numpy()
            im = ax.imshow(pattern, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(token_labels)
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(token_labels)
            ax.set_title(f"Layer {l}, Head {h}")
            ax.set_xlabel("Key (attended to)")
            ax.set_ylabel("Query (attending)")
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved attention patterns to {save_path}")
    plt.close()


def load_model_from_checkpoint(ckpt_path: str, device: str = "cpu"):
    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_cfg = saved["cfg"]
    model_cfg = HookedTransformerConfig(
        n_layers=int(raw_cfg["n_layers"]),
        n_heads=int(raw_cfg["n_heads"]),
        d_model=int(raw_cfg["d_model"]),
        d_head=int(raw_cfg["d_model"]) // int(raw_cfg["n_heads"]),
        d_mlp=int(raw_cfg["d_mlp"]),
        d_vocab=int(raw_cfg["p"]) + 1,
        n_ctx=3,
        act_fn="relu",
        init_weights=False,
        normalization_type="LN",
        attention_dir="bidirectional",
        default_prepend_bos=False,
    )
    model = HookedTransformer(model_cfg)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/checkpoints/best_val_acc.pt")
    parser.add_argument("--config", type=str,
                        default="configs/analysis/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results/attention")
    parser.add_argument("--n_samples", type=int, default=5)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model_from_checkpoint(args.checkpoint, device=device)
    p = int(cfg.get("p", 113))
    print(f"Loaded model from {args.checkpoint}")

    rng = torch.Generator().manual_seed(cfg.get("seed", 42))
    for i in range(args.n_samples):
        a = torch.randint(0, p, (1,), generator=rng)
        b = torch.randint(0, p, (1,), generator=rng)
        eq = torch.tensor([p])
        tokens = torch.stack([a, b, eq], dim=1).to(device)

        attn = get_attention_patterns(model, tokens)
        save_path = os.path.join(args.output_dir, f"attention_sample_{i}.png")
        visualize_attention_patterns(
            attn, tokens,
            title=f"Sample {i}: a={a.item()}, b={b.item()}",
            save_path=save_path,
        )

    print("Attention pattern visualisation complete.")
