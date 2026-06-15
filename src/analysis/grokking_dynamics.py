import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from transformer_lens import HookedTransformer

from src.data.modular_addition import ModularAdditionDataset, train_test_split
from src.utils.config import load_yaml


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    import torch.nn.functional as F
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        logits = logits[:, -1, :]
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * len(inputs)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(inputs)
    model.train()
    return total_loss / total, correct / total


def load_checkpoint_and_evaluate(ckpt_path, model_cfg, dataloader, device):
    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HookedTransformer(model_cfg)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)
    loss, acc = evaluate_model(model, dataloader, device)
    step = saved.get("step", 0)
    del model
    return step, loss, acc


def analyse_grokking_dynamics(
    checkpoint_dir: str,
    p: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_mlp: int,
    val_split: float,
    seed: int,
    device: str,
    output_dir: str = "results",
):
    # Data
    full_dataset = ModularAdditionDataset(p=p, seed=seed)
    _, val_dataset = train_test_split(full_dataset, val_split=val_split, seed=seed)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    # Model config (must match training)
    from transformer_lens import HookedTransformerConfig
    model_cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_model // n_heads,
        d_mlp=d_mlp,
        d_vocab=p + 1,
        n_ctx=3,
        act_fn="relu",
        use_attn_mask=False,
        init_weights=False,
        normalization_type="LN",
    )

    # Find all checkpoint files
    ckpt_files = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("step_") and f.endswith(".pt")
    ], key=lambda x: int(x.split("_")[1].split(".")[0]))

    if not ckpt_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    steps = []
    val_losses = []
    val_accs = []

    print(f"Found {len(ckpt_files)} checkpoints, evaluating...")
    for ckpt_file in ckpt_files:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        step, loss, acc = load_checkpoint_and_evaluate(
            ckpt_path, model_cfg, val_loader, device
        )
        steps.append(step)
        val_losses.append(loss)
        val_accs.append(acc)
        print(f"  Step {step:6d}: val_loss={loss:.4f}, val_acc={acc:.4f}")

    # Sort by step
    order = np.argsort(steps)
    steps = np.array(steps)[order]
    val_losses = np.array(val_losses)[order]
    val_accs = np.array(val_accs)[order]

    # Find grokking point (first step where val_acc > 0.95)
    grokking_idx = np.where(val_accs > 0.95)[0]
    grokking_step = steps[grokking_idx[0]] if len(grokking_idx) > 0 else None

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_loss = "tab:blue"
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss", color=color_loss)
    ax1.plot(steps, val_losses, color=color_loss, marker="o", markersize=4,
             label="Validation Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax2 = ax1.twinx()
    color_acc = "tab:green"
    ax2.set_ylabel("Accuracy", color=color_acc)
    ax2.plot(steps, val_accs, color=color_acc, marker="s", markersize=4,
             label="Validation Accuracy")
    ax2.tick_params(axis="y", labelcolor=color_acc)

    if grokking_step is not None:
        ax1.axvline(x=grokking_step, color="red", linestyle="--", alpha=0.7,
                    label=f"Grokking Point (step {grokking_step})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title("Grokking Dynamics on Modular Addition")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "grokking_dynamics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved grokking dynamics plot to {path}")
    plt.close()

    return steps, val_losses, val_accs, grokking_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str,
                        default="results/checkpoints")
    parser.add_argument("--config", type=str,
                        default="configs/analysis/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    analyse_grokking_dynamics(
        checkpoint_dir=args.checkpoint_dir,
        p=cfg.get("p", 113),
        d_model=cfg.get("d_model", 128),
        n_layers=cfg.get("n_layers", 1),
        n_heads=cfg.get("n_heads", 4),
        d_mlp=cfg.get("d_mlp", 512),
        val_split=0.3,
        seed=cfg.get("seed", 42),
        device=device,
        output_dir=args.output_dir,
    )
