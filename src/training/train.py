import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
from torch import optim
from transformer_lens import HookedTransformer, HookedTransformerConfig
import wandb

from src.data.modular_addition import ModularAdditionDataset, train_test_split
from src.utils.config import parse_args_and_config


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> HookedTransformer:
    model_cfg = HookedTransformerConfig(
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_model=cfg["d_model"],
        d_head=cfg["d_model"] // cfg["n_heads"],
        d_mlp=cfg["d_mlp"],
        d_vocab=cfg["p"] + 1,
        n_ctx=3,
        act_fn="relu",
        init_weights=True,
        normalization_type="LN",
        attention_dir="bidirectional",
        default_prepend_bos=False,
        seed=cfg["seed"],
    )
    model = HookedTransformer(model_cfg)
    return model


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def evaluate(model: HookedTransformer, dataloader, device: str) -> tuple:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        logits = logits[:, -1, :]
        loss = F.cross_entropy(logits, labels)
        acc = compute_accuracy(logits, labels)
        total_loss += loss.item()
        total_acc += acc
        n_batches += 1
    model.train()
    return total_loss / n_batches, total_acc / n_batches


def train(cfg: dict):
    device = cfg["device"]
    set_seed(cfg["seed"])

    # Data
    full_dataset = ModularAdditionDataset(p=cfg["p"], seed=cfg["seed"])
    train_dataset, val_dataset = train_test_split(
        full_dataset, val_split=cfg["val_split"], seed=cfg["seed"]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False
    )

    # Model
    model = build_model(cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # W&B (disabled if project is empty or "none")
    run = None
    wandb_project = cfg.get("wandb_project")
    if wandb_project and wandb_project.lower() != "none":
        try:
            run = wandb.init(
                project=wandb_project,
                name=cfg.get("wandb_run_name"),
                config=cfg,
            )
        except Exception:
            pass

    # Checkpoint dir
    ckpt_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "results", "checkpoints"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    save_steps_list = cfg.get("save_steps")
    if save_steps_list is None:
        checkpoint_every = cfg.get("checkpoint_every", 5000)
        save_steps_list = list(range(0, cfg["n_steps"] + 1, checkpoint_every))
    save_steps = set(save_steps_list)

    step = 0
    best_val_acc = 0.0

    # Tracking history for grokking dynamics
    history = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    while step < cfg["n_steps"]:
        for inputs, labels in train_loader:
            if step >= cfg["n_steps"]:
                break

            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)
            logits = logits[:, -1, :]
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = compute_accuracy(logits, labels)

            if step % cfg["log_every"] == 0:
                val_loss, val_acc = evaluate(model, val_loader, device)

                history["step"].append(step)
                history["train_loss"].append(loss.item())
                history["val_loss"].append(val_loss)
                history["train_acc"].append(acc)
                history["val_acc"].append(val_acc)

                print(
                    f"Step {step:6d} | "
                    f"Train Loss {loss.item():.4f} Acc {acc:.4f} | "
                    f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}"
                )

                if run:
                    wandb.log({
                        "train_loss": loss.item(),
                        "val_loss": val_loss,
                        "train_acc": acc,
                        "val_acc": val_acc,
                    }, step=step)

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_path = os.path.join(ckpt_dir, "best_val_acc.pt")
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "val_acc": val_acc,
                        "cfg": cfg,
                    }, best_path)

            # Periodic checkpoint
            if step in save_steps:
                ckpt_path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_acc": history["val_acc"][-1] if history["val_acc"] else 0.0,
                    "train_acc": history["train_acc"][-1] if history["train_acc"] else 0.0,
                    "cfg": cfg,
                }, ckpt_path)
                print(f"  -> Saved checkpoint: {ckpt_path}")

            step += 1

    # Save final checkpoint
    final_path = os.path.join(ckpt_dir, "final.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "val_acc": history["val_acc"][-1] if history["val_acc"] else 0.0,
        "cfg": cfg,
    }, final_path)

    if run:
        run.finish()

    return model, history


if __name__ == "__main__":
    cfg = parse_args_and_config()
    model, history = train(cfg)
    print("Training complete.")
