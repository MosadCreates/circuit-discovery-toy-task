import argparse
import os
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged = base.copy()
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str, cli_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    config = load_yaml(config_path)
    if cli_overrides:
        filtered = {k: v for k, v in cli_overrides.items() if v is not None}
        config = deep_merge(config, filtered)
    return config


def parse_args_and_config() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Circuit Discovery on a Toy Task — Modular Addition"
    )
    parser.add_argument("--config", type=str, default="configs/training/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--p", type=int, default=None, help="Prime modulus")
    parser.add_argument("--n_layers", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=None, help="Number of attention heads per layer")
    parser.add_argument("--d_model", type=int, default=None, help="Residual stream dimension")
    parser.add_argument("--d_mlp", type=int, default=None, help="MLP hidden dimension")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="AdamW weight decay")
    parser.add_argument("--n_steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--val_split", type=float, default=None, help="Validation set fraction")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    args = parser.parse_args()
    config = load_yaml(args.config)

    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    config = deep_merge(config, overrides)

    # Resolve device
    if config.get("device") == "auto" or config.get("device") is None:
        import torch
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve wandb env vars
    for key in ("wandb_project", "wandb_run_name", "wandb_entity"):
        env_key = f"WANDB_{key.split('_')[-1].upper()}"
        if config.get(key) is None:
            config[key] = os.environ.get(env_key)

    return config
