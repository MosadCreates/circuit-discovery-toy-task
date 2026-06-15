import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from src.data.modular_addition import ModularAdditionDataset
from src.analysis.attention_patterns import load_model_from_checkpoint
from src.utils.config import load_yaml


def logit_diff_fn(logits: torch.Tensor, correct_tokens: torch.Tensor, p: int):
    """
    Compute the logit difference metric:
    diff = logit(correct) - mean(logit(incorrect))

    Args:
        logits: [batch, d_vocab] logits at the final position
        correct_tokens: [batch] correct answer token IDs
        p: the prime modulus (number of answer tokens)

    Returns:
        diff: [batch] logit difference values
    """
    batch_size = logits.shape[0]
    correct_logits = logits[torch.arange(batch_size), correct_tokens]  # [batch]
    # Mask out the = token (token p) from incorrect set — only consider 0..p-1
    all_logits = logits[:, :p]  # [batch, p]
    # Create mask: True for incorrect tokens
    mask = torch.ones_like(all_logits, dtype=torch.bool)
    mask[torch.arange(batch_size), correct_tokens] = False
    incorrect_logits = all_logits[mask].reshape(batch_size, p - 1)  # [batch, p-1]
    mean_incorrect = incorrect_logits.mean(dim=1)  # [batch]
    return correct_logits - mean_incorrect


def recovery_score(
    patched_diff: float, clean_diff: float, corrupted_diff: float
) -> float:
    """Compute recovery as fraction of clean-corrupted gap restored."""
    denom = clean_diff - corrupted_diff
    if abs(denom) < 1e-8:
        return 0.0
    return (patched_diff - corrupted_diff) / denom


class ActivationPatcher:
    """
    Activation patching infrastructure using TransformerLens hooks.

    Runs clean and corrupted forward passes, caches activations,
    and supports patching any named activation from clean into corrupted.
    """

    def __init__(self, model: HookedTransformer, p: int, device: str = "cpu"):
        self.model = model
        self.p = p
        self.device = device
        self.clean_cache = None
        self.corrupted_cache = None

    def make_inputs(self, a: int, b: int):
        """Create a batch of 1 clean input [a, b, =]."""
        tokens = torch.tensor([[a, b, self.p]], dtype=torch.long, device=self.device)
        return tokens

    def make_corrupted_inputs(self, a: int, b: int, strategy: str = "random"):
        """Create a corrupted version of (a,b) with the same answer or different."""
        p = self.p
        if strategy == "random":
            # Replace a with a random different value
            a_new = (a + torch.randint(1, p, (1,)).item()) % p
            b_new = b
        elif strategy == "symmetric":
            # Replace (a,b) with different values that sum to the same answer
            shift = torch.randint(1, p, (1,)).item()
            a_new = (a + shift) % p
            b_new = (b - shift) % p
        elif strategy == "zero_ablation":
            # Return the same input — we'll ablate activations to zero in hooks
            return self.make_inputs(a, b)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return self.make_inputs(a_new, b_new)

    def run_and_cache(self, tokens: torch.Tensor):
        """Run model and return (logits, cache_dict)."""
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(
                tokens, return_cache_object=True
            )
        return logits, cache

    def compute_metrics(self, logits: torch.Tensor, correct_token: int):
        """Compute logit difference for a single example."""
        diff = logit_diff_fn(logits, torch.tensor([correct_token], device=self.device), self.p)
        pred = logits[0, :self.p].argmax().item()
        acc = 1.0 if pred == correct_token else 0.0
        return diff.item(), acc, pred

    def patch_and_run(
        self,
        corrupted_tokens: torch.Tensor,
        correct_token: int,
        hook_name: str = None,
        patch_positions: list = None,
        patch_fn: callable = None,
    ):
        """
        Run corrupted input with a specific activation patched from clean.

        Args:
            corrupted_tokens: [1, 3] corrupted input
            correct_token: correct answer token ID
            hook_name: name of the hook to patch (e.g., "blocks.0.attn.hook_z")
            patch_positions: list of positions to patch (None = all)
            patch_fn: custom patch function (overrides default)
                      signature: (corrupted_activation, hook) -> patched_activation

        Returns:
            diff: logit difference after patching
            acc: accuracy after patching
        """
        assert self.clean_cache is not None, "Must run clean pass first"

        def default_patch_fn(activations, hook):
            clean_act = self.clean_cache[hook.name]
            if patch_positions is not None:
                for pos in patch_positions:
                    activations[:, pos] = clean_act[:, pos]
                return activations
            return clean_act

        fn = patch_fn if patch_fn is not None else default_patch_fn

        with torch.no_grad():
            patched_logits = self.model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, fn)] if hook_name else [],
            )

        return self.compute_metrics(patched_logits[:, -1, :], correct_token)

    def run_clean_and_corrupted(
        self, a: int, b: int, strategy: str = "random"
    ):
        """Run clean and corrupted passes and cache both."""
        clean_tokens = self.make_inputs(a, b)
        corrupted_tokens = self.make_corrupted_inputs(a, b, strategy=strategy)

        self.clean_logits, self.clean_cache = self.run_and_cache(clean_tokens)
        self.corrupted_logits, self.corrupted_cache = self.run_and_cache(
            corrupted_tokens
        )

        correct_token = (a + b) % self.p
        clean_diff, clean_acc, _ = self.compute_metrics(
            self.clean_logits[:, -1, :], correct_token
        )
        corr_diff, corr_acc, _ = self.compute_metrics(
            self.corrupted_logits[:, -1, :], correct_token
        )

        self._current_metrics = {
            "a": a,
            "b": b,
            "correct_token": correct_token,
            "clean_diff": clean_diff,
            "clean_acc": clean_acc,
            "corrupted_diff": corr_diff,
            "corrupted_acc": corr_acc,
        }

        return clean_tokens, corrupted_tokens, correct_token

    def get_hook_names(self):
        """Return the full list of relevant hook names for the model."""
        hooks = [
            "hook_embed",
            "hook_pos_embed",
        ]
        for l in range(self.model.cfg.n_layers):
            hooks.extend([
                f"blocks.{l}.hook_resid_pre",
                f"blocks.{l}.ln1.hook_normalized",
                f"blocks.{l}.attn.hook_q",
                f"blocks.{l}.attn.hook_k",
                f"blocks.{l}.attn.hook_v",
                f"blocks.{l}.attn.hook_z",
                f"blocks.{l}.attn.hook_attn_scores",
                f"blocks.{l}.attn.hook_pattern",
                f"blocks.{l}.hook_attn_out",
                f"blocks.{l}.hook_resid_mid",
                f"blocks.{l}.ln2.hook_normalized",
                f"blocks.{l}.mlp.hook_pre",
                f"blocks.{l}.mlp.hook_post",
                f"blocks.{l}.hook_mlp_out",
                f"blocks.{l}.hook_resid_post",
            ])
        hooks.extend([
            "ln_final.hook_normalized",
            "unembed.hook_in",
            "unembed.hook_out",
        ])
        return hooks


def get_all_hook_names(model: HookedTransformer):
    """Utility: print all available hook names."""
    patcher = ActivationPatcher(model, 0)
    hooks = patcher.get_hook_names()
    for h in hooks:
        print(h)
    return hooks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/checkpoints/best_val_acc.pt")
    parser.add_argument("--config", type=str,
                        default="configs/patching/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results/patching")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = int(cfg.get("p", 113))

    model = load_model_from_checkpoint(args.checkpoint, device=device)
    patcher = ActivationPatcher(model, p, device=device)

    print("Available hook names:")
    for h in patcher.get_hook_names():
        print(f"  {h}")

    # Quick test: patch head output for one example
    a, b = 5, 3
    strategy = "random"
    clean_tokens, corrupted_tokens, correct_token = patcher.run_clean_and_corrupted(
        a, b, strategy=strategy
    )
    m = patcher._current_metrics
    print(f"\nClean (a={a}, b={b}, answer={correct_token}): "
          f"diff={m['clean_diff']:.3f}, acc={m['clean_acc']:.2f}")
    print(f"Corrupted: diff={m['corrupted_diff']:.3f}, acc={m['corrupted_acc']:.2f}")

    # Test patching the full residual stream at the = position
    hook_name = "blocks.0.hook_resid_mid"
    diff, acc, pred = patcher.patch_and_run(
        corrupted_tokens, correct_token,
        hook_name=hook_name,
        patch_positions=[2],  # = position
    )
    rec = recovery_score(diff, m["clean_diff"], m["corrupted_diff"])
    print(f"Patch resid_mid[=]: diff={diff:.3f}, acc={acc:.2f}, recovery={rec:.2%}")

    print("Patcher test complete.")
