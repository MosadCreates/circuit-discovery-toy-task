import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from transformer_lens import HookedTransformer

from src.analysis.attention_patterns import load_model_from_checkpoint
from src.data.modular_addition import ModularAdditionDataset, train_test_split
from src.patching.patcher import ActivationPatcher, recovery_score
from src.utils.config import load_yaml


def compute_component_importance(
    patcher: ActivationPatcher,
    n_samples: int = 200,
    strategy: str = "random",
):
    p = patcher.p
    n_heads = patcher.model.cfg.n_heads
    d_mlp = patcher.model.cfg.d_mlp

    head_accum = torch.zeros(n_heads)
    head_counts = torch.zeros(n_heads)
    neuron_accum = torch.zeros(d_mlp)
    neuron_counts = torch.zeros(d_mlp)
    mlp_accum = 0.0
    mlp_count = 0

    rng = torch.Generator().manual_seed(42)

    for _ in range(n_samples):
        a_val = torch.randint(0, p, (1,), generator=rng).item()
        b_val = torch.randint(0, p, (1,), generator=rng).item()
        _, corrupted_tokens, correct_token = (
            patcher.run_clean_and_corrupted(a_val, b_val, strategy=strategy)
        )
        m = patcher._current_metrics
        denom = m["clean_diff"] - m["corrupted_diff"]
        if abs(denom) < 1e-8:
            continue

        diff, _, _ = patcher.patch_and_run(
            corrupted_tokens, correct_token,
            hook_name="blocks.0.hook_mlp_out",
            patch_positions=[2],
        )
        mlp_accum += recovery_score(diff, m["clean_diff"], m["corrupted_diff"])
        mlp_count += 1

        for h in range(n_heads):
            def make_head_fn(clean_cache, head_idx):
                def fn(activations, hook):
                    clean_act = clean_cache[hook.name]
                    activations[:, 2, head_idx:head_idx+1, :] = \
                        clean_act[:, 2, head_idx:head_idx+1, :]
                    return activations
                return fn

            diff_h, _, _ = patcher.patch_and_run(
                corrupted_tokens, correct_token,
                hook_name="blocks.0.attn.hook_z",
                patch_fn=make_head_fn(patcher.clean_cache, h),
            )
            head_accum[h] += recovery_score(
                diff_h, m["clean_diff"], m["corrupted_diff"]
            )
            head_counts[h] += 1

        for n in range(d_mlp):
            def make_neuron_fn(clean_cache, neuron_idx):
                def fn(activations, hook):
                    clean_act = clean_cache[hook.name]
                    activations[:, 2, neuron_idx] = clean_act[:, 2, neuron_idx]
                    return activations
                return fn

            diff_n, _, _ = patcher.patch_and_run(
                corrupted_tokens, correct_token,
                hook_name="blocks.0.mlp.hook_post",
                patch_fn=make_neuron_fn(patcher.clean_cache, n),
            )
            neuron_accum[n] += recovery_score(
                diff_n, m["clean_diff"], m["corrupted_diff"]
            )
            neuron_counts[n] += 1

    head_importance = head_accum / head_counts.clamp(min=1)
    neuron_importance = neuron_accum / neuron_counts.clamp(min=1)
    mlp_importance = mlp_accum / max(mlp_count, 1)

    return {
        "head_importance": head_importance,
        "neuron_importance": neuron_importance,
        "mlp_importance": mlp_importance,
    }


def evaluate_ablated(
    model: HookedTransformer,
    val_loader,
    ablate_heads: list = None,
    ablate_neurons: list = None,
    p_prime: int = 113,
    device: str = "cpu",
):
    model.eval()
    correct = 0
    total = 0
    hooks = []

    if ablate_heads:
        for h in ablate_heads:
            def make_head_ablate(head_idx):
                def fn(activations, hook):
                    activations[:, :, head_idx, :] = 0.0
                    return activations
                return fn
            hooks.append(
                ("blocks.0.attn.hook_z", make_head_ablate(h))
            )

    if ablate_neurons:
        def make_neuron_ablate(neuron_indices):
            def fn(activations, hook):
                activations[:, 2, neuron_indices] = 0.0
                return activations
            return fn
        hooks.append(
            ("blocks.0.mlp.hook_post", make_neuron_ablate(ablate_neurons))
        )

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model.run_with_hooks(inputs, fwd_hooks=hooks)
            preds = logits[:, -1, :p_prime].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    model.train()
    return correct / max(total, 1)


def plot_circuit_accuracy(
    accuracy_data: dict,
    full_acc: float,
    save_path: str = "results/circuit_accuracy_vs_components.png",
):
    fig, ax = plt.subplots(figsize=(8, 5))
    components = sorted(accuracy_data.keys())
    accuracies = [accuracy_data[c] for c in components]

    ax.plot(components, [a * 100 for a in accuracies],
            "o-", color="steelblue", linewidth=2, markersize=6)
    ax.axhline(y=full_acc * 100, color="green", linestyle="--",
               label=f"Full model ({full_acc:.1%})", alpha=0.7)

    threshold = 0.9 * full_acc
    ax.axhline(y=threshold * 100, color="red", linestyle=":",
               label=f"90% threshold ({threshold:.1%})", alpha=0.7)

    ax.set_xlabel("Number of circuit components (heads + MLP neurons)")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Minimal Circuit: Accuracy vs Number of Components")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved circuit accuracy plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/checkpoints/best_val_acc.pt")
    parser.add_argument("--config", type=str,
                        default="configs/analysis/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--n_samples_importance", type=int, default="50")
    parser.add_argument("--strategy", type=str, default="random")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p_val = int(cfg.get("p", 113))

    model = load_model_from_checkpoint(args.checkpoint, device=device)
    patcher = ActivationPatcher(model, p_val, device=device)
    n_heads = model.cfg.n_heads
    d_mlp = model.cfg.d_mlp

    importance = compute_component_importance(
        patcher, n_samples=int(args.n_samples_importance),
        strategy=args.strategy
    )

    head_imp = importance["head_importance"]
    neuron_imp = importance["neuron_importance"]
    mlp_imp = importance["mlp_importance"]

    print(f"\nComponent importance (recovery %):")
    print(f"  Full MLP: {mlp_imp:.2%}")
    for h in range(n_heads):
        print(f"  Head {h}: {head_imp[h]:.2%}")
    top_n = min(10, d_mlp)
    top_neurons = neuron_imp.argsort(descending=True)[:top_n]
    print(f"  Top-{top_n} neurons: {top_neurons.tolist()} "
          f"(mean recovery: {neuron_imp[top_neurons].mean():.2%})")

    head_order = head_imp.argsort(descending=True).tolist()
    neuron_order = neuron_imp.argsort(descending=True).tolist()

    full_dataset = ModularAdditionDataset(p=p_val, seed=cfg.get("seed", 42))
    _, val_dataset = train_test_split(
        full_dataset, val_split=0.3, seed=cfg.get("seed", 42)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False
    )

    full_acc = evaluate_ablated(
        model, val_loader, ablate_heads=[], ablate_neurons=[],
        p_prime=p_val, device=device
    )
    print(f"\nFull model validation accuracy: {full_acc:.2%}")

    results = {}
    circuit_heads = set()
    circuit_neurons = set()
    all_heads = list(range(n_heads))
    all_neurons = list(range(d_mlp))

    n_comp = 0

    acc = evaluate_ablated(
        model, val_loader,
        ablate_heads=all_heads, ablate_neurons=all_neurons,
        p_prime=p_val, device=device,
    )
    results[n_comp] = acc
    print(f"  All ablated: acc = {acc:.2%}")

    for h in head_order:
        circuit_heads.add(h)
        n_comp += 1
        inactive_heads = [hi for hi in all_heads if hi not in circuit_heads]
        acc = evaluate_ablated(
            model, val_loader,
            ablate_heads=inactive_heads, ablate_neurons=all_neurons,
            p_prime=p_val, device=device,
        )
        results[n_comp] = acc
        print(f"  +Head {h}: acc = {acc:.2%}")

    max_add_neurons = min(50, d_mlp)
    for n in neuron_order[:max_add_neurons]:
        circuit_neurons.add(n)
        n_comp += 1
        inactive_neurons = [ni for ni in all_neurons if ni not in circuit_neurons]
        # Keep heads active
        acc = evaluate_ablated(
            model, val_loader,
            ablate_heads=[], ablate_neurons=inactive_neurons,
            p_prime=p_val, device=device,
        )
        results[n_comp] = acc
        if n_comp <= 60:
            print(f"  +Neuron {n}: acc = {acc:.2%}")

    threshold = 0.9 * full_acc
    reached = False
    best_n = 0
    for n, acc in sorted(results.items()):
        if acc >= threshold:
            best_n = n
            reached = True
            break

    if reached:
        print(f"\nMinimal circuit: {best_n} components "
              f"achieve {results[best_n]:.2%} "
              f"(threshold: {threshold:.2%})")
    else:
        comp = max(results.keys())
        print(f"\nCircuit with {comp} components: "
              f"{results[comp]:.2%} (threshold: {threshold:.2%})")

    plot_circuit_accuracy(results, full_acc,
                          save_path=os.path.join(args.output_dir,
                                                  "circuit_accuracy_vs_components.png"))
    print("Minimal circuit analysis complete.")
