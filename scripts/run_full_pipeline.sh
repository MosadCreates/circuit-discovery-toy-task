#!/usr/bin/env bash
set -euo pipefail

# Run the full pipeline: train + all analysis + build docs

echo "=== Stage 1: Training ==="
python src/training/train.py --config configs/training/default.yaml

echo "=== Stage 2: Grokking Dynamics ==="
python src/analysis/grokking_dynamics.py --config configs/analysis/default.yaml

echo "=== Stage 3: Attention Patterns ==="
python src/analysis/attention_patterns.py --config configs/analysis/default.yaml
python src/analysis/head_roles.py --config configs/analysis/default.yaml
python src/analysis/attention_variation.py --config configs/analysis/default.yaml

echo "=== Stage 4: Fourier Analysis ==="
python src/analysis/fourier.py --config configs/analysis/default.yaml
python src/analysis/embedding_analysis.py --config configs/analysis/default.yaml
python src/analysis/neuron_analysis.py --config configs/analysis/default.yaml
python src/analysis/logit_fourier.py --config configs/analysis/default.yaml

echo "=== Stage 5: Activation Patching ==="
python src/patching/patcher.py --config configs/patching/default.yaml
python src/patching/residual_stream_patch.py --config configs/patching/default.yaml
python src/patching/head_patch.py --config configs/patching/default.yaml
python src/patching/mlp_patch.py --config configs/patching/default.yaml
python src/patching/path_patch.py --config configs/patching/default.yaml

echo "=== Stage 6: Logit Attribution ==="
python src/analysis/logit_lens.py --config configs/analysis/default.yaml
python src/analysis/direct_logit_attribution.py --config configs/analysis/default.yaml
python src/analysis/weight_analysis.py --config configs/analysis/default.yaml

echo "=== Stage 7: Minimal Circuit ==="
python src/analysis/minimal_circuit.py --config configs/analysis/default.yaml

echo "=== Stage 8: Build Docs ==="
mkdocs build --clean -d site

echo "=== Pipeline complete! ==="
