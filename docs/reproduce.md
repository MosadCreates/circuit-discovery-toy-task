# Reproduce

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.12 |
| PyTorch | 2.0+ | 2.5+ |
| GPU | T4 (16GB) | A100 (40GB) |
| Disk | 1 GB | 2 GB |
| Time | 1 hour | 30 min |

## Setup (Windows)

```powershell
# Clone and enter
git clone https://github.com/yourusername/circuit-discovery-toy-task
cd circuit-discovery-toy-task

# Run setup
.\setup.ps1
```

## Setup (Linux/macOS)

```bash
# Clone and enter
git clone https://github.com/yourusername/circuit-discovery-toy-task
cd circuit-discovery-toy-task

# Create environment and install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train the Model

```bash
# Full training (p=113, 50k steps, ~30 min on T4)
python src/training/train.py --config configs/training/default.yaml

# Fast debug (p=13, 5k steps, ~2 min on CPU)
python src/training/train.py --config configs/training/fast-debug.yaml
```

## Run All Analysis

```bash
# All analyses in sequence
bash scripts/run_full_pipeline.sh

# Or step by step:
export RESULTS=results/figures

# Attention patterns
python src/analysis/attention_patterns.py --config configs/analysis/default.yaml --output $RESULTS

# Fourier analysis
python src/analysis/fourier.py --config configs/analysis/default.yaml --output $RESULTS

# Neuron 2D Fourier
python src/analysis/neuron_analysis.py --config configs/analysis/default.yaml --output $RESULTS

# Activation patching
python src/patching/residual_stream_patch.py --config configs/patching/default.yaml --output $RESULTS
python src/patching/head_patch.py --config configs/patching/default.yaml --output $RESULTS
python src/patching/mlp_patch.py --config configs/patching/default.yaml --output $RESULTS
python src/patching/path_patch.py --config configs/patching/default.yaml --output $RESULTS

# Logit attribution
python src/analysis/direct_logit_attribution.py --config configs/analysis/default.yaml --output $RESULTS
python src/analysis/logit_lens.py --config configs/analysis/default.yaml --output $RESULTS
python src/analysis/weight_analysis.py --config configs/analysis/default.yaml --output $RESULTS

# Minimal circuit
python src/analysis/minimal_circuit.py --checkpoint results/checkpoints/best_val_acc.pt --output $RESULTS

# Build this site
cp -r results/figures/* docs/figures/
mkdocs build --clean -d site
```

## Run Tests

```bash
pytest tests/ -v
```

## Run on Colab

Open [notebooks/colab_demo.ipynb](https://colab.research.google.com/github/yourusername/circuit-discovery-toy-task/blob/main/notebooks/colab_demo.ipynb) and run all cells.

## Configuration

All hyperparameters are in YAML files:

| File | Purpose |
|------|---------|
| `configs/training/default.yaml` | Full training (p=113) |
| `configs/training/fast-debug.yaml` | Quick test (p=13) |
| `configs/analysis/default.yaml` | Analysis parameters |
| `configs/patching/default.yaml` | Patching parameters |

## Expected Outputs

```
results/
├── checkpoints/
│   ├── best_val_acc.pt        # Best model checkpoint
│   ├── step_50000.pt          # Final checkpoint
│   └── step_*.pt              # Intermediate checkpoints
├── figures/
│   ├── Figure_1_grokking_dynamics.png
│   ├── Figure_2_embedding_fourier_spectrum.png
│   ├── Figure_3_neuron_2d_fourier.png
│   ├── Figure_4_attention_patterns.png
│   ├── Figure_5_residual_patching.png
│   ├── Figure_6_head_patching.png
│   ├── Figure_7_neuron_patching.png
│   ├── Figure_8_direct_logit_attribution.png
│   └── Figure_9_ablation_curve.png
└── metrics/
    └── training_metrics.json
```

## Docker

```bash
docker build -t circuit-discovery -f docker/Dockerfile .
docker run --gpus all -v $(pwd)/results:/app/results circuit-discovery python src/training/train.py
```
