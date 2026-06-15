#!/usr/bin/env bash
set -euo pipefail

# Circuit Discovery on a Toy Task — Setup Script (Linux/macOS)
# Usage: bash setup.sh          # GPU (CUDA)
#        bash setup.sh --cpu    # CPU only

USE_CPU="${1:-}"

if [ "$USE_CPU" = "--cpu" ]; then
    REQ_FILE="requirements-cpu.txt"
else
    REQ_FILE="requirements.txt"
fi

echo "=== Creating Python virtual environment ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing dependencies from $REQ_FILE ==="
pip install -r "$REQ_FILE"

echo "=== Verifying installation ==="
python -c "
import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
import transformer_lens; print(f'TransformerLens {transformer_lens.__version__}')
import einops; print(f'einops {einops.__version__}')
import matplotlib; print(f'matplotlib {matplotlib.__version__}')
import numpy; print(f'numpy {numpy.__version__}')
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch.nn.functional as F
cfg = HookedTransformerConfig(n_layers=1, n_heads=2, d_model=16, d_head=8, d_mlp=64,
    d_vocab=14, n_ctx=3, act_fn='relu', init_weights=True, seed=42)
model = HookedTransformer(cfg)
tokens = torch.tensor([[0, 1, 13]])
logits = model(tokens)
loss = F.cross_entropy(logits[:, -1, :], torch.tensor([2]))
print(f'Forward pass OK — loss={loss.item():.4f}')
print('TransformerLens verified: forward pass works correctly.')
"

echo "=== Setup complete! ==="
echo "Activate the environment: source .venv/bin/activate"
