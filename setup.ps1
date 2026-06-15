param(
    [switch]$CPU
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

Write-Host "=== Circuit Discovery on a Toy Task - Setup ===" -ForegroundColor Cyan

# Check Python version
$pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Python version: $pyVersion" -ForegroundColor Green

if ($pyVersion -notin @("3.11", "3.12")) {
    Write-Host "WARNING: Recommended Python 3.11 or 3.12. You have $pyVersion." -ForegroundColor Yellow
}

# Check for existing venv
if (Test-Path -LiteralPath ".venv") {
    $choice = Read-Host -Prompt "Virtual environment '.venv' already exists. Recreate? (y/N)"
    if ($choice -eq "y") {
        Remove-Item -Recurse -Force ".venv"
        python -m venv .venv
        Write-Host "Virtual environment recreated." -ForegroundColor Green
    }
} else {
    python -m venv .venv
    Write-Host "Virtual environment created." -ForegroundColor Green
}

# Determine pip path
$pipExe = if (Test-Path -LiteralPath ".venv\Scripts\pip.exe") { ".\.venv\Scripts\pip.exe" } else { ".\.venv\Scripts\python.exe" }
$pipArgs = if (Test-Path -LiteralPath ".venv\Scripts\pip.exe") { @() } else { @("-m", "pip") }

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
& "$pipExe" $pipArgs install --upgrade pip

# Install requirements
$reqFile = if ($CPU) { "requirements-cpu.txt" } else { "requirements.txt" }
Write-Host "`nInstalling from $reqFile ..." -ForegroundColor Yellow
& "$pipExe" $pipArgs install -r "$reqFile"
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip install failed." -ForegroundColor Red
    exit 1
}

# Verify installation
Write-Host "`n=== Verifying installation ===" -ForegroundColor Cyan
.\.venv\Scripts\python.exe -c "
import torch, transformer_lens, einops, numpy, yaml
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch.nn.functional as F

print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
print(f'TransformerLens {transformer_lens.__version__}')

# Build a minimal model and run a forward pass
cfg = HookedTransformerConfig(
    n_layers=1, n_heads=2, d_model=16, d_head=8, d_mlp=64,
    d_vocab=14, n_ctx=3, act_fn='relu', init_weights=True, seed=42
)
model = HookedTransformer(cfg)
tokens = torch.tensor([[0, 1, 13]])
logits = model(tokens)
loss = F.cross_entropy(logits[:, -1, :], torch.tensor([2]))
print(f'Forward pass OK - loss={loss.item():.4f}')
print('TransformerLens verified: forward pass works correctly.')
"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== Setup complete! ===" -ForegroundColor Green
    Write-Host "Activate with:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
} else {
    Write-Host "`n=== Setup failed ===" -ForegroundColor Red
    exit 1
}
