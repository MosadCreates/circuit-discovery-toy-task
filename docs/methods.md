# Methods

## Task Definition

The task is $(a + b) \bmod p$ for $p = 113$ (prime). Inputs are tokenised as $[a, b, =]$, where token IDs are the integers themselves, and the $=$ token has ID $p$. The vocabulary size is $p + 1 = 114$. The label is the correct answer token $(a+b) \bmod p$. There are exactly $p^2 = 12,769$ possible input-output pairs.

**Why a prime modulus?** When $p$ is prime, $\mathbb{Z}/p\mathbb{Z}$ is a field, guaranteeing that the $p$-dimensional Fourier basis forms a complete orthonormal basis. The Fourier basis diagonalises the cyclic convolution operator, which is exactly the operation the model needs to compute.

## Model Architecture

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Layers | 1 | Sufficient for the task |
| Attention heads | 4 | Nanda et al. found 3 key heads |
| $d_{\text{model}}$ | 128 | Sufficient residual stream capacity |
| $d_{\text{mlp}}$ | 512 | MLP must implement nonlinear trig identity |
| $d_{\text{head}}$ | 32 | $d_{\text{model}} / n_{\text{heads}}$ |
| Activation | ReLU | Standard GPT-2 style |
| Attention | Bidirectional | Task is not autoregressive |
| Vocab size | 114 | Tokens 0–113, token 113 is $=$ |
| Context length | 3 | $[a, b, =]$ |

## Training Setup

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Optimiser | AdamW | Standard; weight decay is critical for grokking |
| Weight decay | 1.0 | Penalises memorisation; promotes compression |
| Learning rate | 0.001 | Standard for AdamW on small tasks |
| Batch size | 12,769 (full batch) | Full-batch gives cleaner dynamics |
| Steps | 50,000 | Grokking at 20,000–40,000 steps |
| Seed | 42 | Fixed for reproducibility |
| Val split | 30% | Random split, fixed seed, no data leakage |

## Analysis Tools

### Attention Pattern Visualisation

We use TransformerLens hooks to extract the attention pattern `blocks.0.attn.hook_pattern` from every head on every input. Patterns are averaged over 50 random $(a, b)$ pairs.

### Activation Patching

For each component, we:

1. Run the model on a **clean input** $(a,b)$ and cache activations.
2. Run the model on a **corrupted input** $(a',b)$ where $a' \neq a$ and cache activations.
3. Re-run the corrupted input with the component's activation replaced by the clean version.
4. Measure **recovery** as the fraction of the clean-corrupted logit difference gap that is restored.

### Fourier Analysis

We construct the $p$-dimensional orthonormal Fourier basis over $\mathbb{Z}/p\mathbb{Z}$:

$$
f_0[n] = \frac{1}{\sqrt{p}}, \quad
f_k^{\text{cos}}[n] = \sqrt{\frac{2}{p}} \cos\left(\frac{2\pi k n}{p}\right), \quad
f_k^{\text{sin}}[n] = \sqrt{\frac{2}{p}} \sin\left(\frac{2\pi k n}{p}\right)
$$

We project model weights and activations onto this basis to measure Fourier concentration.

### Direct Logit Attribution

Using the linearity of the residual stream, we decompose the final logit for the correct answer as:

$$
\text{logit}_{\text{correct}} = \sum_{\text{components } c} \left( \text{output}_c[=] \cdot W_U[:, \text{correct}] \right)
$$

where $\text{output}_c[=]$ is component $c$'s output at the $=$ position.

## Reproducibility

All random seeds are fixed throughout (data split, model initialisation, training, analysis). The complete pipeline runs in under 1 hour on a single T4 GPU.
