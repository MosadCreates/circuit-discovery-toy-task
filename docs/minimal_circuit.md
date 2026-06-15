# Minimal Circuit

**Verdict — Hypothesis H4:**
> A circuit of 2 attention heads + 15 MLP neurons ($\sim 3\%$ of components) achieves $>95\%$ of full model accuracy.

## Ablation Methodology

We evaluate the model after systematically zeroing out components:
- **Heads:** Zero the `hook_z` output.
- **Neurons:** Zero the `hook_post` activation.
- All ablations are performed at the **==** position only.

## Performance Breakdown

| Configuration | Accuracy | % of Full |
|---------------|----------|-----------|
| Full model | 70.0% | 100% |
| No heads | 38.7% | 55% |
| Head 0 only | 45.2% | 65% |
| Head 2 only | 42.1% | 60% |
| Heads 0 + 2 | 52.4% | 75% |
| Heads 0 + 2 + 3 | 58.8% | 84% |
| All 4 heads | 62.3% | 89% |
| 2 heads + 15 neurons | **66.8%** | **95%** |
| 2 heads + 30 neurons | 68.2% | 97% |
| Heads alone (no MLP) | 5.1% | 7% |

## Ablation Curve

<div style="text-align: center; margin: 2em 0;">
    <img src="figures/Figure_9_ablation_curve.png" alt="Ablation Curve" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
    <p class="figure-caption"><strong>Figure 9:</strong> Accuracy vs number of remaining components. Adding 2 heads then 15 neurons recovers >95% of performance. The sharp knee at 5–15 neurons indicates a sparse set of critical neurons.</p>
</div>

## Algorithm Reconstruction

The model's algorithm can be reconstructed in 4 steps:

### Step 1: Embedding to Fourier Basis (Embedding)

The embedding matrix $W_E \in \mathbb{R}^{114 \times 128}$ maps each token ID to a vector that is a linear combination of Fourier basis functions. A small number of frequencies (dominated by $k=5$) accounts for most of the variance.

### Step 2: Frequency Routing (Heads 0, 2)

Attention heads 0 and 2 attend from the == position to the **a** and **b** positions respectively. Their outputs $z_0, z_2 \in \mathbb{R}^{d_{\text{head}}}$ at the == position carry the Fourier components of $a$ and $b$ into the residual stream at the output position.

Head 3 provides broad amplification, while head 1 appears to participate in identity computation.

### Step 3: Trigonometric Identity (MLP)

The MLP receives the sum of head outputs at the == position and applies $W_{\text{in}} \in \mathbb{R}^{128 \times 512}$, ReLU, then $W_{\text{out}} \in \mathbb{R}^{512 \times 128}$. The 15 critical neurons are those whose first-layer weights are most aligned with specific frequency pairs $(k_a, k_b)$.

These neurons implement the identity:

$$
\cos(k(a+b)) = \cos(ka)\cos(kb) - \sin(ka)\sin(kb)
$$

The ReLU nonlinearity enables the cancellation of the cross-terms, effectively isolating the cos/sin of the sum.

### Step 4: Unembedding to Answer (Unembedding + Logit Lens)

The unembedding matrix $W_U \in \mathbb{R}^{128 \times 114}$ projects the final residual stream at the == position to logits. The logit for answer $c$ is maximised when the Fourier components of the residual match the Fourier components of token $c$, which they do precisely when $c = (a+b) \bmod p$.

## The Identified Circuit

```
a ---> Embed ---> Head 0 ----+
                              |
= ---> Embed ---> ... --------+--> MLP (15 neurons) --> Unembed --> logit
                              |
b ---> Embed ---> Head 2 ----+
                              |
                   Head 3 ----+
                              |
                   Head 1 ----+
```

## Key Insight

The model composes **two-dimensional Fourier coefficients** ($\cos(ka), \sin(ka)$ and $\cos(kb), \sin(kb)$) via attention routing and the MLP's nonlinearity to produce $\cos(k(a+b)), \sin(k(a+b))$, which the unembedding reads out. This is exactly the discrete Fourier transform's **convolution theorem** implemented in a learned neural circuit.
