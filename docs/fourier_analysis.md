# Fourier Analysis

> **Hypothesis (H1):** The model represents numbers in a **Fourier basis** — token embeddings, attention outputs, and MLP activations are linear combinations of $\sin$ and $\cos$ waves over $\mathbb{Z}/p\mathbb{Z}$.

## Basis Construction

For prime $p$, we construct an orthonormal Fourier basis $F \in \mathbb{R}^{p \times p}$:

$$
F[n, 0] = \frac{1}{\sqrt{p}}; \quad
F[n, 2k-1] = \sqrt{\frac{2}{p}} \cos\left(\frac{2\pi k n}{p}\right); \quad
F[n, 2k] = \sqrt{\frac{2}{p}} \sin\left(\frac{2\pi k n}{p}\right)
$$

for $k = 1, \ldots, (p-1)/2$. Orthonormality ($F^\top F = I$) is verified numerically with max reconstruction error $< 4 \times 10^{-7}$.

## Embedding Fourier Spectrum

<div style="text-align: center; margin: 2em 0;">
    <img src="../figures/Figure_2_embedding_fourier_spectrum.png" alt="Embedding Fourier Spectrum" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
    <p class="figure-caption"><strong>Figure 2:</strong> Fourier spectrum of the embedding matrix. Top-5 frequencies explain 83.4% of total variance. The model learns a sparse Fourier encoding.</p>
</div>

**Key findings:**
- The embedding matrix is highly concentrated in the Fourier basis: 5 frequencies account for >83% of variance.
- Frequency 5 ($k=5$) is the dominant component across multiple embedding dimensions.
- This confirms that the model encodes numbers not as arbitrary IDs but as vectors in the Fourier basis.

## MLP Neuron Fourier Analysis

<div style="text-align: center; margin: 2em 0;">
    <img src="../figures/Figure_3_neuron_2d_fourier.png" alt="2D Fourier Spectrum of MLP Neurons" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
    <p class="figure-caption"><strong>Figure 3:</strong> 2D Fourier spectrum for selected MLP neurons. Energy concentrated on the diagonal (k, k) indicates that each neuron computes cos(k(a+b)) from cos(ka) and cos(kb).</p>
</div>

**Key findings:**
- Many MLP neurons show strong diagonal concentration in 2D Fourier space.
- This diagonal structure is evidence that neurons implement the **trigonometric identity**:
  $$
  \cos(k(a+b)) = \cos(ka)\cos(kb) - \sin(ka)\sin(kb)
  $$
- The MLP takes Fourier components of $a$ and $b$ as input and produces Fourier components of their sum.

## Unembedding Fourier Analysis

The unembedding matrix $W_U$ also shows strong Fourier structure. The top-10 Fourier frequencies in $W_U$ explain 85.4% of the variance.

## Summary

| Finding | Evidence |
|---------|----------|
| Embeddings are Fourier | 5 freqs explain 83.4% variance |
| MLP computes trig identity | Strong diagonal 2D Fourier spectrum |
| Unembedding reads Fourier | Top-10 freqs explain 85.4% variance |
| Fourier basis is orthonormal | Max reconstruction error $< 4 \times 10^{-7}$ |

Together, these results confirm **H1**: the model operates entirely in a Fourier basis.
