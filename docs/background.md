# Background

## 1. The Modular Addition Task

### Definition

Modular addition is the problem of computing

$$
(a + b) \bmod p
$$

where $a$ and $b$ are integers in $\{0, 1, \dots, p-1\}$ and $p$ is a prime. We represent the input to the transformer as a sequence of three tokens:

$$
[\; a,\; b,\; \texttt{=} \;]
$$

where $a$ and $b$ are embedded directly as their integer token IDs (i.e. token $0$ represents the integer $0$, token $1$ represents $1$, etc.), and $\texttt{=}$ is an auxiliary token (token ID $p$) marking the position where the model must output the answer. The vocabulary size is $p + 1$: the integers $0$ through $p-1$ plus the $\texttt{=}$ token. The label is the correct answer token $(a + b) \bmod p$.

There are exactly $p^2$ possible inputs. For $p = 113$, the full dataset contains $113^2 = 12,\!769$ examples — small enough to fit in a single batch even on a free-tier GPU.

### Why This Task

Modular addition has become a canonical benchmark in mechanistic interpretability for several reasons:

1. **Known ground-truth solution.** Unlike most neural network behaviours, the optimal algorithm for this task is known analytically: it is the discrete Fourier transform over $\mathbb{Z}/p\mathbb{Z}$. A transformer can solve modular addition by learning to represent numbers in a Fourier basis and using trigonometric identities to compute the sum. This means a researcher can *verify* whether the model has discovered the correct algorithm, rather than just describing what it does without ground truth.

2. **Non-trivial but discoverable.** The algorithm is not trivial (it requires learning a frequency representation, attention-based summation, and nonlinear feature interaction), but it is small enough that a single-layer, 4-head transformer can express it. This makes full reverse-engineering feasible.

3. **Strong empirical signature.** The model exhibits *grokking* (sudden generalisation after prolonged memorisation) on this task, which provides a clear behavioural signal that an algorithmic solution has been discovered.

4. **Clean mathematics.** The group $\mathbb{Z}/p\mathbb{Z}$ has a rich but well-understood structure. The Fourier basis diagonalises the convolution operator, which is exactly the operation the model needs to compute (since $(a+b) \bmod p$ is a convolution in the group algebra). This mathematical elegance makes the analysis rigorous.

### Why a Prime Modulus

When $p$ is prime, $\mathbb{Z}/p\mathbb{Z}$ is a *field* (every non-zero element has a multiplicative inverse). This gives the Fourier basis two important properties:

- **Completeness:** The $p$ Fourier basis vectors $\{ \mathbf{f}_k \}_{k=0}^{p-1}$ form an orthonormal basis for $\mathbb{R}^p$ (or $\mathbb{C}^p$).
- **Convolution theorem:** Convolution in the spatial domain corresponds to pointwise multiplication in the Fourier domain. Since $(a+b) \bmod p$ is a convolution of the delta functions at $a$ and $b$, the Fourier representation factors the computation: the Fourier transform of the answer is the elementwise product of the Fourier transforms of the inputs.

For composite $p$, the Fourier analysis is messier (the basis may not be complete, and the group structure has zero divisors). Prime $p$ gives the cleanest theoretical setting.

### Why It Is a Good Circuit Discovery Target

| Property | Implication |
|----------|-------------|
| Fully synthetic data | No distribution shift, no labelling noise, infinite supply |
| 100% ground truth labels | Every metric is exact, not statistical |
| Small model (1 layer, 4 heads) | Full circuit description fits in a diagram |
| Known solution in literature | Findings can be validated against Nanda et al. (2023) |
| Reproducible grokking | The behavioural phase change is a strong signal that a new algorithm has been "found" by the model |

---

## 2. The Grokking Phenomenon

### Definition

Grokking, introduced by Power et al. (2022), is the phenomenon where a neural network first *memorises* the training data and only later *generalises* to the validation set, despite continued training with no new data. The characteristic signature is:

1. **Memorisation phase:** Training accuracy reaches $\sim 100\%$ while validation accuracy remains near chance.
2. **Generalisation phase:** After many additional gradient steps, validation accuracy suddenly jumps to $\sim 100\%$, often within a few hundred steps.
3. **Post-generalisation:** Both training and validation accuracy remain at ceiling.

The loss curve shows two distinct phases: a fast initial drop (memorisation), then a long plateau, followed by a second sharp drop (generalisation). This "double descent" in validation loss occurs without any change to the data or learning rate.

<div style="text-align: center;">
<em>Figure 1 will show the characteristic grokking dynamics: train/val loss and accuracy vs. training step, with the grokking point marked.</em>
</div>

### Why Grokking Happens

The leading hypothesis, supported by Nanda et al. (2023), is that the model simultaneously learns two solutions:

- **A memorising solution** that stores individual training examples in its parameters. This solution learns quickly because it requires only rote storage, but it requires high-weight-norm parameters (since each example must be stored with enough precision to be retrieved correctly).
- **A generalising solution** that compresses the underlying algorithm. This solution learns slowly because it requires discovering the Fourier structure of the task, but it is more parameter-efficient (lower weight norm).

Weight decay penalises large weights. The memorisation solution has high weight norm, so it is strongly penalised. The generalising solution has low weight norm, so it is preferred by the optimiser. Initially, the memorisation solution dominates because it learns faster. But as weight decay continuously suppresses it, the generalising solution slowly becomes dominant. When the generalising solution crosses a threshold, validation accuracy suddenly jumps — the model has "groked."

This is why high weight decay (typically $1.0$) is critical for reproducing grokking: without it, the memorisation solution would never be sufficiently suppressed.

### Foundational References

1. **Power et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets."** *arXiv:2201.02177.*
   - First described the grokking phenomenon.
   - Demonstrated it on modular addition, modular subtraction, and other algorithmic tasks.
   - Showed that grokking occurs across multiple architectures (transformers, MLPs, LSTMs).

2. **Nanda et al. (2023). "Progress Measures for Grokking via Mechanistic Interpretability."** *ICLR 2023.*
   - Reverse-engineered the Fourier algorithm in a 1-layer transformer trained on modular addition.
   - Introduced the "effective rank" progress measure: the rank of the embedding and unembedding matrices decreases during grokking as the model compresses its representation.
   - Showed that the model uses a small set of Fourier frequencies and that attention heads compute the sum via a trigonometric identity.
   - This paper is the direct precursor to the present project.

### What This Project Adds

Where Nanda et al. (2023) focused on the progress measure (a scalar quantity that predicts grokking before it happens), this project provides:

- A **complete end-to-end circuit description** with causal validation via activation patching.
- **Publication-quality figures** for every step of the analysis.
- A **minimal circuit** that accounts for >90% of model performance with <50% of components.
- A **self-contained, reproducible pipeline** usable as a template for future circuit discovery projects.

---

## 3. What Circuit Discovery Means

### Circuits

A *circuit* is a minimal subgraph of the transformer's computational graph that is causally responsible for the model's behaviour on a specific task. The full model has many parameters, but only a subset are used for any given task. The goal of circuit discovery is to identify this subset and explain how it implements the computation.

Formally, a circuit consists of:

- **Nodes:** Specific attention heads, MLP neurons, or embedding/unembedding dimensions.
- **Edges:** The flow of information through the residual stream, attention, and MLP computations.

A good circuit explanation satisfies three criteria:

1. **Faithfulness:** If you ablate all components outside the circuit, the model still performs the task at near-original accuracy.
2. **Completeness:** The circuit accounts for most (>90%) of the model's performance.
3. **Mechanistic understanding:** Each component's function is described in terms of the algorithm it implements, not just in terms of its activation patterns.

### The Residual Stream as a Communication Channel

Elhage et al. (2021) introduced the "residual stream" perspective: in a transformer, every sublayer (attention head, MLP) reads from the same residual stream and writes back to it. Concretely, for a layer $l$ with attention sublayer $\text{Attn}_l$ and MLP sublayer $\text{MLP}_l$:

$$
x_{l+1} = x_l + \text{Attn}_l(x_l) + \text{MLP}_l(x_l + \text{Attn}_l(x_l))
$$

where $x_l$ is the residual stream state at layer $l$ (a $[\text{seq\_len}, d_\text{model}]$ tensor). Each component's output is *added* to the stream, not multiplied. This linearity is the key architectural fact that makes mechanistic interpretability tractable: the final output is a linear sum of component contributions.

The residual stream is the "communication bus": any component can read from any position's residual stream (via attention) and write to any position's residual stream. This is in contrast to strictly feed-forward architectures where information flows in one direction only.

### The Three Analysis Tools

#### 1. Attention Pattern Visualisation

Attention heads compute a weighted sum of value vectors at different positions. The attention weights (the softmax of query-key dot products) reveal *where* each head is reading from. For a head $h$ at layer $l$:

$$
\text{attn}_{l,h}[i, j] = \text{softmax}_j \left( \frac{(x_i W_Q) (x_j W_K)^\top}{\sqrt{d_\text{head}}} \right)
$$

This is a $[\text{seq\_len}, \text{seq\_len}]$ matrix where row $i$ shows how much position $i$ attends to each position $j$. Visualising these patterns tells us which positions the model considers relevant for each computation.

#### 2. Activation Patching (Causal Tracing)

Activation patching is the most direct method for establishing causal necessity. The procedure is:

1. Run the model on a **clean input** (e.g. $(a, b) = (5, 3)$) and cache all intermediate activations.
2. Run the model on a **corrupted input** (e.g. $(a', b') = (17, 3)$ where $a' \neq a$) and cache all intermediate activations.
3. Re-run the corrupted model, but at a specific hook point (e.g. the output of head 2 at layer 1), replace the corrupted activation with the clean activation.
4. Measure how much the model's performance recovers towards the clean output.

If patching component $X$ restores $> 50\%$ of performance, then $X$ is causally important: the computation that determines the answer passes through $X$.

Unlike zero-ablation (setting activations to zero), which can push the model into a distribution it never sees during training, activation patching uses a *meaningful counterfactual* — the activation from a different input — giving cleaner causal attribution.

#### 3. Logit Lens and Direct Logit Attribution

Because the residual stream is linear, the final logits can be decomposed as a sum of contributions from each component:

$$
\text{logits} = W_U \cdot \text{LayerNorm}\left( \sum_{\text{components } c} \text{output}_c \right)
$$

The **logit lens** applies the unembedding $W_U$ at every layer to see how the model's prediction evolves through the depth. This reveals whether the answer is "written in" early (embedding layer directly encodes the answer) or late (requires full computation through attention and MLP).

**Direct logit attribution** decomposes the logit for the correct answer as a sum over components:

$$
\log P(\text{correct}) = \sum_c \underbrace{W_U[\text{correct}] \cdot \text{output}_c}_{\text{direct contribution of } c}
$$

This tells us which components contribute positively to the correct answer and which contribute negatively (e.g. suppressing incorrect answers).

---

## 4. Specific Claims (Hypotheses to Verify)

### H1: The model uses Fourier features

The model represents $a$ and $b$ as linear combinations of $\sin(2\pi k a / p)$ and $\cos(2\pi k a / p)$ for a small set of frequencies $k$.

**Evidence required:** The embedding matrix $W_E$ (shape $[p, d_\text{model}]$) should have a Fourier spectrum concentrated on a few frequencies. The fraction of variance explained by the top $K$ Fourier frequencies should exceed $80\%$ for $K \leq 5$.

### H2: Specific attention heads implement the key computation

Not all 4 attention heads are equally important. Some attend from the $\texttt{=}$ position to the $a$ or $b$ positions and contribute to computing the sum representation.

**Evidence required:** Attention pattern analysis shows which heads attend to which positions. Activation patching shows that only 2–3 heads are causally necessary. The minimal circuit excludes at least one head without significant performance loss.

### H3: The MLP implements a frequency-selective nonlinearity

The MLP neurons act as "frequency detectors": each neuron responds strongly to inputs where $a + b$ has a specific value mod $p$, and its 2D Fourier spectrum over $(a, b)$ space is concentrated on a single frequency $k$.

**Evidence required:** The 2D Fourier spectrum of top MLP neurons shows energy concentrated at specific $(k_a, k_b)$ pairs corresponding to $\cos(2\pi k (a+b) / p)$ and $\sin(2\pi k (a+b) / p)$.

### H4: The circuit can be reduced to a small set of components

A minimal set of 2–3 attention heads and 10–20 MLP neurons accounts for $>90\%$ of model accuracy.

**Evidence required:** Ablating all non-circuit components leaves accuracy at $>90\%$ of the full model. The ablation curve (accuracy vs. number of components) saturates quickly.

---

## References

- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits." *Transformer Circuits Thread.*
- Nanda, N., et al. (2023). "Progress Measures for Grokking via Mechanistic Interpretability." *ICLR 2023.*
- Power, A., et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets." *arXiv:2201.02177.*

---

*This document is Section 1 of the research writeup for "Circuit Discovery on a Toy Task: Reverse-Engineering a Grokked Transformer on Modular Addition."*
