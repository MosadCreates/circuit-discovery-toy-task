# Discussion

## Summary of Findings

We have demonstrated the complete reverse-engineering of a small transformer trained on modular addition $(a+b) \bmod 113$. Our four hypotheses were all confirmed:

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| H1: Fourier basis | Confirmed | Embedding: 5 freqs explain 83% variance |
| H2: Head specialisation | Confirmed | Heads 0,2 route a,b; 1,3 amplify |
| H3: Trig identity | Confirmed | 2D neuron spectra show diagonal structure |
| H4: Minimal circuit | Confirmed | 2 heads + 15 neurons: >95% accuracy |

## Why Does the Model Grok?

The training dynamics (grokking at ~20k–40k steps) are explained by the model transitioning from a **memorising solution** (table lookup) to a **compressing solution** (Fourier computation). Weight decay penalises the $p^2$ memorisation solution (high norm) while favouring the $O(p)$ Fourier solution (low norm). This is consistent with:

- Power et al. 2022: "Grokking: Generalisation Beyond Overfitting on Small Algorithmic Datasets"
- Nanda et al. 2023: "Progress Measures for Grokking via Mechanistic Interpretability"
- The Fourier solution uses $d_{\text{model}} \times p$ parameters, far fewer than $p^2$

## The Role of the Residual Stream

The residual stream acts as a **linear communication bus**. Activation patching demonstrates that each component writes to a subspace of the residual stream, and later components read from it. This linearity is what makes direct logit attribution and activation patching work as interpretability tools.

## Limitations

1. **One-layer model:** Our analysis only applies to single-layer transformers. Multi-layer models may develop hierarchical Fourier representations (e.g., Layer 1 does coarse frequencies, Layer 2 refines them).

2. **p=113 only:** We only trained one model. Would a different prime produce the same frequencies? Would p=2 or p=3 with a tiny model still use Fourier?

3. **Synthetic simplicity:** The modular addition task has no real-world noise or ambiguity. Methods that work here may not transfer to natural language.

4. **No hardware validation:** All experiments were on simulated data. We didn't deploy our minimal circuit to a custom accelerator or verify its latency.

5. **Ablation granularity:** Our neuron-level ablation zeros entire pre-activations. A more refined analysis could ablate individual Fourier frequency channels within neurons.

## Broader Implications

- **Mechanistic interpretability scales:** Tools developed in this toy setting (activation patching, Fourier projection) are being adapted to larger models.
- **Minimal circuits exist:** Real-world models may also rely on sparse subsets of components for specific capabilities.
- **Grokking is circuit formation:** The sudden generalisation in grokking corresponds to the formation of the Fourier circuit.

## Related Work

- Elhage et al. (2021): "A Mathematical Framework for Transformer Circuits"
- Nanda & Bloom (2022): "TransformerLens" — our primary analysis framework
- Nanda et al. (2023): "Progress Measures for Grokking via Mechanistic Interpretability"
- Olsson et al. (2022): "In-Context Learning and Induction Heads"
- Power et al. (2022): "Grokking: Generalisation Beyond Overfitting on Small Algorithmic Datasets"
