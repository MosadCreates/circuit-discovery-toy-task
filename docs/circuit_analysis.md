# Circuit Analysis

## Attention Pattern Visualisation

We extract attention patterns from all 4 heads on 50 random $(a,b)$ pairs.

<div style="text-align: center; margin: 2em 0;">
    <img src="figures/Figure_4_attention_patterns.png" alt="Attention Pattern Heatmaps" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
    <p class="figure-caption"><strong>Figure 4:</strong> Attention pattern heatmaps for all 4 heads. Heads 1 and 3 attend predominantly to the <code>=</code> position. Heads 0 and 2 distribute attention to <code>a</code> and <code>b</code>.</p>
</div>

**Head roles:**
- **Head 0 (attends to a):** Routing the embedding of $a$ to the output position.
- **Head 1 (attends to =):** Self-attention at the output position (possibly for identity).
- **Head 2 (attends to b):** Routing the embedding of $b$ to the output position.
- **Head 3 (uniform):** Broad attention, potentially serving as a residual connection bypass.

## Residual Stream Causal Tracing

We perform causal tracing on the residual stream by patching every residual position independently.

<div style="text-align: center; margin: 2em 0;">
    <img src="figures/Figure_5_residual_patching.png" alt="Residual Stream Patching Heatmap" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
    <p class="figure-caption"><strong>Figure 5:</strong> Residual stream causal tracing. Dark green indicates high recovery — the patched component restored most of the performance. The <code>=</code> position after the attention layer is where the computation converges.</p>
</div>

## Head-Level Patching

We systematically ablate each attention head by replacing its output with the corrupted version.

<div style="text-align: center; margin: 2em 0;">
    <img src="figures/Figure_6_head_patching.png" alt="Head-Level Patching Results" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
    <p class="figure-caption"><strong>Figure 6:</strong> Head-level activation patching results. Darker cells indicate higher recovery. Heads 0 and 2 are the most important for routing a and b information.</p>
</div>

**Patching method:**
- Clean input: $(a, b)$ → cache activations.
- Corrupted input: $(a', b)$ with $a' \neq a$ → cache activations.
- Patch: replace corrupted head output with clean head output at the $=$ position.
- **Recovery score:** $\frac{\text{logit}_{\text{patched}} - \text{logit}_{\text{corrupted}}}{\text{logit}_{\text{clean}} - \text{logit}_{\text{corrupted}}}$.

## MLP and Neuron-Level Patching

<div style="text-align: center; margin: 2em 0;">
    <img src="figures/Figure_7_neuron_patching.png" alt="Neuron-Level Patching Results" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
    <p class="figure-caption"><strong>Figure 7:</strong> Neuron-level activation patching. Each pixel represents the recovery score when that neuron's activation at the <code>=</code> position is patched. A sparse set of 15 neurons shows high recovery.</p>
</div>

## Direct Logit Attribution

<div style="text-align: center; margin: 2em 0;">
    <img src="figures/Figure_8_direct_logit_attribution.png" alt="Direct Logit Attribution" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
    <p class="figure-caption"><strong>Figure 8:</strong> Direct logit attribution breakdown per component. The MLP contributes the largest positive component to the correct logit, while attention heads modulate routing.</p>
</div>

## Path Patching

We apply path patching to isolate the causal effect of specific attention-head-to-MLP connection paths. This analysis reveals which head outputs feed into which MLP neurons to enable the trigonometric identity computation. Full results are shown in Figure S1 in the Appendix.

## Summary

| Finding | Evidence |
|---------|----------|
| Heads 0,2 route a,b | Attention patterns + patching |
| Head 3 amplifies signal | Uniform attention + positive DLA |
| 15 critical neurons | Neuron patching identifies them |
| MLP dominant contributor | DLA: MLP contributes ~60% of logit |
