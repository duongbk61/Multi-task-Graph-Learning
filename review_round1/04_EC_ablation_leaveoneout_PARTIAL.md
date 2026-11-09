# Ablation table (leave-one-out of the fusion novelties)

Values are mean±std ×100. **Micro-F1** = accuracy, **Macro-F1** = macro average.
Protocol: fixed split (SPLIT_SEED=14) + 5 random restarts, batch_size=64 — same as the
`UnifiedHMSL (full)` row, so each variant differs from Full by exactly one novelty.

| Variant | Ponzi Micro-F1 | Ponzi Macro-F1 | Phish Micro-F1 | Phish Macro-F1 |
|---|---|---|---|---|
| **UnifiedHMSL (full)** | 91.08±0.71 | 79.66±1.76 | 97.29±0.64 | 97.24±0.64 |
| w/o CPA | 90.41±0.64 | 79.54±1.41 | 96.78±0.95 | 96.74±0.95 |
| w/o TCG (fixed fusion) | 90.33±0.85 | 77.54±2.04 | 96.93±0.41 | 96.89±0.41 |
| w/o Expert Rules | 91.23±0.80 | 80.65±1.03 | 97.00±0.18 | 96.95±0.19 |

## Δ vs Full (removing a component; negative = the component helps)

| Removed | ΔPonzi Macro-F1 | ΔPhish Macro-F1 | Verdict |
|---|---|---|---|
| CPA | −0.12 | −0.50 | marginal (within noise on Ponzi) |
| TCG | **−2.12** | −0.35 | **clearly helps Ponzi** (strongest, robust) |
| Expert Rules | **+0.99** | −0.29 | on Ponzi it slightly *hurts* here; helps Phish a touch |

### Honest reading
- **TCG is the one robust win**: removing it drops Ponzi Macro-F1 by ~2.1 — the largest, consistent effect.
- **CPA is marginal** on Ponzi (−0.12, well inside the ±1–2 run-to-run band) but does help the
  saturated Phish slightly (−0.50).
- **Expert Rules is NOT a reliable win**: in this fixed-split protocol, removing it *improves*
  Ponzi Macro-F1 (+0.99). An earlier 3-seed run had shown Expert helping Ponzi — so the sign
  **flips with seed/protocol**, i.e. the effect is within noise, not robust. Report it as
  "≈ neutral on Ponzi, small help on Phish", do not claim it as a clear gain.
- On **Phish (saturated ~97)** the Full model is best and every component adds a small amount
  (0.3–0.5), but these are within noise.

> Nondeterminism caveat: `torch_scatter`/`torch_sparse` GPU atomics make Ponzi Macro-F1 fluctuate
> ≈ ±1–2 across identical re-runs, so only the TCG effect clears the noise floor decisively.
