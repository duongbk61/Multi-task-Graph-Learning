# Meta-IFD — Result Tables (export for Drive)

All numbers are macro/micro F1 etc. as **mean±std ×100** unless stated. Generated on this
machine (GTX 1660, 6 GB) with `batch_size=64` (the paper uses 512 — see caveat below).

## Headline number (use this in the thesis)
**5-fold stratified cross-validation** of the Full model (`03_cv_headline.md`):
- Ponzi macro-F1 = **81.9 ± 1.2** (or 80.95 on a second run — GPU nondeterminism ±1)
- Phish macro-F1 = **96.4 ± 0.7**

## Files

| File | What it is |
|---|---|
| `01_comparison_full.md` / `.csv` | Full comparison: 12 paper baselines + Meta-IFD (paper) + our two rows (fixed-split & 5-fold CV). |
| `02_comparison_grouped.md` / `.csv` | Same data in the grouped layout (Machine learning / Single-task homo / hetero / Multi-task), incl. `MT-HGNN (shared enc.)` baseline vs `UnifiedHMSL (ours)`. |
| `03_cv_headline.md` | 5-fold CV of the Full model — the defensible headline result. |
| `04_ablation_components.md` / `.csv` | Leave-one-out ablation (w/o CPA / TCG / Expert), Micro-F1 + Macro-F1, fixed-split + 5 restarts. |
| `05_ablation_4novelties.md` | Earlier 4-novelty ablation (3-seed): Multi-task, CPA, TCG, Expert. |
| `06_split_sensitivity.md` | Robustness: Full model over 16 splits (distribution + lucky-split ceiling). |
| `raw/` | All raw per-seed / per-fold JSON behind the tables (for reproducibility). |

## Key honest caveats (please keep in mind when presenting)

1. **Batch size.** Our runs use `batch_size=64` (6 GB GPU); the paper uses 512. This is the main
   reason our reproduction sits a few points below the paper's Meta-IFD on Ponzi.
2. **Protocol.** Paper baselines use *fixed split + K restarts*. Our `fixed-split` rows match that
   (directly comparable); our `5-fold CV` rows are a stricter, different protocol (robustness).
3. **Nondeterminism.** GPU atomic adds make Ponzi macro-F1 fluctuate ≈ ±1–2 across identical runs;
   only the **TCG** ablation effect clears that noise floor decisively. CPA and Expert effects are
   marginal / within noise (Expert even flips sign by protocol) — do not overstate them.
4. **Never select the split for a high number** — that is test-set leakage. The `06_split_sensitivity`
   MAX is a diagnostic ceiling only, not a reportable metric.
