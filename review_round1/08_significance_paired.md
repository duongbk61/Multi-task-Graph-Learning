# Paired significance (B1–B4) — n=15, rule-free + fix, cùng 15 seed

Seeds (paired, identical across models): 14,21,42,77,100,7,11,33,55,88,99,123,777,2024,314.
Macro-F1 %. 95% CI = Δ ± t·std(d)/√n với t = 2.145 (df=14). Paired t-test + Wilcoxon signed-rank.

| Comparison | Task | Δ Macro | 95% CI | p(t) | p(Wilcoxon) | Significant? |
|---|---|---|---|---|---|---|
| UnifiedHMSL − MT-HGNN | Ponzi | +0.34 | [−1.02, +1.71] | 0.598 | 0.553 | No |
| UnifiedHMSL − MT-HGNN | Phish | +0.15 | [−0.37, +0.66] | 0.548 | 0.561 | No |
| UnifiedHMSL − Single-task | Ponzi | −0.90 | [−2.10, +0.30] | 0.131 | 0.084 | No |
| UnifiedHMSL − Single-task | Phish | +0.10 | [−0.44, +0.63] | 0.709 | 0.720 | No |
| UnifiedHMSL − Meta-IFD | — | N/A | — | — | — | Not testable (quoted, no per-seed) |

Means: UnifiedHMSL 81.24 / 97.21 · MT-HGNN 80.90 / 97.06 · Single-task 82.14 / 97.12.

## Conclusion (B4)
All four comparisons: 95% CI contains 0 and p ≫ 0.05 → **no statistically significant difference**.
- Fusion (CPA+TCG) does NOT significantly beat MT-HGNN (Δ≈0). Does NOT match the expected +3.36 —
  that value came from the V1 pipeline (with ExpertRules, no scaling fix), a different configuration.
- Multi-task ≈ single-task (Ponzi even −0.90). Consistent with "benefit is one unified model, not accuracy".
- UnifiedHMSL vs Meta-IFD not testable (paper-quoted, no per-seed). A self-trained Meta-IFD (A8) is
  being computed; a paired test against it is possible if desired.

Config note: everything here uses the NEW standard config (expert_mode=none + signed-log fix). Mixing
V1 numbers (e.g. +3.36) with this config in one table is not valid — pick one config for the whole paper.
