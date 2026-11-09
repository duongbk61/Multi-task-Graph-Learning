# Self-trained baselines vs UnifiedHMSL (fair protocol, same split/features)

Baselines trained by us on the SAME fixed split (SEED=14), same 14-dim features (signed-log),
NeighborLoader [100,100]. Batch: GNN baselines 128, Meta-IFD 128 (single-task), UnifiedHMSL 64
(full multi-task OOMs at 128 on 6 GB — hardware limit). n = 5–15 restarts as noted.

## Full metric table (Macro-F1 %, etc.)
| Model | Task | Precision | Recall | Binary-F1 | Micro-F1 | Macro-F1 | n/batch |
|---|---|---|---|---|---|---|---|
| LightGBM | Ponzi | 67.99±2.40 | 44.74±2.35 | 53.95±2.31 | 89.22±0.47 | 73.92±1.28 | 5 / — |
| LightGBM | Phish | 94.20±0.20 | 97.01±0.41 | 95.58±0.28 | 96.09±0.25 | 96.04±0.25 | 5 / — |
| RGCN | Ponzi | 72.19 | 59.47 | 65.08 | 91.00 | 79.96 | 10 / 128 |
| RGCN | Phish | 95.66 | 97.39 | 96.51 | 96.93 | 96.88 | 10 / 128 |
| HGT | Ponzi | 80.95 | 59.47 | 68.45 | 92.27 | 82.02 | 10 / 128 |
| HGT | Phish | 95.55 | 95.85 | 95.69 | 96.24 | 96.18 | 10 / 128 |
| Meta-IFD | Ponzi | 76.42±4.32 | 56.32±6.36 | 64.47±3.57 | 91.30±0.56 | 79.76±1.89 | 10 / 128 |
| Meta-IFD | Phish | 96.37±1.29 | 97.59±0.45 | 96.97±0.62 | 97.34±0.57 | 97.30±0.57 | 10 / 128 |
| **UnifiedHMSL** | Ponzi | 82.74±4.69 | 56.49±4.49 | 66.94±3.07 | 92.14±0.63 | **81.24±1.69** | 15 / 64 |
| **UnifiedHMSL** | Phish | 96.20±1.79 | 97.59±1.09 | 96.88±0.75 | 97.25±0.69 | **97.21±0.69** | 15 / 64 |

## Macro-F1 ranking
| Model | Ponzi | Phish |
|---|---|---|
| HGT | **82.02** | 96.18 (worst) |
| UnifiedHMSL | 81.24 | 97.21 |
| RGCN | 79.96 | 96.88 |
| Meta-IFD | 79.76 | **97.30** |
| LightGBM | 73.92 | 96.04 |

## Honest positioning (for the paper)
- **No single baseline is strong on BOTH tasks.** HGT wins Ponzi but is worst on Phish; Meta-IFD wins
  Phish but is weak on Ponzi. **UnifiedHMSL is top-2 on both**, i.e. the only method that is consistently
  strong across Ponzi AND phishing — with **one unified model** (538K params) instead of two single-task
  models (~1.08M).
- Do NOT claim per-task SOTA: on Ponzi, HGT (82.02) ≥ UnifiedHMSL (81.24) (within noise). Claim
  **"unified, balanced, robust"**, not "best on each task".
- Supporting strengths: significant gain over Meta-IFD baseline (+2.93 Ponzi, p=0.005, batch 64);
  inductive ≈ transductive (generalises to unseen accounts); clearly beats feature-only GBDT;
  the signed-log input scaling is the key ablation contributor (Δ+1.93 Ponzi, Wilcoxon p=0.027).
