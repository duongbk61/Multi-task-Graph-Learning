# Ethereum Fraud Detection — Comparison Table

Baselines transcribed from the original paper (values are mean±std ×100).
- **Precision / Recall / Binary-F1** = positive (fraud) class.
- **Micro-F1** = accuracy. **Macro-F1** = macro average. **Rank** = per-dataset method rank from the paper.
- Our row (bottom) = **5-fold stratified CV** of the Full model (MT+CPA+TCG+Expert), no split selection.
  Rank/Avg-Rank/Group-Rank are left blank for our row (they depend on the full ranking and on which rows you keep).

> Editing note: this is a plain Markdown table — delete any row you don't need by removing its line.
> The last two rows are `Meta-IFD (paper)` (as printed in the paper) and `Meta-IFD (ours, 5-fold CV)` (our reproduction).

## Ponzi Scheme Dataset

| Method | Precision | Recall | Binary-F1 | Micro-F1 | Macro-F1 | Rank |
|---|---|---|---|---|---|---|
| LightGBM | 72.29±11.91 | 38.42±2.11 | 49.97±4.18 | 89.07±1.30 | 71.92±2.46 | 8.2 |
| Trans2Vec | 91.16±3.24 | 48.42±4.59 | 63.12±4.08 | 92.04±0.69 | 79.33±2.22 | 2.8 |
| GCN | 67.61±8.70 | 27.89±6.78 | 38.44±5.22 | 87.66±0.49 | 65.78±2.51 | 12.2 |
| GAT | 77.62±6.48 | 47.37±6.66 | 58.40±5.01 | 90.56±0.80 | 76.53±2.68 | 5.0 |
| SAGE | 70.88±7.06 | 37.37±3.07 | 48.78±3.26 | 88.92±0.79 | 71.29±1.82 | 9.6 |
| FA-GNN | 66.69±6.00 | 42.11±4.40 | 51.17±1.76 | 88.70±0.50 | 72.39±0.76 | 9.0 |
| HGT | 58.47±5.69 | 45.26±1.97 | 50.83±2.26 | 87.58±1.26 | 71.86±1.49 | 10.4 |
| RGCN | 69.61±5.64 | 53.16±5.10 | 60.26±5.29 | 90.11±1.26 | 77.31±3.00 | 5.6 |
| ieHGCN | 61.49±5.34 | 38.42±11.36 | 46.59±10.69 | 88.03±1.21 | 69.92±5.62 | 11.4 |
| HTSGCN | 67.47±5.79 | 42.11±3.33 | 51.59±2.34 | 88.85±0.78 | 72.64±1.30 | 8.2 |
| BERT4ETH | 72.83±3.75 | 65.26±4.21 | 68.67±2.18 | 91.60±0.60 | 81.91±1.23 | 2.4 |
| ZipZap | 69.64±8.44 | 56.22±4.65 | 61.77±3.68 | 90.37±1.28 | 78.13±2.17 | 4.6 |
| Meta-IFD (paper) | 79.71±3.66 | 61.58±1.29 | 69.46±2.05 | 92.34±0.60 | 82.54±1.20 | 1.4 |
| **Meta-IFD (ours, fixed split)** | 73.91±4.05 | 57.37±4.53 | 64.43±3.19 | 91.08±0.71 | 79.66±1.76 | — |
| **Meta-IFD (ours, 5-fold CV)** | 69.09±4.15 | 65.44±4.56 | 67.18±4.12 | 90.91±1.12 | 80.95±2.38 | — |

## Phish Scam Dataset

| Method | Precision | Recall | Binary-F1 | Micro-F1 | Macro-F1 | Rank |
|---|---|---|---|---|---|---|
| LightGBM | 94.13±0.67 | 97.01±1.13 | 95.54±0.37 | 96.06±0.31 | 96.00±0.32 | 4.8 |
| Trans2Vec | 77.21±1.85 | 74.19±2.11 | 75.63±0.96 | 79.17±0.83 | 78.71±0.82 | 13.0 |
| GCN | 92.05±1.35 | 88.38±1.14 | 90.18±1.19 | 91.61±1.03 | 91.43±1.05 | 11.6 |
| GAT | 90.20±1.52 | 95.19±0.86 | 92.62±0.95 | 93.38±0.90 | 93.31±0.90 | 9.6 |
| SAGE | 93.46±0.45 | 96.02±0.62 | 94.72±0.44 | 95.33±0.39 | 95.27±0.39 | 6.2 |
| FA-GNN | 96.92±0.20 | 96.60±0.31 | 96.76±0.21 | 97.18±0.18 | 97.13±0.19 | 2.6 |
| HGT | 90.89±2.16 | 92.20±1.16 | 91.53±1.33 | 92.55±1.22 | 92.44±1.22 | 11.0 |
| RGCN | 94.95±1.57 | 94.36±1.07 | 94.64±0.45 | 95.35±0.43 | 95.25±0.43 | 6.4 |
| ieHGCN | 94.38±0.83 | 94.36±3.63 | 94.32±1.75 | 95.08±1.39 | 94.99±1.45 | 7.4 |
| HTSGCN | 97.00±0.68 | 96.27±0.95 | 96.63±0.37 | 97.07±0.31 | 97.02±0.31 | 3.2 |
| BERT4ETH | 92.84±1.64 | 92.95±1.17 | 92.87±0.42 | 93.78±0.42 | 93.68±0.41 | 9.2 |
| ZipZap | 98.27±1.24 | 94.22±1.61 | 96.19±0.47 | 96.96±0.35 | 96.83±0.37 | 4.4 |
| Meta-IFD (paper) | 97.37±0.55 | 98.01±0.17 | 97.68±0.27 | 97.97±0.24 | 97.94±0.24 | 1.2 |
| **Meta-IFD (ours, fixed split)** | 97.02±1.30 | 96.76±0.31 | 96.89±0.72 | 97.29±0.64 | 97.24±0.64 | — |
| **Meta-IFD (ours, 5-fold CV)** | 96.98±0.48 | 95.86±0.86 | 96.41±0.54 | 96.89±0.46 | 96.83±0.47 | — |

## Overall Rank (from paper)

| Method | Average Rank | Group Rank |
|---|---|---|
| LightGBM | 6.5 | 6.50 |
| Trans2Vec | 7.9 | — |
| GCN | 11.9 | 8.14 |
| GAT | 7.3 | (group) |
| SAGE | 7.9 | (group) |
| FA-GNN | 5.8 | (group) |
| HGT | 10.7 | 7.95 |
| RGCN | 6.0 | (group) |
| ieHGCN | 9.4 | (group) |
| HTSGCN | 5.8 | (group) |
| BERT4ETH | 5.8 | 5.15 |
| ZipZap | 4.5 | (group) |
| Meta-IFD (paper) | 1.3 | 1.30 |

## ⚠️ Read before using our rows (honesty notes)

There are **two** "ours" rows. Pick the one that matches your framing:

- **`Meta-IFD (ours, fixed split)`** — same protocol as EVERY paper baseline: the baked
  split (`SPLIT_SEED=14`) + K=5 random restarts, mean±std over restarts. **This is the
  row to place next to the baselines** — it is protocol-comparable.
- **`Meta-IFD (ours, 5-fold CV)`** — stratified 5-fold cross-validation (each labeled node
  tested once). More rigorous / harder, but a *different* protocol from the baselines, so
  larger std and not strictly cell-comparable. Best used as a robustness result.

Three caveats that apply to both:

1. **Reproduction gap vs the paper's own Meta-IFD.** Our reproduction lands a few points below
   the paper's Meta-IFD on Ponzi (fixed-split Macro-F1 79.7 vs 82.5; CV 81.0). The main reason:
   **batch size**. The paper trains at `batch_size=512`; our GPU (GTX 1660, 6 GB) forces
   `batch_size=64`. Phish is essentially matched (97.2–96.8 vs 97.9).
2. **Run-to-run nondeterminism.** `torch_scatter`/`torch_sparse` use GPU atomic adds → not
   bit-reproducible; Ponzi Macro-F1 fluctuates ≈ ±0.01 between identical re-runs (we observed
   CV 80.95 and 81.90 on two runs). Treat Ponzi as ≈ 80 ± 2.
3. **Phish is saturated** (~96–98 for most strong methods), so differences there are within noise.

Rank / Avg-Rank / Group-Rank are blank for our rows — they depend on the full ranking and on
which rows you keep after trimming. Recompute them yourself if you need a rank for our method.

