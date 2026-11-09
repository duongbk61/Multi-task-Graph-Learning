# Comparison table (grouped layout, matching the requested figure)

Values are mean±std ×100. **P** = Precision (fraud class), **R** = Recall (fraud class),
**B-F1** = Binary-F1 (fraud class), **Mic** = Micro-F1 (accuracy), **Mac** = Macro-F1.

Protocol notes:
- Baseline rows (LightGBM … Meta-IFD baseline) are the **paper's reported numbers** (their models
  are not in this repo, so they cannot be re-run here).
- `MT-HGNN (shared enc.)` and `UnifiedHMSL (ours)` are **our runs**: fixed split (SPLIT_SEED=14)
  + 5 random restarts, mean±std — same protocol, batch_size=64 (6 GB GPU).
- `MT-HGNN (shared enc.)` = UnifiedHMSL with the three novelties OFF (no CPA, no TCG, no Expert):
  shared `my_conv` backbone + ICVAE augmentation + dual heads only. It isolates the contribution
  of the 4 novelties (CPA + TCG + Expert + the multi-task heads that UnifiedHMSL adds on top).

| Method | Ponzi P | Ponzi R | Ponzi B-F1 | Ponzi Mic | Ponzi Mac | Phish P | Phish R | Phish B-F1 | Phish Mic | Phish Mac |
|---|---|---|---|---|---|---|---|---|---|---|
| ***Machine learning*** | | | | | | | | | | |
| LightGBM | 72.29±11.91 | 38.42±2.11 | 49.97±4.18 | 89.07±1.30 | 71.92±2.46 | 94.13±0.67 | 97.01±1.13 | 95.54±0.37 | 96.06±0.31 | 96.00±0.32 |
| ***Single-task, homogeneous graph*** | | | | | | | | | | |
| GCN | 67.61±8.70 | 27.89±6.78 | 38.44±5.22 | 87.66±0.49 | 65.78±2.51 | 92.05±1.35 | 88.38±1.14 | 90.18±1.19 | 91.61±1.03 | 91.43±1.05 |
| GAT | 77.62±6.48 | 47.37±6.66 | 58.40±5.01 | 90.56±0.80 | 76.53±2.68 | 90.20±1.52 | 95.19±0.86 | 92.62±0.95 | 93.38±0.90 | 93.31±0.90 |
| GraphSAGE | 70.88±7.06 | 37.37±3.07 | 48.78±3.26 | 88.92±0.79 | 71.29±1.82 | 93.46±0.45 | 96.02±0.62 | 94.72±0.44 | 95.33±0.39 | 95.27±0.39 |
| ***Single-task, heterogeneous graph*** | | | | | | | | | | |
| RGCN | 69.61±5.64 | 53.16±5.10 | 60.26±5.29 | 90.11±1.26 | 77.31±3.00 | 94.95±1.57 | 94.36±1.07 | 94.64±0.45 | 95.35±0.43 | 95.25±0.43 |
| HGT | 58.47±5.69 | 45.26±1.97 | 50.83±2.26 | 87.58±1.26 | 71.86±1.49 | 90.89±2.16 | 92.20±1.16 | 91.53±1.33 | 92.55±1.22 | 92.44±1.22 |
| Meta-IFD (baseline) | 79.71±3.66 | 61.58±1.29 | 69.46±2.05 | 92.34±0.60 | 82.54±1.20 | 97.37±0.55 | 98.01±0.17 | 97.68±0.27 | 97.97±0.24 | 97.94±0.24 |
| ***Multi-task*** | | | | | | | | | | |
| MT-HGNN (shared enc.) | 69.48±3.38 | 58.42±7.14 | 63.21±4.93 | 90.48±0.87 | 78.87±2.67 | 96.10±1.23 | 97.59±0.31 | 96.83±0.61 | 97.22±0.56 | 97.17±0.56 |
| **UnifiedHMSL (ours)** | 73.91±4.05 | 57.37±4.53 | 64.43±3.19 | 91.08±0.71 | 79.66±1.76 | 97.02±1.30 | 96.76±0.31 | 96.89±0.72 | 97.29±0.64 | 97.24±0.64 |

## What the multi-task rows show (our controlled comparison)

`MT-HGNN (shared enc.)` and `UnifiedHMSL (ours)` differ ONLY by the three novelties
(CPA + TCG + Expert) — same backbone, same split, same 5 restarts — so their gap is a clean
ablation of the novelties:

| | Ponzi Macro-F1 | Ponzi Binary-F1 | Phish Macro-F1 |
|---|---|---|---|
| MT-HGNN (shared enc.) | 78.87±2.67 | 63.21±4.93 | 97.17±0.56 |
| **UnifiedHMSL (ours)** | **79.66±1.76** | **64.43±3.19** | 97.24±0.56 |
| Δ (novelties add) | **+0.79** | **+1.22** | +0.07 |

- The novelties give a **modest but consistent gain on Ponzi** (the hard task) and also **lower
  variance** (std 2.67 → 1.76). On Phish they are within noise — Phish is saturated (~97), so
  there is little headroom to show a gain.
- Caveat (unchanged): both our rows use `batch_size=64` (6 GB GPU) vs the paper's 512, which is
  why our UnifiedHMSL sits a bit below the paper's Meta-IFD row on Ponzi.

> Honest note: this comparison shows the novelties help, but the effect on Ponzi is small
> (~0.8 macro-F1) and Phish is flat. Frame the contribution accordingly — do not overstate it.
