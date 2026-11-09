## Ablation study — macro-F1 (mean ± std over 3 seeds: [14, 21, 42])

Leave-one-out: each row removes ONE novelty from the Full model. A **drop vs Full** means the component contributes; Δ = (ablated − Full).

| Configuration | Ponzi F1 | ΔPonzi | Phish F1 | ΔPhish |
|---|---|---|---|---|
| Full (MT+CPA+TCG+Expert) | 0.8047 ± 0.0054 |  | 0.9670 ± 0.0053 |  |
| (-) Expert Rules | 0.7830 ± 0.0138 | -0.0218 | 0.9664 ± 0.0031 | -0.0006 |
| (-) CPA | 0.7938 ± 0.0177 | -0.0109 | 0.9712 ± 0.0017 | +0.0042 |
| (-) TCG | 0.7727 ± 0.0222 | -0.0320 | 0.9652 ± 0.0039 | -0.0018 |
| (-) Multi-task | 0.8238 ± 0.0141 | +0.0191 | 0.9718 ± 0.0008 | +0.0048 |

### Robustness — seeds (out of 3) where Full > ablated

| Removed component | Ponzi wins | Phish wins |
|---|---|---|
| Expert Rules | 3/3 | 2/3 |
| CPA | 2/3 | 0/3 |
| TCG | 3/3 | 3/3 |
| Multi-task | 1/3 | 0/3 |

### Appendix — per-seed test macro-F1

| Configuration | seed 14 (P/H) | seed 21 (P/H) | seed 42 (P/H) |
|---|---|---|---|
| Full (MT+CPA+TCG+Expert) | 0.8112 / 0.9688 | 0.8049 / 0.9598 | 0.7980 / 0.9723 |
| (-) Expert Rules | 0.7972 / 0.9651 | 0.7874 / 0.9706 | 0.7643 / 0.9634 |
| (-) CPA | 0.8076 / 0.9724 | 0.8049 / 0.9688 | 0.7689 / 0.9723 |
| (-) TCG | 0.7504 / 0.9688 | 0.8029 / 0.9597 | 0.7647 / 0.9670 |
| (-) Multi-task [Ponzi] | 0.8039 / — | 0.8358 / — | 0.8317 / — |
| (-) Multi-task [Phish] | — / 0.9724 | — / 0.9706 | — / 0.9724 |
