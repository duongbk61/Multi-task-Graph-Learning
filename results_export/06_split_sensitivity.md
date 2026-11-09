# Split-sensitivity study (Full model, 16 splits, fixed train_seed)

Trains the Full model on 16 different stratified splits (only the data partition varies).
The DISTRIBUTION characterises sensitivity to the split; **the MAX is a lucky-split upper
bound (diagnostic), NOT a reportable metric** (selecting on test = leakage).

| Statistic | Ponzi macro-F1 | Phish macro-F1 |
|---|---|---|
| min | 77.53 | 94.61 |
| p25 | 79.71 | 96.70 |
| median | 82.33 | 97.24 |
| mean | 82.26 | 97.14 |
| std | 2.86 | 0.96 |
| MAX (ceiling) | 87.12 | 98.35 |

Mean±std ×100 — Ponzi 82.26±2.86, Phish 97.14±0.96.
Luckiest split for Ponzi: seed 42 -> 87.12 (upper envelope, do NOT report as performance).

Note: split-mean (82.26) matches the 5-fold CV mean, cross-validating the CV number.
