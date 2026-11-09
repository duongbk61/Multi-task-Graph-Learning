# Analysis round 2 (Nov 2026)

Scripts and raw numbers behind the answers to the second advisor review.
Run every script from the repo root, e.g. `python analysis_round2/bench2.py train joint`.
Outputs are written next to the script. Model checkpoints (`*.pth`) and logs are not committed.

| File | What it holds |
|------|---------------|
| `check_triplet.py` | Triplet gradients, ICVAE draw diversity, attention vs plain mean |
| `gate_stats.py`, `gate_init.py`, `gate_stats.json` | CPA/TCG gate values on the test nodes (seeds 14, 21, 42) |
| `wo_tcg_rest.json`, `wo_tcg_rest_row.json` | The 6 extra CPA-only seeds (88, 99, 123, 777, 2024, 314); not merged into `A_wo_tcg.json` |
| `fusion_2x2.json` | Per-seed confusion counts and macro-F1 for mean / CPA / TCG / CPA+TCG, 15 paired seeds |
| `gbdt_check.json` | LightGBM rerun used to check the baseline row |
| `bench.py`, `bench_train.json`, `bench_mean.json` | Train/inference cost at batch 64, seeds 14/21/42, plus mean fusion |
| `bench2.py`, `bench2_train_*.json`, `bench2_infer.jsonl` | Paper cost table: joint vs single-task, batch 128, seed 14 |

These are measurement runs. Accuracy numbers for the paper come from the `A_*.json` files in the repo root.
