"""Regenerate a clean UTF-8 ablation report (main table with deltas + per-seed appendix)."""
import json
import numpy as np

d = json.load(open('ablation_results.json', encoding='utf-8'))
raw = d['raw']
seeds = d['args']['seeds']


def ms(v):
    return (np.mean(v), np.std(v)) if v else (float('nan'), float('nan'))


full_p = np.mean(raw["Full (MT+CPA+TCG+Expert)"]['ponzi_f1'])
full_h = np.mean(raw["Full (MT+CPA+TCG+Expert)"]['phish_f1'])

# (label, ponzi_list, phish_list)
rows = [
    ("Full (MT+CPA+TCG+Expert)", raw["Full (MT+CPA+TCG+Expert)"]['ponzi_f1'], raw["Full (MT+CPA+TCG+Expert)"]['phish_f1']),
    ("(-) Expert Rules",         raw["(-) Expert Rules"]['ponzi_f1'],         raw["(-) Expert Rules"]['phish_f1']),
    ("(-) CPA",                  raw["(-) CPA"]['ponzi_f1'],                  raw["(-) CPA"]['phish_f1']),
    ("(-) TCG",                  raw["(-) TCG"]['ponzi_f1'],                  raw["(-) TCG"]['phish_f1']),
    ("(-) Multi-task",           raw["(-) Multi-task [Ponzi]"]['ponzi_f1'],   raw["(-) Multi-task [Phish]"]['phish_f1']),
]

lines = []
lines.append(f"## Ablation study — macro-F1 (mean ± std over {len(seeds)} seeds: {seeds})\n")
lines.append("Leave-one-out: each row removes ONE novelty from the Full model. "
             "A **drop vs Full** means the component contributes; Δ = (ablated − Full).\n")
lines.append("| Configuration | Ponzi F1 | ΔPonzi | Phish F1 | ΔPhish |")
lines.append("|---|---|---|---|---|")
for label, pl, hl in rows:
    pm, ps = ms(pl); hm, hs = ms(hl)
    dp = "" if label.startswith("Full") else f"{pm-full_p:+.4f}"
    dh = "" if label.startswith("Full") else f"{hm-full_h:+.4f}"
    lines.append(f"| {label} | {pm:.4f} ± {ps:.4f} | {dp} | {hm:.4f} ± {hs:.4f} | {dh} |")

# robustness: in how many seeds does Full beat the ablated variant (per task)
lines.append("\n### Robustness — seeds (out of %d) where Full > ablated\n" % len(seeds))
lines.append("| Removed component | Ponzi wins | Phish wins |")
lines.append("|---|---|---|")
fp = raw["Full (MT+CPA+TCG+Expert)"]['ponzi_f1']
fh = raw["Full (MT+CPA+TCG+Expert)"]['phish_f1']
for label, key_p, key_h in [
    ("Expert Rules", ("(-) Expert Rules", 'ponzi_f1'), ("(-) Expert Rules", 'phish_f1')),
    ("CPA",          ("(-) CPA", 'ponzi_f1'),          ("(-) CPA", 'phish_f1')),
    ("TCG",          ("(-) TCG", 'ponzi_f1'),          ("(-) TCG", 'phish_f1')),
    ("Multi-task",   ("(-) Multi-task [Ponzi]", 'ponzi_f1'), ("(-) Multi-task [Phish]", 'phish_f1')),
]:
    ap = raw[key_p[0]][key_p[1]]
    ah = raw[key_h[0]][key_h[1]]
    wp = sum(1 for a, b in zip(fp, ap) if a > b)
    wh = sum(1 for a, b in zip(fh, ah) if a > b)
    lines.append(f"| {label} | {wp}/{len(seeds)} | {wh}/{len(seeds)} |")

# per-seed appendix
lines.append("\n### Appendix — per-seed test macro-F1\n")
lines.append("| Configuration | " + " | ".join(f"seed {s} (P/H)" for s in seeds) + " |")
lines.append("|---|" + "---|" * len(seeds))
for name in raw:
    pl = raw[name]['ponzi_f1']; hl = raw[name]['phish_f1']
    cells = []
    for i in range(len(seeds)):
        p = f"{pl[i]:.4f}" if i < len(pl) else "—"
        h = f"{hl[i]:.4f}" if i < len(hl) else "—"
        cells.append(f"{p} / {h}")
    lines.append(f"| {name} | " + " | ".join(cells) + " |")

out = "\n".join(lines) + "\n"
open('ablation_report.md', 'w', encoding='utf-8').write(out)
print(out)
