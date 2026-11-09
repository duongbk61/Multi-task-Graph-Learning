"""
Ablation harness for the four Meta-IFD novelties:
  (1) Multi-task dual-head  -> task_mode: 'multi' vs single-task ('ponzi'/'phish')
  (2) CPA  (CrossPathAttention) -> use_cpa
  (3) TCG  (TaskGate)           -> use_tcg
  (4) Expert Rules             -> expert_mode: 'feature' vs 'none'

Design:
  * FIXED data split (dataset.py SPLIT_SEED). --seeds only reseeds weight init +
    stochastic ICVAE augmentation, i.e. reproducible random restarts. Small data ->
    report mean +/- std over several seeds.
  * Every config gets the SAME budget (--epochs, --patience) and batch_size for fairness.
  * Model selection on validation macro-F1 (of the task(s) the config trains); the
    reported number is TEST macro-F1 at the best-val epoch.

Leave-one-out table (default): Full model minus each novelty in turn.
Each config's row is filled from the appropriate run(s):
  - multi-task configs give both Ponzi and Phish F1 from one run
  - the "-Multi-task" row takes Ponzi F1 from a ponzi-only run and Phish F1 from a phish-only run

Usage:
  python run_ablation.py --seeds 14 21 42 --epochs 60 --patience 20 --batch_size 64 --gpu 0
"""
import os
import copy
import json
import time
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score
from dataset import Ponzi, Phish
from unified_model import UnifiedHMSL
from run_unified import get_augmented_data
from utils import seed_everything

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Config matrix: name -> dict of switches. task_mode in {multi, ponzi, phish}. #
# The "-Multi-task" row is assembled from two single-task configs.            #
# --------------------------------------------------------------------------- #
def build_configs():
    full = dict(use_cpa=True, use_tcg=True, expert_mode='feature', task_mode='multi')
    return [
        ("Full (MT+CPA+TCG+Expert)", dict(full)),
        ("(-) Expert Rules",         {**full, 'expert_mode': 'none'}),
        ("(-) CPA",                  {**full, 'use_cpa': False}),
        ("(-) TCG",                  {**full, 'use_tcg': False}),
        # single-task runs for the multi-task ablation (everything else on):
        ("(-) Multi-task [Ponzi]",   {**full, 'task_mode': 'ponzi'}),
        ("(-) Multi-task [Phish]",   {**full, 'task_mode': 'phish'}),
    ]


@torch.no_grad()
def eval_task(model, loader, aug_model, target_node, aug_args, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = get_augmented_data(batch.to(device), aug_model, aug_args, device)
        bs = batch[target_node].batch_size
        out_ponzi, out_phish, *_ = model(batch.x_dict_new, batch.edge_index_dict,
                                         raw_x_dict=batch.x_dict)
        out = out_ponzi if target_node == 'CA' else out_phish
        preds.extend(out[:bs].argmax(1).cpu().numpy())
        labels.extend(batch[target_node].y[:bs].cpu().numpy())
    return f1_score(labels, preds, average='macro')


def train_one(cfg, seed, ctx, args):
    """Train a single model under cfg with the given seed; return dict of test F1."""
    device = ctx['device']
    seed_everything(seed)
    aug_args = argparse.Namespace(concat=args.concat)

    model = UnifiedHMSL(hidden=args.hidden, out_channels=2, data=ctx['data_ponzi'],
                        concat=args.concat, expert_mode=cfg['expert_mode'],
                        use_cpa=cfg['use_cpa'], use_tcg=cfg['use_tcg']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    task = cfg['task_mode']
    use_ponzi = task in ('multi', 'ponzi')
    use_phish = task in ('multi', 'phish')

    best_val = -1.0
    best_test = {}
    patience = 0

    for epoch in range(args.epochs):
        # ---- train ----
        model.train()
        if task == 'multi':
            iterator = zip(ctx['tr_ponzi'], ctx['tr_phish'])
        elif task == 'ponzi':
            iterator = ((b, None) for b in ctx['tr_ponzi'])
        else:
            iterator = ((None, b) for b in ctx['tr_phish'])

        for bp, bh in iterator:
            optimizer.zero_grad()
            loss = 0.0
            if bp is not None:
                bp = get_augmented_data(bp.to(device), ctx['aug_p'], aug_args, device)
                op, _, co_p, exp_p, _ = model(bp.x_dict_new, bp.edge_index_dict, raw_x_dict=bp.x_dict)
                sp = bp['CA'].batch_size
                loss = loss + F.cross_entropy(op[:sp], bp['CA'].y[:sp]) + args.loss_train * co_p
            if bh is not None:
                bh = get_augmented_data(bh.to(device), ctx['aug_h'], aug_args, device)
                _, oh, co_h, _, exp_h = model(bh.x_dict_new, bh.edge_index_dict, raw_x_dict=bh.x_dict)
                sh = bh['EOA'].batch_size
                loss = loss + F.cross_entropy(oh[:sh], bh['EOA'].y[:sh]) + args.loss_train * co_h
            loss.backward()
            optimizer.step()

        # ---- validate + track best (test frozen at best-val epoch) ----
        val_score = 0.0
        test_row = {}
        if use_ponzi:
            vp = eval_task(model, ctx['va_ponzi'], ctx['aug_p'], 'CA', aug_args, device)
            val_score += vp
            test_row['ponzi_f1'] = eval_task(model, ctx['te_ponzi'], ctx['aug_p'], 'CA', aug_args, device)
        if use_phish:
            vh = eval_task(model, ctx['va_phish'], ctx['aug_h'], 'EOA', aug_args, device)
            val_score += vh
            test_row['phish_f1'] = eval_task(model, ctx['te_phish'], ctx['aug_h'], 'EOA', aug_args, device)

        if val_score > best_val:
            best_val = val_score
            best_test = test_row
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    del model
    torch.cuda.empty_cache()
    return best_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[14, 21, 42])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--loss_train', type=float, default=0.1)
    ap.add_argument('--concat', type=int, default=3)
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--num_neighbors', type=int, nargs='+', default=[100, 100])
    ap.add_argument('--out', type=str, default='ablation_results')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | seeds={args.seeds} | epochs={args.epochs} "
          f"patience={args.patience} batch={args.batch_size} neighbors={args.num_neighbors}")

    data_ponzi = Ponzi('./data/Ponzi/')[0]
    data_phish = Phish('./data/Phish/')[0]
    aug_p = torch.load('./pretrain_model/icvae_Ponzi.pkl', map_location=device)
    aug_h = torch.load('./pretrain_model/icvae_Phish.pkl', map_location=device)

    nb = args.num_neighbors
    kw = dict(num_neighbors=nb, batch_size=args.batch_size)
    ctx = dict(
        device=device, data_ponzi=data_ponzi, data_phish=data_phish, aug_p=aug_p, aug_h=aug_h,
        tr_ponzi=NeighborLoader(data_ponzi, shuffle=True,  input_nodes=('CA', data_ponzi['CA'].train_mask), **kw),
        va_ponzi=NeighborLoader(data_ponzi, shuffle=False, input_nodes=('CA', data_ponzi['CA'].val_mask),   **kw),
        te_ponzi=NeighborLoader(data_ponzi, shuffle=False, input_nodes=('CA', data_ponzi['CA'].test_mask),  **kw),
        tr_phish=NeighborLoader(data_phish, shuffle=True,  input_nodes=('EOA', data_phish['EOA'].train_mask), **kw),
        va_phish=NeighborLoader(data_phish, shuffle=False, input_nodes=('EOA', data_phish['EOA'].val_mask),   **kw),
        te_phish=NeighborLoader(data_phish, shuffle=False, input_nodes=('EOA', data_phish['EOA'].test_mask),  **kw),
    )

    configs = build_configs()
    # raw[config_name][metric] = list of per-seed values
    raw = {name: {'ponzi_f1': [], 'phish_f1': []} for name, _ in configs}

    t_start = time.time()
    for seed in args.seeds:
        for name, cfg in configs:
            t0 = time.time()
            res = train_one(cfg, seed, ctx, args)
            for k, v in res.items():
                raw[name][k].append(v)
            msg = " ".join(f"{k}={v:.4f}" for k, v in res.items())
            print(f"[seed {seed}] {name:<28} {msg}  ({time.time()-t0:.0f}s)", flush=True)
            # checkpoint partial results after every run
            with open(f"{args.out}.json", 'w') as f:
                json.dump({'args': vars(args), 'raw': raw}, f, indent=2)

    # ---- assemble leave-one-out table ----
    def ms(vals):
        return (np.mean(vals), np.std(vals)) if vals else (float('nan'), float('nan'))

    rows = []  # (label, ponzi_mean, ponzi_std, phish_mean, phish_std)
    for name in ["Full (MT+CPA+TCG+Expert)", "(-) Expert Rules", "(-) CPA", "(-) TCG"]:
        pm, ps = ms(raw[name]['ponzi_f1']); hm, hs = ms(raw[name]['phish_f1'])
        rows.append((name, pm, ps, hm, hs))
    # multi-task ablation: ponzi from ponzi-only, phish from phish-only
    pm, ps = ms(raw["(-) Multi-task [Ponzi]"]['ponzi_f1'])
    hm, hs = ms(raw["(-) Multi-task [Phish]"]['phish_f1'])
    rows.append(("(-) Multi-task", pm, ps, hm, hs))

    n = len(args.seeds)
    header = f"\n## Ablation (macro-F1, mean±std over {n} seeds: {args.seeds})\n"
    table = "| Configuration | Ponzi F1 | Phish F1 |\n|---|---|---|\n"
    for label, pm, ps, hm, hs in rows:
        table += f"| {label} | {pm:.4f} ± {ps:.4f} | {hm:.4f} ± {hs:.4f} |\n"
    print(header + table)
    with open(f"{args.out}.md", 'w') as f:
        f.write(header + table)
    print(f"Saved {args.out}.md and {args.out}.json  |  total {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
