"""
Fixed-split (paper protocol) evaluation of the Full model, producing the full
paper-format metric set. Uses the baked dataset.py split (SPLIT_SEED=14) and
K random restarts (varying train_seed = weight init + ICVAE augmentation), then
reports mean±std over restarts — exactly the protocol every paper baseline uses.
This row IS directly comparable to the paper table.

  Precision / Recall / Binary-F1 = positive (fraud) class
  Micro-F1 = accuracy ; Macro-F1 = macro average

Usage:
  python run_fixed_table.py --restarts 14 21 42 77 100 --epochs 60 --patience 20 --gpu 0
"""
import json
import argparse
import warnings
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler
from dataset import Ponzi, Phish
from unified_model import UnifiedHMSL
from run_unified import get_augmented_data
from run_cv_table import eval_full
from utils import seed_everything

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--restarts', type=int, nargs='+', default=[14, 21, 42, 77, 100])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--loss_train', type=float, default=0.1)
    ap.add_argument('--concat', type=int, default=3)
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--num_neighbors', type=int, nargs='+', default=[100, 100])
    ap.add_argument('--use_cpa', type=int, default=1)
    ap.add_argument('--use_tcg', type=int, default=1)
    ap.add_argument('--use_generative', type=int, default=1)
    ap.add_argument('--use_contrastive', type=int, default=1)
    ap.add_argument('--raw_transform', type=str, default='none', choices=['none', 'signed_log'])
    ap.add_argument('--expert_mode', type=str, default='feature', choices=['feature', 'loss', 'none'])
    ap.add_argument('--out', type=str, default='fixed_table')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    dp = Ponzi('./data/Ponzi/')[0]
    dh = Phish('./data/Phish/')[0]
    aug_p = torch.load('./pretrain_model/icvae_Ponzi.pkl', map_location=device)
    aug_h = torch.load('./pretrain_model/icvae_Phish.pkl', map_location=device)
    ns_p = NeighborSampler(dp, num_neighbors=args.num_neighbors)
    ns_h = NeighborSampler(dh, num_neighbors=args.num_neighbors)
    aug_args = argparse.Namespace(concat=args.concat)
    bs = dict(batch_size=args.batch_size, num_neighbors=args.num_neighbors)

    # baked SPLIT_SEED=14 masks straight from the dataset
    tr_p = NeighborLoader(dp, shuffle=True,  input_nodes=('CA',  dp['CA'].train_mask), neighbor_sampler=ns_p, **bs)
    va_p = NeighborLoader(dp, shuffle=False, input_nodes=('CA',  dp['CA'].val_mask),   neighbor_sampler=ns_p, **bs)
    te_p = NeighborLoader(dp, shuffle=False, input_nodes=('CA',  dp['CA'].test_mask),  neighbor_sampler=ns_p, **bs)
    tr_h = NeighborLoader(dh, shuffle=True,  input_nodes=('EOA', dh['EOA'].train_mask), neighbor_sampler=ns_h, **bs)
    va_h = NeighborLoader(dh, shuffle=False, input_nodes=('EOA', dh['EOA'].val_mask),   neighbor_sampler=ns_h, **bs)
    te_h = NeighborLoader(dh, shuffle=False, input_nodes=('EOA', dh['EOA'].test_mask),  neighbor_sampler=ns_h, **bs)
    print(f"fixed split: test CA={int(dp['CA'].test_mask.sum())} EOA={int(dh['EOA'].test_mask.sum())} "
          f"| restarts={args.restarts}", flush=True)

    runs = []
    for rs in args.restarts:
        seed_everything(rs)
        model = UnifiedHMSL(hidden=args.hidden, out_channels=2, data=dp, concat=args.concat,
                            expert_mode=args.expert_mode,
                            use_cpa=bool(args.use_cpa), use_tcg=bool(args.use_tcg),
                            use_generative=bool(args.use_generative),
                            use_contrastive=bool(args.use_contrastive),
                            raw_transform=args.raw_transform).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        best_val, best = -1.0, None
        patience = 0
        for epoch in range(args.epochs):
            model.train()
            for bp, bh in zip(tr_p, tr_h):
                opt.zero_grad()
                bp = get_augmented_data(bp.to(device), aug_p, aug_args, device)
                op, _, co_p, _, _ = model(bp.x_dict_new, bp.edge_index_dict, raw_x_dict=bp.x_dict)
                sp = bp['CA'].batch_size
                bh = get_augmented_data(bh.to(device), aug_h, aug_args, device)
                _, oh, co_h, _, _ = model(bh.x_dict_new, bh.edge_index_dict, raw_x_dict=bh.x_dict)
                sh = bh['EOA'].batch_size
                loss = (F.cross_entropy(op[:sp], bp['CA'].y[:sp]) + F.cross_entropy(oh[:sh], bh['EOA'].y[:sh])
                        + args.loss_train * (co_p + co_h))
                loss.backward(); opt.step()
            vp = eval_full(model, va_p, aug_p, 'CA', aug_args, device)['macro_f1']
            vh = eval_full(model, va_h, aug_h, 'EOA', aug_args, device)['macro_f1']
            if vp + vh > best_val:
                best_val = vp + vh
                best = dict(ponzi=eval_full(model, te_p, aug_p, 'CA', aug_args, device),
                            phish=eval_full(model, te_h, aug_h, 'EOA', aug_args, device))
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    break
        runs.append(best)
        pz = best['ponzi']
        print(f"[restart {rs}] Ponzi macroF1={pz['macro_f1']:.4f} binF1={pz['binary_f1']:.4f} "
              f"P={pz['precision']:.4f} R={pz['recall']:.4f} micro={pz['micro_f1']:.4f} | "
              f"Phish macroF1={best['phish']['macro_f1']:.4f}", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
        with open(f"{args.out}.json", 'w') as f:
            json.dump({'args': vars(args), 'runs': runs}, f, indent=2)

    order = [('precision', 'Precision'), ('recall', 'Recall'), ('binary_f1', 'Binary-F1'),
             ('micro_f1', 'Micro-F1'), ('macro_f1', 'Macro-F1'),
             ('pr_auc', 'PR-AUC'), ('roc_auc', 'ROC-AUC'),
             ('balanced_acc', 'Bal-Acc'), ('fpr', 'FPR')]

    def agg(task, key):
        v = np.array([r[task][key] for r in runs]) * 100
        return v.mean(), v.std()

    print("\n===== OUR FIXED-SPLIT ROW (paper protocol, mean±std ×100) =====", flush=True)
    row = {}
    for task, tag in [('ponzi', 'Ponzi'), ('phish', 'Phish')]:
        cells = [tag]
        for key, name in order:
            m, s = agg(task, key)
            row[f"{task}_{key}"] = [round(m, 2), round(s, 2)]
            cells.append(f"{name}={m:.2f}±{s:.2f}")
        print("  " + "  ".join(cells), flush=True)
    # representative confusion matrix = FIRST restart (seed order); store raw counts
    rep = args.restarts[0]
    row['_confusion_seed'] = rep
    for task, nt in [('ponzi', 'CA'), ('phish', 'EOA')]:
        c = runs[0][task]
        row[f'{task}_confusion'] = dict(tp=c['tp'], fp=c['fp'], fn=c['fn'], tn=c['tn'])
        print(f"  [confusion {task} seed {rep}] TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']}", flush=True)
    with open(f"{args.out}_row.json", 'w') as f:
        json.dump(row, f, indent=2)
    print(f"\nSaved {args.out}.json and {args.out}_row.json", flush=True)


if __name__ == '__main__':
    main()
