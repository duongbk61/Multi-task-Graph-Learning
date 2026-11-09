"""
Stratified K-fold cross-validation for the FULL model (MT+CPA+TCG+Expert).

Motivation: the baked split (dataset.py SPLIT_SEED=14) is a single stratified
partition; with only ~38 positive Ponzi nodes in a 20% test slice, macro-F1 is
highly sensitive to WHICH nodes land in test. The honest way to characterise that
sensitivity is cross-validation — NOT searching for a favourable split (which would
be test-set selection / leakage). Every labeled node is tested exactly once across
folds; we report mean±std over folds.

Protocol (per task, independently, then trained jointly):
  * StratifiedKFold(K, shuffle, random_state=CV_SEED) over labeled nodes (y != 2).
  * Within each fold's train pool, carve a stratified val slice for model selection.
  * Fixed training seed per fold (weight init + ICVAE aug); model-select on val
    (Ponzi+Phish macro-F1); report TEST macro-F1 at best-val epoch.

Caveat logged for the report: the DT-distilled ExpertRules are FIXED constants in
code (domain knowledge), not re-fit per fold — treated as a fixed heuristic.

Usage:
  python run_cv.py --folds 5 --epochs 60 --patience 20 --train_seed 42 --gpu 0
"""
import json
import time
import copy
import argparse
import warnings
import gc
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler
from sklearn.metrics import f1_score, precision_score, recall_score
from dataset import Ponzi, Phish
from unified_model import UnifiedHMSL
from run_unified import get_augmented_data
from utils import seed_everything

warnings.filterwarnings("ignore")


def labeled_indices(data, ntype):
    y = data[ntype].y
    idx = (y != 2).nonzero(as_tuple=True)[0].cpu().numpy()
    return idx, y[idx].cpu().numpy()


def mask_from(idx, size):
    m = torch.zeros(size, dtype=torch.bool)
    m[torch.as_tensor(idx, dtype=torch.long)] = True
    return m


@torch.no_grad()
def eval_task(model, loader, aug_model, target_node, aug_args, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = get_augmented_data(batch.to(device), aug_model, aug_args, device)
        bs = batch[target_node].batch_size
        op, oh, *_ = model(batch.x_dict_new, batch.edge_index_dict, raw_x_dict=batch.x_dict)
        out = op if target_node == 'CA' else oh
        preds.extend(out[:bs].argmax(1).cpu().numpy())
        labels.extend(batch[target_node].y[:bs].cpu().numpy())
    return dict(f1=f1_score(labels, preds, average='macro'),
                precision=precision_score(labels, preds, average='macro', zero_division=0),
                recall=recall_score(labels, preds, average='macro', zero_division=0))


def train_fold(masks, ctx, args):
    device = ctx['device']
    seed_everything(args.train_seed)
    aug_args = argparse.Namespace(concat=args.concat)
    dp, dh = ctx['data_ponzi'], ctx['data_phish']
    # Reuse ONE prebuilt sampler per graph (graph-only; seed nodes passed per batch)
    # so we don't rebuild a full-graph CSC for every loader/fold -> avoids CPU-RAM blowup.
    bs = dict(batch_size=args.batch_size, num_neighbors=args.num_neighbors)
    tr_p = NeighborLoader(dp, shuffle=True,  input_nodes=('CA',  masks['ca_train']), neighbor_sampler=ctx['ns_p'], **bs)
    va_p = NeighborLoader(dp, shuffle=False, input_nodes=('CA',  masks['ca_val']),   neighbor_sampler=ctx['ns_p'], **bs)
    te_p = NeighborLoader(dp, shuffle=False, input_nodes=('CA',  masks['ca_test']),  neighbor_sampler=ctx['ns_p'], **bs)
    tr_h = NeighborLoader(dh, shuffle=True,  input_nodes=('EOA', masks['eoa_train']), neighbor_sampler=ctx['ns_h'], **bs)
    va_h = NeighborLoader(dh, shuffle=False, input_nodes=('EOA', masks['eoa_val']),   neighbor_sampler=ctx['ns_h'], **bs)
    te_h = NeighborLoader(dh, shuffle=False, input_nodes=('EOA', masks['eoa_test']),  neighbor_sampler=ctx['ns_h'], **bs)

    model = UnifiedHMSL(hidden=args.hidden, out_channels=2, data=dp, concat=args.concat,
                        expert_mode='feature', use_cpa=True, use_tcg=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_val = -1.0
    best = {}
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        for bp, bh in zip(tr_p, tr_h):
            opt.zero_grad()
            bp = get_augmented_data(bp.to(device), ctx['aug_p'], aug_args, device)
            op, _, co_p, _, _ = model(bp.x_dict_new, bp.edge_index_dict, raw_x_dict=bp.x_dict)
            sp = bp['CA'].batch_size
            bh = get_augmented_data(bh.to(device), ctx['aug_h'], aug_args, device)
            _, oh, co_h, _, _ = model(bh.x_dict_new, bh.edge_index_dict, raw_x_dict=bh.x_dict)
            sh = bh['EOA'].batch_size
            loss = (F.cross_entropy(op[:sp], bp['CA'].y[:sp])
                    + F.cross_entropy(oh[:sh], bh['EOA'].y[:sh])
                    + args.loss_train * (co_p + co_h))
            loss.backward()
            opt.step()

        vp = eval_task(model, va_p, ctx['aug_p'], 'CA', aug_args, device)['f1']
        vh = eval_task(model, va_h, ctx['aug_h'], 'EOA', aug_args, device)['f1']
        if vp + vh > best_val:
            best_val = vp + vh
            tp = eval_task(model, te_p, ctx['aug_p'], 'CA', aug_args, device)
            th = eval_task(model, te_h, ctx['aug_h'], 'EOA', aug_args, device)
            best = dict(ponzi=tp, phish=th, epoch=epoch)
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    del model, tr_p, va_p, te_p, tr_h, va_h, te_h
    gc.collect()
    torch.cuda.empty_cache()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--cv_seed', type=int, default=42)
    ap.add_argument('--train_seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--loss_train', type=float, default=0.1)
    ap.add_argument('--concat', type=int, default=3)
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--num_neighbors', type=int, nargs='+', default=[100, 100])
    ap.add_argument('--val_frac', type=float, default=0.25)  # of the fold train pool
    ap.add_argument('--out', type=str, default='cv_results')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | {args.folds}-fold CV | cv_seed={args.cv_seed} "
          f"train_seed={args.train_seed} epochs={args.epochs} patience={args.patience}", flush=True)

    dp = Ponzi('./data/Ponzi/')[0]
    dh = Phish('./data/Phish/')[0]
    aug_p = torch.load('./pretrain_model/icvae_Ponzi.pkl', map_location=device)
    aug_h = torch.load('./pretrain_model/icvae_Phish.pkl', map_location=device)
    # Build one sampler per graph, reused across all folds/loaders.
    ns_p = NeighborSampler(dp, num_neighbors=args.num_neighbors)
    ns_h = NeighborSampler(dh, num_neighbors=args.num_neighbors)
    ctx = dict(device=device, data_ponzi=dp, data_phish=dh, aug_p=aug_p, aug_h=aug_h,
               ns_p=ns_p, ns_h=ns_h)

    ca_idx, ca_y = labeled_indices(dp, 'CA')
    eoa_idx, eoa_y = labeled_indices(dh, 'EOA')
    n_ca, n_eoa = dp['CA'].y.size(0), dh['EOA'].y.size(0)
    print(f"labeled: CA={len(ca_idx)} (pos={int(ca_y.sum())})  EOA={len(eoa_idx)} (pos={int(eoa_y.sum())})", flush=True)

    skf_ca = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.cv_seed)
    skf_eoa = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.cv_seed)
    ca_folds = list(skf_ca.split(ca_idx, ca_y))
    eoa_folds = list(skf_eoa.split(eoa_idx, eoa_y))

    fold_results = []
    t0 = time.time()
    for k in range(args.folds):
        ca_tr_pool, ca_te = ca_folds[k]
        eoa_tr_pool, eoa_te = eoa_folds[k]
        # stratified val carve-out from the train pool
        ca_tr, ca_va = train_test_split(ca_tr_pool, test_size=args.val_frac,
                                        random_state=args.cv_seed, stratify=ca_y[ca_tr_pool])
        eoa_tr, eoa_va = train_test_split(eoa_tr_pool, test_size=args.val_frac,
                                          random_state=args.cv_seed, stratify=eoa_y[eoa_tr_pool])
        masks = dict(
            ca_train=mask_from(ca_idx[ca_tr], n_ca), ca_val=mask_from(ca_idx[ca_va], n_ca),
            ca_test=mask_from(ca_idx[ca_te], n_ca),
            eoa_train=mask_from(eoa_idx[eoa_tr], n_eoa), eoa_val=mask_from(eoa_idx[eoa_va], n_eoa),
            eoa_test=mask_from(eoa_idx[eoa_te], n_eoa),
        )
        ts = time.time()
        res = train_fold(masks, ctx, args)
        fold_results.append(res)
        print(f"[fold {k}] Ponzi F1={res['ponzi']['f1']:.4f} "
              f"(P={res['ponzi']['precision']:.4f} R={res['ponzi']['recall']:.4f}) | "
              f"Phish F1={res['phish']['f1']:.4f} "
              f"(P={res['phish']['precision']:.4f} R={res['phish']['recall']:.4f}) "
              f"| test n: CA={int(masks['ca_test'].sum())} EOA={int(masks['eoa_test'].sum())} "
              f"({time.time()-ts:.0f}s)", flush=True)
        with open(f"{args.out}.json", 'w') as f:
            json.dump({'args': vars(args), 'folds': fold_results}, f, indent=2)

    pf = [r['ponzi']['f1'] for r in fold_results]
    hf = [r['phish']['f1'] for r in fold_results]
    print("\n==================== 5-FOLD CV — FULL MODEL ====================", flush=True)
    print(f"Ponzi macro-F1: {np.mean(pf):.4f} ± {np.std(pf):.4f}  (folds: {[round(x,4) for x in pf]})")
    print(f"Phish macro-F1: {np.mean(hf):.4f} ± {np.std(hf):.4f}  (folds: {[round(x,4) for x in hf]})")
    print(f"total {time.time()-t0:.0f}s")

    md = (f"## 5-fold CV — Full model (macro-F1, mean±std)\n\n"
          f"| Task | macro-F1 | Precision | Recall |\n|---|---|---|---|\n")
    for tag, key in [("Ponzi", 'ponzi'), ("Phish", 'phish')]:
        f1s = [r[key]['f1'] for r in fold_results]
        ps = [r[key]['precision'] for r in fold_results]
        rs = [r[key]['recall'] for r in fold_results]
        md += (f"| {tag} | {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | "
               f"{np.mean(ps):.4f} ± {np.std(ps):.4f} | {np.mean(rs):.4f} ± {np.std(rs):.4f} |\n")
    md += f"\nPer-fold Ponzi F1: {[round(x,4) for x in pf]}\n"
    open(f"{args.out}.md", 'w', encoding='utf-8').write(md)
    print(f"Saved {args.out}.md / {args.out}.json")


if __name__ == '__main__':
    main()
