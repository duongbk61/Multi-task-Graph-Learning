"""
Split-sensitivity study for the FULL model.

Trains the model on MANY different stratified train/val/test partitions (varying
the SPLIT seed) while holding the training seed fixed, so the only thing that
changes is the data partition. Reports the full DISTRIBUTION of test macro-F1
(min / p25 / median / mean / max).

IMPORTANT — how to read this:
  * The DISTRIBUTION (and its mean±std) characterises how sensitive the model is
    to the train/test partition. That is legitimate to report.
  * The MAX is the "luckiest-split ceiling" (upper envelope). It is NOT a valid
    performance metric and MUST NOT be reported as "the model achieves X" — that
    would be selecting on the test set (data leakage). Use 5-fold CV (run_cv.py)
    for the headline number.

Usage:
  python run_split_variance.py --split_seeds 0 1 7 11 14 21 33 42 55 77 88 99 100 123 777 2024 --train_seed 42 --gpu 0
"""
import json
import time
import argparse
import warnings
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.sampler import NeighborSampler
from dataset import Ponzi, Phish
from run_cv import labeled_indices, mask_from, train_fold

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split_seeds', type=int, nargs='+',
                    default=[0, 1, 7, 11, 14, 21, 33, 42, 55, 77, 88, 99, 100, 123, 777, 2024])
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
    ap.add_argument('--out', type=str, default='split_variance')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | {len(args.split_seeds)} splits | train_seed={args.train_seed} "
          f"(FIXED) epochs={args.epochs} patience={args.patience}", flush=True)

    dp = Ponzi('./data/Ponzi/')[0]
    dh = Phish('./data/Phish/')[0]
    aug_p = torch.load('./pretrain_model/icvae_Ponzi.pkl', map_location=device)
    aug_h = torch.load('./pretrain_model/icvae_Phish.pkl', map_location=device)
    ctx = dict(device=device, data_ponzi=dp, data_phish=dh, aug_p=aug_p, aug_h=aug_h,
               ns_p=NeighborSampler(dp, num_neighbors=args.num_neighbors),
               ns_h=NeighborSampler(dh, num_neighbors=args.num_neighbors))

    ca_idx, ca_y = labeled_indices(dp, 'CA')
    eoa_idx, eoa_y = labeled_indices(dh, 'EOA')
    n_ca, n_eoa = dp['CA'].y.size(0), dh['EOA'].y.size(0)

    results = {}
    t0 = time.time()
    for s in args.split_seeds:
        # stratified test(0.2) then val(0.25 of train) — same proportions as dataset.py
        ca_tr_pool, ca_te = train_test_split(np.arange(len(ca_idx)), test_size=0.2,
                                             random_state=s, stratify=ca_y)
        ca_tr, ca_va = train_test_split(ca_tr_pool, test_size=0.25,
                                        random_state=s, stratify=ca_y[ca_tr_pool])
        eoa_tr_pool, eoa_te = train_test_split(np.arange(len(eoa_idx)), test_size=0.2,
                                               random_state=s, stratify=eoa_y)
        eoa_tr, eoa_va = train_test_split(eoa_tr_pool, test_size=0.25,
                                          random_state=s, stratify=eoa_y[eoa_tr_pool])
        masks = dict(
            ca_train=mask_from(ca_idx[ca_tr], n_ca), ca_val=mask_from(ca_idx[ca_va], n_ca),
            ca_test=mask_from(ca_idx[ca_te], n_ca),
            eoa_train=mask_from(eoa_idx[eoa_tr], n_eoa), eoa_val=mask_from(eoa_idx[eoa_va], n_eoa),
            eoa_test=mask_from(eoa_idx[eoa_te], n_eoa),
        )
        ts = time.time()
        res = train_fold(masks, ctx, args)   # uses fixed args.train_seed
        pf, hf = res['ponzi']['f1'], res['phish']['f1']
        results[s] = dict(ponzi_f1=pf, phish_f1=hf)
        print(f"[split {s:>4}] Ponzi={pf:.4f}  Phish={hf:.4f}  ({time.time()-ts:.0f}s)", flush=True)
        with open(f"{args.out}.json", 'w') as f:
            json.dump({'args': vars(args), 'results': {str(k): v for k, v in results.items()}}, f, indent=2)

    pf = np.array([r['ponzi_f1'] for r in results.values()])
    hf = np.array([r['phish_f1'] for r in results.values()])
    best_split = max(results, key=lambda k: results[k]['ponzi_f1'])

    def stats(a):
        return (f"min={a.min():.4f}  p25={np.percentile(a,25):.4f}  median={np.median(a):.4f}  "
                f"mean={a.mean():.4f}±{a.std():.4f}  MAX={a.max():.4f}")

    print("\n============ SPLIT-SENSITIVITY (Full model) ============", flush=True)
    print(f"Ponzi F1: {stats(pf)}")
    print(f"Phish F1: {stats(hf)}")
    print(f"Luckiest split for Ponzi: seed {best_split} -> {results[best_split]['ponzi_f1']:.4f} "
          f"(upper envelope, NOT reportable as performance)")
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
