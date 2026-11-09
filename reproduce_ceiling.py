"""
=====================================================================
  CEILING REPRODUCTION SCRIPT  (lucky-split upper envelope)
=====================================================================
Reproduces the single BEST split from the split-sensitivity study:
  split_seed = 42  ->  Ponzi macro-F1 ~= 0.871, Phish ~= 0.978

  *** READ THIS ***
  This number is the "luckiest-split ceiling". It is obtained by using the
  train/test partition that happens to give the highest Ponzi F1. It is a
  DIAGNOSTIC upper bound only — it tells you how high the current architecture
  can score on a favourable split. It MUST NOT be reported in the thesis as
  "the model achieves 0.871", because the split was chosen using the test set
  (data leakage). The reportable headline is the 5-fold CV number from
  run_cv.py:  Ponzi 0.819 +/- 0.012,  Phish 0.964 +/- 0.007.

Run:
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  python reproduce_ceiling.py --split_seed 42 --train_seed 42 --gpu 0
"""
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler
from dataset import Ponzi, Phish
from unified_model import UnifiedHMSL
from run_unified import get_augmented_data
from run_cv import labeled_indices, mask_from, eval_task
from utils import seed_everything

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split_seed', type=int, default=42, help='data partition (42 = ceiling)')
    ap.add_argument('--train_seed', type=int, default=42, help='weight init + ICVAE aug')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--loss_train', type=float, default=0.1)
    ap.add_argument('--concat', type=int, default=3)
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--num_neighbors', type=int, nargs='+', default=[100, 100])
    ap.add_argument('--ckpt', type=str, default='ceiling_split42_model.pth')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print("  CEILING REPRODUCTION  (diagnostic upper bound — NOT reportable)")
    print(f"  split_seed={args.split_seed}  train_seed={args.train_seed}  "
          f"epochs={args.epochs} patience={args.patience}")
    print("=" * 65, flush=True)

    dp = Ponzi('./data/Ponzi/')[0]
    dh = Phish('./data/Phish/')[0]
    aug_p = torch.load('./pretrain_model/icvae_Ponzi.pkl', map_location=device)
    aug_h = torch.load('./pretrain_model/icvae_Phish.pkl', map_location=device)
    ns_p = NeighborSampler(dp, num_neighbors=args.num_neighbors)
    ns_h = NeighborSampler(dh, num_neighbors=args.num_neighbors)

    ca_idx, ca_y = labeled_indices(dp, 'CA')
    eoa_idx, eoa_y = labeled_indices(dh, 'EOA')
    n_ca, n_eoa = dp['CA'].y.size(0), dh['EOA'].y.size(0)

    # ---- rebuild the exact split for split_seed (same recipe as the study) ----
    s = args.split_seed
    ca_trp, ca_te = train_test_split(np.arange(len(ca_idx)), test_size=0.2, random_state=s, stratify=ca_y)
    ca_tr, ca_va = train_test_split(ca_trp, test_size=0.25, random_state=s, stratify=ca_y[ca_trp])
    eoa_trp, eoa_te = train_test_split(np.arange(len(eoa_idx)), test_size=0.2, random_state=s, stratify=eoa_y)
    eoa_tr, eoa_va = train_test_split(eoa_trp, test_size=0.25, random_state=s, stratify=eoa_y[eoa_trp])

    bs = dict(batch_size=args.batch_size, num_neighbors=args.num_neighbors)
    tr_p = NeighborLoader(dp, shuffle=True,  input_nodes=('CA',  mask_from(ca_idx[ca_tr], n_ca)),  neighbor_sampler=ns_p, **bs)
    va_p = NeighborLoader(dp, shuffle=False, input_nodes=('CA',  mask_from(ca_idx[ca_va], n_ca)),  neighbor_sampler=ns_p, **bs)
    te_p = NeighborLoader(dp, shuffle=False, input_nodes=('CA',  mask_from(ca_idx[ca_te], n_ca)),  neighbor_sampler=ns_p, **bs)
    tr_h = NeighborLoader(dh, shuffle=True,  input_nodes=('EOA', mask_from(eoa_idx[eoa_tr], n_eoa)), neighbor_sampler=ns_h, **bs)
    va_h = NeighborLoader(dh, shuffle=False, input_nodes=('EOA', mask_from(eoa_idx[eoa_va], n_eoa)), neighbor_sampler=ns_h, **bs)
    te_h = NeighborLoader(dh, shuffle=False, input_nodes=('EOA', mask_from(eoa_idx[eoa_te], n_eoa)), neighbor_sampler=ns_h, **bs)
    print(f"test sizes: CA={len(ca_te)}  EOA={len(eoa_te)}", flush=True)

    seed_everything(args.train_seed)
    aug_args = argparse.Namespace(concat=args.concat)
    model = UnifiedHMSL(hidden=args.hidden, out_channels=2, data=dp, concat=args.concat,
                        expert_mode='feature', use_cpa=True, use_tcg=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_val = -1.0
    best = None
    best_state = None
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
            loss = (F.cross_entropy(op[:sp], bp['CA'].y[:sp])
                    + F.cross_entropy(oh[:sh], bh['EOA'].y[:sh])
                    + args.loss_train * (co_p + co_h))
            loss.backward()
            opt.step()

        vp = eval_task(model, va_p, aug_p, 'CA', aug_args, device)['f1']
        vh = eval_task(model, va_h, aug_h, 'EOA', aug_args, device)['f1']
        tp = eval_task(model, te_p, aug_p, 'CA', aug_args, device)
        th = eval_task(model, te_h, aug_h, 'EOA', aug_args, device)
        flag = ""
        if vp + vh > best_val:
            best_val = vp + vh
            best = dict(ponzi=tp, phish=th, epoch=epoch)
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = 0
            flag = "  <-- best-val (test frozen here)"
        else:
            patience += 1
        print(f"Epoch {epoch:03d} | valP={vp:.4f} valH={vh:.4f} || "
              f"testPonzi={tp['f1']:.4f} testPhish={th['f1']:.4f}{flag}", flush=True)
        if patience >= args.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    torch.save(best_state, args.ckpt)
    print("\n" + "=" * 65)
    print(f"  CEILING RESULT (best-val epoch {best['epoch']}):")
    print(f"    Ponzi macro-F1 = {best['ponzi']['f1']:.4f} "
          f"(P={best['ponzi']['precision']:.4f} R={best['ponzi']['recall']:.4f})")
    print(f"    Phish macro-F1 = {best['phish']['f1']:.4f} "
          f"(P={best['phish']['precision']:.4f} R={best['phish']['recall']:.4f})")
    print(f"  checkpoint -> {args.ckpt}")
    print("  REMINDER: diagnostic upper bound, NOT a reportable metric.")
    print("  Report the 5-fold CV: Ponzi 0.819+/-0.012, Phish 0.964+/-0.007.")
    print("=" * 65, flush=True)


if __name__ == '__main__':
    main()
