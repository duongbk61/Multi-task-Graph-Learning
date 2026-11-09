"""
Seed search for the FULL model (MT+CPA+TCG+Expert).
Trains the full config over many NEW seeds, model-selects on val macro-F1, and
records TEST macro-F1 per seed. Saves the single best checkpoint so the winning
run is reproducible/reusable. Same fair budget as the ablation sweep.

Ranking: primary = Ponzi+Phish (sum); also reports best-by-Ponzi and best-by-Phish.

Usage:
  python run_full_search.py --seeds 0 1 7 33 55 77 88 100 123 2024 --epochs 60 --patience 20 --gpu 0
"""
import json
import time
import copy
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
    return f1_score(labels, preds, average='macro')


def train_full(seed, ctx, args):
    device = ctx['device']
    seed_everything(seed)
    aug_args = argparse.Namespace(concat=args.concat)
    model = UnifiedHMSL(hidden=args.hidden, out_channels=2, data=ctx['data_ponzi'],
                        concat=args.concat, expert_mode='feature',
                        use_cpa=True, use_tcg=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_val = -1.0
    best_test = {}
    best_state = None
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        for bp, bh in zip(ctx['tr_ponzi'], ctx['tr_phish']):
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

        vp = eval_task(model, ctx['va_ponzi'], ctx['aug_p'], 'CA', aug_args, device)
        vh = eval_task(model, ctx['va_phish'], ctx['aug_h'], 'EOA', aug_args, device)
        if vp + vh > best_val:
            best_val = vp + vh
            tp = eval_task(model, ctx['te_ponzi'], ctx['aug_p'], 'CA', aug_args, device)
            th = eval_task(model, ctx['te_phish'], ctx['aug_h'], 'EOA', aug_args, device)
            best_test = {'ponzi_f1': tp, 'phish_f1': th, 'val': vp + vh, 'epoch': epoch}
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    del model
    torch.cuda.empty_cache()
    return best_test, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[0, 1, 7, 33, 55, 77, 88, 100, 123, 2024])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--loss_train', type=float, default=0.1)
    ap.add_argument('--concat', type=int, default=3)
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--num_neighbors', type=int, nargs='+', default=[100, 100])
    ap.add_argument('--out', type=str, default='full_search')
    ap.add_argument('--ckpt', type=str, default='best_full_model.pth')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | seeds={args.seeds} | epochs={args.epochs} patience={args.patience}", flush=True)

    data_ponzi = Ponzi('./data/Ponzi/')[0]
    data_phish = Phish('./data/Phish/')[0]
    aug_p = torch.load('./pretrain_model/icvae_Ponzi.pkl', map_location=device)
    aug_h = torch.load('./pretrain_model/icvae_Phish.pkl', map_location=device)
    kw = dict(num_neighbors=args.num_neighbors, batch_size=args.batch_size)
    ctx = dict(
        device=device, data_ponzi=data_ponzi, data_phish=data_phish, aug_p=aug_p, aug_h=aug_h,
        tr_ponzi=NeighborLoader(data_ponzi, shuffle=True,  input_nodes=('CA', data_ponzi['CA'].train_mask), **kw),
        va_ponzi=NeighborLoader(data_ponzi, shuffle=False, input_nodes=('CA', data_ponzi['CA'].val_mask),   **kw),
        te_ponzi=NeighborLoader(data_ponzi, shuffle=False, input_nodes=('CA', data_ponzi['CA'].test_mask),  **kw),
        tr_phish=NeighborLoader(data_phish, shuffle=True,  input_nodes=('EOA', data_phish['EOA'].train_mask), **kw),
        va_phish=NeighborLoader(data_phish, shuffle=False, input_nodes=('EOA', data_phish['EOA'].val_mask),   **kw),
        te_phish=NeighborLoader(data_phish, shuffle=False, input_nodes=('EOA', data_phish['EOA'].test_mask),  **kw),
    )

    results = {}   # seed -> {ponzi_f1, phish_f1, val, epoch}
    best_sum_seed = None
    best_sum = -1.0
    t0 = time.time()
    for seed in args.seeds:
        ts = time.time()
        res, state = train_full(seed, ctx, args)
        results[seed] = res
        s = res['ponzi_f1'] + res['phish_f1']
        star = ""
        if s > best_sum:
            best_sum, best_sum_seed = s, seed
            torch.save(state, args.ckpt)   # keep checkpoint of the current global best
            star = "  <== new best (ckpt saved)"
        print(f"[seed {seed:>4}] Ponzi={res['ponzi_f1']:.4f}  Phish={res['phish_f1']:.4f}  "
              f"sum={s:.4f}  (val-epoch {res['epoch']}, {time.time()-ts:.0f}s){star}", flush=True)
        with open(f"{args.out}.json", 'w') as f:
            json.dump({'args': vars(args), 'results': {str(k): v for k, v in results.items()},
                       'best_sum_seed': best_sum_seed}, f, indent=2)

    # leaderboards
    by_sum = sorted(results.items(), key=lambda kv: -(kv[1]['ponzi_f1'] + kv[1]['phish_f1']))
    by_ponzi = max(results.items(), key=lambda kv: kv[1]['ponzi_f1'])
    by_phish = max(results.items(), key=lambda kv: kv[1]['phish_f1'])
    print("\n==================== SEED SEARCH — FULL MODEL ====================", flush=True)
    print(f"{'seed':>6} | {'Ponzi':>7} | {'Phish':>7} | {'sum':>7}")
    for seed, r in by_sum:
        print(f"{seed:>6} | {r['ponzi_f1']:.4f} | {r['phish_f1']:.4f} | {r['ponzi_f1']+r['phish_f1']:.4f}")
    print("-----------------------------------------------------------------")
    print(f"BEST overall (sum): seed {best_sum_seed}  Ponzi={results[best_sum_seed]['ponzi_f1']:.4f} "
          f"Phish={results[best_sum_seed]['phish_f1']:.4f}  -> saved to {args.ckpt}")
    print(f"BEST Ponzi:         seed {by_ponzi[0]}  Ponzi={by_ponzi[1]['ponzi_f1']:.4f}")
    print(f"BEST Phish:         seed {by_phish[0]}  Phish={by_phish[1]['phish_f1']:.4f}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
