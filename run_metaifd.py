"""
Self-trained Meta-IFD baseline (SINGLE-TASK, original form).
Represents the pre-UnifiedHMSL baseline: two SEPARATE single-task models (one Ponzi/CA,
one Phish/EOA), base architecture = ICVAE augmentation + triplet + expert rules, but
WITHOUT our additions (no CPA, no TCG) and WITHOUT the signed-log fix (raw features).
Same fixed split (SEED=14), same protocol (batch 64, 100x2 neighbours, 5 restarts).
Reports fraud-class P/R/Binary-F1, Micro-F1(=acc), Macro-F1, mean±std.
"""
import json, argparse, warnings, gc
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from dataset import Ponzi, Phish
from unified_model import UnifiedHMSL
from run_unified import get_augmented_data
from utils import seed_everything

warnings.filterwarnings("ignore")


@torch.no_grad()
def evaluate(model, loader, aug_model, nt, aug_args, device):
    model.eval(); preds, labels = [], []
    for b in loader:
        b = get_augmented_data(b.to(device), aug_model, aug_args, device)
        bs = b[nt].batch_size
        op, oh, *_ = model(b.x_dict_new, b.edge_index_dict, raw_x_dict=b.x_dict)
        out = op if nt == 'CA' else oh
        preds.extend(out[:bs].argmax(1).cpu().numpy()); labels.extend(b[nt].y[:bs].cpu().numpy())
    return dict(precision=precision_score(labels, preds, pos_label=1, zero_division=0) * 100,
                recall=recall_score(labels, preds, pos_label=1, zero_division=0) * 100,
                binary_f1=f1_score(labels, preds, pos_label=1, zero_division=0) * 100,
                micro_f1=accuracy_score(labels, preds) * 100,
                macro_f1=f1_score(labels, preds, average='macro') * 100)


def train_task(task, ctx, args):
    device = ctx['device']
    data = ctx['dp'] if task == 'ponzi' else ctx['dh']
    nt = 'CA' if task == 'ponzi' else 'EOA'
    aug = ctx['aug_p'] if task == 'ponzi' else ctx['aug_h']
    ns = ctx['ns_p'] if task == 'ponzi' else ctx['ns_h']
    aug_args = argparse.Namespace(concat=args.concat)
    kw = dict(batch_size=args.batch_size, num_neighbors=args.num_neighbors)
    tr = NeighborLoader(data, shuffle=True,  input_nodes=(nt, data[nt].train_mask), neighbor_sampler=ns, **kw)
    va = NeighborLoader(data, shuffle=False, input_nodes=(nt, data[nt].val_mask),   neighbor_sampler=ns, **kw)
    te = NeighborLoader(data, shuffle=False, input_nodes=(nt, data[nt].test_mask),  neighbor_sampler=ns, **kw)

    runs = []
    for rs in args.seeds:
        seed_everything(rs)
        model = UnifiedHMSL(hidden=args.hidden, out_channels=2, data=data, concat=args.concat,
                            expert_mode=args.expert_mode, raw_transform=args.raw_transform,
                            use_cpa=bool(args.use_cpa), use_tcg=bool(args.use_tcg)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        best_val, best = -1.0, None; patience = 0
        for epoch in range(args.epochs):
            model.train()
            for b in tr:
                opt.zero_grad()
                b = get_augmented_data(b.to(device), aug, aug_args, device)
                op, oh, co, _, _ = model(b.x_dict_new, b.edge_index_dict, raw_x_dict=b.x_dict)
                out = op if nt == 'CA' else oh
                s = b[nt].batch_size
                loss = F.cross_entropy(out[:s], b[nt].y[:s]) + args.loss_train * co
                loss.backward(); opt.step()
            vm = evaluate(model, va, aug, nt, aug_args, device)['macro_f1']
            if vm > best_val:
                best_val = vm; best = evaluate(model, te, aug, nt, aug_args, device); patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    break
        runs.append(best)
        print(f"  [{task} seed {rs}] Macro={best['macro_f1']:.2f} BinF1={best['binary_f1']:.2f}", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[14, 21, 42, 77, 100])
    ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64); ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001); ap.add_argument('--loss_train', type=float, default=0.1)
    ap.add_argument('--concat', type=int, default=3); ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--num_neighbors', type=int, nargs='+', default=[100, 100])
    ap.add_argument('--expert_mode', type=str, default='feature', choices=['feature', 'loss', 'none'])
    ap.add_argument('--raw_transform', type=str, default='none', choices=['none', 'signed_log'])
    ap.add_argument('--use_cpa', type=int, default=0)
    ap.add_argument('--use_tcg', type=int, default=0)
    ap.add_argument('--out', type=str, default='metaifd_baseline')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    dp = Ponzi('./data/Ponzi/')[0]; dh = Phish('./data/Phish/')[0]
    ctx = dict(device=device, dp=dp, dh=dh,
               aug_p=torch.load('./pretrain_model/icvae_Ponzi.pkl', map_location=device),
               aug_h=torch.load('./pretrain_model/icvae_Phish.pkl', map_location=device),
               ns_p=NeighborSampler(dp, num_neighbors=args.num_neighbors),
               ns_h=NeighborSampler(dh, num_neighbors=args.num_neighbors))
    out = {}
    for task in ['ponzi', 'phish']:
        print(f"=== Meta-IFD single-task: {task} ===", flush=True)
        runs = train_task(task, ctx, args)
        agg = {k: (np.mean([r[k] for r in runs]), np.std([r[k] for r in runs])) for k in runs[0]}
        out[task] = {'agg': {k: list(v) for k, v in agg.items()}, 'runs': runs}  # per-seed persisted
        c = lambda k: f"{agg[k][0]:.2f}±{agg[k][1]:.2f}"
        print(f"Meta-IFD |{task:6s}| P {c('precision')} | R {c('recall')} | BinF1 {c('binary_f1')} | "
              f"Mic {c('micro_f1')} | Mac {c('macro_f1')}", flush=True)
        json.dump({'args': vars(args), 'tasks': out}, open(f"{args.out}.json", 'w'), indent=2)
    print(f"saved {args.out}.json")


if __name__ == '__main__':
    main()
