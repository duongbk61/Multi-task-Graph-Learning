"""
(A9) INDUCTIVE split for UnifiedHMSL (standard config: rule-free + signed-log fix).

Training: build a train-subgraph with the TEST nodes and ALL edges incident to them
REMOVED, so the encoder never sees test nodes (they cannot be sampled as neighbours).
Testing: use the ORIGINAL full graph and forward the test nodes once (their edges present).

Reports INDUCTIVE Macro-F1 + PR-AUC for both tasks. The TRANSDUCTIVE row is reused from
the A1 Full run (A_full_row.json) if present; otherwise pass --with_transductive to compute it.

Usage:
  python run_inductive.py --seeds 14 21 42 77 100 --gpu 0
"""
import json, argparse, warnings, gc, copy
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from dataset import Ponzi, Phish
from unified_model import UnifiedHMSL
from run_unified import get_augmented_data
from run_cv_table import eval_full
from utils import seed_everything

warnings.filterwarnings("ignore")


def inductive_graph(data, target_ntype, test_mask):
    """Return a NEW HeteroData with all edges incident to test nodes (of target_ntype) removed.
    Node features/masks are shared (not modified); only edge stores are rebuilt."""
    test_ids = test_mask.nonzero(as_tuple=True)[0]
    d = HeteroData()
    for nt in data.node_types:
        d[nt].x = data[nt].x
        for attr in ['y', 'train_mask', 'val_mask', 'test_mask', 'num_nodes']:
            if attr in data[nt]:
                d[nt][attr] = data[nt][attr]
    for et in data.edge_types:
        src_t, _, dst_t = et
        ei = data[et].edge_index
        keep = torch.ones(ei.size(1), dtype=torch.bool)
        if src_t == target_ntype:
            keep &= ~torch.isin(ei[0], test_ids)
        if dst_t == target_ntype:
            keep &= ~torch.isin(ei[1], test_ids)
        d[et].edge_index = ei[:, keep]
        if 'edge_attr' in data[et]:
            d[et].edge_attr = data[et].edge_attr[keep]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[14, 21, 42, 77, 100])
    ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64); ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001); ap.add_argument('--loss_train', type=float, default=0.1)
    ap.add_argument('--concat', type=int, default=3); ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--num_neighbors', type=int, nargs='+', default=[100, 100])
    ap.add_argument('--out', type=str, default='inductive_results')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    aug_args = argparse.Namespace(concat=args.concat)

    dp = Ponzi('./data/Ponzi/')[0]; dh = Phish('./data/Phish/')[0]
    aug_p = torch.load('./pretrain_model/icvae_Ponzi.pkl', map_location=device)
    aug_h = torch.load('./pretrain_model/icvae_Phish.pkl', map_location=device)

    print("building inductive train-subgraphs (removing test nodes + their edges)...", flush=True)
    dp_ind = inductive_graph(dp, 'CA', dp['CA'].test_mask)
    dh_ind = inductive_graph(dh, 'EOA', dh['EOA'].test_mask)
    for tag, orig, ind, et in [('Ponzi', dp, dp_ind, ('EOA', 'trans', 'CA')),
                               ('Phish', dh, dh_ind, ('EOA', 'trans', 'EOA'))]:
        print(f"  {tag}: edges {et} {orig[et].edge_index.size(1)} -> {ind[et].edge_index.size(1)}", flush=True)

    kw = dict(batch_size=args.batch_size, num_neighbors=args.num_neighbors)
    # TRAIN/VAL loaders on the inductive (edge-removed) graphs; TEST loaders on the FULL graphs.
    tr_p = NeighborLoader(dp_ind, shuffle=True,  input_nodes=('CA',  dp_ind['CA'].train_mask), **kw)
    va_p = NeighborLoader(dp_ind, shuffle=False, input_nodes=('CA',  dp_ind['CA'].val_mask),   **kw)
    te_p = NeighborLoader(dp,     shuffle=False, input_nodes=('CA',  dp['CA'].test_mask),       **kw)
    tr_h = NeighborLoader(dh_ind, shuffle=True,  input_nodes=('EOA', dh_ind['EOA'].train_mask), **kw)
    va_h = NeighborLoader(dh_ind, shuffle=False, input_nodes=('EOA', dh_ind['EOA'].val_mask),   **kw)
    te_h = NeighborLoader(dh,     shuffle=False, input_nodes=('EOA', dh['EOA'].test_mask),       **kw)

    runs = []
    for rs in args.seeds:
        seed_everything(rs)
        model = UnifiedHMSL(hidden=args.hidden, out_channels=2, data=dp, concat=args.concat,
                            expert_mode='none', raw_transform='signed_log',
                            use_cpa=True, use_tcg=True).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        best_val, best = -1.0, None; patience = 0
        for epoch in range(args.epochs):
            model.train()
            for bp, bh in zip(tr_p, tr_h):
                opt.zero_grad()
                bp = get_augmented_data(bp.to(device), aug_p, aug_args, device)
                op, _, co_p, _, _ = model(bp.x_dict_new, bp.edge_index_dict, raw_x_dict=bp.x_dict); sp = bp['CA'].batch_size
                bh = get_augmented_data(bh.to(device), aug_h, aug_args, device)
                _, oh, co_h, _, _ = model(bh.x_dict_new, bh.edge_index_dict, raw_x_dict=bh.x_dict); sh = bh['EOA'].batch_size
                loss = F.cross_entropy(op[:sp], bp['CA'].y[:sp]) + F.cross_entropy(oh[:sh], bh['EOA'].y[:sh]) + args.loss_train * (co_p + co_h)
                loss.backward(); opt.step()
            vp = eval_full(model, va_p, aug_p, 'CA', aug_args, device)['macro_f1']
            vh = eval_full(model, va_h, aug_h, 'EOA', aug_args, device)['macro_f1']
            if vp + vh > best_val:
                best_val = vp + vh
                best = dict(ponzi=eval_full(model, te_p, aug_p, 'CA', aug_args, device),
                            phish=eval_full(model, te_h, aug_h, 'EOA', aug_args, device)); patience = 0
            else:
                patience += 1
                if patience >= args.patience: break
        runs.append(best)
        print(f"[inductive seed {rs}] Ponzi Mac={best['ponzi']['macro_f1']*100:.2f} PR-AUC={best['ponzi']['pr_auc']*100:.2f} "
              f"| Phish Mac={best['phish']['macro_f1']*100:.2f} PR-AUC={best['phish']['pr_auc']*100:.2f}", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
        json.dump({'args': vars(args), 'runs': runs}, open(f"{args.out}.json", 'w'), indent=2)

    def ms(task, key):
        v = np.array([r[task][key] for r in runs]) * 100; return v.mean(), v.std()
    print("\n===== A9 INDUCTIVE (mean±std ×100, n=%d) =====" % len(args.seeds))
    for task, tag in [('ponzi', 'Ponzi'), ('phish', 'Phish')]:
        mm, ms_ = ms(task, 'macro_f1'); pm, ps = ms(task, 'pr_auc')
        print(f"  {tag}: Macro-F1 {mm:.2f}±{ms_:.2f} | PR-AUC {pm:.2f}±{ps:.2f}")


if __name__ == '__main__':
    main()
