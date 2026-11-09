"""
Self-trained heterogeneous GNN baselines: RGCN and HGT.
Same fixed split (SEED=14), same 14-dim node features (with signed-log, fair to our model),
same protocol (NeighborLoader [100,100], batch 128, N restarts). Per-task (Ponzi=CA, Phish=EOA).
Reports fraud-class P/R/Binary-F1, Micro-F1(=acc), Macro-F1, mean±std.

RGCN = HeteroConv with one SAGEConv per relation (relation-specific weights) — the relational-GNN
       mechanism. HGT = HGTConv (heterogeneous graph transformer).
"""
import json, argparse, warnings, gc
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler
from torch_geometric.nn import HeteroConv, SAGEConv, HGTConv, Linear
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from dataset import Ponzi, Phish
from utils import seed_everything

warnings.filterwarnings("ignore")


def signed_log(x):
    return torch.sign(x) * torch.log1p(x.abs())


class RGCN(torch.nn.Module):
    def __init__(self, metadata, hidden=128, out=2):
        super().__init__()
        self.lin = torch.nn.ModuleDict({nt: Linear(-1, hidden) for nt in metadata[0]})
        self.conv1 = HeteroConv({et: SAGEConv((-1, -1), hidden) for et in metadata[1]}, aggr='sum')
        self.conv2 = HeteroConv({et: SAGEConv((-1, -1), hidden) for et in metadata[1]}, aggr='sum')
        self.head = torch.nn.ModuleDict({nt: Linear(hidden, out) for nt in metadata[0]})

    def forward(self, x_dict, ei_dict, target):
        x = {nt: F.relu(self.lin[nt](signed_log(x))) for nt, x in x_dict.items()}
        x = {k: F.relu(v) for k, v in self.conv1(x, ei_dict).items()}
        x = self.conv2(x, ei_dict)
        return self.head[target](x[target])


class HGT(torch.nn.Module):
    def __init__(self, metadata, hidden=128, out=2, heads=2):
        super().__init__()
        self.lin = torch.nn.ModuleDict({nt: Linear(-1, hidden) for nt in metadata[0]})
        self.conv1 = HGTConv(hidden, hidden, metadata, heads)
        self.conv2 = HGTConv(hidden, hidden, metadata, heads)
        self.head = torch.nn.ModuleDict({nt: Linear(hidden, out) for nt in metadata[0]})

    def forward(self, x_dict, ei_dict, target):
        x = {nt: F.relu(self.lin[nt](signed_log(x))) for nt, x in x_dict.items()}
        x = self.conv1(x, ei_dict)
        x = self.conv2(x, ei_dict)
        return self.head[target](x[target])


def metrics(y, p):
    return dict(precision=precision_score(y, p, pos_label=1, zero_division=0) * 100,
                recall=recall_score(y, p, pos_label=1, zero_division=0) * 100,
                binary_f1=f1_score(y, p, pos_label=1, zero_division=0) * 100,
                micro_f1=accuracy_score(y, p) * 100,
                macro_f1=f1_score(y, p, average='macro') * 100)


@torch.no_grad()
def evaluate(model, loader, nt, device):
    model.eval(); preds, labels = [], []
    for b in loader:
        b = b.to(device); bs = b[nt].batch_size
        out = model(b.x_dict, b.edge_index_dict, nt)
        preds.extend(out[:bs].argmax(1).cpu().numpy()); labels.extend(b[nt].y[:bs].cpu().numpy())
    return metrics(np.array(labels), np.array(preds))


def train_model(model_name, data, nt, ns, args, device):
    kw = dict(batch_size=args.batch_size, num_neighbors=args.num_neighbors)
    tr = NeighborLoader(data, shuffle=True,  input_nodes=(nt, data[nt].train_mask), neighbor_sampler=ns, **kw)
    va = NeighborLoader(data, shuffle=False, input_nodes=(nt, data[nt].val_mask),   neighbor_sampler=ns, **kw)
    te = NeighborLoader(data, shuffle=False, input_nodes=(nt, data[nt].test_mask),  neighbor_sampler=ns, **kw)
    runs = []
    for rs in args.seeds:
        seed_everything(rs)
        model = (RGCN if model_name == 'RGCN' else HGT)(data.metadata(), hidden=args.hidden).to(device)
        # lazy init: one forward to materialize
        for b in tr:
            model(b.to(device).x_dict, b.edge_index_dict, nt); break
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        best_val, best = -1.0, None; patience = 0
        for epoch in range(args.epochs):
            model.train()
            for b in tr:
                b = b.to(device); opt.zero_grad(); s = b[nt].batch_size
                out = model(b.x_dict, b.edge_index_dict, nt)
                loss = F.cross_entropy(out[:s], b[nt].y[:s]); loss.backward(); opt.step()
            vm = evaluate(model, va, nt, device)['macro_f1']
            if vm > best_val:
                best_val = vm; best = evaluate(model, te, nt, device); patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    break
        runs.append(best)
        print(f"  [{model_name} {nt} seed {rs}] Macro={best['macro_f1']:.2f} BinF1={best['binary_f1']:.2f}", flush=True)
        del model, opt; gc.collect(); torch.cuda.empty_cache()
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[14, 21, 42, 77, 100, 7, 11, 33, 55, 88])
    ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=128); ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001); ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--num_neighbors', type=int, nargs='+', default=[100, 100])
    ap.add_argument('--models', type=str, nargs='+', default=['RGCN', 'HGT'])
    ap.add_argument('--out', type=str, default='hetero_baselines')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    dp = Ponzi('./data/Ponzi/')[0]; dh = Phish('./data/Phish/')[0]
    ns_p = NeighborSampler(dp, num_neighbors=args.num_neighbors); ns_h = NeighborSampler(dh, num_neighbors=args.num_neighbors)
    out = {}
    for model_name in args.models:
        for tag, data, nt, ns in [('Ponzi', dp, 'CA', ns_p), ('Phish', dh, 'EOA', ns_h)]:
            print(f"=== {model_name} on {tag} ===", flush=True)
            runs = train_model(model_name, data, nt, ns, args, device)
            agg = {k: (np.mean([r[k] for r in runs]), np.std([r[k] for r in runs])) for k in runs[0]}
            out[f'{model_name}|{tag}'] = {'agg': {k: list(v) for k, v in agg.items()}, 'runs': runs}
            c = lambda k: f"{agg[k][0]:.2f}±{agg[k][1]:.2f}"
            print(f"{model_name} |{tag:6s}| P {c('precision')} | R {c('recall')} | BinF1 {c('binary_f1')} | "
                  f"Mic {c('micro_f1')} | Mac {c('macro_f1')}", flush=True)
            json.dump({'args': vars(args), 'results': out}, open(f"{args.out}.json", 'w'), indent=2)
    print("HETERO_BASELINES_DONE", flush=True)


if __name__ == '__main__':
    main()
