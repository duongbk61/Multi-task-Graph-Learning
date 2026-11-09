"""Gate statistics for CPA/TCG on the test nodes. Writes only to the scratchpad."""
import sys, os, json, copy, argparse, warnings, gc
sys.path.insert(0, '.')
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
OUT = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEEDS = [int(s) for s in sys.argv[1:]] or [14, 21, 42]

dp = Ponzi('./data/Ponzi/')[0]
dh = Phish('./data/Phish/')[0]
aug_p = torch.load('./pretrain_model/icvae_Ponzi.pkl', map_location=device)
aug_h = torch.load('./pretrain_model/icvae_Phish.pkl', map_location=device)
aug_args = argparse.Namespace(concat=3)
ns_p = NeighborSampler(dp, num_neighbors=[100, 100])
ns_h = NeighborSampler(dh, num_neighbors=[100, 100])
bs = dict(batch_size=64, num_neighbors=[100, 100])
tr_p = NeighborLoader(dp, shuffle=True, input_nodes=('CA', dp['CA'].train_mask), neighbor_sampler=ns_p, **bs)
va_p = NeighborLoader(dp, shuffle=False, input_nodes=('CA', dp['CA'].val_mask), neighbor_sampler=ns_p, **bs)
te_p = NeighborLoader(dp, shuffle=False, input_nodes=('CA', dp['CA'].test_mask), neighbor_sampler=ns_p, **bs)
tr_h = NeighborLoader(dh, shuffle=True, input_nodes=('EOA', dh['EOA'].train_mask), neighbor_sampler=ns_h, **bs)
va_h = NeighborLoader(dh, shuffle=False, input_nodes=('EOA', dh['EOA'].val_mask), neighbor_sampler=ns_h, **bs)
te_h = NeighborLoader(dh, shuffle=False, input_nodes=('EOA', dh['EOA'].test_mask), neighbor_sampler=ns_h, **bs)


@torch.no_grad()
def gates(model, loader, aug, nt):
    """Recompute the alphas from each gate's real inputs, keeping the seed (test) nodes only."""
    model.eval()
    cur = {'bs': 0}
    rec = {'cpa_gc': [], 'cpa_cg': [], 'tcg': [], 'y': []}

    def pre_cpa(m, inp):
        h_gen, h_cont = inp
        s = cur['bs']
        rec['cpa_gc'].append(torch.sigmoid((m.q_gen(h_gen) * m.k_cont(h_cont)).sum(-1) / m.scale)[:s].cpu())
        rec['cpa_cg'].append(torch.sigmoid((m.q_cont(h_cont) * m.k_gen(h_gen)).sum(-1) / m.scale)[:s].cpu())

    def pre_tcg(m, inp):
        s = cur['bs']
        rec['tcg'].append(torch.sigmoid(m.fc2(m.dropout(F.relu(m.fc1(inp[0]))))).squeeze(-1)[:s].cpu())

    hs = [model.cross_attn[nt].register_forward_pre_hook(pre_cpa),
          model.task_gate[nt].register_forward_pre_hook(pre_tcg)]
    for b in loader:
        b = get_augmented_data(b.to(device), aug, aug_args, device)
        cur['bs'] = b[nt].batch_size
        model(b.x_dict_new, b.edge_index_dict, raw_x_dict=b.x_dict)
        rec['y'].append(b[nt].y[:cur['bs']].cpu())
    for h in hs:
        h.remove()
    return {k: torch.cat(v).numpy() for k, v in rec.items()}


def report(tag, r):
    y = r.pop('y')
    print(f'  {tag}: n_test={len(y)} fraud={int((y == 1).sum())} normal={int((y == 0).sum())}')
    out = {}
    for name, a in r.items():
        row = dict(mean=a.mean(), std=a.std(), min=a.min(), max=a.max(),
                   p05=np.percentile(a, 5), p95=np.percentile(a, 95),
                   frac_lt_01=(a < 0.01).mean(), frac_gt_99=(a > 0.99).mean(),
                   mean_fraud=a[y == 1].mean(), mean_normal=a[y == 0].mean())
        out[name] = {k: float(v) for k, v in row.items()}
        print('    {:7s} mean={mean:.4f} std={std:.4f} min={min:.4f} max={max:.4f} p05={p05:.4f} p95={p95:.4f} '
              '<.01={frac_lt_01:.3f} >.99={frac_gt_99:.3f} fraud={mean_fraud:.4f} normal={mean_normal:.4f}'.format(name, **row),
              flush=True)
    return out


def both(model, tag):
    print(f'\n=== {tag} ===', flush=True)
    res = {}
    res['CA'] = report('CA (Ponzi test)', gates(model, te_p, aug_p, 'CA'))
    res['EOA'] = report('EOA (Phish test)', gates(model, te_h, aug_h, 'EOA'))
    return res


for rs in SEEDS:
    seed_everything(rs)
    model = UnifiedHMSL(hidden=128, out_channels=2, data=dp, concat=3, expert_mode='none',
                        use_cpa=True, use_tcg=True, raw_transform='signed_log').to(device)
    b0 = get_augmented_data(next(iter(te_p)).to(device), aug_p, aug_args, device)
    model.eval()
    with torch.no_grad():
        model(b0.x_dict_new, b0.edge_index_dict, raw_x_dict=b0.x_dict)
    both(model, f'UNTRAINED init, seed {rs}')
    del model; gc.collect(); torch.cuda.empty_cache()
