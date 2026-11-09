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


# ---- 1. existing checkpoint: rule-mode, no signed-log (NOT the reported config) ----
all_res = {}
seed_everything(14)
old = UnifiedHMSL(hidden=128, out_channels=2, data=dp, concat=3, expert_mode='feature').to(device)
b0 = get_augmented_data(next(iter(te_p)).to(device), aug_p, aug_args, device)
with torch.no_grad():
    old(b0.x_dict_new, b0.edge_index_dict, raw_x_dict=b0.x_dict)
old.load_state_dict(torch.load('best_full_model.pth', map_location=device))
pz = eval_full(old, te_p, aug_p, 'CA', aug_args, device)
ph = eval_full(old, te_h, aug_h, 'EOA', aug_args, device)
print(f"best_full_model.pth test macroF1 Ponzi={pz['macro_f1']:.4f} Phish={ph['macro_f1']:.4f}", flush=True)
all_res['best_full_model'] = both(old, 'best_full_model.pth (rule on, raw input to gates)')
del old; gc.collect(); torch.cuda.empty_cache()

# ---- 2. reported config, same training order as run_fixed_table.py ----
ref = json.load(open('A_full.json'))
ref_map = dict(zip(ref['args']['restarts'], ref['runs']))
for rs in SEEDS:
    seed_everything(rs)
    model = UnifiedHMSL(hidden=128, out_channels=2, data=dp, concat=3, expert_mode='none',
                        use_cpa=True, use_tcg=True, use_generative=True, use_contrastive=True,
                        raw_transform='signed_log').to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    best_val, best, best_state, best_epoch, patience = -1.0, None, None, -1, 0
    for epoch in range(60):
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
                    + 0.1 * (co_p + co_h))
            loss.backward(); opt.step()
        vp = eval_full(model, va_p, aug_p, 'CA', aug_args, device)['macro_f1']
        vh = eval_full(model, va_h, aug_h, 'EOA', aug_args, device)['macro_f1']
        if vp + vh > best_val:
            best_val = vp + vh
            best = dict(ponzi=eval_full(model, te_p, aug_p, 'CA', aug_args, device),
                        phish=eval_full(model, te_h, aug_h, 'EOA', aug_args, device))
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break
    r = ref_map[rs]
    print(f"\n[seed {rs}] best_epoch={best_epoch} test macroF1 Ponzi={best['ponzi']['macro_f1']:.4f} "
          f"Phish={best['phish']['macro_f1']:.4f} | A_full.json Ponzi={r['ponzi']['macro_f1']:.4f} "
          f"Phish={r['phish']['macro_f1']:.4f}", flush=True)
    torch.save(best_state, os.path.join(OUT, f'std_seed{rs}.pth'))
    model.load_state_dict(best_state)
    all_res[f'seed{rs}'] = both(model, f'reported config, seed {rs}, best-val checkpoint')
    all_res[f'seed{rs}']['f1'] = dict(ponzi=best['ponzi']['macro_f1'], phish=best['phish']['macro_f1'],
                                      ref_ponzi=r['ponzi']['macro_f1'], ref_phish=r['phish']['macro_f1'],
                                      best_epoch=best_epoch)
    with open(os.path.join(OUT, 'gate_stats.json'), 'w') as f:
        json.dump(all_res, f, indent=2)
    del model, opt; gc.collect(); torch.cuda.empty_cache()
print('GATE_DONE', flush=True)
