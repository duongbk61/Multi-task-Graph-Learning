"""Time / GPU / RAM benchmark of the reported config. Writes only to the scratchpad.

  bench.py train     full training runs + inference, as in run_fixed_table.py
  bench.py pretrain  real ICVAE pretrain, result discarded (pretrain_model/ untouched)
"""
import sys, os, json, copy, time, argparse, warnings, gc, threading, subprocess
sys.path.insert(0, '.')
import numpy as np
import psutil
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
MODE = sys.argv[1]
SEEDS = [] if MODE == 'mean' else ([int(s) for s in sys.argv[2:]] or [14, 21, 42])
device = torch.device('cuda:0')
GB = 1024 ** 3
MB = 1024 ** 2


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def smi():
    q = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'], text=True)
    m, u = q.strip().split(',')
    return float(m), float(u)


class Sampler(threading.Thread):
    """Polls nvidia-smi once per second, tagging each sample with the current phase."""
    def __init__(self):
        super().__init__(daemon=True)
        self.phase, self.rows, self.stop_flag = 'idle', [], False

    def run(self):
        while not self.stop_flag:
            m, u = smi()
            self.rows.append((self.phase, m, u))
            time.sleep(1.0)

    def summary(self, phase):
        r = [x for x in self.rows if x[0] == phase]
        if not r:
            return {}
        return dict(n=len(r), mem_used_peak_mb=max(x[1] for x in r), util_mean=float(np.mean([x[2] for x in r])))


def peak_reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_read():
    return dict(alloc_mb=torch.cuda.max_memory_allocated() / MB, reserved_mb=torch.cuda.max_memory_reserved() / MB)


def ram_peak_gb():
    return psutil.Process().memory_info().peak_wset / GB


def ram_now_gb():
    return psutil.Process().memory_info().rss / GB


if MODE == 'pretrain':
    print('PRETRAIN_DONE skipped on request', flush=True)
    sys.exit(0)
    import icvae_pretrain as ip
    res = {}
    WARM, N = 200, 20000

    class Enough(Exception):
        pass

    orig_loss = ip.loss_fn
    st = {}

    def counted(*a):
        """Stops the pretrain after a fixed number of batches; the full run takes hours."""
        st['n'] += 1
        if st['n'] == 1:
            st['setup'] = tick() - st['start']
        if st['n'] == WARM:
            st['t0'] = tick()
        if st['n'] == WARM + N:
            st['t1'] = tick()
            raise Enough
        return orig_loss(*a)

    ip.loss_fn = counted
    for name, Cls, path, nt in [('Ponzi', Ponzi, './data/Ponzi', 'CA'), ('Phish', Phish, './data/Phish', 'EOA')]:
        data = Cls(path)[0]
        ip.target_node = nt
        n_edges = int(sum(data.edge_stores[i]['edge_index'].shape[1] for i in range(len(data.edge_types))))
        seed_everything(14)
        peak_reset()
        st.clear()
        st.update(n=0, start=tick())
        try:
            ip.generated_generator_sixedges(argparse.Namespace(batch_size=512, pretrain_epochs=3), data, device)
        except Enough:
            pass
        rate = N / (st['t1'] - st['t0'])
        batches = int(np.ceil(n_edges / 512))
        res[name] = dict(edges=n_edges, batches_per_epoch=batches, setup_s=st['setup'], batches_per_s=rate,
                         estimated_3_epochs_s=st['setup'] + 3 * batches / rate, measured_batches=N,
                         **peak_read(), ram_peak_gb=ram_peak_gb())
        print(name, res[name], flush=True)
        del data
        gc.collect()
    json.dump(res, open(os.path.join(OUT, 'bench_pretrain.json'), 'w'), indent=2)
    print('PRETRAIN_DONE', flush=True)
    sys.exit(0)

# ---------------- train + inference ----------------
base_mem, _ = smi()
print(f'gpu baseline used before torch work: {base_mem:.0f} MiB', flush=True)
samp = Sampler()
samp.start()

t0 = time.perf_counter()
dp = Ponzi('./data/Ponzi/')[0]
dh = Phish('./data/Phish/')[0]
t_data = time.perf_counter() - t0
t0 = time.perf_counter()
aug_p = torch.load('./pretrain_model/icvae_Ponzi.pkl', map_location=device)
aug_h = torch.load('./pretrain_model/icvae_Phish.pkl', map_location=device)
t_aug = time.perf_counter() - t0
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
one = dict(batch_size=1, num_neighbors=[100, 100])
te1_p = NeighborLoader(dp, shuffle=False, input_nodes=('CA', dp['CA'].test_mask), neighbor_sampler=ns_p, **one)
te1_h = NeighborLoader(dh, shuffle=False, input_nodes=('EOA', dh['EOA'].test_mask), neighbor_sampler=ns_h, **one)

info = dict(
    t_load_data_s=t_data, t_load_icvae_s=t_aug,
    sizes=dict(ponzi=dict(train=int(dp['CA'].train_mask.sum()), val=int(dp['CA'].val_mask.sum()), test=int(dp['CA'].test_mask.sum()),
                          ca=dp['CA'].num_nodes, eoa=dp['EOA'].num_nodes, edges=int(sum(s.num_edges for s in dp.edge_stores))),
               phish=dict(train=int(dh['EOA'].train_mask.sum()), val=int(dh['EOA'].val_mask.sum()), test=int(dh['EOA'].test_mask.sum()),
                          ca=dh['CA'].num_nodes, eoa=dh['EOA'].num_nodes, edges=int(sum(s.num_edges for s in dh.edge_stores)))),
    steps_per_epoch=min(len(tr_p), len(tr_h)), icvae_params=sum(p.numel() for p in aug_p.parameters()))
print(info, flush=True)


def build(use_cpa, use_tcg):
    return UnifiedHMSL(hidden=128, out_channels=2, data=dp, concat=3, expert_mode='none',
                       use_cpa=use_cpa, use_tcg=use_tcg, use_generative=True, use_contrastive=True,
                       raw_transform='signed_log').to(device)


def train_epoch(model, opt):
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


@torch.no_grad()
def infer_pass(model, loader, aug, nt, limit=None):
    """One pass over the loader, split into neighbour sampling, ICVAE and GNN time."""
    model.eval()
    t_s = t_a = t_f = 0.0
    n, sampled = 0, []
    it = iter(loader)
    idx = 0
    while limit is None or idx < limit:
        a = tick()
        try:
            b = next(it)
        except StopIteration:
            break
        c = tick(); t_s += c - a
        b = get_augmented_data(b.to(device), aug, aug_args, device)
        d = tick(); t_a += d - c
        k = b[nt].batch_size
        out = model(b.x_dict_new, b.edge_index_dict, raw_x_dict=b.x_dict)[0 if nt == 'CA' else 1]
        torch.softmax(out[:k], 1).cpu()
        e = tick(); t_f += e - d
        n += k; idx += 1
        sampled.append(b['CA'].num_nodes + b['EOA'].num_nodes)
    return dict(nodes=n, batches=idx, sampling_s=t_s, icvae_s=t_a, gnn_s=t_f, total_s=t_s + t_a + t_f,
                sampled_nodes_mean=float(np.mean(sampled)), sampled_nodes_max=int(np.max(sampled)))


def inference(model, tag):
    out = {}
    for name, l, l1, aug, nt in [('ponzi', te_p, te1_p, aug_p, 'CA'), ('phish', te_h, te1_h, aug_h, 'EOA')]:
        samp.phase = f'infer_{name}'
        peak_reset()
        passes = [infer_pass(model, l, aug, nt) for _ in range(6)]
        peak = peak_read()
        keep = passes[1:]  # the first pass warms caches
        row = {k: float(np.mean([p[k] for p in keep])) for k in keep[0]}
        row['total_s_std'] = float(np.std([p['total_s'] for p in keep]))
        row['first_pass_s'] = passes[0]['total_s']
        row['ms_per_node'] = 1000 * row['total_s'] / row['nodes']
        row['nodes_per_s'] = row['nodes'] / row['total_s']
        peak_reset()
        single = infer_pass(model, l1, aug, nt, limit=100)
        row['batch1_ms_per_node'] = 1000 * single['total_s'] / single['nodes']
        row['batch1_sampled_nodes_mean'] = single['sampled_nodes_mean']
        row['batch1_peak'] = peak_read()
        row['peak'] = peak
        samp.phase = 'idle'
        row['smi'] = samp.summary(f'infer_{name}')
        out[name] = row
        print(f"[{tag}] infer {name}: {row['total_s']:.2f}s/{int(row['nodes'])} nodes, {row['ms_per_node']:.2f} ms/node, "
              f"alloc {peak['alloc_mb']:.0f} reserved {peak['reserved_mb']:.0f}", flush=True)
    return out


results = dict(info=info, base_mem=base_mem, runs=[])
for rs in SEEDS:
    seed_everything(rs)
    model = build(True, True)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    best_val, best_state, best_epoch, patience = -1.0, None, -1, 0
    tr_t, va_t, te_t = [], [], []
    tr_peak = dict(alloc_mb=0.0, reserved_mb=0.0)
    ev_peak = dict(alloc_mb=0.0, reserved_mb=0.0)
    t_run = tick()
    epochs = 0
    for epoch in range(60):
        epochs += 1
        torch.cuda.reset_peak_memory_stats()
        samp.phase = 'train'
        a = tick(); train_epoch(model, opt); tr_t.append(tick() - a)
        p = peak_read()
        tr_peak = {k: max(tr_peak[k], p[k]) for k in p}
        torch.cuda.reset_peak_memory_stats()
        samp.phase = 'eval'
        a = tick()
        vp = eval_full(model, va_p, aug_p, 'CA', aug_args, device)['macro_f1']
        vh = eval_full(model, va_h, aug_h, 'EOA', aug_args, device)['macro_f1']
        va_t.append(tick() - a)
        if vp + vh > best_val:
            best_val = vp + vh
            a = tick()
            eval_full(model, te_p, aug_p, 'CA', aug_args, device)
            eval_full(model, te_h, aug_h, 'EOA', aug_args, device)
            te_t.append(tick() - a)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break
        p = peak_read()
        ev_peak = {k: max(ev_peak[k], p[k]) for k in p}
        samp.phase = 'idle'
    run_s = tick() - t_run
    samp.phase = 'idle'
    model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters())  # lazy params exist only after a forward
    row = dict(seed=rs, params=n_params, epochs=epochs, best_epoch=best_epoch, run_s=run_s,
               train_epoch_s=float(np.mean(tr_t)), train_epoch_std=float(np.std(tr_t)),
               val_s=float(np.mean(va_t)), test_s=float(np.mean(te_t)), test_evals=len(te_t),
               train_peak=tr_peak, eval_peak=ev_peak, train_smi=samp.summary('train'),
               eval_smi=samp.summary('eval'), ram_peak_gb=ram_peak_gb(), ram_now_gb=ram_now_gb(),
               infer=inference(model, f'seed{rs}'))
    results['runs'].append(row)
    print(f"[seed {rs}] {epochs} epochs, best {best_epoch}, {run_s:.0f}s, train {row['train_epoch_s']:.2f}s/ep, "
          f"val {row['val_s']:.2f}s, test {row['test_s']:.2f}s, train alloc {tr_peak['alloc_mb']:.0f} "
          f"reserved {tr_peak['reserved_mb']:.0f}, ram {row['ram_peak_gb']:.2f}GB", flush=True)
    json.dump(results, open(os.path.join(OUT, 'bench_train.json'), 'w'), indent=2)
    samp.rows = []
    del model, opt, best_state; gc.collect(); torch.cuda.empty_cache()

# ---- plain mean fusion, same machine: 5 epochs are enough for a per-epoch cost ----
seed_everything(14)
model = build(False, False)
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
times = []
torch.cuda.reset_peak_memory_stats()
for _ in range(5):
    a = tick(); train_epoch(model, opt); times.append(tick() - a)
results['mean_fusion'] = dict(
    params=sum(p.numel() for p in model.parameters() if not isinstance(p, torch.nn.parameter.UninitializedParameter)),
    train_epoch_s=float(np.mean(times)), train_peak=peak_read(), infer=inference(model, 'mean'))
json.dump(results, open(os.path.join(OUT, 'bench_mean.json'), 'w'), indent=2)
samp.stop_flag = True
print('TRAIN_DONE', results['mean_fusion']['params'], results['mean_fusion']['train_epoch_s'], flush=True)
