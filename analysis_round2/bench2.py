"""Joint vs single-task cost table, batch 128. Writes only to the scratchpad.

  bench2.py train joint|ponzi|phish   one full run, saves the best-val checkpoint
  bench2.py infer joint|ponzi|phish   fresh process: load from file, then predict the test set(s)
"""
import sys, os, json, copy, time, argparse, warnings, gc
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler
from dataset import Ponzi, Phish
from unified_model import UnifiedHMSL
from run_unified import get_augmented_data
from utils import seed_everything

warnings.filterwarnings("ignore")
OUT = os.path.dirname(os.path.abspath(__file__))
MODE, VAR = sys.argv[1], sys.argv[2]
SEED, BATCH = 14, 128
device = torch.device('cuda:0')
aug_args = argparse.Namespace(concat=3)
GIB = 1024 ** 3
USE_P, USE_H = VAR in ('joint', 'ponzi'), VAR in ('joint', 'phish')
CKPT = os.path.join(OUT, f'bench2_{VAR}.pth')


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def loaders(data, nt, sampler, kinds):
    m = {'tr': 'train_mask', 'va': 'val_mask', 'te': 'test_mask'}
    return {k: NeighborLoader(data, shuffle=k == 'tr', input_nodes=(nt, data[nt][m[k]]), neighbor_sampler=sampler,
                              batch_size=BATCH, num_neighbors=[100, 100]) for k in kinds}


def build(data):
    return UnifiedHMSL(hidden=128, out_channels=2, data=data, concat=3, expert_mode='none',
                       use_cpa=True, use_tcg=True, use_generative=True, use_contrastive=True,
                       raw_transform='signed_log').to(device)


@torch.no_grad()
def predict(model, loader, aug, nt, collect=False):
    model.eval()
    preds, labels = [], []
    for b in loader:
        b = get_augmented_data(b.to(device), aug, aug_args, device)
        k = b[nt].batch_size
        out = model(b.x_dict_new, b.edge_index_dict, raw_x_dict=b.x_dict)[0 if nt == 'CA' else 1]
        p = out[:k].argmax(1).cpu()
        if collect:
            preds.append(p); labels.append(b[nt].y[:k].cpu())
    if collect:
        return f1_score(torch.cat(labels), torch.cat(preds), average='macro')


kinds = ['tr', 'va', 'te'] if MODE == 'train' else ['te']
dp = dh = lp = lh = None
if USE_P:
    dp = Ponzi('./data/Ponzi/')[0]
    lp = loaders(dp, 'CA', NeighborSampler(dp, num_neighbors=[100, 100]), kinds)
if USE_H:
    dh = Phish('./data/Phish/')[0]
    lh = loaders(dh, 'EOA', NeighborSampler(dh, num_neighbors=[100, 100]), kinds)
meta = dp if dp is not None else dh

if MODE == 'train':
    aug_p = torch.load('./pretrain_model/icvae_Ponzi.pkl', map_location=device) if USE_P else None
    aug_h = torch.load('./pretrain_model/icvae_Phish.pkl', map_location=device) if USE_H else None
    seed_everything(SEED)
    model = build(meta)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    torch.cuda.reset_peak_memory_stats()
    best_val, best_state, best_epoch, patience, best_test = -1.0, None, -1, 0, {}
    t_train = t_val = t_test = 0.0
    tr_epochs = []
    epochs = 0
    for epoch in range(60):
        epochs += 1
        a = tick()
        model.train()
        if VAR == 'joint':
            it = zip(lp['tr'], lh['tr'])
        elif VAR == 'ponzi':
            it = ((b, None) for b in lp['tr'])
        else:
            it = ((None, b) for b in lh['tr'])
        steps = 0
        for bp, bh in it:
            opt.zero_grad()
            loss = 0.0
            if bp is not None:
                bp = get_augmented_data(bp.to(device), aug_p, aug_args, device)
                op, _, co_p, _, _ = model(bp.x_dict_new, bp.edge_index_dict, raw_x_dict=bp.x_dict)
                sp = bp['CA'].batch_size
                loss = loss + F.cross_entropy(op[:sp], bp['CA'].y[:sp]) + 0.1 * co_p
            if bh is not None:
                bh = get_augmented_data(bh.to(device), aug_h, aug_args, device)
                _, oh, co_h, _, _ = model(bh.x_dict_new, bh.edge_index_dict, raw_x_dict=bh.x_dict)
                sh = bh['EOA'].batch_size
                loss = loss + F.cross_entropy(oh[:sh], bh['EOA'].y[:sh]) + 0.1 * co_h
            loss.backward(); opt.step()
            steps += 1
        dt = tick() - a
        t_train += dt; tr_epochs.append(dt)
        a = tick()
        val = 0.0
        if USE_P:
            val += predict(model, lp['va'], aug_p, 'CA', collect=True)
        if USE_H:
            val += predict(model, lh['va'], aug_h, 'EOA', collect=True)
        t_val += tick() - a
        if val > best_val:
            best_val = val
            a = tick()
            best_test = {}
            if USE_P:
                best_test['ponzi'] = predict(model, lp['te'], aug_p, 'CA', collect=True)
            if USE_H:
                best_test['phish'] = predict(model, lh['te'], aug_h, 'EOA', collect=True)
            t_test += tick() - a
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    torch.save(best_state, CKPT)
    n_params = sum(p.numel() for p in model.parameters())
    res = dict(variant=VAR, seed=SEED, batch=BATCH, epochs=epochs, best_epoch=best_epoch, steps_per_epoch=steps,
               params=n_params, train_s=t_train, val_s=t_val, test_s=t_test, train_val_s=t_train + t_val,
               s_per_epoch=float(np.mean(tr_epochs)), peak_alloc_bytes=peak, peak_alloc_gib=peak / GIB,
               peak_alloc_gb=peak / 1e9, peak_reserved_gib=reserved / GIB, test_macro_f1=best_test,
               ckpt_mb=os.path.getsize(CKPT) / 1024 ** 2)
    print('TRAIN', json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(OUT, f'bench2_train_{VAR}.json'), 'w'), indent=2)
    sys.exit(0)

# ---- inference: model and ICVAE start from files, data and loaders are already up ----
torch.zeros(1, device=device)  # CUDA start-up is not model loading
tasks = [t for t, u in (('ponzi', USE_P), ('phish', USE_H)) if u]
first = {'ponzi': next(iter(lp['te'])) if USE_P else None, 'phish': next(iter(lh['te'])) if USE_H else None}
a = tick()
aug = {}
if USE_P:
    aug['ponzi'] = torch.load('./pretrain_model/icvae_Ponzi.pkl', map_location=device)
if USE_H:
    aug['phish'] = torch.load('./pretrain_model/icvae_Phish.pkl', map_location=device)
t_icvae = tick() - a
a = tick()
state = torch.load(CKPT, map_location=device)
model = build(meta)
b = get_augmented_data(first[tasks[0]].to(device), aug[tasks[0]], aug_args, device)
with torch.no_grad():
    model(b.x_dict_new, b.edge_index_dict, raw_x_dict=b.x_dict)  # materialise lazy layers
model.load_state_dict(state)
t_gnn = tick() - a
del b, state

cfg = {'ponzi': (lp, 'CA'), 'phish': (lh, 'EOA')}
gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
passes, f1 = [], {}
for i in range(6):
    row = {}
    for t in tasks:
        a = tick()
        predict(model, cfg[t][0]['te'], aug[t], cfg[t][1])
        row[t] = tick() - a
    passes.append(row)
peak = torch.cuda.max_memory_allocated()
for t in tasks:
    f1[t] = predict(model, cfg[t][0]['te'], aug[t], cfg[t][1], collect=True)
res = dict(variant=VAR, batch=BATCH, load_icvae_s=t_icvae, load_gnn_s=t_gnn, load_s=t_icvae + t_gnn,
           first_pass_s=sum(passes[0].values()),
           infer_s=float(np.mean([sum(p.values()) for p in passes[1:]])),
           infer_std=float(np.std([sum(p.values()) for p in passes[1:]])),
           per_task_s={t: float(np.mean([p[t] for p in passes[1:]])) for t in tasks},
           infer_peak_gib=peak / GIB, test_macro_f1=f1)
print('INFER', json.dumps(res), flush=True)
with open(os.path.join(OUT, 'bench2_infer.jsonl'), 'a') as f:
    f.write(json.dumps(res) + '\n')
