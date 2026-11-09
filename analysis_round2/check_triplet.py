"""Read-only diagnostics for the advisor's questions (no project file is written)."""
import sys, argparse, warnings
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler
from dataset import Ponzi, Phish
from unified_model import UnifiedHMSL
from run_unified import get_augmented_data
from utils import seed_everything, one_hot

warnings.filterwarnings("ignore")
SEED = 14
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device, flush=True)

dp = Ponzi('./data/Ponzi/')[0]
dh = Phish('./data/Phish/')[0]
print('Ponzi edge_types', dp.edge_types, flush=True)
print('Phish edge_types', dh.edge_types, flush=True)
aug_p = torch.load('./pretrain_model/icvae_Ponzi.pkl', map_location=device)
aug_h = torch.load('./pretrain_model/icvae_Phish.pkl', map_location=device)
aug_args = argparse.Namespace(concat=3)
kw = dict(batch_size=64, num_neighbors=[100, 100])
ns_p = NeighborSampler(dp, num_neighbors=[100, 100])
ns_h = NeighborSampler(dh, num_neighbors=[100, 100])
tr_p = NeighborLoader(dp, shuffle=True, input_nodes=('CA', dp['CA'].train_mask), neighbor_sampler=ns_p, **kw)
tr_h = NeighborLoader(dh, shuffle=True, input_nodes=('EOA', dh['EOA'].train_mask), neighbor_sampler=ns_h, **kw)

seed_everything(SEED)
model = UnifiedHMSL(hidden=128, out_channels=2, data=dp, concat=3, expert_mode='none',
                    use_cpa=True, use_tcg=True, raw_transform='signed_log').to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


def forward_pair():
    bp = get_augmented_data(next(iter(tr_p)).to(device), aug_p, aug_args, device)
    bh = get_augmented_data(next(iter(tr_h)).to(device), aug_h, aug_args, device)
    _, _, triplet_ponzi, _, _ = model(bp.x_dict_new, bp.edge_index_dict, raw_x_dict=bp.x_dict)
    _, _, triplet_phish, _, _ = model(bh.x_dict_new, bh.edge_index_dict, raw_x_dict=bh.x_dict)
    return bp, bh, triplet_ponzi, triplet_phish


def advisor_snippet(tag):
    print(f'\n===== ADVISOR SNIPPET [{tag}] =====', flush=True)
    model.train()
    bp, bh, triplet_ponzi, triplet_phish = forward_pair()
    print('triplet_ponzi', float(triplet_ponzi), 'triplet_phish', float(triplet_phish), flush=True)
    model.zero_grad()
    loss_tri = triplet_ponzi + triplet_phish
    loss_tri.backward(retain_graph=True)
    for n, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            print(n, tuple(p.shape))
    print('--- end of printed names ---', flush=True)
    groups = {}
    for n, p in model.named_parameters():
        key = '.'.join(n.split('.')[:2])
        g = 0.0 if p.grad is None else float(p.grad.norm())
        groups[key] = max(groups.get(key, 0.0), g)
    print('max grad norm per group:', {k: round(v, 6) for k, v in groups.items()}, flush=True)
    model.zero_grad()
    return bp, bh


def hinge_stats(z_list, nt, tag):
    """z_list: list (K) of [V, N, H]. Mirrors contrast_module; reports hinge activity."""
    a = model.attention([z[1] for z in z_list], nt)
    p = model.attention([z[2] for z in z_list], nt)
    n = model.attention([z[0] for z in z_list], nt)
    ap = (a - p).norm(dim=1)
    an = (a - n).norm(dim=1)
    active = ((ap - an + 0.3) > 0).float().mean()
    print(f'  {tag} {nt}: V={z_list[0].shape[0]} N={a.shape[0]} d(a,p)={ap.mean():.4f} d(a,n)={an.mean():.4f} '
          f'active_hinge={active:.3f}', flush=True)


@torch.no_grad()
def view_diag(tag):
    print(f'\n===== VIEW DIAG [{tag}] =====', flush=True)
    model.eval()
    for name, loader, aug, nt in [('Ponzi', tr_p, aug_p, 'CA'), ('Phish', tr_h, aug_h, 'EOA')]:
        b = get_augmented_data(next(iter(loader)).to(device), aug, aug_args, device)
        x_list = b.x_dict_new[nt]
        print(f'{name}: x_dict_new[{nt}] len(K)={len(x_list)} each shape [V,N,F]={tuple(x_list[0].shape)}')
        print(f'  L1 diff of generated views between draws k0/k1 per relation:',
              [round(float((x_list[0][v] - x_list[1][v]).abs().sum(1).mean()), 6) for v in range(x_list[0].shape[0])])
        print(f'  L1 diff across relations in draw 0 (v0-v1, v1-v2):',
              round(float((x_list[0][0] - x_list[0][1]).abs().sum(1).mean()), 6),
              round(float((x_list[0][1] - x_list[0][2]).abs().sum(1).mean()), 6))
        z_list = [torch.tanh(model.lin_dict[nt](x)) for x in x_list]
        sat = float((z_list[0][3].abs() > 0.99).float().mean())
        sat_gen = float((z_list[0][0].abs() > 0.99).float().mean())
        print(f'  tanh saturation |h|>0.99: original view={sat:.3f}, generated view={sat_gen:.3f}')
        hinge_stats(z_list, nt, tag)
        pooled = [torch.tanh(model.lin_dict_mean[nt](z.mean(0))) for z in z_list]
        s = torch.stack(pooled, 0).permute(1, 0, 2)
        q, k, v = model.q_lin[nt](s), model.k_lin[nt](s), model.v_lin[nt](s)
        att = F.softmax(k @ q.transpose(1, 2), -1)
        print(f'  gen-path input shape [N,K,H]={tuple(s.shape)} attention matrix shape={tuple(att.shape)}')
        print('  mean attention matrix:', np.round(att.mean(0).cpu().numpy(), 3).tolist(), flush=True)
        dev = (att - 1.0 / 3).abs()
        print(f'  per-node |att-1/3|: mean={dev.mean():.6f} p99={torch.quantile(dev.flatten()[:200000], 0.99):.6f} max={dev.max():.6f}')
        out_att = (att @ v).mean(1)
        out_mean = v.mean(1)
        print(f'  ||attention_output - plain_mean_of_v||/||output|| = {float((out_att - out_mean).norm(dim=1).mean() / out_att.norm(dim=1).mean()):.6f}')
        between = (s[:, 0] - s[:, 1]).norm(dim=1).mean() / s[:, 0].norm(dim=1).mean()
        print(f'  relative gap between pooled draws k0/k1 = {float(between):.6f}', flush=True)


@torch.no_grad()
def icvae_diag():
    print('\n===== ICVAE CHECKPOINT DIAG =====', flush=True)
    for name, data, aug in [('Ponzi', dp, aug_p), ('Phish', dh, aug_h)]:
        print(name, type(aug).__module__, type(aug).__name__, 'latent', aug.latent_size)
        for i, et in enumerate(data.edge_types):
            ei = data.edge_stores[i]['edge_index']
            idx = torch.randperm(ei.size(1))[:200000]
            head = data[et[0]].x[ei[0][idx]].float().to(device)
            tail = data[et[-1]].x[ei[1][idx]].float().to(device)
            keep = (head - tail).abs().sum(1) > 0   # only pairs where head and tail differ
            head, tail = head[keep], tail[keep]
            e = one_hot(torch.LongTensor([i]), len(data.edge_types), dtype=torch.float32).repeat(head.size(0), 1).to(device)
            z = torch.randn(head.size(0), aug.latent_size, device=device)
            rec = aug.inference(z, head, e)
            mae_tail = float((rec - torch.sigmoid(tail)).abs().mean())
            mae_head = float((rec - torch.sigmoid(head)).abs().mean())
            print(f'  {et[0]}-{et[1]}-{et[2]} n={head.size(0)} MAE(rec, sigmoid(tail))={mae_tail:.4f} '
                  f'MAE(rec, sigmoid(head))={mae_head:.4f}', flush=True)


icvae_diag()
advisor_snippet('init')
view_diag('init')

for epoch in range(EPOCHS):
    model.train()
    co_sum, n = 0.0, 0
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
        co_sum += float(co_p + co_h); n += 1
    print(f'epoch {epoch} mean triplet(ponzi+phish)={co_sum / n:.5f}', flush=True)

advisor_snippet(f'after {EPOCHS} epochs')
view_diag(f'after {EPOCHS} epochs')
print('DONE', flush=True)
