"""
Run ONLY the E-A config #5 (VAE, no interaction-condition) with the pretrained
vaenc_{Ponzi,Phish}.pkl models. Same fixed-split + 5-restart protocol as run_ea.py.
Augmentation = ICVAE-style generation but with a ZERO edge-type vector (no relation cond).
"""
import json, argparse, warnings, gc
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler
from sklearn.metrics import f1_score
from dataset import Ponzi, Phish
from unified_model import UnifiedHMSL
from utils import feature_tensor_normalize, seed_everything

warnings.filterwarnings("ignore")


def gen_zerocond(data, aug_model, device):
    """Like get_augmented_features but edge-type one-hot is ZERO (no interaction cond)."""
    CA_ls, EOA_ls = [], []
    n_rel = len(data.edge_types)
    for i in range(n_rel):
        src = data.edge_types[i][0]
        zero_e = torch.zeros((data[src].num_nodes, n_rel), device=device)
        z = torch.randn([data[src].num_nodes, aug_model.latent_size]).to(device)
        aug = aug_model.inference(z, data[src].x, zero_e).detach()
        aug = feature_tensor_normalize(aug).detach()
        (CA_ls if src == 'CA' else EOA_ls).append(aug)
    return CA_ls, EOA_ls


def augment(data, aug_model, concat, device):
    data.x_dict_new = {'CA': [], 'EOA': []}
    ca_o, eoa_o = data.x_dict['CA'], data.x_dict['EOA']
    for _ in range(concat):
        CA_x, EOA_x = gen_zerocond(data, aug_model, device)
        data.x_dict_new['CA'].append(torch.stack(CA_x + [ca_o]))
        data.x_dict_new['EOA'].append(torch.stack(EOA_x + [eoa_o]))
    return data


def ev(model, loader, aug_model, nt, concat, device):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for b in loader:
            b = augment(b.to(device), aug_model, concat, device)
            bs = b[nt].batch_size
            op, oh, *_ = model(b.x_dict_new, b.edge_index_dict, raw_x_dict=b.x_dict)
            out = op if nt == 'CA' else oh
            preds.extend(out[:bs].argmax(1).cpu().numpy()); labels.extend(b[nt].y[:bs].cpu().numpy())
    return f1_score(labels, preds, average='macro')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--restarts', type=int, nargs='+', default=[14, 21, 42, 77, 100])
    ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64); ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001); ap.add_argument('--concat', type=int, default=3)
    ap.add_argument('--loss_train', type=float, default=0.1)
    ap.add_argument('--gpu', type=str, default='0'); ap.add_argument('--num_neighbors', type=int, nargs='+', default=[100, 100])
    ap.add_argument('--expert_mode', type=str, default='feature', choices=['feature', 'loss', 'none'])
    ap.add_argument('--raw_transform', type=str, default='none', choices=['none', 'signed_log'])
    ap.add_argument('--out', type=str, default='ea_vaenc')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    dp = Ponzi('./data/Ponzi/')[0]; dh = Phish('./data/Phish/')[0]
    aug_p = torch.load('./pretrain_model/vaenc_Ponzi.pkl', map_location=device)
    aug_h = torch.load('./pretrain_model/vaenc_Phish.pkl', map_location=device)
    ns_p = NeighborSampler(dp, num_neighbors=args.num_neighbors); ns_h = NeighborSampler(dh, num_neighbors=args.num_neighbors)
    bs = dict(batch_size=args.batch_size, num_neighbors=args.num_neighbors)
    L = lambda d, nt, mk, ns: NeighborLoader(d, shuffle=(mk == 'train_mask'), input_nodes=(nt, getattr(d[nt], mk)), neighbor_sampler=ns, **bs)
    tr_p, va_p, te_p = L(dp, 'CA', 'train_mask', ns_p), L(dp, 'CA', 'val_mask', ns_p), L(dp, 'CA', 'test_mask', ns_p)
    tr_h, va_h, te_h = L(dh, 'EOA', 'train_mask', ns_h), L(dh, 'EOA', 'val_mask', ns_h), L(dh, 'EOA', 'test_mask', ns_h)

    pj, hj = [], []
    for rs in args.restarts:
        seed_everything(rs)
        model = UnifiedHMSL(hidden=args.hidden, out_channels=2, data=dp, concat=args.concat,
                            expert_mode=args.expert_mode, raw_transform=args.raw_transform,
                            use_cpa=True, use_tcg=True).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        best_val, best = -1.0, None; patience = 0
        for epoch in range(args.epochs):
            model.train()
            for bp, bh in zip(tr_p, tr_h):
                opt.zero_grad()
                bp = augment(bp.to(device), aug_p, args.concat, device)
                op, _, co_p, _, _ = model(bp.x_dict_new, bp.edge_index_dict, raw_x_dict=bp.x_dict); sp = bp['CA'].batch_size
                bh = augment(bh.to(device), aug_h, args.concat, device)
                _, oh, co_h, _, _ = model(bh.x_dict_new, bh.edge_index_dict, raw_x_dict=bh.x_dict); sh = bh['EOA'].batch_size
                loss = F.cross_entropy(op[:sp], bp['CA'].y[:sp]) + F.cross_entropy(oh[:sh], bh['EOA'].y[:sh]) + args.loss_train * (co_p + co_h)
                loss.backward(); opt.step()
            vp = ev(model, va_p, aug_p, 'CA', args.concat, device); vh = ev(model, va_h, aug_h, 'EOA', args.concat, device)
            if vp + vh > best_val:
                best_val = vp + vh
                best = (ev(model, te_p, aug_p, 'CA', args.concat, device), ev(model, te_h, aug_h, 'EOA', args.concat, device)); patience = 0
            else:
                patience += 1
                if patience >= args.patience: break
        pj.append(best[0]); hj.append(best[1])
        print(f"[seed {rs}] Ponzi={best[0]:.4f} Phish={best[1]:.4f}", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
        json.dump({'ponzi': pj, 'phish': hj}, open(f"{args.out}.json", 'w'), indent=2)
    p, h = np.array(pj) * 100, np.array(hj) * 100
    print(f"\nVAE (no interaction-condition) | Ponzi {p.mean():.2f}±{p.std():.2f} | Phish {h.mean():.2f}±{h.std():.2f}")


if __name__ == '__main__':
    main()
