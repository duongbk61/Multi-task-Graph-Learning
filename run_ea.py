"""
(E-A) Ablation Generative — fixed-split + 5 restarts, Macro-F1 mean±std per task.

Augmentation modes (replace the ICVAE generative augmentation):
  icvae    : original ICVAE inference (3 relation views), row-L1 normalized  [= Full]
  raw      : no generation — 3 views = normalized raw feature (isolates "− ICVAE")
  gaussian : normalized raw + N(0, sigma) per view (naive noise baseline)

Path switches (model): use_generative / use_contrastive isolate each representation path.
Triplet switch: --triplet 0 removes the triplet loss.

Configs run here (Full & ICVAE-cond are reused from fixed_table_row.json to save time):
  - ICVAE      : aug=raw
  - triplet    : aug=icvae, triplet=0
  Gaussian     : aug=gaussian
  - generative : gen=0 (contrastive only), triplet=0
  - contrastive: cont=0 (generative only)
VAE-no-condition (#5) needs a separately pretrained model -> handled in run_vae_nocond.py.
"""
import json, argparse, warnings, gc, copy
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler
from dataset import Ponzi, Phish
from unified_model import UnifiedHMSL
from run_unified import get_augmented_features
from run_cv_table import eval_full
from utils import feature_tensor_normalize, seed_everything

warnings.filterwarnings("ignore")


def augment(data, aug_model, concat, mode, device, noise_std=0.1):
    data.x_dict_new = {}
    ca_orig = data.x_dict['CA']; eoa_orig = data.x_dict['EOA']
    data.x_dict_new['CA'] = []; data.x_dict_new['EOA'] = []
    for _ in range(concat):
        if mode == 'icvae':
            CA_x, EOA_x = get_augmented_features(data, aug_model, device)      # 3 views each, normalized
        elif mode == 'raw':
            CA_x = [feature_tensor_normalize(ca_orig).to(device) for _ in range(3)]
            EOA_x = [feature_tensor_normalize(eoa_orig).to(device) for _ in range(3)]
        elif mode == 'gaussian':
            CA_x = [feature_tensor_normalize(ca_orig).to(device) + torch.randn_like(ca_orig) * noise_std
                    for _ in range(3)]
            EOA_x = [feature_tensor_normalize(eoa_orig).to(device) + torch.randn_like(eoa_orig) * noise_std
                     for _ in range(3)]
        else:
            raise ValueError(mode)
        data.x_dict_new['CA'].append(torch.stack(CA_x + [ca_orig]))
        data.x_dict_new['EOA'].append(torch.stack(EOA_x + [eoa_orig]))
    return data


def run_config(cfg, ctx, args):
    device = ctx['device']
    macro = {'ponzi': [], 'phish': []}
    for rs in args.restarts:
        seed_everything(rs)
        model = UnifiedHMSL(hidden=args.hidden, out_channels=2, data=ctx['dp'], concat=args.concat,
                            expert_mode=args.expert_mode, raw_transform=args.raw_transform,
                            use_cpa=True, use_tcg=True,
                            use_generative=cfg['gen'], use_contrastive=cfg['cont']).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        mode, tw = cfg['aug'], cfg['triplet']

        def ev(loader, aug_model, nt):
            # eval_full expects model(...) with x_dict_new; wrap augment with this cfg's mode
            model.eval(); preds = []; labels = []
            from sklearn.metrics import f1_score
            with torch.no_grad():
                for b in loader:
                    b = augment(b.to(device), aug_model, args.concat, mode, device, args.noise_std)
                    bs = b[nt].batch_size
                    op, oh, *_ = model(b.x_dict_new, b.edge_index_dict, raw_x_dict=b.x_dict)
                    out = op if nt == 'CA' else oh
                    preds.extend(out[:bs].argmax(1).cpu().numpy()); labels.extend(b[nt].y[:bs].cpu().numpy())
            return f1_score(labels, preds, average='macro')

        best_val, best = -1.0, None; patience = 0
        for epoch in range(args.epochs):
            model.train()
            for bp, bh in zip(ctx['tr_p'], ctx['tr_h']):
                opt.zero_grad()
                bp = augment(bp.to(device), ctx['aug_p'], args.concat, mode, device, args.noise_std)
                op, _, co_p, _, _ = model(bp.x_dict_new, bp.edge_index_dict, raw_x_dict=bp.x_dict)
                sp = bp['CA'].batch_size
                bh = augment(bh.to(device), ctx['aug_h'], args.concat, mode, device, args.noise_std)
                _, oh, co_h, _, _ = model(bh.x_dict_new, bh.edge_index_dict, raw_x_dict=bh.x_dict)
                sh = bh['EOA'].batch_size
                loss = (F.cross_entropy(op[:sp], bp['CA'].y[:sp]) + F.cross_entropy(oh[:sh], bh['EOA'].y[:sh])
                        + tw * (co_p + co_h))
                loss.backward(); opt.step()
            vp = ev(ctx['va_p'], ctx['aug_p'], 'CA'); vh = ev(ctx['va_h'], ctx['aug_h'], 'EOA')
            if vp + vh > best_val:
                best_val = vp + vh
                best = (ev(ctx['te_p'], ctx['aug_p'], 'CA'), ev(ctx['te_h'], ctx['aug_h'], 'EOA'))
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    break
        macro['ponzi'].append(best[0]); macro['phish'].append(best[1])
        print(f"  [seed {rs}] Ponzi={best[0]:.4f} Phish={best[1]:.4f}", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    return macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--restarts', type=int, nargs='+', default=[14, 21, 42, 77, 100])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--concat', type=int, default=3)
    ap.add_argument('--noise_std', type=float, default=0.1)
    ap.add_argument('--expert_mode', type=str, default='feature', choices=['feature', 'loss', 'none'])
    ap.add_argument('--raw_transform', type=str, default='none', choices=['none', 'signed_log'])
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--num_neighbors', type=int, nargs='+', default=[100, 100])
    ap.add_argument('--out', type=str, default='ea_results')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    dp = Ponzi('./data/Ponzi/')[0]; dh = Phish('./data/Phish/')[0]
    aug_p = torch.load('./pretrain_model/icvae_Ponzi.pkl', map_location=device)
    aug_h = torch.load('./pretrain_model/icvae_Phish.pkl', map_location=device)
    ns_p = NeighborSampler(dp, num_neighbors=args.num_neighbors); ns_h = NeighborSampler(dh, num_neighbors=args.num_neighbors)
    bs = dict(batch_size=args.batch_size, num_neighbors=args.num_neighbors)
    ctx = dict(device=device, dp=dp, dh=dh, aug_p=aug_p, aug_h=aug_h,
               tr_p=NeighborLoader(dp, shuffle=True,  input_nodes=('CA',  dp['CA'].train_mask), neighbor_sampler=ns_p, **bs),
               va_p=NeighborLoader(dp, shuffle=False, input_nodes=('CA',  dp['CA'].val_mask),   neighbor_sampler=ns_p, **bs),
               te_p=NeighborLoader(dp, shuffle=False, input_nodes=('CA',  dp['CA'].test_mask),  neighbor_sampler=ns_p, **bs),
               tr_h=NeighborLoader(dh, shuffle=True,  input_nodes=('EOA', dh['EOA'].train_mask), neighbor_sampler=ns_h, **bs),
               va_h=NeighborLoader(dh, shuffle=False, input_nodes=('EOA', dh['EOA'].val_mask),   neighbor_sampler=ns_h, **bs),
               te_h=NeighborLoader(dh, shuffle=False, input_nodes=('EOA', dh['EOA'].test_mask),  neighbor_sampler=ns_h, **bs))

    configs = {
        '- ICVAE (raw only)':       dict(aug='raw',      gen=True,  cont=True,  triplet=0.1),
        '- triplet':                dict(aug='icvae',    gen=True,  cont=True,  triplet=0.0),
        'Gaussian-noise aug':       dict(aug='gaussian', gen=True,  cont=True,  triplet=0.1),
        '- generative (cont only)': dict(aug='icvae',    gen=False, cont=True,  triplet=0.0),
        '- contrastive (gen only)': dict(aug='icvae',    gen=True,  cont=False, triplet=0.1),
    }
    results = {}
    for name, cfg in configs.items():
        print(f"=== {name} ===", flush=True)
        m = run_config(cfg, ctx, args)
        results[name] = {'ponzi': m['ponzi'], 'phish': m['phish']}
        with open(f"{args.out}.json", 'w') as f:
            json.dump({'args': vars(args), 'results': results}, f, indent=2)

    print("\n===== E-A RESULTS (Macro-F1 %, mean±std) =====", flush=True)
    for name, r in results.items():
        p = np.array(r['ponzi']) * 100; h = np.array(r['phish']) * 100
        print(f"{name:28s} | Ponzi {p.mean():.2f}±{p.std():.2f} | Phish {h.mean():.2f}±{h.std():.2f}", flush=True)
    print("(Full & ICVAE-interaction-cond = reuse fixed_table: Ponzi 79.66±1.76, Phish 97.24±0.64)")


if __name__ == '__main__':
    main()
