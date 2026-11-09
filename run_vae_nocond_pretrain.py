"""
Pretrain a VAE WITHOUT interaction-type conditioning (E-A config #5).

Reuses the exact ICVAE architecture and pretraining recipe as icvae_pretrain.py, but
feeds a ZERO edge-type vector everywhere (same 6-dim width, but carrying no relation
information) during pretraining. At inference (run_ea.py mode 'vaenc') the edge-type is
also zeroed, so the generator is conditioned only on the node's own feature `c`, never on
which relation — a clean ablation of the interaction conditioning that ICVAE adds.

Saves ./pretrain_model/vaenc_{Ponzi,Phish}.pkl.

Usage:
  python run_vae_nocond_pretrain.py --dataset Ponzi --pretrain_epochs 3 --gpu 0
  python run_vae_nocond_pretrain.py --dataset Phish --pretrain_epochs 3 --gpu 0 --max_edges 5000000
"""
import gc, argparse, warnings
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import trange
from dataset import Ponzi, Phish
from icvae import ICVAE
from utils import mkdir, seed_everything

warnings.filterwarnings("ignore")


def loss_fn(recon_x, x, mean, log_var):
    x = torch.sigmoid(x)
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['Ponzi', 'Phish'], required=True)
    ap.add_argument('--pretrain_epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--seed', type=int, default=14)
    ap.add_argument('--max_edges', type=int, default=0, help='0 = use all edges; else cap total edges')
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'Ponzi':
        data = Ponzi('./data/Ponzi/')[0]; target = 'CA'
    else:
        data = Phish('./data/Phish/')[0]; target = 'EOA'
    n_rel = len(data.edge_types)

    x_list, c_list = [], []
    for i in trange(n_rel, desc='build features'):
        h_t = data.edge_types[i][0]; t_t = data.edge_types[i][-1]
        hi = data.edge_stores[i]['edge_index'][0]; ti = data.edge_stores[i]['edge_index'][-1]
        c_list.append(data[h_t].x[hi]); x_list.append(data[t_t].x[ti])
    fx = np.vstack(x_list).astype(np.float32); fc = np.vstack(c_list).astype(np.float32)
    del x_list, c_list; gc.collect()
    if args.max_edges and fx.shape[0] > args.max_edges:
        idx = np.random.choice(fx.shape[0], args.max_edges, replace=False)
        fx, fc = fx[idx], fc[idx]
        print(f'subsampled edges -> {fx.shape[0]}')
    fx = torch.tensor(fx); fc = torch.tensor(fc)
    fe = torch.zeros((fx.shape[0], n_rel), dtype=torch.float32)   # ZERO edge-type (no interaction cond)
    print(f'{args.dataset}: {fx.shape[0]} edge-samples, feat dim {fx.shape[1]}')

    ds = TensorDataset(fx, fc, fe)
    dl = DataLoader(ds, sampler=RandomSampler(ds), batch_size=args.batch_size)
    vae = ICVAE(encoder_layer_sizes=[data[target].x.shape[1], 64], latent_size=50,
                decoder_layer_sizes=[64, data[target].x.shape[1]],
                edge_type_size=data.edge_types, conditional_size=data[target].x.shape[1]).to(device)
    opt = optim.Adam(vae.parameters(), lr=0.01)
    for _ in trange(args.pretrain_epochs, desc='pretrain VAE-nocond'):
        for x, c, e in dl:
            vae.train()
            x, c, e = x.to(device), c.to(device), e.to(device)
            recon, mean, log_var, _ = vae(x, c, e)
            l = loss_fn(recon, x, mean, log_var)
            opt.zero_grad(); l.backward(); opt.step()
    out = mkdir('./pretrain_model/') + f'vaenc_{args.dataset}.pkl'
    torch.save(vae, out)
    print('saved', out)


if __name__ == '__main__':
    main()
