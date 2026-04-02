import gc
import os.path as osp
import sys
import warnings
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import trange
from dataset import Ponzi, Phish
from diffusion import Diffuser
from utils import get_parser, one_hot, mkdir

warnings.filterwarnings("ignore")
exc_path = sys.path[0]


def generated_generator_sixedges_ddpm(args, data, device, target_node):
    x_list, c_list, category_list = [], [], []
    e_type_index = {}
    for index, e in enumerate(data.edge_types):
        e_type_index[e] = index
    for i in trange(len(data.edge_types)):
        head_node_type = data.edge_types[i][0]
        tail_node_type = data.edge_types[i][-1]
        head_node_index = data.edge_stores[i]['edge_index'][0]
        tail_node_index = data.edge_stores[i]['edge_index'][-1]
        c_list.append(data[head_node_type].x[head_node_index])
        x_list.append(data[tail_node_type].x[tail_node_index])
        edge_onehot = one_hot(torch.LongTensor([e_type_index[data.edge_types[i]]]), len(e_type_index),
                              dtype=float).repeat(
            len(head_node_index), 1)
        category_list.append(edge_onehot)
    features_x = np.vstack(x_list).astype(np.float32)
    features_c = np.vstack(c_list).astype(np.float32)
    features_e_type = np.vstack(category_list).astype(np.float32)
    del x_list
    del c_list
    del category_list
    gc.collect()
    features_x = torch.tensor(features_x, dtype=torch.float32)
    features_c = torch.tensor(features_c, dtype=torch.float32)
    features_e_type = torch.tensor(features_e_type, dtype=torch.float32)
    
    diffusion_dataset = TensorDataset(features_x, features_c, features_e_type)
    diffusion_dataset_sampler = RandomSampler(diffusion_dataset)
    diffusion_dataset_dataloader = DataLoader(diffusion_dataset, sampler=diffusion_dataset_sampler, batch_size=args.batch_size)

    # Pretrain Diffusion
    x_dim = data[target_node].x.shape[1]
    e_dim = len(data.edge_types)
    c_dim = data[target_node].x.shape[1]
    
    ddpm = Diffuser(x_dim=x_dim, cond_dim=c_dim, edge_type_dim=e_dim, num_steps=100)
    ddpm_optimizer = optim.Adam(ddpm.parameters(), lr=0.001)
    ddpm.to(device)

    for _ in trange(args.pretrain_epochs, desc='Run DDPM Train'):
        for _, (x, c, e) in enumerate(diffusion_dataset_dataloader):
            ddpm.train()
            x, c, e = x.to(device), c.to(device), e.to(device)
            loss = ddpm(x, c, e)
            ddpm_optimizer.zero_grad()
            loss.backward()
            ddpm_optimizer.step()
            
    return ddpm


if __name__ == '__main__':
    args = get_parser()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if 'Ponzi' in args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)), f'./data/Ponzi')
        dataset = Ponzi(path)
        data = dataset[0]
        print(data)
        target_node = 'CA'
    elif 'Phish' in args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)), f'./data/Phish')
        dataset = Phish(path)
        data = dataset[0]
        print(data)
        target_node = 'EOA'
    e_type_index = {}
    for index, e in enumerate(data.edge_types):
        e_type_index[e] = index

    model = generated_generator_sixedges_ddpm(args, data, device, target_node)
    if 'Ponzi' in args.dataset:
        torch.save(model, mkdir(f"./pretrain_model/") + f'diffusion_Ponzi.pkl')
    elif 'Phish' in args.dataset:
        torch.save(model, mkdir(f"./pretrain_model/") + f'diffusion_Phish.pkl')
