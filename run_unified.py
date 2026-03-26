import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import trange
import os.path as osp
from utils import get_parser, one_hot, mkdir, feature_tensor_normalize
from torch_geometric.loader import NeighborLoader
from dataset import Ponzi, Phish
from unified_model import UnifiedHMSL
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

def get_augmented_features(data, dataset_name, device):
    CA_ls = []
    EOA_ls = []
    cvae_model = torch.load(
        f"./pretrain_model/icvae_{dataset_name}.pkl",
        map_location=device)
    e_type_index = {}
    for index, e in enumerate(data.edge_types):
        e_type_index[e] = index
    for i in range(6):
        src_node = data.edge_types[i][0]
        edge_onehot = one_hot(torch.LongTensor([e_type_index[data.edge_types[i]]]), len(e_type_index)).repeat(
            data[f'{src_node}'].num_nodes, 1).to(device)
        z = torch.randn([data[f'{src_node}'].num_nodes, cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, data[f'{src_node}'].x, edge_onehot).detach()
        augmented_features = feature_tensor_normalize(augmented_features).detach()
        if src_node == 'CA':
            CA_ls.append(augmented_features)
        else:
            EOA_ls.append(augmented_features)
    del cvae_model
    return CA_ls, EOA_ls

def get_augmented_data(data, dataset_name, args, device):
    data.x_dict_new = data.x_dict
    data.x_dict_new['EOA'] = []
    data.x_dict_new['CA'] = []
    for _ in range(args.concat):
        CA_x, EOA_x = get_augmented_features(data, dataset_name, device)
        data.x_dict_new['EOA'].append(EOA_x)
        data.x_dict_new['EOA'][_].append(data.x_dict['EOA'])
        data.x_dict_new['CA'].append(CA_x)
        data.x_dict_new['CA'][_].append(data.x_dict['CA'])
        data.x_dict_new['CA'][_] = torch.stack(data.x_dict_new["CA"][_])
        data.x_dict_new['EOA'][_] = torch.stack(data.x_dict_new["EOA"][_])
    return data

def train_step(model, optimizer, ponzi_loader, phish_loader, args, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Iterate through both loaders
    # For simplicity in this version, we alternate or zip them.
    # If they have different lengths, we cycle the shorter one or stop at the shortest.
    for batch_ponzi, batch_phish in zip(ponzi_loader, phish_loader):
        optimizer.zero_grad()
        
        # Move to device and augment
        batch_ponzi = get_augmented_data(batch_ponzi.to(device), 'Ponzi', args, device)
        batch_phish = get_augmented_data(batch_phish.to(device), 'Phish', args, device)
        
        # Forward pass
        # Ponzi Head task
        out_ponzi_p, out_phish_p, loss_co_p, expert_ponzi_p, expert_phish_p = model(batch_ponzi.x_dict_new, batch_ponzi.edge_index_dict, raw_x_dict=batch_ponzi.x_dict)
        size_ponzi = batch_ponzi['CA'].batch_size
        loss_ponzi = F.cross_entropy(out_ponzi_p[:size_ponzi], batch_ponzi['CA'].y[:size_ponzi])
        
        # Phish Head task
        out_ponzi_h, out_phish_h, loss_co_h, expert_ponzi_h, expert_phish_h = model(batch_phish.x_dict_new, batch_phish.edge_index_dict, raw_x_dict=batch_phish.x_dict)
        size_phish = batch_phish['EOA'].batch_size
        loss_phish = F.cross_entropy(out_phish_h[:size_phish], batch_phish['EOA'].y[:size_phish])
        
        # Combined Loss
        loss = loss_ponzi + loss_phish + args.loss_train * (loss_co_p + loss_co_h)

        if args.expert_mode == 'loss':
            prob_ponzi_p = torch.softmax(out_ponzi_p[:size_ponzi], dim=1)[:, 1]
            loss_expert_p = F.mse_loss(prob_ponzi_p, expert_ponzi_p[:size_ponzi].squeeze(-1))
            
            prob_phish_h = torch.softmax(out_phish_h[:size_phish], dim=1)[:, 1]
            loss_expert_h = F.mse_loss(prob_phish_h, expert_phish_h[:size_phish].squeeze(-1))
            
            loss += 0.5 * (loss_expert_p + loss_expert_h)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / num_batches if num_batches > 0 else 0

@torch.no_grad()
def evaluate(model, loader, target_node, task_type, args, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    total_examples = 0
    
    predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()

    for batch in loader:
        batch = get_augmented_data(batch.to(device), task_type, args, device)
        batch_size = batch[target_node].batch_size
        out_ponzi, out_phish, loss_co, expert_ponzi, expert_phish = model(batch.x_dict_new, batch.edge_index_dict, raw_x_dict=batch.x_dict)
        
        output = out_ponzi if target_node == 'CA' else out_phish
        
        y = batch[target_node].y[:batch_size]
        
        loss = F.cross_entropy(output[:batch_size], y) + args.loss_train * loss_co
        
        if args.expert_mode == 'loss':
            expert_out = expert_ponzi if target_node == 'CA' else expert_phish
            prob = torch.softmax(output[:batch_size], dim=1)[:, 1]
            loss_expert = F.mse_loss(prob, expert_out[:batch_size].squeeze(-1))
            loss += 0.5 * loss_expert
            
        total_loss += loss.item()
        total_examples += 1
        
        preds = predict_fn(F.log_softmax(output[:batch_size], dim=1)).numpy()
        labels = y.detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
        
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    return total_loss / total_examples, f1_macro

if __name__ == '__main__':
    args = get_parser()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Load Datasets
    path_ponzi = osp.join(osp.dirname(osp.realpath(__file__)), './data/Ponzi/')
    dataset_ponzi = Ponzi(path_ponzi)
    
    path_phish = osp.join(osp.dirname(osp.realpath(__file__)), './data/Phish/')
    dataset_phish = Phish(path_phish)
    
    data_ponzi = dataset_ponzi[0]
    data_phish = dataset_phish[0]
    
    # Loaders
    kwargs = {'batch_size': args.batch_size}
    train_loader_ponzi = NeighborLoader(data_ponzi, num_neighbors=[100] * 2, shuffle=True,
                                        input_nodes=('CA', data_ponzi['CA'].train_mask), **kwargs)
    val_loader_ponzi = NeighborLoader(data_ponzi, num_neighbors=[100] * 2, shuffle=False,
                                      input_nodes=('CA', data_ponzi['CA'].val_mask), **kwargs)
    
    train_loader_phish = NeighborLoader(data_phish, num_neighbors=[100] * 2, shuffle=True,
                                        input_nodes=('EOA', data_phish['EOA'].train_mask), **kwargs)
    val_loader_phish = NeighborLoader(data_phish, num_neighbors=[100] * 2, shuffle=False,
                                      input_nodes=('EOA', data_phish['EOA'].val_mask), **kwargs)


    test_loader_ponzi = NeighborLoader(data_ponzi, num_neighbors=[100] * 2, shuffle=False,
                                        input_nodes=('CA', data_ponzi['CA'].test_mask), **kwargs)
    test_loader_phish = NeighborLoader(data_phish, num_neighbors=[100] * 2, shuffle=False,
                                        input_nodes=('EOA', data_phish['EOA'].test_mask), **kwargs)
    # Note: Using metadata from one dataset (assuming types are similar)
    model = UnifiedHMSL(hidden=args.hidden, out_channels=2, data=data_ponzi, concat=args.concat, expert_mode=args.expert_mode)
    model = model.to(device)
    
    print("--- Unified Multi-task Training Started ---")
    
    best_ponzi_f1 = 0
    best_phish_f1 = 0
    
    best_val_f1_sum = 0
    best_model_state = None

    save_path = f"best_unified_model_{args.expert_mode}.pth"

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    for epoch in range(args.epochs):
        loss_train = train_step(model, optimizer, train_loader_ponzi, train_loader_phish, args, device)
        
        loss_val_ponzi, f1_ponzi = evaluate(model, val_loader_ponzi, 'CA', 'Ponzi', args, device)
        loss_val_phish, f1_phish = evaluate(model, val_loader_phish, 'EOA', 'Phish', args, device)
        
        # Lưu model nếu tổng F1 của cả 2 task là tốt nhất (đảm bảo cân bằng 2 task)
        current_val_f1 = f1_ponzi + f1_phish
        if current_val_f1 > best_val_f1_sum:
            best_val_f1_sum = current_val_f1
            torch.save(model.state_dict(), save_path)
            print(f"--- Saved Best Model to {save_path} ---")
            best_model_state = copy.deepcopy(model.state_dict())
            best_ponzi_f1 = f1_ponzi
            best_phish_f1 = f1_phish
            print(f"--- Best Model Updated at Epoch {epoch} ---")
        print(f"Epoch {epoch:03d} | Loss: {loss_train:.4f} | Ponzi Val F1: {f1_ponzi:.4f} | Phish Val F1: {f1_phish:.4f}")
    # --- ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST ---
    print("\n--- Final Evaluation on TEST SET ---")
    model.load_state_dict(best_model_state)
    loss_test_ponzi, f1_test_ponzi = evaluate(model, test_loader_ponzi, 'CA', 'Ponzi', args, device)
    loss_test_phish, f1_test_phish = evaluate(model, test_loader_phish, 'EOA', 'Phish', args, device)
    print(f"Test Ponzi F1: {f1_test_ponzi:.4f}")
    print(f"Test Phish F1: {f1_test_phish:.4f}")
    