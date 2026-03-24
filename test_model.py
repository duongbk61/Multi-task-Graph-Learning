import torch
import os.path as osp
import numpy as np
from torch_geometric.loader import NeighborLoader
from dataset import Ponzi, Phish
from unified_model import UnifiedHMSL
from model_old import HMSL # Model đơn nhiệm
from utils import get_parser, one_hot, feature_tensor_normalize
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.nn.functional as F

# --- Hàm bổ trợ Giao diện ---
def get_augmented_features(data, dataset_name, device):
    # Load CVAE model corresponding to the dataset
    cvae_model = torch.load(f"./pretrain_model/icvae_{dataset_name}.pkl", map_location=device)
    cvae_model.eval()
    
    CA_ls = []
    EOA_ls = []
    e_type_index = {e: index for index, e in enumerate(data.edge_types)}
    
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

@torch.no_grad()
def evaluate_model(model, loader, target_node, task_type, args, device, is_unified=True):
    model.eval()
    all_preds = []
    all_labels = []
    predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()

    for batch in loader:
        batch = get_augmented_data(batch.to(device), task_type, args, device)
        batch_size = batch[target_node].batch_size
        
        if is_unified:
            out_ponzi, out_phish, _ = model(batch.x_dict_new, batch.edge_index_dict)
            output = out_ponzi if target_node == 'CA' else out_phish
        else:
            output, _ = model(batch.x_dict_new, batch.edge_index_dict)
            
        y = batch[target_node].y[:batch_size]
        preds = predict_fn(F.log_softmax(output[:batch_size], dim=1)).numpy()
        labels = y.detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
        
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    
    return {
        'f1': f1_macro,
        'precision': precision,
        'recall': recall
    }

def run_comparison():
    args = get_parser()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    path_ponzi = osp.join(osp.dirname(osp.realpath(__file__)), './data/Ponzi/')
    data_ponzi = Ponzi(path_ponzi)[0]
    
    path_phish = osp.join(osp.dirname(osp.realpath(__file__)), './data/Phish/')
    data_phish = Phish(path_phish)[0]

    kwargs = {'batch_size': args.batch_size}
    test_loader_ponzi = NeighborLoader(data_ponzi, num_neighbors=[100] * 2, shuffle=False,
                                        input_nodes=('CA', data_ponzi['CA'].test_mask), **kwargs)
    test_loader_phish = NeighborLoader(data_phish, num_neighbors=[100] * 2, shuffle=False,
                                        input_nodes=('EOA', data_phish['EOA'].test_mask), **kwargs)

    results = {}

    # 2. Test Single Models
    print("\n--- Testing Single-task Models ---")
    
    # Ponzi Single
    if osp.exists("best_single_model_Ponzi.pth"):
        model_s_ponzi = HMSL(hidden=args.hidden, out_channels=2, data=data_ponzi, concat=args.concat, target_node='CA').to(device)
        model_s_ponzi.load_state_dict(torch.load("best_single_model_Ponzi.pth", map_location=device))
        metrics = evaluate_model(model_s_ponzi, test_loader_ponzi, 'CA', 'Ponzi', args, device, is_unified=False)
        results['Single_Ponzi'] = metrics
        print(f"Single Task Ponzi - F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    else:
        print("Single Ponzi model not found.")

    # Phish Single
    if osp.exists("best_single_model_Phish.pth"):
        model_s_phish = HMSL(hidden=args.hidden, out_channels=2, data=data_phish, concat=args.concat, target_node='EOA').to(device)
        model_s_phish.load_state_dict(torch.load("best_single_model_Phish.pth", map_location=device))
        metrics = evaluate_model(model_s_phish, test_loader_phish, 'EOA', 'Phish', args, device, is_unified=False)
        results['Single_Phish'] = metrics
        print(f"Single Task Phish - F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    else:
        print("Single Phish model not found.")

    # 3. Test Unified Model
    print("\n--- Testing Unified Model ---")
    if osp.exists("best_unified_model.pth"):
        model_u = UnifiedHMSL(hidden=args.hidden, out_channels=2, data=data_ponzi, concat=args.concat).to(device)
        model_u.load_state_dict(torch.load("best_unified_model.pth", map_location=device))
        
        m_p = evaluate_model(model_u, test_loader_ponzi, 'CA', 'Ponzi', args, device, is_unified=True)
        m_h = evaluate_model(model_u, test_loader_phish, 'EOA', 'Phish', args, device, is_unified=True)
        
        results['Unified_Ponzi'] = m_p
        results['Unified_Phish'] = m_h
        print(f"Unified Ponzi - F1: {m_p['f1']:.4f}, Precision: {m_p['precision']:.4f}, Recall: {m_p['recall']:.4f}")
        print(f"Unified Phish - F1: {m_h['f1']:.4f}, Precision: {m_h['precision']:.4f}, Recall: {m_h['recall']:.4f}")
    else:
        print("Unified model not found.")

    # 4. Summary Table
    print("\n" + "="*60)
    print(f"{'Metric':<10} | {'Ponzi-S':<10} | {'Ponzi-U':<10} | {'Phish-S':<10} | {'Phish-U':<10}")
    print("-" * 60)
    
    for metric_name in ['f1', 'precision', 'recall']:
        p_s = results.get('Single_Ponzi', {}).get(metric_name, 0.0)
        p_u = results.get('Unified_Ponzi', {}).get(metric_name, 0.0)
        h_s = results.get('Single_Phish', {}).get(metric_name, 0.0)
        h_u = results.get('Unified_Phish', {}).get(metric_name, 0.0)
        
        print(f"{metric_name.upper():<10} | {p_s:<10.4f} | {p_u:<10.4f} | {h_s:<10.4f} | {h_u:<10.4f}")
    print("="*60)

if __name__ == '__main__':
    run_comparison()
