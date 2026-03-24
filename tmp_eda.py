import torch
import sys
import os.path as osp
from dataset import Ponzi, Phish

def analyze(data, node_type):
    x = data[node_type].x
    y = data[node_type].y
    
    # We only care about labeled nodes
    mask = (y == 0) | (y == 1)
    x = x[mask]
    y = y[mask]
    
    pos_x = x[y == 1]
    neg_x = x[y == 0]
    
    print(f"[{node_type}] Total features: {x.shape[1]} | Pos: {pos_x.shape[0]} | Neg: {neg_x.shape[0]}")
    for i in range(x.shape[1]):
        pos_mean = pos_x[:, i].mean().item()
        neg_mean = neg_x[:, i].mean().item()
        
        # calculate quartiles instead of just median for robust skew analysis
        pos_q25 = torch.quantile(pos_x[:, i], 0.25).item()
        pos_q50 = torch.quantile(pos_x[:, i], 0.50).item()
        pos_q75 = torch.quantile(pos_x[:, i], 0.75).item()
        
        neg_q25 = torch.quantile(neg_x[:, i], 0.25).item()
        neg_q50 = torch.quantile(neg_x[:, i], 0.50).item()
        neg_q75 = torch.quantile(neg_x[:, i], 0.75).item()
        
        pos_std = pos_x[:, i].std().item()
        neg_std = neg_x[:, i].std().item()
        
        # Print features with significant structural differences
        diff = abs(pos_mean - neg_mean) / (pos_std + neg_std + 1e-8)
        med_diff = abs(pos_q50 - neg_q50)
        
        if diff > 0.4 or med_diff > 0.5 or (pos_q75 < neg_q25) or (pos_q25 > neg_q75):
            print(f"Feature {i:02d}: ")
            print(f"  Pos -> Mean={pos_mean:.2f}, Q=[{pos_q25:.2f}, {pos_q50:.2f}, {pos_q75:.2f}]")
            print(f"  Neg -> Mean={neg_mean:.2f}, Q=[{neg_q25:.2f}, {neg_q50:.2f}, {neg_q75:.2f}]")
            print(f"  Score: Diff={diff:.2f}")

try:
    print("=== Ponzi ===")
    dataset_ponzi = Ponzi('./data/Ponzi/')
    analyze(dataset_ponzi[0], 'CA')

    print("\n=== Phish ===")
    dataset_phish = Phish('./data/Phish/')
    analyze(dataset_phish[0], 'EOA')
except Exception as e:
    import traceback
    traceback.print_exc()
