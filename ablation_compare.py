"""
Module ablation across the 3 brought-back checkpoints.
Each checkpoint is loaded into ITS matching architecture:
  - best_unified_model.pth           -> legacy UnifiedHMSL, expert_mode='none'   (Base)
  - best_unified_model_feature.pth   -> legacy UnifiedHMSL, expert_mode='feature' (+Expert)
  - best_unified_model_feature_icvae.pth -> current UnifiedHMSL, expert_mode='feature' (+CPA/TCG)
Same test split + same RNG seed for CVAE augmentation => as-fair-as-possible compare.
"""
import torch, warnings, copy
warnings.filterwarnings("ignore")
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from dataset import Ponzi, Phish
from run_unified import get_augmented_data
import unified_model as cur
import unified_model_legacy as leg

DEVICE = "cpu"
NEI = [15, 15]
BATCH = 128
SEED = 14

class Args: pass
args = Args(); args.concat = 3

@torch.no_grad()
def evaluate(model, loader, aug_model, target_node):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = get_augmented_data(batch.to(DEVICE), aug_model, args, DEVICE)
        bs = batch[target_node].batch_size
        out_ponzi, out_phish, *_ = model(batch.x_dict_new, batch.edge_index_dict, raw_x_dict=batch.x_dict)
        out = out_ponzi if target_node == "CA" else out_phish
        preds.extend(out[:bs].argmax(1).cpu().numpy())
        labels.extend(batch[target_node].y[:bs].cpu().numpy())
    return {
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
    }

def build(model_mod, expert_mode, data, ckpt):
    m = model_mod.UnifiedHMSL(hidden=128, out_channels=2, data=data, concat=3, expert_mode=expert_mode).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return m

def main():
    print("loading data + ICVAE...", flush=True)
    dp = Ponzi("./data/Ponzi/")[0]; dh = Phish("./data/Phish/")[0]
    ap = torch.load("./pretrain_model/icvae_Ponzi.pkl", map_location=DEVICE)
    ah = torch.load("./pretrain_model/icvae_Phish.pkl", map_location=DEVICE)

    configs = [
        ("Base (unified)",        leg, "none",    "best_unified_model.pth"),
        ("+ Expert (feature)",    leg, "feature", "best_unified_model_feature.pth"),
        ("+ CPA/TCG (full)",      cur, "feature", "best_unified_model_feature_icvae.pth"),
    ]

    rows = []
    for name, mod, mode, ckpt in configs:
        print(f"\n=== {name}  [{ckpt}] ===", flush=True)
        try:
            mp = build(mod, mode, dp, ckpt)
            mh = build(mod, mode, dh, ckpt)
        except Exception as e:
            print(f"  LOAD FAILED: {e}", flush=True); continue
        # same seed => identical stochastic augmentation across models
        torch.manual_seed(SEED)
        tp = NeighborLoader(dp, num_neighbors=NEI, shuffle=False, input_nodes=("CA", dp["CA"].test_mask), batch_size=BATCH)
        rp = evaluate(mp, tp, ap, "CA")
        torch.manual_seed(SEED)
        th = NeighborLoader(dh, num_neighbors=NEI, shuffle=False, input_nodes=("EOA", dh["EOA"].test_mask), batch_size=BATCH)
        rh = evaluate(mh, th, ah, "EOA")
        rows.append((name, rp, rh))
        print(f"  Ponzi  F1={rp['f1']:.4f} P={rp['precision']:.4f} R={rp['recall']:.4f}", flush=True)
        print(f"  Phish  F1={rh['f1']:.4f} P={rh['precision']:.4f} R={rh['recall']:.4f}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'Configuration':<22} | {'Ponzi F1':>8} {'Ponzi P':>8} {'Ponzi R':>8} | {'Phish F1':>8} {'Phish P':>8} {'Phish R':>8}")
    print("-" * 78)
    for name, rp, rh in rows:
        print(f"{name:<22} | {rp['f1']:>8.4f} {rp['precision']:>8.4f} {rp['recall']:>8.4f} | {rh['f1']:>8.4f} {rh['precision']:>8.4f} {rh['recall']:>8.4f}")
    print("=" * 78)
    print("DONE")

if __name__ == "__main__":
    main()
