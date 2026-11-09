"""
5-fold CV producing the FULL paper-format metric set for the Full model, so our
row is column-comparable with the paper table:
  Precision / Recall / Binary-F1  = POSITIVE (fraud) class
  Micro-F1                        = accuracy
  Macro-F1                        = macro average
Reports mean±std (×100) per task. No leakage (stratified K-fold, node tested once).
"""
import json
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler
from dataset import Ponzi, Phish
from unified_model import UnifiedHMSL
from run_unified import get_augmented_data
from run_cv import labeled_indices, mask_from
from utils import seed_everything

warnings.filterwarnings("ignore")


@torch.no_grad()
def eval_full(model, loader, aug_model, target_node, aug_args, device):
    import numpy as np
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 balanced_accuracy_score, confusion_matrix)
    model.eval()
    preds, labels, probs = [], [], []
    for batch in loader:
        batch = get_augmented_data(batch.to(device), aug_model, aug_args, device)
        bs = batch[target_node].batch_size
        op, oh, *_ = model(batch.x_dict_new, batch.edge_index_dict, raw_x_dict=batch.x_dict)
        out = op if target_node == 'CA' else oh
        prob1 = torch.softmax(out[:bs], dim=1)[:, 1].cpu().numpy()
        preds.extend(out[:bs].argmax(1).cpu().numpy())
        probs.extend(prob1)
        labels.extend(batch[target_node].y[:bs].cpu().numpy())
    labels = np.array(labels); preds = np.array(preds); probs = np.array(probs)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # AUCs need both classes present
    both = len(np.unique(labels)) > 1
    return dict(
        precision=precision_score(labels, preds, pos_label=1, zero_division=0),  # fraud class
        recall=recall_score(labels, preds, pos_label=1, zero_division=0),
        binary_f1=f1_score(labels, preds, pos_label=1, zero_division=0),  # = fraud-class F1
        micro_f1=accuracy_score(labels, preds),
        macro_f1=f1_score(labels, preds, average='macro'),
        pr_auc=average_precision_score(labels, probs) if both else 0.0,
        roc_auc=roc_auc_score(labels, probs) if both else 0.0,
        balanced_acc=balanced_accuracy_score(labels, preds),
        fpr=fpr,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--cv_seed', type=int, default=42)
    ap.add_argument('--train_seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--loss_train', type=float, default=0.1)
    ap.add_argument('--concat', type=int, default=3)
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--num_neighbors', type=int, nargs='+', default=[100, 100])
    ap.add_argument('--val_frac', type=float, default=0.25)
    ap.add_argument('--use_cpa', type=int, default=1)
    ap.add_argument('--use_tcg', type=int, default=1)
    ap.add_argument('--expert_mode', type=str, default='feature', choices=['feature', 'loss', 'none'])
    ap.add_argument('--raw_transform', type=str, default='none', choices=['none', 'signed_log'])
    ap.add_argument('--out', type=str, default='cv_table')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    dp = Ponzi('./data/Ponzi/')[0]
    dh = Phish('./data/Phish/')[0]
    aug_p = torch.load('./pretrain_model/icvae_Ponzi.pkl', map_location=device)
    aug_h = torch.load('./pretrain_model/icvae_Phish.pkl', map_location=device)
    ns_p = NeighborSampler(dp, num_neighbors=args.num_neighbors)
    ns_h = NeighborSampler(dh, num_neighbors=args.num_neighbors)
    ca_idx, ca_y = labeled_indices(dp, 'CA')
    eoa_idx, eoa_y = labeled_indices(dh, 'EOA')
    n_ca, n_eoa = dp['CA'].y.size(0), dh['EOA'].y.size(0)

    ca_folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=args.cv_seed).split(ca_idx, ca_y))
    eoa_folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=args.cv_seed).split(eoa_idx, eoa_y))
    bs = dict(batch_size=args.batch_size, num_neighbors=args.num_neighbors)
    aug_args = argparse.Namespace(concat=args.concat)

    fold_rows = []
    for k in range(args.folds):
        ca_trp, ca_te = ca_folds[k]; eoa_trp, eoa_te = eoa_folds[k]
        ca_tr, ca_va = train_test_split(ca_trp, test_size=args.val_frac, random_state=args.cv_seed, stratify=ca_y[ca_trp])
        eoa_tr, eoa_va = train_test_split(eoa_trp, test_size=args.val_frac, random_state=args.cv_seed, stratify=eoa_y[eoa_trp])
        tr_p = NeighborLoader(dp, shuffle=True,  input_nodes=('CA',  mask_from(ca_idx[ca_tr], n_ca)),  neighbor_sampler=ns_p, **bs)
        va_p = NeighborLoader(dp, shuffle=False, input_nodes=('CA',  mask_from(ca_idx[ca_va], n_ca)),  neighbor_sampler=ns_p, **bs)
        te_p = NeighborLoader(dp, shuffle=False, input_nodes=('CA',  mask_from(ca_idx[ca_te], n_ca)),  neighbor_sampler=ns_p, **bs)
        tr_h = NeighborLoader(dh, shuffle=True,  input_nodes=('EOA', mask_from(eoa_idx[eoa_tr], n_eoa)), neighbor_sampler=ns_h, **bs)
        va_h = NeighborLoader(dh, shuffle=False, input_nodes=('EOA', mask_from(eoa_idx[eoa_va], n_eoa)), neighbor_sampler=ns_h, **bs)
        te_h = NeighborLoader(dh, shuffle=False, input_nodes=('EOA', mask_from(eoa_idx[eoa_te], n_eoa)), neighbor_sampler=ns_h, **bs)

        seed_everything(args.train_seed)
        model = UnifiedHMSL(hidden=args.hidden, out_channels=2, data=dp, concat=args.concat,
                            expert_mode=args.expert_mode, raw_transform=args.raw_transform,
                            use_cpa=bool(args.use_cpa), use_tcg=bool(args.use_tcg)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        best_val, best = -1.0, None; patience = 0
        for epoch in range(args.epochs):
            model.train()
            for bp, bh in zip(tr_p, tr_h):
                opt.zero_grad()
                bp = get_augmented_data(bp.to(device), aug_p, aug_args, device)
                op, _, co_p, _, _ = model(bp.x_dict_new, bp.edge_index_dict, raw_x_dict=bp.x_dict)
                sp = bp['CA'].batch_size
                bh = get_augmented_data(bh.to(device), aug_h, aug_args, device)
                _, oh, co_h, _, _ = model(bh.x_dict_new, bh.edge_index_dict, raw_x_dict=bh.x_dict)
                sh = bh['EOA'].batch_size
                loss = (F.cross_entropy(op[:sp], bp['CA'].y[:sp]) + F.cross_entropy(oh[:sh], bh['EOA'].y[:sh])
                        + args.loss_train * (co_p + co_h))
                loss.backward(); opt.step()
            vp = eval_full(model, va_p, aug_p, 'CA', aug_args, device)['macro_f1']
            vh = eval_full(model, va_h, aug_h, 'EOA', aug_args, device)['macro_f1']
            if vp + vh > best_val:
                best_val = vp + vh
                best = dict(ponzi=eval_full(model, te_p, aug_p, 'CA', aug_args, device),
                            phish=eval_full(model, te_h, aug_h, 'EOA', aug_args, device))
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    break
        fold_rows.append(best)
        pz, ph = best['ponzi'], best['phish']
        print(f"[fold {k}] Ponzi macroF1={pz['macro_f1']:.4f} binF1={pz['binary_f1']:.4f} "
              f"P={pz['precision']:.4f} R={pz['recall']:.4f} micro={pz['micro_f1']:.4f} | "
              f"Phish macroF1={ph['macro_f1']:.4f} binF1={ph['binary_f1']:.4f}", flush=True)
        del model; import gc; gc.collect(); torch.cuda.empty_cache()
        with open(f"{args.out}.json", 'w') as f:
            json.dump({'args': vars(args), 'folds': fold_rows}, f, indent=2)

    def agg(task, key):
        v = np.array([r[task][key] for r in fold_rows]) * 100
        return v.mean(), v.std()

    print("\n===== OUR ROW (paper format, mean±std ×100) =====", flush=True)
    order = [('precision', 'Precision'), ('recall', 'Recall'), ('binary_f1', 'Binary-F1'),
             ('micro_f1', 'Micro-F1'), ('macro_f1', 'Macro-F1')]
    for task, tag in [('ponzi', 'Ponzi'), ('phish', 'Phish')]:
        cells = [f"{tag}"]
        for key, name in order:
            m, s = agg(task, key)
            cells.append(f"{name}={m:.2f}±{s:.2f}")
        print("  " + "  ".join(cells), flush=True)

    # emit ready-to-paste values
    row = {}
    for task in ['ponzi', 'phish']:
        for key, _ in order:
            m, s = agg(task, key)
            row[f"{task}_{key}"] = [round(m, 2), round(s, 2)]
    with open(f"{args.out}_row.json", 'w') as f:
        json.dump(row, f, indent=2)
    print(f"\nSaved {args.out}.json and {args.out}_row.json", flush=True)


if __name__ == '__main__':
    main()
