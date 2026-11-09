"""
Self-trained feature-only baselines: XGBoost & LightGBM.
Same fixed split (SPLIT_SEED=14) + same 14-dim node features as UnifiedHMSL, with the
SAME signed-log transform for fair comparison. Graph-free (features only).
Reports positive(fraud)-class P/R/Binary-F1, Micro-F1(=acc), Macro-F1, mean±std over seeds.
"""
import json, argparse, warnings
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from dataset import Ponzi, Phish

warnings.filterwarnings("ignore")


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def get_split(data, nt):
    x = data[nt].x.numpy().astype(np.float32)
    y = data[nt].y.numpy()
    tr = data[nt].train_mask.numpy(); va = data[nt].val_mask.numpy(); te = data[nt].test_mask.numpy()
    # train on train+val (baseline convention: no early stopping needed), test on test
    trmask = tr | va
    Xtr, ytr = signed_log(x[trmask]), y[trmask]
    Xte, yte = signed_log(x[te]), y[te]
    return Xtr, ytr, Xte, yte


def metrics(y, p):
    return dict(precision=precision_score(y, p, pos_label=1, zero_division=0) * 100,
                recall=recall_score(y, p, pos_label=1, zero_division=0) * 100,
                binary_f1=f1_score(y, p, pos_label=1, zero_division=0) * 100,
                micro_f1=accuracy_score(y, p) * 100,
                macro_f1=f1_score(y, p, average='macro') * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[14, 21, 42, 77, 100])
    ap.add_argument('--out', type=str, default='gbdt_baselines')
    args = ap.parse_args()

    dp = Ponzi('./data/Ponzi/')[0]; dh = Phish('./data/Phish/')[0]
    tasks = {'Ponzi': get_split(dp, 'CA'), 'Phish': get_split(dh, 'EOA')}

    def make(model_name, seed):
        if model_name == 'XGBoost':
            return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, random_state=seed,
                                 eval_metric='logloss', n_jobs=4, verbosity=0)
        return LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, random_state=seed,
                              n_jobs=4, verbose=-1)

    results = {}
    for model_name in ['XGBoost', 'LightGBM']:
        for task, (Xtr, ytr, Xte, yte) in tasks.items():
            accs = []
            for seed in args.seeds:
                clf = make(model_name, seed)
                clf.fit(Xtr, ytr)
                accs.append(metrics(yte, clf.predict(Xte)))
            agg = {k: (np.mean([a[k] for a in accs]), np.std([a[k] for a in accs]))
                   for k in accs[0]}
            results[f'{model_name}|{task}'] = agg
            c = lambda k: f"{agg[k][0]:.2f}±{agg[k][1]:.2f}"
            print(f"{model_name:9s}|{task:6s}| P {c('precision')} | R {c('recall')} | "
                  f"BinF1 {c('binary_f1')} | Mic {c('micro_f1')} | Mac {c('macro_f1')}", flush=True)
    json.dump({k: {m: list(v) for m, v in d.items()} for k, d in results.items()},
              open(f"{args.out}.json", 'w'), indent=2)
    print(f"saved {args.out}.json  (n={len(args.seeds)} seeds {args.seeds})")


if __name__ == '__main__':
    main()
