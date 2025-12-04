# probe_heads_fit.py
# -*- coding: utf-8 -*-
import argparse, json, warnings
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.exceptions import ConvergenceWarning

def choose_cv(y, requested_cv, seed):
    counts = np.bincount(y)
    counts = counts[counts > 0]
    min_c = int(counts.min()) if len(counts) else 0
    if min_c < 2:
        # 너무 적으면 KFold 대신 ShuffleSplit로 다회 샘플링
        splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
        use_cv = "SSS-5x20%"
    else:
        n_splits = max(2, min(requested_cv, min_c))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        use_cv = f"SKFold-{n_splits}"
    return splitter, use_cv, min_c

def eval_cv(X, y, splitter, max_iter=3000, C=1.0, tol=1e-3, seed=42):
    accs, f1s, qwks = [], [], []
    for tr, te in splitter.split(X, y):
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(
                solver="saga",
                max_iter=max_iter,
                C=C,
                tol=tol,
                class_weight="balanced",
                random_state=seed,
            ),
        )
        clf.fit(X[tr], y[tr])
        yp = clf.predict(X[te])
        accs.append(accuracy_score(y[te], yp))
        f1s.append(f1_score(y[te], yp, average="macro"))
        qwks.append(cohen_kappa_score(y[te], yp, weights="quadratic"))
    return dict(
        acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
        f1_mean=float(np.mean(f1s)),   f1_std=float(np.std(f1s)),
        qwk_mean=float(np.mean(qwks)), qwk_std=float(np.std(qwks)),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--cv", type=int, default=5, help="요청 폴드 수(자동 한도 적용)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--also-concat", action="store_true")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--suppress-warnings", action="store_true")
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    if args.suppress_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    z = np.load(args.npz, allow_pickle=True)
    X, y = z["X"], z["y"]
    meta = z["meta"].item()
    N, H, D = X.shape
    print(f"Loaded: X={X.shape}, y={y.shape}, meta={meta}")

    splitter, use_cv, min_c = choose_cv(y, args.cv, args.seed)
    print(f"[info] CV mode: {use_cv} (min_class_count={min_c}, requested={args.cv})")

    # 1) Head-wise
    rows = []
    for h in range(H):
        res = eval_cv(
            X[:, h, :], y, splitter,
            max_iter=args.max_iter, C=args.C, tol=args.tol, seed=args.seed
        )
        rows.append(dict(head=h, **res))
    rows.sort(key=lambda r: r["qwk_mean"], reverse=True)

    print("\n== Head-wise (top K by QWK) ==")
    for r in rows[:args.topk]:
        print(f"head {r['head']:>2d} | QWK {r['qwk_mean']:.3f}±{r['qwk_std']:.3f} "
              f"| F1 {r['f1_mean']:.3f}±{r['f1_std']:.3f} "
              f"| Acc {r['acc_mean']:.3f}±{r['acc_std']:.3f}")

    out = dict(meta=meta, headwise=rows, cv_mode=use_cv, min_class_count=min_c)

    # 2) Concat (옵션)
    if args.also_concat:
        Xc = X.reshape(N, H*D)
        res_c = eval_cv(Xc, y, splitter,
                        max_iter=args.max_iter, C=args.C, tol=args.tol, seed=args.seed)
        print("\n== Concat (all heads) ==")
        print(json.dumps(res_c, indent=2))
        out["concat"] = res_c

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()