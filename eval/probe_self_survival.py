import argparse, json
from pathlib import Path
import numpy as np


def auc_from_scores(y_true, scores):
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float32)
    if y.size < 8:
        return float('nan')
    pos = (y == 1)
    neg = (y == 0)
    n_pos = int(pos.sum()); n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(1, y.size + 1, dtype=np.float32)
    # average ranks on ties
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for ui, c in enumerate(counts):
            if c <= 1:
                continue
            idxs = np.where(inv == ui)[0]
            ranks[idxs] = float(ranks[idxs].mean())
    sum_ranks_pos = float(ranks[pos].sum())
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg))


def ridge_score_probe(X, y, l2=1e-2, seed=0):
    """Fit ridge regression to y in {0,1} and return scores on test split."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n = X.shape[0]
    if n < 20:
        return None
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    k = int(0.7 * n)
    tr = idx[:k]; te = idx[k:]
    Xtr = X[tr]; ytr = y[tr]
    Xte = X[te]; yte = y[te]
    # center for stability
    mu = Xtr.mean(axis=0, keepdims=True)
    Xtr = Xtr - mu
    Xte = Xte - mu
    XtX = (Xtr.T @ Xtr).astype(np.float32)
    XtX.flat[::XtX.shape[0] + 1] += float(l2)
    w = np.linalg.solve(XtX, (Xtr.T @ ytr).astype(np.float32))
    scores = (Xte @ w).astype(np.float32)
    return yte, scores


def load_features(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r.get('ep', 0)))
    return rows


def build_labels(rows, K, S_fail, D_fail, cap_fail, integ_fail, scar_fail):
    n = len(rows)
    # per-episode fail
    fail = np.zeros((n,), dtype=np.int32)
    for i, r in enumerate(rows):
        sw = int(r.get('switches', 0))
        dr = float(r.get('self_slow_drift', 0.0))
        cap = float(r.get('capital_end', 1.0))
        integ = float(r.get('integrity_end', 1.0))
        scar = int(r.get('scar_end', 0))
        death = bool(r.get('death_end', False))
        if death or sw > S_fail or dr > D_fail or cap < cap_fail or integ < integ_fail or scar >= scar_fail:
            fail[i] = 1
    # future-window label
    Y = np.zeros((n,), dtype=np.int32)
    for i in range(n):
        j = min(n, i + int(K))
        Y[i] = 1 if fail[i:j].max(initial=0) == 1 else 0
    return Y


def build_labels_soft_quantile(rows, K, q=0.2, prefer='score'):
    """Soft label: mark bottom-q fraction of future-window quality as positive.

    - prefer='score': uses future mean of `score` (lower is worse)
    - prefer='switches': uses future mean of `switches` (higher is worse)

    This is designed to avoid degenerate labels when hard failures are rare.
    """
    n = len(rows)
    K = int(K)

    def future_mean(key, i):
        j = min(n, i + K)
        vals = []
        for t in range(i, j):
            v = rows[t].get(key, None)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        if not vals:
            return float('nan')
        return float(np.mean(vals))

    if prefer not in ('score', 'switches'):
        prefer = 'score'

    metric_key = 'score' if prefer == 'score' else 'switches'
    m = np.array([future_mean(metric_key, i) for i in range(n)], dtype=np.float32)

    # If metric is missing/constant, fall back to the other one.
    if not np.isfinite(m).any() or np.nanstd(m) < 1e-8:
        metric_key = 'switches' if metric_key == 'score' else 'score'
        m = np.array([future_mean(metric_key, i) for i in range(n)], dtype=np.float32)

    # Replace non-finite with median to keep quantile stable.
    med = float(np.nanmedian(m)) if np.isfinite(m).any() else 0.0
    m = np.where(np.isfinite(m), m, med)

    # For score, lower is worse -> bottom-q is positive.
    # For switches, higher is worse -> top-q is positive.
    if metric_key == 'score':
        thr = float(np.quantile(m, q))
        Y = (m <= thr).astype(np.int32)
    else:
        thr = float(np.quantile(m, 1.0 - q))
        Y = (m >= thr).astype(np.int32)
    return Y, {'metric': metric_key, 'q': float(q), 'thr': thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', type=str, default=str(Path(__file__).resolve().parents[1] / 'logs' / 'episode_features.jsonl'))
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--S_fail', type=int, default=22)
    ap.add_argument('--D_fail', type=float, default=0.015)
    ap.add_argument('--cap_fail', type=float, default=0.18)
    ap.add_argument('--integ_fail', type=float, default=0.35)
    ap.add_argument('--scar_fail', type=int, default=10)
    ap.add_argument('--label', type=str, default='hard', choices=['hard', 'soft_q'])
    ap.add_argument('--soft_q', type=float, default=0.2)
    ap.add_argument('--soft_prefer', type=str, default='score', choices=['score', 'switches'])
    ap.add_argument('--l2', type=float, default=1e-2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    def _auto_find_features(default_path: Path):
        """Try to locate episode_features.jsonl in common locations.

        Some users run the training script in a different version folder and
        then want to analyze those logs here. We try to be helpful:
        - if default exists, use it
        - else, search under ./logs/** for episode_features*.jsonl
        - else, return None
        """
        if default_path.exists():
            return default_path
        root = default_path.parents[0]  # logs/
        if not root.exists():
            return None
        candidates = sorted(root.rglob('episode_features*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    path = _auto_find_features(Path(args.path))
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            "episode_features.jsonl not found. Run `python run.py` first to generate logs, "
            "or pass --path to an existing episode_features.jsonl (e.g., from v10.4p)."
        )

    path = Path(path)

    rows = load_features(path)
    if len(rows) < 40:
        print(f'Not enough rows: {len(rows)}')
        return

    label_info = None
    if args.label == 'hard':
        Y = build_labels(rows, args.K, args.S_fail, args.D_fail, args.cap_fail, args.integ_fail, args.scar_fail)
    else:
        Y, label_info = build_labels_soft_quantile(rows, args.K, q=args.soft_q, prefer=args.soft_prefer)

    pos_rate = float(np.mean(Y))
    if label_info is not None:
        print(f"[label] mode=soft_q metric={label_info['metric']} q={label_info['q']:.2f} thr={label_info['thr']:.6f} pos_rate={pos_rate:.3f} (K={args.K}, N={len(rows)})")
    else:
        print(f"[label] mode=hard pos_rate={pos_rate:.3f} (K={args.K}, N={len(rows)})")

    # Baseline observable features
    Xb = []
    Xs = []
    for r in rows:
        Xb.append([
            float(r.get('switches', 0)),
            float(r.get('self_slow_drift', 0.0)),
            float(r.get('endowment', 0.0)),
            float(r.get('bank', 0.0)),
            float(r.get('capital_end', 0.0)),
            float(r.get('integrity_end', 1.0)),
            float(r.get('scar_end', 0)),
        ])
        sp = r.get('self_slow_proj', None)
        if sp is None:
            Xs.append([0.0])
        else:
            Xs.append([float(x) for x in sp])

    Xb = np.asarray(Xb, dtype=np.float32)
    Xs = np.asarray(Xs, dtype=np.float32)
    Xc = np.concatenate([Xb, Xs], axis=1)

    for name, X in [('baseline', Xb), ('self_proj', Xs), ('concat', Xc)]:
        out = ridge_score_probe(X, Y, l2=args.l2, seed=args.seed)
        if out is None:
            print(f'{name}: insufficient data')
            continue
        yte, scores = out
        auc = auc_from_scores(yte, scores)
        n_pos = int((yte == 1).sum()); n_neg = int((yte == 0).sum())
        if np.isnan(auc):
            print(f'[{name}] AUC=nan  (K={args.K}, N={len(rows)}, test_pos={n_pos}, test_neg={n_neg})')
        else:
            print(f'[{name}] AUC={auc:.4f}  (K={args.K}, N={len(rows)}, test_pos={n_pos}, test_neg={n_neg})')


if __name__ == '__main__':
    main()
