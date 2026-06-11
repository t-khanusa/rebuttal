from __future__ import annotations

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr, kendalltau, wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
_EPS = 1e-10
RNG = np.random.default_rng(0)


# ════════════════════════════════════════════════════════════════
# 0.  GEOMETRY  (mirrors HICL_TBI_v2.PoincareBall; kept local so this
#     module has no TabPFN / CUDA import dependency)
# ════════════════════════════════════════════════════════════════

class PoincareBall:
    def __init__(self, c: float = 1.0):
        self.c = c
        self._sqrt_c = np.sqrt(c)
        self._boundary = 1.0 / self._sqrt_c - 1e-5

    def exp_map_zero(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        safe = np.maximum(n, _EPS)
        return np.tanh(self._sqrt_c * safe) / (self._sqrt_c * safe) * v

    def clip(self, x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return np.where(n > self._boundary, x * self._boundary / n, x)

    def dist_one_to_many(self, u: np.ndarray, V: np.ndarray) -> np.ndarray:
        sq_u = np.sum(u ** 2)
        sq_V = np.sum(V ** 2, axis=1)
        sq_d = np.sum((u[None, :] - V) ** 2, axis=1)
        denom = np.maximum((1 - self.c * sq_u) * (1 - self.c * sq_V), _EPS)
        arg = np.maximum(1 + 2 * self.c * sq_d / denom, 1.0 + _EPS)
        return np.arccosh(arg) / self._sqrt_c

    def dist_matrix(self, Z: np.ndarray) -> np.ndarray:
        n = len(Z)
        D = np.zeros((n, n))
        for i in range(n):
            D[i] = self.dist_one_to_many(Z[i], Z)
        np.fill_diagonal(D, 0.0)
        return 0.5 * (D + D.T)


# ════════════════════════════════════════════════════════════════
# 1.  DATA LOADING / PREPROCESSING
# ════════════════════════════════════════════════════════════════

def load_full_from_folds(folder: str):
    """Reconstruct the full dataset from the 10 disjoint CV test folds."""
    dfs = []
    for i in range(1, 11):
        p = os.path.join(folder, f"test_fold_{i}.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if not dfs:
        raise FileNotFoundError(f"No test_fold_*.csv found in {folder}")
    df = pd.concat(dfs, axis=0, ignore_index=True)
    X = df.values[:, :-1].astype(float)
    y = df.values[:, -1]
    return X, y


def load_fold(folder: str, fold: int = 1):
    tr = pd.read_csv(os.path.join(folder, f"train_fold_{fold}.csv"))
    te = pd.read_csv(os.path.join(folder, f"test_fold_{fold}.csv"))
    Xtr = tr.values[:, :-1].astype(float)
    ytr = tr.values[:, -1]
    Xte = te.values[:, :-1].astype(float)
    yte = te.values[:, -1]
    return Xtr, ytr, Xte, yte


def ordinal_codes(y: np.ndarray) -> np.ndarray:
    """Map labels to ordinal integer codes (ascending)."""
    try:
        return y.astype(float)
    except (ValueError, TypeError):
        cats = sorted(pd.unique(pd.Series(y)), key=lambda v: str(v))
        m = {c: i for i, c in enumerate(cats)}
        return np.array([m[v] for v in y], dtype=float)


def mean_fill(X: np.ndarray) -> np.ndarray:
    out = X.copy()
    col = np.nanmean(out, axis=0)
    col = np.where(np.isnan(col), 0.0, col)
    idx = np.where(np.isnan(out))
    out[idx] = np.take(col, idx[1])
    return out


def project(X_raw: np.ndarray, ball: PoincareBall, scaler: StandardScaler | None = None):
    """mean-fill -> standardize -> exp map -> Poincare ball."""
    Xf = mean_fill(X_raw)
    if scaler is None:
        scaler = StandardScaler().fit(Xf)
    Xs = scaler.transform(Xf)
    Z = ball.clip(ball.exp_map_zero(Xs))
    return Xs, Z, scaler


def subsample(n: int, cap: int):
    if n <= cap:
        return np.arange(n)
    return RNG.choice(n, size=cap, replace=False)


# ════════════════════════════════════════════════════════════════
# A.  GROMOV  delta-HYPERBOLICITY
# ════════════════════════════════════════════════════════════════

def _maxmin_product(G: np.ndarray) -> np.ndarray:
    n = G.shape[0]
    out = np.empty_like(G)
    for x in range(n):
        out[x] = np.minimum(G[x][:, None], G).max(axis=0)
    return out


def gromov_delta(D: np.ndarray, n_basepoints: int = 8):
    """Worst-case Gromov 4-point delta and delta_rel = 2*delta/diam.

    Lower delta_rel  ->  more tree-like  ->  better suited to hyperbolic
    embedding (this is exactly the claim the paper asserts but never measures).
    """
    n = D.shape[0]
    diam = float(D.max())
    bps = RNG.choice(n, size=min(n_basepoints, n), replace=False)
    deltas = []
    for w in bps:
        row = D[w]
        G = 0.5 * (row[:, None] + row[None, :] - D)
        deltas.append(float((_maxmin_product(G) - G).max()))
    delta = float(np.max(deltas))
    return {"delta": delta, "diam": diam,
            "delta_rel": (2.0 * delta / diam) if diam > 0 else float("nan")}


def analysis_A_delta(X_raw, y, ball, cap=140):
    Xs, Z, _ = project(X_raw, ball)
    idx = subsample(len(Xs), cap)
    De = np.linalg.norm(Xs[idx][:, None, :] - Xs[idx][None, :, :], axis=-1)
    Dh = ball.dist_matrix(Z[idx])
    euc = gromov_delta(De)
    hyp = gromov_delta(Dh)
    return {"n_used": int(len(idx)),
            "euclidean": euc, "hyperbolic": hyp,
            "tree_likeness_gain": euc["delta_rel"] - hyp["delta_rel"]}


# ════════════════════════════════════════════════════════════════
# B.  SEVERITY  <->  HYPERBOLIC RADIUS   (validates the core premise)
# ════════════════════════════════════════════════════════════════

def analysis_B_radius(X_raw, y, ball):
    _, Z, _ = project(X_raw, ball)
    r = np.linalg.norm(Z, axis=1)
    sev = ordinal_codes(y)
    rho, p = spearmanr(r, sev)
    classes = np.unique(sev)
    per_class = {float(c): float(r[sev == c].mean()) for c in classes}
    # ordered by ascending class code; monotone increase supports the premise
    return {"spearman_rho": float(rho), "p_value": float(p),
            "abs_rho": float(abs(rho)),
            "mean_radius_per_class": per_class,
            "radius_overall_mean": float(r.mean())}


# ════════════════════════════════════════════════════════════════
# C.  CLASS SEPARATION / IMPLICIT MARGIN
# ════════════════════════════════════════════════════════════════

def _norm_offdiag(D):
    m = np.median(D[np.triu_indices_from(D, k=1)])
    return D / (m + _EPS)


def analysis_C_margin(X_raw, y, ball, cap=1200):
    Xs, Z, _ = project(X_raw, ball)
    idx = subsample(len(Xs), cap)
    Xs, Z, sev = Xs[idx], Z[idx], ordinal_codes(y)[idx]
    r = np.linalg.norm(Z, axis=1)

    De = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=-1)
    Dh = ball.dist_matrix(Z)
    Den, Dhn = _norm_offdiag(De), _norm_offdiag(Dh)

    classes = np.unique(sev)
    # global silhouette (higher = better-separated clusters)
    sil_e = float(silhouette_score(De, sev, metric="precomputed")) if len(classes) > 1 else float("nan")
    sil_h = float(silhouette_score(Dh, sev, metric="precomputed")) if len(classes) > 1 else float("nan")

    # per class-pair: normalized inter-class distance vs. the pair's mean radius
    pair_radius, pair_ratio = [], []
    for a_i in range(len(classes)):
        for b_i in range(a_i + 1, len(classes)):
            a, b = classes[a_i], classes[b_i]
            ma, mb = sev == a, sev == b
            inter_e = Den[np.ix_(ma, mb)].mean()
            inter_h = Dhn[np.ix_(ma, mb)].mean()
            rad = 0.5 * (r[ma].mean() + r[mb].mean())
            pair_radius.append(rad)
            pair_ratio.append(inter_h / (inter_e + _EPS))
    pair_radius, pair_ratio = np.asarray(pair_radius), np.asarray(pair_ratio)
    rho, p = (spearmanr(pair_radius, pair_ratio) if len(pair_ratio) > 2 else (np.nan, np.nan))

    return {"n_used": int(len(idx)),
            "silhouette_euclidean": sil_e,
            "silhouette_hyperbolic": sil_h,
            "margin_radius_spearman": float(rho),
            "margin_radius_p": float(p),
            "interpretation": "positive margin_radius_spearman => hyperbolic "
                              "amplifies inter-class separation MORE in the "
                              "high-radius tail (supports implicit margin claim)"}


# ════════════════════════════════════════════════════════════════
# D.  RETRIEVAL REORDERING   (hyperbolic vs Euclidean kNN)
# ════════════════════════════════════════════════════════════════

def analysis_D_retrieval(folder, ball, fold=1, k=20):
    Xtr, ytr, Xte, yte = load_fold(folder, fold)
    Xtr_f, Xte_f = mean_fill(Xtr), mean_fill(Xte)
    scaler = StandardScaler().fit(Xtr_f)
    Xtr_s, Xte_s = scaler.transform(Xtr_f), scaler.transform(Xte_f)
    Ztr = ball.clip(ball.exp_map_zero(Xtr_s))
    Zte = ball.clip(ball.exp_map_zero(Xte_s))
    q_rad = np.linalg.norm(Zte, axis=1)

    jacc, taus = [], []
    for i in range(len(Xte_s)):
        de = np.linalg.norm(Xtr_s - Xte_s[i][None, :], axis=1)
        dh = ball.dist_one_to_many(Zte[i], Ztr)
        ne = set(np.argsort(de)[:k])
        nh = set(np.argsort(dh)[:k])
        jacc.append(len(ne & nh) / len(ne | nh))
        taus.append(kendalltau(de, dh).correlation)
    jacc, taus, q_rad = np.asarray(jacc), np.asarray(taus), q_rad

    # stratify by query radius tercile (core / mid / tail)
    terc = np.quantile(q_rad, [1/3, 2/3])
    grp = np.digitize(q_rad, terc)
    strat = {("core", "mid", "tail")[g]:
             {"jaccard": float(jacc[grp == g].mean()),
              "kendall_tau": float(np.nanmean(taus[grp == g])),
              "n": int((grp == g).sum())}
             for g in np.unique(grp)}
    return {"k": k, "fold": fold,
            "topk_jaccard_overall": float(jacc.mean()),
            "kendall_tau_overall": float(np.nanmean(taus)),
            "by_query_radius": strat,
            "interpretation": "lower jaccard / tau in 'tail' => hyperbolic "
                              "retrieval diverges from L2 mostly for severe "
                              "(high-radius) queries"}


# ════════════════════════════════════════════════════════════════
# E.  CONTEXT QUALITY   (the real mechanism of 'asymmetric inference')
# ════════════════════════════════════════════════════════════════

def analysis_E_context(folder, ball, fold=1, k=20):
    Xtr, ytr, Xte, yte = load_fold(folder, fold)
    Xtr_f, Xte_f = mean_fill(Xtr), mean_fill(Xte)
    scaler = StandardScaler().fit(Xtr_f)
    Xtr_s, Xte_s = scaler.transform(Xtr_f), scaler.transform(Xte_f)
    Ztr = ball.clip(ball.exp_map_zero(Xtr_s))
    Zte = ball.clip(ball.exp_map_zero(Xte_s))

    sev_tr = ordinal_codes(ytr)
    sev_te = ordinal_codes(yte)
    classes, counts = np.unique(sev_tr, return_counts=True)
    tail = set(classes[np.argsort(counts)][:max(1, len(classes) // 2)])

    def _purity_cov(neighbor_fn):
        pur, cov = [], []
        for i in range(len(Xte_s)):
            nn = neighbor_fn(i)
            labs = sev_tr[nn]
            pur.append(float(np.mean(labs == sev_te[i])))
            if sev_te[i] in tail:
                cov.append(float(np.mean(np.isin(labs, list(tail)))))
        return float(np.mean(pur)), (float(np.mean(cov)) if cov else float("nan"))

    def euc_nn(i):
        d = np.linalg.norm(Xtr_s - Xte_s[i][None, :], axis=1)
        return np.argsort(d)[:k]

    def hyp_nn(i):
        d = ball.dist_one_to_many(Zte[i], Ztr)
        return np.argsort(d)[:k]

    pe, ce = _purity_cov(euc_nn)
    ph, ch = _purity_cov(hyp_nn)
    return {"k": k, "fold": fold,
            "context_label_purity": {"euclidean": pe, "hyperbolic": ph,
                                      "gain": ph - pe},
            "tail_class_coverage_for_tail_queries":
                {"euclidean": ce, "hyperbolic": ch, "gain": ch - ce},
            "interpretation": "higher purity / tail-coverage for hyperbolic "
                              "explains TabPFN gains even though TabPFN decides "
                              "in Euclidean space (context-quality argument)"}


# ════════════════════════════════════════════════════════════════
# F.  STATISTICS UTILITIES  (make the result tables defensible)
# ════════════════════════════════════════════════════════════════

def paired_fold_test(acc_a, acc_b, n_boot=10000):
    """Paired Wilcoxon + bootstrap CI of the mean per-fold difference.

    acc_a, acc_b : array-like of per-fold scores (same folds, same order).
    """
    a, b = np.asarray(acc_a, float), np.asarray(acc_b, float)
    diff = a - b
    try:
        w_stat, w_p = wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    boot = np.array([RNG.choice(diff, size=len(diff), replace=True).mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()),
            "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
            "ci95": [float(lo), float(hi)],
            "significant_0.05": bool(w_p < 0.05) if w_p == w_p else False}


def paired_test_from_csv(csv_a, csv_b, metric="accuracy", fold_col="fold"):
    """Compare two per-fold result logs (as written by HICL_TBI.py)."""
    da = pd.read_csv(csv_a).sort_values(fold_col)
    db = pd.read_csv(csv_b).sort_values(fold_col)
    return paired_fold_test(da[metric].values, db[metric].values)


def expected_calibration_error(y_true, probs, n_bins=15):
    """ECE for multi-class predicted-probability matrix probs (n, C).

    Wire this into HICL_TBI.py by replacing `clf.predict` with
    `clf.predict_proba` and saving the probability rows.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ════════════════════════════════════════════════════════════════
# DRIVER
# ════════════════════════════════════════════════════════════════

DATASETS = {
    # name : folder containing train_fold_*.csv / test_fold_*.csv
    "ODC-TBI-8class": "/project/khanhnt/TBI/odc-tbi_1168/8classes",
    "TBI-MH103":      "/project/khanhnt/TBI/k_folds_tbi/data",
}

CURVATURE = 1.0
TOPK = 20


def run_dataset(name, folder, c=CURVATURE):
    ball = PoincareBall(c)
    print(f"\n{'='*70}\nDATASET: {name}   (folder={folder}, c={c})\n{'='*70}")
    X, y = load_full_from_folds(folder)
    print(f"  N={len(X)}  d={X.shape[1]}  classes={len(np.unique(y))}  "
          f"missing={np.isnan(X).mean()*100:.1f}%")

    out = {"dataset": name, "n": int(len(X)), "d": int(X.shape[1]),
           "curvature": c}
    out["A_delta_hyperbolicity"] = analysis_A_delta(X, y, ball)
    out["B_severity_radius"] = analysis_B_radius(X, y, ball)
    out["C_margin"] = analysis_C_margin(X, y, ball)
    try:
        out["D_retrieval"] = analysis_D_retrieval(folder, ball, fold=1, k=TOPK)
        out["E_context"] = analysis_E_context(folder, ball, fold=1, k=TOPK)
    except FileNotFoundError as e:
        out["D_retrieval"] = out["E_context"] = {"error": str(e)}

    print(json.dumps(out, indent=2))
    return out


def main():
    results = []
    for name, folder in DATASETS.items():
        if not os.path.isdir(folder):
            print(f"[skip] {name}: folder not found ({folder})")
            continue
        try:
            results.append(run_dataset(name, folder))
        except Exception as e:  # noqa: BLE001
            print(f"[error] {name}: {e}")
    with open("hicl_theory_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> hicl_theory_analysis_results.json")


if __name__ == "__main__":
    main()
