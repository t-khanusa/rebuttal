"""
HICL-TBI v2: Mixed-Curvature Product-Space Retrieval + Hyperbolic Imputation

Two advances over HICL-TBI v1:

1. **Hyperbolic Imputer** – replaces sklearn KNNImputer.  Finds neighbours in
   the Poincaré Ball (after an initial mean-fill), then uses inverse-hyperbolic-
   distance weighting to reconstruct missing values.  Iterates until
   convergence so that imputed geometry and imputed values become mutually
   consistent.

2. **Mixed-Curvature Product-Space Retrieval** – instead of projecting *all*
   features into a single Poincaré Ball, features are automatically split
   (via skewness) into:
     • Hyperbolic subspace  H^p  – high-|skew| features (severity / ordinal)
     • Euclidean  subspace  E^q  – low-|skew| features (demographic / flat)
     • Spherical  subspace  S^r  – (reserved, empty by default)
   The product-space distance is
       d(x,y) = sqrt( α·d_H² + β·d_E² + γ·d_S² )
   ensuring each geometry only governs the features that match its curvature.
"""

import os
import csv
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
from tabpfn import TabPFNClassifier

_EPS = 1e-10


# ════════════════════════════════════════════════════════════════
# I.  GEOMETRIC PRIMITIVES
# ════════════════════════════════════════════════════════════════

class PoincareBall:
    """d-dimensional Poincaré Ball model with curvature -c (c > 0)."""

    def __init__(self, c: float = 1.0):
        self.c = c
        self._sqrt_c = np.sqrt(c)
        self._boundary = 1.0 / self._sqrt_c - 1e-5

    # --- maps ----------------------------------------------------------
    def exp_map_zero(self, v: np.ndarray) -> np.ndarray:
        """Exponential map at the origin  T_0 D^d_c  →  D^d_c ."""
        v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
        safe = np.maximum(v_norm, _EPS)
        return np.tanh(self._sqrt_c * safe) / (self._sqrt_c * safe) * v

    def log_map_zero(self, y: np.ndarray) -> np.ndarray:
        """Logarithmic map at the origin  D^d_c  →  T_0 D^d_c ."""
        y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
        safe = np.clip(y_norm, _EPS, self._boundary)
        return np.arctanh(self._sqrt_c * safe) / (self._sqrt_c * safe) * y

    def clip(self, x: np.ndarray) -> np.ndarray:
        """Project points back inside the open ball."""
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        return np.where(norms > self._boundary, x * self._boundary / norms, x)

    # --- distances -----------------------------------------------------
    def dist(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Poincaré distance — works for matching batch dims."""
        sq_u = np.sum(u ** 2, axis=-1)
        sq_v = np.sum(v ** 2, axis=-1)
        sq_d = np.sum((u - v) ** 2, axis=-1)
        denom = np.maximum((1 - self.c * sq_u) * (1 - self.c * sq_v), _EPS)
        arg = np.maximum(1 + 2 * self.c * sq_d / denom, 1.0 + _EPS)
        return np.arccosh(arg) / self._sqrt_c

    def dist_one_to_many(self, u: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Distance from a single point u (d,) to each row of V (n, d)."""
        sq_u = np.sum(u ** 2)
        sq_V = np.sum(V ** 2, axis=1)
        sq_d = np.sum((u[None, :] - V) ** 2, axis=1)
        denom = np.maximum((1 - self.c * sq_u) * (1 - self.c * sq_V), _EPS)
        arg = np.maximum(1 + 2 * self.c * sq_d / denom, 1.0 + _EPS)
        return np.arccosh(arg) / self._sqrt_c

    # --- aggregation ---------------------------------------------------
    def einstein_midpoint(self, pts: np.ndarray,
                          w: np.ndarray | None = None) -> np.ndarray:
        """Approximate Fréchet mean via the Einstein midpoint formula.

        Parameters
        ----------
        pts : (n, d)  points inside the ball
        w   : (n,)    non-negative weights (will be normalised)
        """
        if w is None:
            w = np.ones(len(pts))
        w = w / w.sum()
        gamma = 1.0 / np.sqrt(
            np.maximum(1 - self.c * np.sum(pts ** 2, axis=1), _EPS)
        )
        num = np.sum((w * gamma)[:, None] * pts, axis=0)
        den = np.sum(w * gamma)
        return self.clip((num / den)[None, :])[0]


class UnitSphere:
    """Operations on the d-dimensional unit sphere S^d."""

    def project(self, x: np.ndarray) -> np.ndarray:
        norms = np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), _EPS)
        return x / norms

    def dist_one_to_many(self, u: np.ndarray, V: np.ndarray) -> np.ndarray:
        cos = np.clip(V @ u, -1.0 + _EPS, 1.0 - _EPS)
        return np.arccos(cos)


# ════════════════════════════════════════════════════════════════
# II.  HYPERBOLIC IMPUTER
# ════════════════════════════════════════════════════════════════

class HyperbolicImputer:
    """KNN imputer whose neighbour graph is built in the Poincaré Ball.

    Algorithm
    ---------
    1.  Initialise every NaN with the column mean (from training data).
    2.  Repeat until convergence:
        a. StandardScale → exponential-map into the Poincaré Ball.
        b. For each sample with missing values, find *k* nearest neighbours
           using the Poincaré distance (among the reference pool).
        c. Impute each missing feature as the inverse-distance-weighted
           average of those neighbours' values for that feature.
    3.  Return the imputed matrix.

    During ``fit`` the training data is both the query and the reference
    (with self-exclusion).  During ``transform`` the already-imputed
    training data serves as a fixed reference.
    """

    def __init__(self, n_neighbors: int = 5, c: float = 1.0,
                 max_iter: int = 10, tol: float = 1e-4):
        self.k = n_neighbors
        self.c = c
        self.max_iter = max_iter
        self.tol = tol
        self.ball = PoincareBall(c)
        self._scaler = StandardScaler()
        self._col_means: np.ndarray | None = None
        self._X_ref: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "HyperbolicImputer":
        self._col_means = np.nanmean(X, axis=0)
        self._scaler.fit(self._mean_fill(X))
        self._X_ref = self._impute(X, exclude_self=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._impute(X, exclude_self=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self._X_ref.copy()

    # ------------------------------------------------------------------
    def _mean_fill(self, X: np.ndarray) -> np.ndarray:
        out = X.copy()
        for j in range(X.shape[1]):
            mask = np.isnan(out[:, j])
            if mask.any():
                out[mask, j] = self._col_means[j]
        return out

    def _to_poincare(self, X_euc: np.ndarray) -> np.ndarray:
        X_s = self._scaler.transform(X_euc)
        return self.ball.clip(self.ball.exp_map_zero(X_s))

    def _impute(self, X: np.ndarray, exclude_self: bool) -> np.ndarray:
        missing = np.isnan(X)
        if not missing.any():
            return X.copy()

        X_imp = self._mean_fill(X)
        ref_raw = X_imp if exclude_self else self._X_ref
        rows_with_nan = np.where(missing.any(axis=1))[0]
        extra_k = 1 if exclude_self else 0

        for _iteration in range(self.max_iter):
            prev = X_imp[missing].copy()

            ref_hyp = self._to_poincare(ref_raw)
            query_hyp = self._to_poincare(X_imp)

            for i in rows_with_nan:
                dists = self.ball.dist_one_to_many(query_hyp[i], ref_hyp)
                if exclude_self:
                    dists[i] = np.inf

                k = min(self.k, len(ref_raw) - extra_k)
                nn = np.argpartition(dists, k)[:k]
                inv_d = 1.0 / np.maximum(dists[nn], _EPS)
                w = inv_d / inv_d.sum()

                for j in np.where(missing[i])[0]:
                    X_imp[i, j] = w @ ref_raw[nn, j]

            if exclude_self:
                ref_raw = X_imp

            delta = np.sum((X_imp[missing] - prev) ** 2)
            if delta < self.tol:
                break

        return X_imp


# ════════════════════════════════════════════════════════════════
# III.  AUTOMATIC FEATURE SPLITTER
# ════════════════════════════════════════════════════════════════

class FeatureSplitter:
    """Assigns each column to a geometric component based on skewness.

    High-|skew| features  →  Hyperbolic  (hierarchical / ordinal severity)
    Low-|skew|  features  →  Euclidean   (demographic / flat)
    (reserved)            →  Spherical   (not used by default)

    If *all* features are below the threshold the top third (by skewness)
    is forced into the hyperbolic component so that both subspaces are
    always populated.
    """

    def __init__(self, skew_threshold: float = 1.0):
        self.thresh = skew_threshold
        self.hyp_idx: np.ndarray | None = None
        self.euc_idx: np.ndarray | None = None
        self.sph_idx: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FeatureSplitter":
        sk = np.abs(skew(X, axis=0, nan_policy='omit'))
        self.hyp_idx = np.where(sk >= self.thresh)[0]
        self.euc_idx = np.where(sk < self.thresh)[0]
        self.sph_idx = np.array([], dtype=int)

        if len(self.hyp_idx) == 0:
            order = np.argsort(sk)[::-1]
            split = max(X.shape[1] // 3, 1)
            self.hyp_idx = order[:split]
            self.euc_idx = order[split:]
        return self

    def split(self, X: np.ndarray):
        def _sel(idx):
            return X[:, idx] if len(idx) else np.empty((len(X), 0))
        return _sel(self.hyp_idx), _sel(self.euc_idx), _sel(self.sph_idx)

    def summary(self) -> str:
        return (f"H={len(self.hyp_idx)}  E={len(self.euc_idx)}  "
                f"S={len(self.sph_idx)}  (of {len(self.hyp_idx)+len(self.euc_idx)+len(self.sph_idx)} total)")


# ════════════════════════════════════════════════════════════════
# IV.  PRODUCT-SPACE RETRIEVER   H^p × E^q × S^r
# ════════════════════════════════════════════════════════════════

class ProductSpaceRetriever:
    """Retrieves k-NN using a mixed-curvature product distance.

    d²(x, y) = α · d_H(x_H, y_H)² + β · d_E(x_E, y_E)² + γ · d_S(x_S, y_S)²

    Each component is independently StandardScaled before projection so
    that no single geometry dominates due to raw feature magnitudes.
    """

    def __init__(self, c: float = 1.0,
                 w_hyp: float = 1.0, w_euc: float = 1.0, w_sph: float = 1.0):
        self.ball = PoincareBall(c)
        self.sphere = UnitSphere()
        self.w = {"h": w_hyp, "e": w_euc, "s": w_sph}

        self._sc_h = StandardScaler()
        self._sc_e = StandardScaler()
        self._sc_s = StandardScaler()

        self._ref_h = self._ref_e = self._ref_s = None
        self._n_train = 0

    # ------------------------------------------------------------------
    def fit(self, X_h: np.ndarray, X_e: np.ndarray,
            X_s: np.ndarray) -> "ProductSpaceRetriever":
        self._n_train = len(X_h) if X_h.shape[1] else len(X_e)

        if X_h.shape[1]:
            scaled = self._sc_h.fit_transform(X_h)
            self._ref_h = self.ball.clip(self.ball.exp_map_zero(scaled))
        else:
            self._ref_h = None

        if X_e.shape[1]:
            self._ref_e = self._sc_e.fit_transform(X_e)
        else:
            self._ref_e = None

        if X_s.shape[1]:
            scaled = self._sc_s.fit_transform(X_s)
            self._ref_s = self.sphere.project(scaled)
        else:
            self._ref_s = None

        return self

    # ------------------------------------------------------------------
    def kneighbors(self, x_h: np.ndarray, x_e: np.ndarray,
                   x_s: np.ndarray, k: int):
        """Return (distances, indices) for a single query point."""
        d2 = np.zeros(self._n_train)

        if self._ref_h is not None and x_h.size:
            q = self.ball.clip(
                self.ball.exp_map_zero(
                    self._sc_h.transform(x_h.reshape(1, -1))
                )
            )[0]
            d2 += self.w["h"] * self.ball.dist_one_to_many(q, self._ref_h) ** 2

        if self._ref_e is not None and x_e.size:
            q = self._sc_e.transform(x_e.reshape(1, -1))[0]
            d2 += self.w["e"] * np.sum((self._ref_e - q[None, :]) ** 2, axis=1)

        if self._ref_s is not None and x_s.size:
            q = self.sphere.project(
                self._sc_s.transform(x_s.reshape(1, -1))
            )[0]
            d2 += self.w["s"] * self.sphere.dist_one_to_many(q, self._ref_s) ** 2

        dist = np.sqrt(np.maximum(d2, 0.0))
        k = min(k, self._n_train)
        if k <= 0:
            return np.array([]), np.array([], dtype=int)
        # np.argpartition requires kth < len(dist), so use k-1.
        idx = np.argpartition(dist, k - 1)[:k]
        order = np.argsort(dist[idx])
        idx = idx[order]
        return dist[idx], idx


# ════════════════════════════════════════════════════════════════
# V.   EVALUATION METRICS
# ════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 2:
        tn, fp, _, _ = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) else 0.0
    else:
        specs = []
        for i in range(cm.shape[0]):
            tn = cm.sum() - (cm[i].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specs.append(tn / (tn + fp) if (tn + fp) else 0.0)
        spec = np.mean(specs)
    return dict(accuracy=acc, precision=prec, sensitivity=rec, specificity=spec,
                f1_score=f1)


# ════════════════════════════════════════════════════════════════
# VI.  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def main():
    folder = "/project/khanhnt/TBI/k_folds_tbi/data"
    log_file = "HICL_TBI_v2_results.csv"

    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(
            ["k_neighbors", "fold", "accuracy", "precision",
             "sensitivity", "specificity"]
        )

    k_values = [10, 60, 110, 160, 210, 260, 310, 360, 410]

    for k_neighbors in k_values:
        fold_acc, fold_prec, fold_sens, fold_spec = [], [], [], []

        for fold in range(1, 11):
            train_df = pd.read_csv(
                os.path.join(folder, f"train_fold_{fold}.csv"))
            test_df = pd.read_csv(
                os.path.join(folder, f"test_fold_{fold}.csv"))

            X_train = train_df.values[:, :-1].astype(float)
            y_train = train_df.values[:, -1]
            X_test = test_df.values[:, :-1].astype(float)
            y_test = test_df.values[:, -1]

            # ── Stage 1: Hyperbolic Imputation ────────────────────────
            imputer = HyperbolicImputer(n_neighbors=5, c=1.0, max_iter=10)
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            # ── Stage 2: Feature Splitting ────────────────────────────
            splitter = FeatureSplitter(skew_threshold=1.0)
            splitter.fit(X_train_imp)
            if fold == 1:
                print(f"  [k={k_neighbors}] Feature split: {splitter.summary()}")

            Xtr_h, Xtr_e, Xtr_s = splitter.split(X_train_imp)
            Xte_h, Xte_e, Xte_s = splitter.split(X_test_imp)

            # ── Stage 3: Product-Space Retrieval ──────────────────────
            retriever = ProductSpaceRetriever(
                c=1.0, w_hyp=1.0, w_euc=1.0, w_sph=1.0)
            retriever.fit(Xtr_h, Xtr_e, Xtr_s)

            # ── Stage 4: In-Context Prediction (TabPFN) ──────────────
            clf = TabPFNClassifier(device="cuda", n_estimators=18)
            predictions = []

            for idx in range(len(X_test_imp)):
                _, nn_idx = retriever.kneighbors(
                    Xte_h[idx], Xte_e[idx], Xte_s[idx], k=k_neighbors)

                clf.fit(X_train_imp[nn_idx], y_train[nn_idx])
                predictions.append(
                    clf.predict(X_test_imp[idx: idx + 1])[0])

            m = compute_metrics(y_test, predictions)
            fold_acc.append(m["accuracy"])
            fold_prec.append(m["precision"])
            fold_sens.append(m["sensitivity"])
            fold_spec.append(m["specificity"])

            print(f"  K={k_neighbors} | Fold {fold:>2d}  "
                  f"Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
                  f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}")

            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    k_neighbors, fold,
                    m["accuracy"], m["precision"],
                    m["sensitivity"], m["specificity"],
                ])

        print(f"  ── K={k_neighbors} MEAN ──  "
              f"Acc={np.mean(fold_acc):.4f}  "
              f"Prec={np.mean(fold_prec):.4f}  "
              f"Sens={np.mean(fold_sens):.4f}  "
              f"Spec={np.mean(fold_spec):.4f}\n")


if __name__ == "__main__":
    main()
