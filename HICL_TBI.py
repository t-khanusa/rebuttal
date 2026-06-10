
import os
import csv
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from HICL_TBI_v2 import HyperbolicImputer


log_file = 'TabPFN_tbi_results_odc_8classes_adaptiveK.csv'
k_log_file = 'TabPFN_tbi_per_query_k_odc_8classes.csv'


def poincare_distance(u, v):
    """
    Tính khoảng cách giữa 2 vector u, v trong hình cầu Poincaré.
    Công thức: d(u, v) = arccosh(1 + 2 * ||u-v||^2 / ((1 - ||u||^2) * (1 - ||v||^2)))
    """
    EPS = 1e-5
    sq_u_norm = np.sum(u ** 2)
    sq_v_norm = np.sum(v ** 2)
    sq_dist = np.sum((u - v) ** 2)
    denom = np.maximum((1 - sq_u_norm) * (1 - sq_v_norm), EPS)
    arg = 1 + 2 * sq_dist / denom
    return np.arccosh(arg)


def project_to_hyperbolic(X: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Exponential map at origin — Euclidean tangent vector → Poincaré Ball."""
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    sqrt_c = np.sqrt(c)
    eps = 1e-10
    X_hyperbolic = np.tanh(sqrt_c * X_norm) * (X / (sqrt_c * X_norm + eps))
    return X_hyperbolic


# ══════════════════════════════════════════════════════════════
#  GEOMETRY-ADAPTIVE k  (density-based, per query)
# ══════════════════════════════════════════════════════════════

def adaptive_k_from_density(sorted_distances: np.ndarray,
                            m_ref: int = 10, alpha: float = 3.0,
                            k_min: int = 10,
                            k_max: int | None = None) -> int:
    """Choose k from the *local density* around the query in Poincaré space.

    Uses a classical Parzen-style variable bandwidth.  The query's local
    scale is σ_q = d_{m_ref}(q), the Poincaré distance to its m_ref-th
    nearest training point.  We then include every training point within
    an adaptive radius r_q = α · σ_q :

        k* = | { x ∈ D_train  :  δ_Poincaré(x, q) ≤ α · σ_q } |

    This adapts the effective bandwidth to the *local density* of the
    training set around the query:
      • in a dense cluster (mild / common cases, small σ_q) the α-scaled
        ball stays compact and includes all truly similar patients;
      • in a sparse region (severe / outlier cases, large σ_q) the ball
        expands automatically so that a statistically sufficient context
        is still retrieved;
      • the returned k* tracks local density — regions with a true cluster
        boundary yield smaller k*, while regions inside a cluster yield
        larger k*.

    Parameters
    ----------
    sorted_distances : (n,) Poincaré distances, already sorted ascending
    m_ref            : reference NN index for the local bandwidth σ_q
    alpha            : radius multiplier (α > 1 → reach beyond σ_q)
    k_min, k_max     : hard bounds on the returned k
    """
    n = len(sorted_distances)
    if k_max is None:
        k_max = n
    if n == 0:
        return k_min
    if n <= m_ref:
        return int(np.clip(n, k_min, k_max))

    sigma_q = float(sorted_distances[m_ref - 1])
    radius = alpha * sigma_q

    k_star = int(np.searchsorted(sorted_distances, radius, side='right'))
    return int(np.clip(k_star, k_min, k_max))


# ══════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    sensitivity = recall

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 2:
        tn, fp, _, _ = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specs = []
        for i in range(cm.shape[0]):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        specificity = float(np.mean(specs))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

folder = '/project/khanhnt/TBI/odc-tbi_1168/8classes'

# ── Density-adaptive k hyperparameters ────────────────────────
K_MIN = 10          # minimum context size for TabPFN stability
K_MAX_CAP = None    # optional hard upper bound; None → use full training size
M_REF = 10          # reference NN index for local bandwidth σ_q = d_{m_ref}(q)
ALPHA = 3.0         # ball radius multiplier r_q = alpha · σ_q

with open(log_file, 'w', newline='') as f:
    csv.writer(f).writerow([
        "fold", "n_test", "k_mean", "k_std", "k_min", "k_max",
        "accuracy", "precision", "sensitivity", "specificity",
    ])
with open(k_log_file, 'w', newline='') as f:
    csv.writer(f).writerow(["fold", "query_idx", "k_star", "y_true", "y_pred"])


accuracy_scores = []
precision_scores = []
sensitivity_scores = []
specificity_scores = []

for i in range(1, 11):
    train_file = os.path.join(folder, f'train_fold_{i}.csv')
    test_file = os.path.join(folder, f'test_fold_{i}.csv')

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    feature_names = train_df.columns[:-1]

    X_train = train_df.values[:, :-1]
    y_train = train_df.values[:, -1]
    X_test = test_df.values[:, :-1]
    y_test = test_df.values[:, -1]

    # ── Stage 1: Hyperbolic Imputation ────────────────────────
    imputer = HyperbolicImputer(n_neighbors=5, c=1.0, max_iter=10)
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # ── Stage 2: Project to Poincaré Ball ─────────────────────
    X_train_hyp = project_to_hyperbolic(X_train_imp)
    X_test_hyp = project_to_hyperbolic(X_test_imp)

    # ── Stage 3: Retrieve ALL training points once per query ─
    n_train = len(X_train_hyp)
    k_retrieve = n_train if K_MAX_CAP is None else min(K_MAX_CAP, n_train)

    knn_hyp = NearestNeighbors(
        n_neighbors=k_retrieve,
        algorithm='brute',
        metric=poincare_distance,
    )
    knn_hyp.fit(X_train_hyp)

    tabpfn_predictions = []
    ks_used = []

    clf = TabPFNClassifier(device='cuda', n_estimators=18)

    for idx, x_query_hyp in enumerate(X_test_hyp):
        distances, neighbor_indices = knn_hyp.kneighbors(
            [x_query_hyp], n_neighbors=k_retrieve, return_distance=True)
        sorted_dists = distances[0]
        sorted_nn = neighbor_indices[0]

        # ── Stage 4: Density-adaptive k around this query ────
        k_star = adaptive_k_from_density(
            sorted_dists,
            m_ref=M_REF, alpha=ALPHA,
            k_min=K_MIN, k_max=k_retrieve,
        )
        ks_used.append(k_star)

        nn = sorted_nn[:k_star]
        X_context = X_train_imp[nn]
        y_context = y_train[nn]

        x_query_original = X_test_imp[idx].reshape(1, -1)
        clf.fit(X_context, y_context)
        y_pred = clf.predict(x_query_original)[0]
        tabpfn_predictions.append(y_pred)

        with open(k_log_file, 'a', newline='') as f:
            csv.writer(f).writerow([i, idx, k_star, y_test[idx], y_pred])

    ks_used = np.asarray(ks_used)
    metrics = compute_metrics(y_test, tabpfn_predictions)
    score = metrics['accuracy']
    precision = metrics['precision']
    sensitivity = metrics['sensitivity']
    specificity = metrics['specificity']

    print(f"Fold {i:>2d} | Adaptive-K  "
          f"k_mean={ks_used.mean():.0f} ± {ks_used.std():.0f}  "
          f"[{ks_used.min()}, {ks_used.max()}]  "
          f"| Acc={score:.4f}  Prec={precision:.4f}  "
          f"Sens={sensitivity:.4f}  Spec={specificity:.4f}")

    accuracy_scores.append(score)
    precision_scores.append(precision)
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)

    with open(log_file, 'a', newline='') as f:
        csv.writer(f).writerow([
            i, len(X_test_hyp),
            float(ks_used.mean()), float(ks_used.std()),
            int(ks_used.min()), int(ks_used.max()),
            score, precision, sensitivity, specificity,
        ])


print("\n--- FINAL RESULT ---")
print(f"Mean Accuracy:    {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"Mean Precision:   {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
print(f"Mean Sensitivity: {np.mean(sensitivity_scores):.4f} ± {np.std(sensitivity_scores):.4f}")
print(f"Mean Specificity: {np.mean(specificity_scores):.4f} ± {np.std(specificity_scores):.4f}")
