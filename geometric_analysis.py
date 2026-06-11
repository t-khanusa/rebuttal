"""
Geometric Analysis of HICL-TBI Embeddings

Produces four analyses that validate the hyperbolic hypothesis:

1. Radial distribution histograms per severity class
2. 2D Poincaré disk embedding with per-class density contours
3. Intra/inter-class distance ratios + silhouette scores (Hyperbolic vs Euclidean)
4. Gromov δ-hyperbolicity (measures how "tree-like" the data is)

All figures are saved as publication-quality PNGs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from itertools import combinations

from HICL_TBI_v2 import PoincareBall, HyperbolicImputer

# ── Configuration ─────────────────────────────────────────────
DATA_FOLDER = "/project/khanhnt/TBI/k_folds_tbi/data"
OUT_DIR = "geometric_analysis_output"
CURVATURE = 1.0

CLASS_NAMES = {0: "Mild", 1: "Moderate", 2: "Severe", 3: "Very Severe"}
CLASS_COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#e74c3c"}

os.makedirs(OUT_DIR, exist_ok=True)
ball = PoincareBall(c=CURVATURE)


# ════════════════════════════════════════════════════════════════
#  Data loading & projection
# ════════════════════════════════════════════════════════════════

def load_and_prepare(target_radius=0.7):
    """Load all folds, impute, project to Euclidean (scaled) and Poincaré.

    The ``target_radius`` parameter controls the dimension-aware tangent-space
    normalization.  After StandardScaling, 64-dimensional vectors have norms
    ~sqrt(64) ≈ 8, which maps *all* points to the Poincaré boundary via tanh.
    We rescale so that the average point sits at the given target radius,
    preserving the radial hierarchy the paper depends on.
    """
    dfs = []
    for fold in range(1, 11):
        for split in ("train", "test"):
            path = os.path.join(DATA_FOLDER, f"{split}_fold_{fold}.csv")
            dfs.append(pd.read_csv(path))
    full = pd.concat(dfs, ignore_index=True).drop_duplicates()

    X_raw = full.values[:, :-1].astype(float)
    y = full.values[:, -1].astype(float)

    imputer = HyperbolicImputer(n_neighbors=5, c=CURVATURE, max_iter=10)
    X_imp = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_euc = scaler.fit_transform(X_imp)

    # Dimension-aware normalization: rescale tangent vectors so that the
    # average sample maps to `target_radius` inside the Poincaré Ball.
    # tanh(||v||) = r  →  ||v|| = atanh(r)
    tangent_norms = np.linalg.norm(X_euc, axis=1)
    desired_tangent_norm = np.arctanh(target_radius)
    scale_factor = desired_tangent_norm / np.mean(tangent_norms)
    X_tangent_normed = X_euc * scale_factor

    X_hyp = ball.clip(ball.exp_map_zero(X_tangent_normed))

    print(f"Loaded {len(X_imp)} unique samples, {X_imp.shape[1]} features")
    print(f"  Tangent-space scaling: factor={scale_factor:.4f}  "
          f"(avg norm {np.mean(tangent_norms):.2f} → {desired_tangent_norm:.4f})")
    for cls in sorted(np.unique(y)):
        print(f"  {CLASS_NAMES.get(cls, cls)}: {(y == cls).sum()}")

    return X_imp, X_euc, X_hyp, y


# ════════════════════════════════════════════════════════════════
#  Analysis 1 – Radial Distribution
# ════════════════════════════════════════════════════════════════

def plot_radial_distribution(X_hyp, y):
    """Histogram of Poincaré radius ||z|| per class."""
    norms = np.linalg.norm(X_hyp, axis=1)
    classes = sorted(np.unique(y))

    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in classes:
        mask = y == cls
        ax.hist(norms[mask], bins=40, alpha=0.55, density=True,
                color=CLASS_COLORS.get(cls, "gray"),
                label=f"{CLASS_NAMES.get(cls, cls)}  (n={mask.sum()})",
                edgecolor="white", linewidth=0.4)

    ax.set_xlabel("Poincaré Radius  $\\|\\mathbf{z}\\|$", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Radial Distribution of Severity Classes\nin the Poincaré Ball",
                 fontsize=14)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_xlim(0, 1.0)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "radial_distribution.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")

    print("  Per-class radius statistics:")
    for cls in classes:
        r = norms[y == cls]
        print(f"    {CLASS_NAMES.get(cls, cls):>12s}:  "
              f"mean={r.mean():.4f}  std={r.std():.4f}  "
              f"min={r.min():.4f}  max={r.max():.4f}")


# ════════════════════════════════════════════════════════════════
#  Analysis 2 – 2D Poincaré Disk Embedding
# ════════════════════════════════════════════════════════════════

def plot_poincare_disk(X_hyp, y):
    """PCA→2D in tangent space, re-project to Poincaré disk, scatter + contours."""
    X_tangent = ball.log_map_zero(X_hyp)
    pca = PCA(n_components=2, random_state=42)
    X_2d_tangent = pca.fit_transform(X_tangent)
    X_2d_hyp = ball.clip(ball.exp_map_zero(X_2d_tangent))

    classes = sorted(np.unique(y))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left panel: Euclidean 2D (PCA of scaled features) ---
    ax = axes[0]
    pca_euc = PCA(n_components=2, random_state=42)
    X_2d_euc = pca_euc.fit_transform(X_tangent)
    for cls in classes:
        mask = y == cls
        ax.scatter(X_2d_euc[mask, 0], X_2d_euc[mask, 1], s=18, alpha=0.6,
                   color=CLASS_COLORS.get(cls, "gray"),
                   label=CLASS_NAMES.get(cls, cls))
    ax.set_title("Euclidean Embedding (PCA)", fontsize=14)
    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.legend(fontsize=10, markerscale=1.5, framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")

    # --- Right panel: Poincaré disk ---
    ax = axes[1]
    circle = plt.Circle((0, 0), 1.0, fill=False, color="black",
                         linewidth=1.5, linestyle="--")
    ax.add_patch(circle)

    for cls in classes:
        mask = y == cls
        ax.scatter(X_2d_hyp[mask, 0], X_2d_hyp[mask, 1], s=18, alpha=0.6,
                   color=CLASS_COLORS.get(cls, "gray"),
                   label=CLASS_NAMES.get(cls, cls))

    # density contours per class
    from scipy.ndimage import gaussian_filter
    grid_res = 300
    for cls in classes:
        mask = y == cls
        pts = X_2d_hyp[mask]
        H, xedges, yedges = np.histogram2d(
            pts[:, 0], pts[:, 1], bins=grid_res,
            range=[[-1.05, 1.05], [-1.05, 1.05]])
        H = gaussian_filter(H.T, sigma=6)
        X_grid = 0.5 * (xedges[:-1] + xedges[1:])
        Y_grid = 0.5 * (yedges[:-1] + yedges[1:])
        ax.contour(X_grid, Y_grid, H, levels=3,
                   colors=[CLASS_COLORS.get(cls, "gray")], alpha=0.7,
                   linewidths=1.0)

    ax.set_title("Poincaré Ball Embedding", fontsize=14)
    ax.set_xlabel("$z_1$", fontsize=12)
    ax.set_ylabel("$z_2$", fontsize=12)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.legend(fontsize=10, markerscale=1.5, framealpha=0.9)

    fig.suptitle("TBI-MH103: Euclidean vs Hyperbolic Embeddings", fontsize=15, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "poincare_disk_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ════════════════════════════════════════════════════════════════
#  Analysis 3 – Intra / Inter Class Distances + Silhouette
# ════════════════════════════════════════════════════════════════

def compute_pairwise_hyp(X_hyp, max_n=500):
    """Full pairwise Poincaré distance matrix (subsample if needed)."""
    n = min(len(X_hyp), max_n)
    idx = np.random.default_rng(42).choice(len(X_hyp), n, replace=False)
    X_sub = X_hyp[idx]
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = ball.dist_one_to_many(X_sub[i], X_sub)
    np.fill_diagonal(D, 0.0)
    return D, idx


def distance_analysis(X_euc, X_hyp, y):
    """Intra/inter-class distances + silhouette in both geometries."""
    D_hyp, idx = compute_pairwise_hyp(X_hyp)
    y_sub = y[idx]
    X_euc_sub = X_euc[idx]

    D_euc = np.zeros_like(D_hyp)
    for i in range(len(X_euc_sub)):
        D_euc[i] = np.linalg.norm(X_euc_sub - X_euc_sub[i], axis=1)
    np.fill_diagonal(D_euc, 0.0)

    classes = sorted(np.unique(y_sub))
    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │           Intra-class / Inter-class Distance Analysis       │")
    print("  ├───────────────┬──────────────────────┬──────────────────────┤")
    print("  │               │      Euclidean       │     Hyperbolic       │")
    print("  │    Class      │  Intra     Inter     │  Intra     Inter     │")
    print("  ├───────────────┼──────────────────────┼──────────────────────┤")

    for cls in classes:
        mask = y_sub == cls
        intra_euc = D_euc[np.ix_(mask, mask)][np.triu_indices(mask.sum(), k=1)]
        inter_euc = D_euc[np.ix_(mask, ~mask)].ravel()
        intra_hyp = D_hyp[np.ix_(mask, mask)][np.triu_indices(mask.sum(), k=1)]
        inter_hyp = D_hyp[np.ix_(mask, ~mask)].ravel()
        name = CLASS_NAMES.get(cls, f"Class {int(cls)}")
        print(f"  │ {name:>13s} │ {intra_euc.mean():7.3f}    {inter_euc.mean():7.3f}   "
              f"│ {intra_hyp.mean():7.3f}    {inter_hyp.mean():7.3f}   │")

    print("  ├───────────────┴──────────────────────┴──────────────────────┤")

    # Ratio: lower = better separation
    intra_e = np.mean([D_euc[np.ix_(y_sub == c, y_sub == c)][np.triu_indices((y_sub == c).sum(), k=1)].mean()
                       for c in classes])
    inter_e = np.mean([D_euc[np.ix_(y_sub == c, y_sub != c)].mean() for c in classes])
    intra_h = np.mean([D_hyp[np.ix_(y_sub == c, y_sub == c)][np.triu_indices((y_sub == c).sum(), k=1)].mean()
                       for c in classes])
    inter_h = np.mean([D_hyp[np.ix_(y_sub == c, y_sub != c)].mean() for c in classes])

    ratio_e = intra_e / inter_e
    ratio_h = intra_h / inter_h

    print(f"  │  Intra/Inter ratio   Euclidean: {ratio_e:.4f}   "
          f"Hyperbolic: {ratio_h:.4f}          │")

    sil_euc = silhouette_score(D_euc, y_sub, metric="precomputed")
    sil_hyp = silhouette_score(D_hyp, y_sub, metric="precomputed")

    print(f"  │  Silhouette score    Euclidean: {sil_euc:.4f}   "
          f"Hyperbolic: {sil_hyp:.4f}          │")
    print("  └────────────────────────────────────────────────────────────┘")

    # Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    x_pos = np.arange(len(classes))
    width = 0.35
    intra_euc_vals, intra_hyp_vals = [], []
    for cls in classes:
        mask = y_sub == cls
        intra_euc_vals.append(
            D_euc[np.ix_(mask, mask)][np.triu_indices(mask.sum(), k=1)].mean())
        intra_hyp_vals.append(
            D_hyp[np.ix_(mask, mask)][np.triu_indices(mask.sum(), k=1)].mean())
    ax.bar(x_pos - width / 2, intra_euc_vals, width, label="Euclidean",
           color="#3498db", alpha=0.8)
    ax.bar(x_pos + width / 2, intra_hyp_vals, width, label="Hyperbolic",
           color="#e74c3c", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([CLASS_NAMES.get(c, c) for c in classes], fontsize=11)
    ax.set_ylabel("Mean Intra-class Distance", fontsize=12)
    ax.set_title("Intra-class Compactness", fontsize=13)
    ax.legend(fontsize=11)

    ax = axes[1]
    metrics = ["Intra/Inter\nRatio ↓", "Silhouette\nScore ↑"]
    euc_vals = [ratio_e, sil_euc]
    hyp_vals = [ratio_h, sil_hyp]
    x_pos = np.arange(len(metrics))
    ax.bar(x_pos - width / 2, euc_vals, width, label="Euclidean",
           color="#3498db", alpha=0.8)
    ax.bar(x_pos + width / 2, hyp_vals, width, label="Hyperbolic",
           color="#e74c3c", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_title("Geometric Separation Quality", fontsize=13)
    ax.legend(fontsize=11)

    for a in axes:
        a.grid(axis="y", alpha=0.3)

    fig.suptitle("Euclidean vs Hyperbolic Distance Analysis — TBI-MH103",
                 fontsize=14)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "distance_analysis.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")

    return D_hyp, D_euc, y_sub


# ════════════════════════════════════════════════════════════════
#  Analysis 4 – Gromov δ-Hyperbolicity
# ════════════════════════════════════════════════════════════════

def gromov_delta(D, n_samples=200, seed=42):
    """Estimate Gromov δ-hyperbolicity from a distance matrix.

    For every quadruple (a,b,c,d) compute the three sums:
       S1 = D[a,b] + D[c,d]
       S2 = D[a,c] + D[b,d]
       S3 = D[a,d] + D[b,c]
    Sort descending.  δ_quad = (S_max - S_second) / 2.
    Global δ = max over all sampled quadruples.

    Lower δ → more tree-like → better fit for hyperbolic geometry.
    """
    rng = np.random.default_rng(seed)
    n = min(len(D), n_samples)
    idx = rng.choice(len(D), n, replace=False)
    D_sub = D[np.ix_(idx, idx)]

    delta_max = 0.0
    indices = np.arange(n)

    n_quads = min(200_000, n * (n - 1) * (n - 2) * (n - 3) // 24)
    quads = set()
    while len(quads) < n_quads:
        q = tuple(sorted(rng.choice(n, 4, replace=False)))
        quads.add(q)

    for a, b, c, d in quads:
        s1 = D_sub[a, b] + D_sub[c, d]
        s2 = D_sub[a, c] + D_sub[b, d]
        s3 = D_sub[a, d] + D_sub[b, c]
        sums = sorted([s1, s2, s3], reverse=True)
        delta_q = (sums[0] - sums[1]) / 2.0
        if delta_q > delta_max:
            delta_max = delta_q

    return delta_max


def hyperbolicity_analysis(D_hyp, D_euc):
    """Compute and compare Gromov δ for both geometries."""
    print("\n  Computing Gromov δ-hyperbolicity (sampling quadruples)...")
    delta_euc = gromov_delta(D_euc, n_samples=200)
    delta_hyp = gromov_delta(D_hyp, n_samples=200)

    diam_euc = D_euc.max()
    diam_hyp = D_hyp.max()
    rel_euc = delta_euc / diam_euc if diam_euc > 0 else 0
    rel_hyp = delta_hyp / diam_hyp if diam_hyp > 0 else 0

    print("\n  ┌───────────────────────────────────────────────────┐")
    print("  │         Gromov δ-Hyperbolicity Analysis           │")
    print("  ├─────────────────────┬─────────────┬───────────────┤")
    print("  │                     │  Euclidean  │  Hyperbolic   │")
    print("  ├─────────────────────┼─────────────┼───────────────┤")
    print(f"  │  δ (absolute)       │  {delta_euc:9.4f}  │  {delta_hyp:11.4f}  │")
    print(f"  │  Diameter           │  {diam_euc:9.4f}  │  {diam_hyp:11.4f}  │")
    print(f"  │  δ / Diameter       │  {rel_euc:9.4f}  │  {rel_hyp:11.4f}  │")
    print("  └─────────────────────┴─────────────┴───────────────┘")
    print("  Interpretation: δ/diam closer to 0 → more tree-like (hyperbolic).")
    print(f"  Euclidean relative δ:  {rel_euc:.4f}")
    print(f"  Hyperbolic relative δ: {rel_hyp:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Euclidean\n$\\delta / \\mathrm{diam}$",
                    "Hyperbolic\n$\\delta / \\mathrm{diam}$"],
                   [rel_euc, rel_hyp],
                   color=["#3498db", "#e74c3c"], alpha=0.85, width=0.5)
    for b, v in zip(bars, [rel_euc, rel_hyp]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Relative δ-hyperbolicity  ($\\delta$ / diameter)", fontsize=12)
    ax.set_title("Gromov δ-Hyperbolicity — TBI-MH103", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gromov_delta.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  GEOMETRIC ANALYSIS — HICL-TBI (TBI-MH103)")
    print("=" * 65)

    print("\n[1/5] Loading & preparing data...")
    X_imp, X_euc, X_hyp, y = load_and_prepare()

    print("\n[2/5] Radial distribution analysis...")
    plot_radial_distribution(X_hyp, y)

    print("\n[3/5] Poincaré disk embedding...")
    plot_poincare_disk(X_hyp, y)

    print("\n[4/5] Distance & silhouette analysis...")
    D_hyp, D_euc, y_sub = distance_analysis(X_euc, X_hyp, y)

    print("\n[5/5] Gromov δ-hyperbolicity...")
    hyperbolicity_analysis(D_hyp, D_euc)

    print("\n" + "=" * 65)
    print(f"  All outputs saved to  ./{OUT_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
