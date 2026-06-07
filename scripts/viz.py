import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def set_axes3d_equal_aspect(ax, xyz: np.ndarray) -> None:
    pts = np.asarray(xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return
    pts = pts[:, :3]
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = float((maxs - mins).max())
    if span <= 0:
        span = 1.0
    ctr = (mins + maxs) * 0.5
    half = span * 0.5
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def set_axes2d_equal_centered(ax, xy: np.ndarray) -> None:
    pts = np.asarray(xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return
    pts = pts[:, :2]
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = float((maxs - mins).max())
    if span <= 0:
        span = 1.0
    ctr = (mins + maxs) * 0.5
    half = span * 0.5
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_aspect("equal", adjustable="box")


def sample_indices(n: int, k: int, seed: int) -> np.ndarray:
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    idx = np.random.default_rng(seed).choice(n, k, replace=False)
    idx.sort()
    return idx.astype(np.int64, copy=False)


def reduce_embedding(
    method: str,
    x: np.ndarray,
    plot_dim: int,
    seed: int,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30.0,
    tsne_iters: int = 1000,
) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=plot_dim, random_state=seed).fit_transform(x)

    if method == "umap":
        n_neighbors = min(umap_neighbors, max(2, x.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=plot_dim,
            n_neighbors=n_neighbors,
            min_dist=umap_min_dist,
            metric="cosine",
            random_state=seed,
        )
        return reducer.fit_transform(x)

    if method == "tsne":
        if x.shape[0] < 4:
            raise ValueError("t-SNE requires at least 4 points.")
        n_pca = min(50, x.shape[1], max(1, x.shape[0] - 1))
        z = PCA(n_components=n_pca, random_state=seed).fit_transform(x)
        z = (z - z.mean(0)) / (z.std(0) + 1e-6)
        perplexity = min(tsne_perplexity, max(2.0, float(x.shape[0] - 1)))
        reducer = TSNE(
            n_components=plot_dim,
            perplexity=perplexity,
            max_iter=tsne_iters,
            random_state=seed,
            init="pca",
            learning_rate="auto",
        )
        return reducer.fit_transform(z)

    raise ValueError(f"Unknown reduction method: {method}")
