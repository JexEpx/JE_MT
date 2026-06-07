"""Per-class Mahalanobis distance for open-set"""

from __future__ import annotations

import torch
import torch.nn.functional as F

OPEN_SET_MAHA_MODES = ("off", "diagonal")
OPEN_SET_MAHA_DISTANCE = ("min_class", "predicted_class")
OPEN_SET_GATE_MODES = ("cosine_margin", "maha", "cosine_margin_maha")


def _regularized_diag_var(embeddings: torch.Tensor, eps: float) -> torch.Tensor:
    #Per-dimension variance with a floor
    if embeddings.shape[0] <= 1:
        return torch.full((embeddings.shape[1],), eps, device=embeddings.device, dtype=embeddings.dtype)
    var = embeddings.var(dim=0, unbiased=False)
    return var.clamp(min=eps)


@torch.no_grad()
def fit_diagonal_class_stats(
    embeddings_per_class: dict[int, torch.Tensor],
    *,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Fit per-class mean and inverse diagonal covariance.
    if not embeddings_per_class:
        raise ValueError("embeddings_per_class is empty")

    class_ids = sorted(embeddings_per_class)
    dim = int(next(iter(embeddings_per_class.values())).shape[-1])
    device = next(iter(embeddings_per_class.values())).device
    dtype = next(iter(embeddings_per_class.values())).dtype

    means = torch.zeros(len(class_ids), dim, device=device, dtype=dtype)
    inv_var = torch.zeros(len(class_ids), dim, device=device, dtype=dtype)
    for i, cidx in enumerate(class_ids):
        emb = F.normalize(embeddings_per_class[cidx].to(device=device, dtype=dtype), dim=1)
        means[i] = F.normalize(emb.mean(dim=0), dim=0)
        var = _regularized_diag_var(emb, eps)
        inv_var[i] = 1.0 / var
    return means, inv_var


@torch.no_grad()
def class_maha_distances(
    z: torch.Tensor,
    means: torch.Tensor,
    inv_var: torch.Tensor,
) -> torch.Tensor:
    #Per-class distances, z is normalized
    d = z.unsqueeze(0) - means
    return (d * d * inv_var).sum(dim=1)
