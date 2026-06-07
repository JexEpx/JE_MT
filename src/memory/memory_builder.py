import warnings

import numpy as np
import torch
import torch.nn.functional as F

from src.memory.open_set_defaults import apply_open_set_defaults
from src.memory.prototype_memory import PrototypeMemory
from src.utils.image_ids import parse_image_id_flexible

DEFAULT_SUPPORT_MIN_COSINE = 0.15


def _crop_image_id(crop):
    try:
        return parse_image_id_flexible(crop.get("image_path", ""))
    except ValueError:
        return None


def _stable_crop_order(crops):
    return sorted(crops, key=lambda c: int(c["feature_index"]))


def filter_pool_by_class_mean_cosine(
    pool,
    features,
    min_cosine,
    *,
    min_required=1,
    log_fn=None,
):
    # Drop crops with cosine similarity to the class mean below min_cosine
    if not pool:
        return []

    min_cosine = float(min_cosine)
    if min_cosine <= 0.0:
        return list(pool)

    idxs = [int(c["feature_index"]) for c in pool]
    emb = torch.from_numpy(np.asarray(features[idxs], dtype=np.float32)).float()
    emb = F.normalize(emb, dim=1)
    mean = F.normalize(emb.mean(dim=0), dim=0)
    sim = emb @ mean

    filtered = [pool[i] for i in range(len(pool)) if float(sim[i]) >= min_cosine]
    if len(filtered) >= max(1, int(min_required)):
        return filtered

    if log_fn is not None:
        log_fn(
            f"[fit_memory] outlier filter kept {len(filtered)}/{len(pool)} crops "
            f"(min_cosine={min_cosine}); using full pool"
        )
    else:
        warnings.warn(
            f"outlier filter kept {len(filtered)}/{len(pool)} crops; using full pool",
            stacklevel=2,
        )
    return list(pool)


def select_indices_kmeans(
    pool,
    features,
    k,
    *,
    random_seed=153,
    min_cosine=DEFAULT_SUPPORT_MIN_COSINE,
    min_required=1,
    log_fn=None,
):
    # Outlier filter, k-means on embeddings, one nearest crop per cluster (one per image)
    if not pool:
        return []

    pool = filter_pool_by_class_mean_cosine(
        pool,
        features,
        min_cosine,
        min_required=min_required,
        log_fn=log_fn,
    )
    if not pool:
        return []

    k = max(1, min(int(k), len(pool)))
    feature_indices = [int(c["feature_index"]) for c in pool]
    emb = torch.from_numpy(np.asarray(features[feature_indices], dtype=np.float32)).float()
    emb = F.normalize(emb, dim=1)

    rng = torch.Generator()
    rng.manual_seed(int(random_seed))
    centroids = PrototypeMemory._kmeans_unit_sphere(emb, k, generator=rng)
    sim = emb @ centroids.T

    chosen: list[int] = []
    used_feat: set[int] = set()
    used_images: set = set()

    for cluster in range(k):
        order = sim[:, cluster].argsort(descending=True).tolist()
        for rank in order:
            fi = feature_indices[rank]
            if fi in used_feat:
                continue
            img_id = _crop_image_id(pool[rank])
            if img_id is not None and img_id in used_images:
                continue
            chosen.append(fi)
            used_feat.add(fi)
            if img_id is not None:
                used_images.add(img_id)
            break

    if len(chosen) < k:
        for rank in sim.max(dim=1).values.argsort(descending=True).tolist():
            fi = feature_indices[rank]
            if fi in used_feat:
                continue
            img_id = _crop_image_id(pool[rank])
            if img_id is not None and img_id in used_images:
                continue
            chosen.append(fi)
            used_feat.add(fi)
            if img_id is not None:
                used_images.add(img_id)
            if len(chosen) >= k:
                break

    return chosen[:k]


def select_class_feature_indices(
    crops,
    *,
    n_support: int,
    exemplar_mode: bool,
    exemplar_max_per_class: int,
    features,
    random_seed: int = 153,
    support_min_cosine: float = DEFAULT_SUPPORT_MIN_COSINE,
    log_fn=None,
):
    if features is None:
        raise ValueError("features array is required for embedding-based selection")

    pool = _stable_crop_order(crops)
    if not exemplar_mode:
        return select_indices_kmeans(
            pool,
            features,
            n_support,
            random_seed=random_seed,
            min_cosine=support_min_cosine,
            min_required=n_support,
            log_fn=log_fn,
        )

    cap = int(exemplar_max_per_class)
    if cap > 0:
        return select_indices_kmeans(
            pool,
            features,
            cap,
            random_seed=random_seed,
            min_cosine=support_min_cosine,
            min_required=min(1, cap),
            log_fn=log_fn,
        )

    filtered = filter_pool_by_class_mean_cosine(
        pool,
        features,
        support_min_cosine,
        min_required=1,
        log_fn=log_fn,
    )
    return [int(c["feature_index"]) for c in filtered]


def build_memory(features, class_crops, class_names, n_support, mem_cfg, log_fn=None):
    mem_cfg = apply_open_set_defaults(mem_cfg)
    exemplar_mode = bool(mem_cfg.get("exemplar_mode", False))
    update_mode = "append_only" if exemplar_mode else "ema"

    memory = PrototypeMemory(
        feature_dim=features.shape[1],
        device=mem_cfg.get("device", "cpu"),
        use_mhn_classify=mem_cfg.get("use_mhn_classify"),
        use_mhn_refine=mem_cfg.get("use_mhn_refine"),
        mhn_beta=float(mem_cfg.get("mhn_beta", 10.0)),
        use_global_threshold=bool(mem_cfg.get("use_global_threshold", True)),
        use_proto_margin=bool(mem_cfg.get("use_proto_margin", True)),
        open_set_threshold_mode=str(mem_cfg.get("open_set_threshold_mode", "known_only")),
        open_set_calib_known_percentile=float(
            mem_cfg.get("open_set_calib_known_percentile", 5.0)
        ),
        open_set_calib_known_margin_percentile=float(
            mem_cfg.get("open_set_calib_known_margin_percentile", 5.0)
        ),
        open_set_margin_mode=str(mem_cfg.get("open_set_margin_mode", "class_aware")),
        open_set_gate_mode=str(mem_cfg.get("open_set_gate_mode", "cosine_margin")),
        open_set_maha_mode=str(mem_cfg.get("open_set_maha_mode", "off")),
        open_set_maha_distance=str(mem_cfg.get("open_set_maha_distance", "min_class")),
        open_set_maha_eps=float(mem_cfg.get("open_set_maha_eps", 1e-4)),
        open_set_calib_maha_percentile=float(mem_cfg.get("open_set_calib_maha_percentile", 95.0)),
        random_seed=int(mem_cfg.get("random_seed", 153)),
        update_mode=update_mode,
    )

    support_set = set()
    support_emb_per_class: dict[int, torch.Tensor] = {}
    num_prototypes = max(1, int(mem_cfg.get("num_prototypes", 1)))
    prototype_init = str(mem_cfg.get("prototype_init", "kmeans"))
    exemplar_max_per_class = int(mem_cfg.get("exemplar_max_per_class", 0))
    random_seed = int(mem_cfg.get("random_seed", 153))
    support_min_cosine = float(mem_cfg.get("support_min_cosine", DEFAULT_SUPPORT_MIN_COSINE))

    for cls in class_names:
        crops = class_crops.get(cls, [])
        idxs = select_class_feature_indices(
            crops,
            n_support=n_support,
            exemplar_mode=exemplar_mode,
            exemplar_max_per_class=exemplar_max_per_class,
            features=features,
            random_seed=random_seed,
            support_min_cosine=support_min_cosine,
            log_fn=log_fn,
        )

        if not idxs:
            continue

        class_num_prototypes = len(idxs) if exemplar_mode else num_prototypes
        class_init = "examples" if exemplar_mode else prototype_init
        emb = torch.from_numpy(features[idxs]).float()
        cidx = memory.add_class(
            emb,
            label=cls,
            num_prototypes=class_num_prototypes,
            prototype_init=class_init,
        )
        support_emb_per_class[cidx] = emb

        support_set.update(idxs)

    if memory.open_set_maha_mode != "off":
        memory.fit_maha_stats(support_emb_per_class)

    return memory, support_set
