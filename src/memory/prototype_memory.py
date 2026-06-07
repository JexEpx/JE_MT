import inspect

import torch
import torch.nn.functional as F

from src.memory.mahalanobis import (
    OPEN_SET_GATE_MODES,
    OPEN_SET_MAHA_DISTANCE,
    OPEN_SET_MAHA_MODES,
    class_maha_distances,
    fit_diagonal_class_stats,
)
from src.memory.mhn import MHN

UNKNOWN = "UNKNOWN"



class PrototypeMemory:

    def __init__(
        self,
        feature_dim,
        device="cpu",
        use_mhn_classify=True,
        use_mhn_refine=True,
        mhn_beta=10.0,
        use_global_threshold=False,
        use_proto_margin=False,
        proto_margin_min=None,
        open_set_threshold_mode="known_only",
        global_top1_threshold=None,
        margin_threshold=None,
        open_set_calib_known_percentile=5.0,
        open_set_calib_known_margin_percentile=5.0,
        open_set_margin_mode="class_aware",
        open_set_gate_mode="cosine_margin",
        open_set_maha_mode="off",
        open_set_maha_distance="min_class",
        open_set_maha_eps=1e-4,
        open_set_calib_maha_percentile=95.0,
        maha_threshold=None,
        random_seed=153,
        update_mode="ema",
    ):
        self.device = torch.device(device)
        self.feature_dim = feature_dim

        self.prototypes = torch.empty((0, feature_dim), device=self.device)
        self.proto_class = torch.empty(0, dtype=torch.long, device=self.device)

        self.labels = []

        self.use_mhn_classify = bool(use_mhn_classify)
        self.use_mhn_refine = bool(use_mhn_refine)
        self.mhn = MHN(beta=mhn_beta)
        self.use_global_threshold = bool(use_global_threshold)
        self.use_proto_margin = bool(use_proto_margin)
        self.proto_margin_min = (
            None if proto_margin_min is None else float(proto_margin_min)
        )
        self.open_set_threshold_mode = str(open_set_threshold_mode)
        self.global_top1_threshold = (
            None
            if global_top1_threshold is None
            else torch.tensor(float(global_top1_threshold), device=self.device)
        )
        self.margin_threshold = (
            None
            if margin_threshold is None
            else torch.tensor(float(margin_threshold), device=self.device)
        )
        self.open_set_calib_known_percentile = float(open_set_calib_known_percentile)
        self.open_set_calib_known_margin_percentile = float(
            open_set_calib_known_margin_percentile
        )
        self.open_set_calib_use_mhn_refine = False
        self.open_set_calib_use_mhn_classify = False
        self.open_set_gate_use_cosine_class = True
        margin_mode = str(open_set_margin_mode or "class_aware").strip().lower()
        if margin_mode not in {"class_aware", "global_proto"}:
            raise ValueError(
                f"open_set_margin_mode must be class_aware|global_proto, got {margin_mode!r}"
            )
        self.open_set_margin_mode = margin_mode
        gate_mode = str(open_set_gate_mode or "cosine_margin").strip().lower()
        if gate_mode not in OPEN_SET_GATE_MODES:
            raise ValueError(
                f"open_set_gate_mode must be one of {OPEN_SET_GATE_MODES}, got {gate_mode!r}"
            )
        self.open_set_gate_mode = gate_mode
        maha_mode = str(open_set_maha_mode or "off").strip().lower()
        if maha_mode not in OPEN_SET_MAHA_MODES:
            raise ValueError(
                f"open_set_maha_mode must be one of {OPEN_SET_MAHA_MODES}, got {maha_mode!r}"
            )
        self.open_set_maha_mode = maha_mode
        maha_dist = str(open_set_maha_distance or "min_class").strip().lower()
        if maha_dist not in OPEN_SET_MAHA_DISTANCE:
            raise ValueError(
                f"open_set_maha_distance must be one of {OPEN_SET_MAHA_DISTANCE}, got {maha_dist!r}"
            )
        self.open_set_maha_distance = maha_dist
        self.open_set_maha_eps = float(open_set_maha_eps)
        self.open_set_calib_maha_percentile = float(open_set_calib_maha_percentile)
        self.maha_threshold = (
            None
            if maha_threshold is None
            else torch.tensor(float(maha_threshold), device=self.device)
        )
        self.class_means = torch.empty((0, feature_dim), device=self.device)
        self.class_inv_var = torch.empty((0, feature_dim), device=self.device)
        self.random_seed = int(random_seed)
        self.update_mode = str(update_mode or "ema").strip().lower()
        if self.update_mode not in {"ema", "append_only"}:
            raise ValueError(f"Unsupported update_mode: {self.update_mode}")

    @property
    def num_classes(self):
        return len(self.labels)

    # KMEANS
    @staticmethod
    def _kmeans_unit_sphere(embeddings, k, n_iter=30, generator=None):
        n = embeddings.shape[0]
        k = max(1, min(k, n))

        indices = torch.randperm(n, generator=generator, device=embeddings.device)[:k]
        centroids = embeddings[indices].clone()

        for _ in range(n_iter):
            sim = embeddings @ centroids.T
            assign = sim.argmax(dim=1)

            new_centroids = []
            for i in range(k):
                mask = assign == i
                if mask.any():
                    mean = embeddings[mask].mean(dim=0)
                    new_centroids.append(F.normalize(mean, dim=0))
                else:
                    new_centroids.append(centroids[i])

            centroids = torch.stack(new_centroids)

        return F.normalize(centroids, dim=1)

    # ADD CLASS
    def add_class(self, embeddings, label=None, num_prototypes=3, prototype_init="kmeans"):
        embeddings = F.normalize(embeddings.to(self.device), dim=1)

        n = embeddings.shape[0]
        k = max(1, min(num_prototypes, n))
        init_mode = str(prototype_init or "kmeans").strip().lower()

        if init_mode == "examples":
            # Keep top-ranked support examples directly as initial prototypes
            new_prototypes = embeddings[:k]
        elif k == 1:
            new_prototypes = embeddings.mean(dim=0, keepdim=True)
        else:
            rng = torch.Generator(device=embeddings.device)
            rng.manual_seed(self.random_seed + len(self.labels))
            new_prototypes = self._kmeans_unit_sphere(embeddings, k, generator=rng)

        new_prototypes = F.normalize(new_prototypes, dim=1)

        class_idx = len(self.labels)

        self.prototypes = torch.cat([self.prototypes, new_prototypes], dim=0)

        class_ids = torch.full(
            (new_prototypes.shape[0],),
            class_idx,
            dtype=torch.long,
            device=self.device
        )

        self.proto_class = torch.cat([self.proto_class, class_ids], dim=0)

        name = label if label else f"class_{class_idx}"
        self.labels.append(name)

        return class_idx

    @torch.no_grad()
    def _prototype_similarities(self, z):
        # Global max cosine and top-1 vs top-2 margin over all prototypes
        sim = z @ self.prototypes.T
        if sim.numel() == 0:
            z0 = z.squeeze(0) if z.dim() > 1 else z
            return (
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
                torch.tensor(0, dtype=torch.long, device=self.device),
            )
        if z.dim() == 1:
            sim = sim.unsqueeze(0)
        k = min(2, int(sim.shape[-1]))
        top = torch.topk(sim, k=k, dim=-1)
        max_sim = top.values[:, 0]
        margin = (
            top.values[:, 0] - top.values[:, 1]
            if k > 1
            else top.values[:, 0]
        )
        proto_idx = top.indices[:, 0]
        class_idx = self.proto_class[proto_idx]
        return max_sim.squeeze(0), margin.squeeze(0), class_idx.squeeze(0)

    @torch.no_grad()
    def _class_index_for_open_set_gate(self, z: torch.Tensor) -> int:
        # Class-aware gate scores are defined on per-class max cosine (not MHN aggregation).
        if bool(getattr(self, "open_set_gate_use_cosine_class", True)):
            _, cidx = self._cosine_class_scores(z)
            return int(cidx)
        cidx, _ = self._predict_class_index(z)
        return int(cidx.item() if isinstance(cidx, torch.Tensor) else cidx)

    @torch.no_grad()
    def _predict_class_index(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Return (class_idx, confidence) for cosine or MHN classification
        if self.use_mhn_classify:
            class_scores, _ = self.mhn.classify(
                z,
                self.prototypes,
                self.proto_class,
                self.num_classes,
            )
            if class_scores.dim() == 1:
                idx = class_scores.argmax()
                confidence = class_scores[idx]
            else:
                idx = class_scores.argmax(dim=-1)
                confidence = class_scores[
                    torch.arange(class_scores.shape[0], device=class_scores.device), idx
                ]
        else:
            class_scores, cidx = self._cosine_class_scores(z)
            idx = torch.tensor(cidx, device=self.device)
            confidence = class_scores[cidx]

        cidx = idx.squeeze(0) if idx.dim() > 0 else idx
        conf_pc = confidence.squeeze(0) if confidence.dim() > 0 else confidence
        return cidx, conf_pc

    @torch.no_grad()
    def open_set_signals(self, z: torch.Tensor, class_idx) -> tuple[torch.Tensor, torch.Tensor]:

        # Top-1 cosine and margin for UNKNOWN gating at the predicted class

        c = int(class_idx.item()) if isinstance(class_idx, torch.Tensor) else int(class_idx)

        if self.open_set_margin_mode == "class_aware":
            sim = z @ self.prototypes.T
            if sim.dim() > 1:
                sim = sim.squeeze(0)
            s_max = self._per_class_max_similarities(sim)
            top1 = s_max[c]
            if self.num_classes > 1:
                margin = self._per_class_competition_margin(s_max)[c]
            else:
                top1_g, margin_g, _ = self._prototype_similarities(z)
                margin = margin_g
            return top1, margin

        top1, margin, _ = self._prototype_similarities(z)
        return top1, margin

    def _uses_maha_gate(self) -> bool:
        return self.open_set_maha_mode != "off" and self.open_set_gate_mode in (
            "maha",
            "cosine_margin_maha",
        )

    def _uses_cosine_margin_gate(self) -> bool:
        return self.open_set_gate_mode in ("cosine_margin", "cosine_margin_maha")

    @torch.no_grad()
    def fit_maha_stats(self, embeddings_per_class: dict[int, torch.Tensor]) -> None:
        # Fit per-class Gaussian stats from support embeddings
        if self.open_set_maha_mode == "off" or not embeddings_per_class:
            return

        means, inv_var = fit_diagonal_class_stats(
            embeddings_per_class,
            eps=self.open_set_maha_eps,
        )

        self.class_means = means.to(self.device)
        self.class_inv_var = inv_var.to(self.device)

    @torch.no_grad()
    def maha_distances(self, z: torch.Tensor) -> torch.Tensor:
        # Per-class Mahalanobis distances
        if self.class_means.numel() == 0:
            return torch.full((self.num_classes,), float("inf"), device=self.device)
        z = z.to(self.device)
        if z.dim() > 1:
            z = z.squeeze(0)
        return class_maha_distances(z, self.class_means, self.class_inv_var)

    @torch.no_grad()
    def maha_open_set_distance(self, z: torch.Tensor, class_idx) -> torch.Tensor:
        # Distance used for UNKNOWN gating
        dists = self.maha_distances(z)
        if self.open_set_maha_distance == "predicted_class":
            c = int(class_idx.item()) if isinstance(class_idx, torch.Tensor) else int(class_idx)
            return dists[c]
        return dists.min()

    @torch.no_grad()
    def prepare_for_inference(self, z: torch.Tensor) -> torch.Tensor:
        # Normalize and optionally MHN-refine
        z = F.normalize(z.to(self.device), dim=-1)
        if (
            self.use_mhn_refine
            and self.num_classes > 0
            and self.prototypes.numel() > 0
        ):
            z, _ = self.mhn.refine(z, self.prototypes)
        return z

    @torch.no_grad()
    def collect_open_set_signal_and_maha(
        self, z: torch.Tensor
    ) -> tuple[float, float, float | None]:
        # Single predict pass for calibration (top1, margin, optional maha)
        z = self.prepare_for_inference(z)
        gate_cidx = self._class_index_for_open_set_gate(z)
        top1, margin = self.open_set_signals(z, gate_cidx)
        maha = (
            float(self.maha_open_set_distance(z, gate_cidx).item())
            if self._uses_maha_gate()
            else None
        )
        return float(top1.item()), float(margin.item()), maha

    @torch.no_grad()
    def _per_class_max_similarities(self, sim: torch.Tensor) -> torch.Tensor:
        scores = torch.full(
            (self.num_classes,),
            -1e9,
            device=self.device,
            dtype=sim.dtype,
        )
        scores.scatter_reduce_(
            0,
            self.proto_class,
            sim,
            reduce="amax",
            include_self=False,
        )
        return scores

    @torch.no_grad()
    def _per_class_competition_margin(self, s_max: torch.Tensor) -> torch.Tensor:
        # Per-class separation from rivals
        n = int(s_max.numel())
        if n <= 1:
            return torch.zeros_like(s_max)
        mat = s_max.unsqueeze(0).expand(n, n)
        eye = torch.eye(n, device=self.device, dtype=torch.bool)
        max_other = mat.masked_fill(eye, -1e9).max(dim=1).values
        return s_max - max_other

    @torch.no_grad()
    def _cosine_class_scores(self, z: torch.Tensor) -> tuple[torch.Tensor, int]:
        # Per-class max cosine
        sim = z @ self.prototypes.T
        if sim.dim() > 1:
            sim = sim.squeeze(0)
        scores = self._per_class_max_similarities(sim)
        return scores, int(scores.argmax().item())

    def apply_open_set_thresholds(
        self,
        global_top1: float,
        margin: float,
        *,
        method: str = "known_only",
        percentile: float | None = None,
        maha: float | None = None,
    ) -> None:
        self.open_set_threshold_mode = str(method)
        self.global_top1_threshold = torch.tensor(float(global_top1), device=self.device)
        self.margin_threshold = torch.tensor(float(margin), device=self.device)
        if maha is not None:
            self.maha_threshold = torch.tensor(float(maha), device=self.device)
        if percentile is not None:
            self.open_set_calib_known_percentile = float(percentile)
        self.open_set_calib_use_mhn_refine = bool(self.use_mhn_refine)
        gate_cosine = bool(getattr(self, "open_set_gate_use_cosine_class", True))
        self.open_set_calib_use_mhn_classify = (
            False if gate_cosine else bool(self.use_mhn_classify)
        )

    @torch.no_grad()
    def fit_open_set_thresholds(
        self,
        features,
        records: list[dict],
        known_names: list[str],
        *,
        method: str = "known_only",
        known_percentile: float | None = None,
        known_margin_percentile: float | None = None,
        support_set: set[int] | None = None,
        log_fn=print,
    ) -> dict:
        from src.memory.open_set_calibration import (
            collect_open_set_scores,
            fit_open_set_thresholds as _fit,
        )

        max_records = getattr(self, "open_set_calib_max_records", None)
        if max_records == 0:
            max_records = None
        calib_seed = int(getattr(self, "open_set_calib_seed", 0))
        batch = collect_open_set_scores(
            self,
            features,
            records,
            known_names,
            support_set=support_set,
            max_records=max_records,
            seed=calib_seed,
        )
        maha_pct = float(getattr(self, "open_set_calib_maha_percentile", 95.0))
        kn_pct = float(
            known_percentile
            if known_percentile is not None
            else self.open_set_calib_known_percentile
        )
        kn_margin_pct = float(
            known_margin_percentile
            if known_margin_percentile is not None
            else self.open_set_calib_known_margin_percentile
        )
        th = _fit(
            batch,
            known_names,
            method=method,
            maha_percentile=maha_pct,
            known_percentile=kn_pct,
            known_margin_percentile=kn_margin_pct,
            memory=self,
        )
        self.apply_open_set_thresholds(
            th.global_top1,
            th.margin,
            method=th.method,
            percentile=th.percentile,
            maha=th.maha,
        )
        log_fn(
            f"[open_set_calib] method={th.method} n={batch.top1.size} "
            f"refine={self.use_mhn_refine} "
            f"tau_top1={th.global_top1:.4f} tau_margin={th.margin:.4f} "
            f"tau_maha={th.maha if th.maha is not None else float('nan'):.4f} "
            f"calib_H={th.calib_metrics.get('harmonic_mean', 0):.3f}"
        )
        return {
            "method": th.method,
            "global_top1_threshold": th.global_top1,
            "margin_threshold": th.margin,
            "maha_threshold": th.maha,
            "percentile": th.percentile,
            "n_calib": int(batch.top1.size),
            "n_unknown_gt": int(batch.is_unknown_gt.sum()),
            "n_known_gt": int((~batch.is_unknown_gt).sum()),
            **th.calib_metrics,
        }

    def _accept_as_known(self, class_idx, max_sim, margin, *, maha_dist=None) -> bool:
        max_sim_f = float(max_sim.item())
        margin_f = float(margin.item())

        if self._uses_cosine_margin_gate():
            if (
                self.use_global_threshold
                and self.global_top1_threshold is not None
                and max_sim_f <= float(self.global_top1_threshold.item())
            ):
                return False

            skip_margin = (
                self.open_set_margin_mode == "class_aware" and self.num_classes <= 1
            )
            margin_cut = (
                float(self.margin_threshold.item())
                if self.margin_threshold is not None
                else float(self.proto_margin_min)
                if self.proto_margin_min is not None
                else None
            )
            if (
                self.use_proto_margin
                and not skip_margin
                and margin_cut is not None
                and margin_f < margin_cut
            ):
                return False

        if self._uses_maha_gate() and self.maha_threshold is not None:
            tau_maha = float(self.maha_threshold.item())
            if maha_dist is None:
                return False
            maha_f = float(
                maha_dist.item() if isinstance(maha_dist, torch.Tensor) else maha_dist
            )
            if maha_f > tau_maha:
                return False

        return True

    # PREDICT
    @torch.no_grad()
    def predict(self, z, *, prepared: bool = False):
        if self.num_classes == 0:
            return UNKNOWN, 0.0

        if not prepared:
            z = self.prepare_for_inference(z)
        else:
            z = z.to(self.device)
        pred_cidx, conf_pc = self._predict_class_index(z)
        gate_cidx = self._class_index_for_open_set_gate(z)
        max_sim, margin = self.open_set_signals(z, gate_cidx)
        maha_dist = (
            self.maha_open_set_distance(z, gate_cidx)
            if self._uses_maha_gate()
            else None
        )

        if self._accept_as_known(
            gate_cidx, max_sim, margin, maha_dist=maha_dist
        ):
            return self.labels[int(pred_cidx.item())], float(conf_pc.item())

        return UNKNOWN, float(conf_pc.item())

    # UPDATE
    @torch.no_grad()
    def update(self, z, class_idx, tau_update=0.7, tau_new=0.5, alpha=0.1):
        z = F.normalize(z.to(self.device), dim=-1)

        mask = self.proto_class == class_idx
        protos = self.prototypes[mask]

        sims = torch.mv(protos, z)
        best_idx = sims.argmax()
        best_sim = sims[best_idx]

        if self.update_mode == "append_only":
            self.prototypes = torch.cat([self.prototypes, z.unsqueeze(0)], dim=0)
            self.proto_class = torch.cat([
                self.proto_class,
                torch.tensor([class_idx], device=self.device)
            ])
            return True

        if best_sim > tau_update:
            updated = (1 - alpha) * protos[best_idx] + alpha * z

            global_idx = torch.nonzero(mask, as_tuple=True)[0][best_idx]
            self.prototypes[global_idx] = F.normalize(updated, dim=0)
            return True

        if best_sim < tau_new:
            self.prototypes = torch.cat([self.prototypes, z.unsqueeze(0)], dim=0)
            self.proto_class = torch.cat([
                self.proto_class,
                torch.tensor([class_idx], device=self.device)
            ])
            return True

        return False

    def to_state_dict(self):
        return {
            "feature_dim": int(self.feature_dim),
            "device": str(self.device),
            "use_mhn_classify": bool(self.use_mhn_classify),
            "use_mhn_refine": bool(self.use_mhn_refine),
            "mhn_beta": float(self.mhn.beta),
            "mhn_alpha": float(self.mhn.alpha),
            "use_global_threshold": bool(self.use_global_threshold),
            "use_proto_margin": bool(self.use_proto_margin),
            "proto_margin_min": self.proto_margin_min,
            "open_set_threshold_mode": self.open_set_threshold_mode,
            "global_top1_threshold": (
                None
                if self.global_top1_threshold is None
                else float(self.global_top1_threshold.item())
            ),
            "margin_threshold": (
                None
                if self.margin_threshold is None
                else float(self.margin_threshold.item())
            ),
            "open_set_calib_known_percentile": float(self.open_set_calib_known_percentile),
            "open_set_calib_known_margin_percentile": float(
                self.open_set_calib_known_margin_percentile
            ),
            "open_set_calib_use_mhn_refine": bool(self.open_set_calib_use_mhn_refine),
            "open_set_calib_use_mhn_classify": bool(self.open_set_calib_use_mhn_classify),
            "open_set_gate_use_cosine_class": bool(
                getattr(self, "open_set_gate_use_cosine_class", True)
            ),
            "open_set_margin_mode": self.open_set_margin_mode,
            "open_set_gate_mode": self.open_set_gate_mode,
            "open_set_maha_mode": self.open_set_maha_mode,
            "open_set_maha_distance": self.open_set_maha_distance,
            "open_set_maha_eps": float(self.open_set_maha_eps),
            "open_set_calib_maha_percentile": float(self.open_set_calib_maha_percentile),
            "maha_threshold": (
                None
                if self.maha_threshold is None
                else float(self.maha_threshold.item())
            ),
            "class_means": self.class_means.detach().cpu(),
            "class_inv_var": self.class_inv_var.detach().cpu(),
            "random_seed": int(self.random_seed),
            "update_mode": str(self.update_mode),
            "prototypes": self.prototypes.detach().cpu(),
            "proto_class": self.proto_class.detach().cpu(),
            "labels": list(self.labels),
        }

    @classmethod
    def from_state_dict(cls, state, device=None):
        target_device = device if device is not None else state.get("device", "cpu")
        obj = cls(
            feature_dim=int(state["feature_dim"]),
            device=target_device,
            use_mhn_classify=bool(state.get("use_mhn_classify", True)),
            use_mhn_refine=bool(state.get("use_mhn_refine", True)),
            mhn_beta=float(state.get("mhn_beta", 10.0)),
            use_global_threshold=bool(state.get("use_global_threshold", False)),
            use_proto_margin=bool(state.get("use_proto_margin", False)),
            proto_margin_min=state.get("proto_margin_min"),
            open_set_threshold_mode=str(
                state.get("open_set_threshold_mode", "known_only")
            ),
            global_top1_threshold=state.get("global_top1_threshold"),
            margin_threshold=state.get("margin_threshold"),
            open_set_calib_known_percentile=float(
                state.get("open_set_calib_known_percentile", 5.0)
            ),
            open_set_calib_known_margin_percentile=float(
                state.get("open_set_calib_known_margin_percentile", 5.0)
            ),
            open_set_margin_mode=str(state.get("open_set_margin_mode", "class_aware")),
            open_set_gate_mode=str(state.get("open_set_gate_mode", "cosine_margin")),
            open_set_maha_mode=str(state.get("open_set_maha_mode", "off")),
            open_set_maha_distance=str(state.get("open_set_maha_distance", "min_class")),
            open_set_maha_eps=float(state.get("open_set_maha_eps", 1e-4)),
            open_set_calib_maha_percentile=float(state.get("open_set_calib_maha_percentile", 95.0)),
            maha_threshold=state.get("maha_threshold"),
            random_seed=int(state.get("random_seed", 153)),
            update_mode=str(state.get("update_mode", "ema")),
        )
        obj.mhn.alpha = float(state.get("mhn_alpha", obj.mhn.alpha))

        obj.prototypes = state["prototypes"].to(obj.device)
        obj.proto_class = state["proto_class"].to(obj.device)
        obj.labels = list(state["labels"])
        if "class_means" in state:
            obj.class_means = state["class_means"].to(obj.device)
            obj.class_inv_var = state["class_inv_var"].to(obj.device)
        obj.open_set_calib_use_mhn_refine = bool(
            state.get("open_set_calib_use_mhn_refine", False)
        )
        obj.open_set_calib_use_mhn_classify = bool(
            state.get("open_set_calib_use_mhn_classify", False)
        )
        obj.open_set_gate_use_cosine_class = bool(
            state.get("open_set_gate_use_cosine_class", True)
        )
        return obj

    def save(self, path):
        torch.save(self.to_state_dict(), path)

    @classmethod
    def load(cls, path, device=None):
        load_kw = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            # Memory checkpoints include non-tensor metadata
            load_kw["weights_only"] = False
        state = torch.load(path, **load_kw)
        return cls.from_state_dict(state, device=device)
