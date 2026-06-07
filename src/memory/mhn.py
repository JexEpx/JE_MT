import torch
import torch.nn.functional as F


class MHN:

    def __init__(self, beta=10.0, alpha=0.5):
        self.beta = beta
        self.alpha = alpha

    def _safe_softmax(self, x):
        x = x - x.max(dim=-1, keepdim=True)[0]
        return torch.softmax(x, dim=-1)

    @staticmethod
    def _is_empty(prototypes):
        return prototypes.numel() == 0 or prototypes.shape[0] == 0

    @staticmethod
    def _normalize_prototypes(prototypes):
        if prototypes.numel() == 0 or prototypes.shape[0] == 0:
            return prototypes
        return F.normalize(prototypes, dim=1)

    def _prepare(self, z, prototypes):
        return F.normalize(z, dim=1), self._normalize_prototypes(prototypes)

    def _weights(self, z, prototypes):
        similarity = torch.matmul(z, prototypes.T)
        scaled = self.beta * similarity
        weights = self._safe_softmax(scaled)
        return weights, similarity

    def refine(self, z, prototypes):
        is_single = (z.dim() == 1)
        if is_single:
            z = z.unsqueeze(0)

        if self._is_empty(prototypes):
            weights = z.new_empty((z.shape[0], 0))
            if is_single:
                return z.squeeze(0), weights.squeeze(0)
            return z, weights

        z, prototypes = self._prepare(z, prototypes)
        weights, _ = self._weights(z, prototypes)
        retrieved = torch.matmul(weights, prototypes)

        refined = (1 - self.alpha) * z + self.alpha * retrieved
        refined = F.normalize(refined, dim=1)

        if is_single:
            return refined.squeeze(0), weights.squeeze(0)

        return refined, weights

    def classify(self, z, prototypes, proto_class, num_classes):
        is_single = (z.dim() == 1)
        if is_single:
            z = z.unsqueeze(0)

        if self._is_empty(prototypes):
            weights = z.new_empty((z.shape[0], 0))
            class_scores = z.new_zeros((z.shape[0], num_classes))
        else:
            z, prototypes = self._prepare(z, prototypes)
            weights, _ = self._weights(z, prototypes)
            one_hot = F.one_hot(proto_class, num_classes=num_classes).float().to(z.device)
            class_scores = torch.matmul(weights, one_hot)

        if is_single:
            return class_scores.squeeze(0), weights.squeeze(0)

        return class_scores, weights
