# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Improved DWA_CORAL.

Key fixes over the v1 implementation:
1. CORAL alignment is normalized by 4 * d^2 (standard CORAL scaling). Without
   this, ||C_e - C_bar||_F^2 is ~O(d^2) and dwarfs the cross-entropy loss,
   so the optimizer mostly minimizes CORAL and ignores the labels.
2. Adaptive weights use a numerically stable softmax (subtract the max of
   the detached losses before exponentiating).
3. A warmup phase keeps the per-domain weights uniform for the first
   `dwa_coral_warmup` steps so the classifier sees every domain equally
   before the adaptive weighting kicks in.
4. Hparam defaults are retuned to match the new normalization.
"""

import torch
import torch.nn.functional as F

from domainbed.algorithms import ERM


ALGORITHMS = [
    "DWA_CORAL",
]


class DWA_CORAL(ERM):
    """Domain-wise Adaptive CORAL (improved)."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DWA_CORAL, self).__init__(
            input_shape, num_classes, num_domains, hparams)
        self.register_buffer(
            "update_count", torch.tensor(0, dtype=torch.long))

    @staticmethod
    def _covariance(features):
        n = features.size(0)
        centered = features - features.mean(dim=0, keepdim=True)
        if n <= 1:
            return centered.t().matmul(centered)
        return centered.t().matmul(centered) / (n - 1)

    def _adaptive_weights(self, domain_losses):
        n_domains = domain_losses.shape[0]
        warmup = int(self.hparams.get("dwa_coral_warmup", 0))
        if warmup > 0 and self.update_count.item() < warmup:
            return domain_losses.new_full((n_domains,), 1.0 / n_domains)

        tau = self.hparams["dwa_coral_tau"]
        det = domain_losses.detach()
        logits = tau * (det - det.max())
        return F.softmax(logits, dim=0)

    def update(self, minibatches, unlabeled=None):
        features = []
        domain_losses = []

        for x_e, y_e in minibatches:
            z_e = self.featurizer(x_e)
            logits_e = self.classifier(z_e)
            loss_e = F.cross_entropy(logits_e, y_e)

            features.append(z_e)
            domain_losses.append(loss_e)

        domain_losses = torch.stack(domain_losses)
        weights = self._adaptive_weights(domain_losses)

        cls_loss = torch.sum(weights * domain_losses)

        feature_dim = features[0].shape[1]
        coral_scale = 4.0 * float(feature_dim) * float(feature_dim)

        covariances = torch.stack([self._covariance(z) for z in features])
        mean_covariance = covariances.mean(dim=0)
        coral_penalties = torch.stack([
            ((c_e - mean_covariance) ** 2).sum() / coral_scale
            for c_e in covariances
        ])
        coral_loss = torch.sum(weights * coral_penalties)

        risk_var = torch.var(domain_losses, unbiased=False)

        loss = (
            cls_loss
            + self.hparams["dwa_coral_lambda"] * coral_loss
            + self.hparams["dwa_coral_beta"] * risk_var
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        return {
            "loss": loss.item(),
            "cls_loss": cls_loss.item(),
            "coral_loss": coral_loss.item(),
            "risk_var": risk_var.item(),
            "min_domain_loss": domain_losses.min().item(),
            "max_domain_loss": domain_losses.max().item(),
        }
