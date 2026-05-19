# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn.functional as F

from domainbed.algorithms import ERM


ALGORITHMS = [
    "DWA_CORAL",
]


class DWA_CORAL(ERM):
    """
    Domain-wise Adaptive CORAL.

    Each minibatch is treated as one source domain. Domains with higher
    detached classification loss receive larger adaptive weights.
    """

    @staticmethod
    def _covariance(features):
        centered = features - features.mean(dim=0, keepdim=True)
        if features.size(0) <= 1:
            return centered.t().matmul(centered)
        return centered.t().matmul(centered) / (features.size(0) - 1)

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
        weights = F.softmax(
            self.hparams["dwa_coral_tau"] * domain_losses.detach(), dim=0)

        cls_loss = torch.sum(weights * domain_losses)

        covariances = torch.stack([
            self._covariance(z_e) for z_e in features
        ])
        mean_covariance = covariances.mean(dim=0)
        coral_penalties = torch.stack([
            torch.linalg.matrix_norm(c_e - mean_covariance, ord="fro").pow(2)
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

        return {
            "loss": loss.item(),
            "cls_loss": cls_loss.item(),
            "coral_loss": coral_loss.item(),
            "risk_var": risk_var.item(),
            "min_domain_loss": domain_losses.min().item(),
            "max_domain_loss": domain_losses.max().item(),
        }
