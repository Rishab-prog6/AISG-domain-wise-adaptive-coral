# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Improved DWA_CORAL and lightweight variants.

DWA_CORAL (v2) key fixes over the original implementation:
1. CORAL alignment is normalized by 4 * d^2 (standard CORAL scaling). Without
   this, ||C_e - C_bar||_F^2 is ~O(d^2) and dwarfs the cross-entropy loss,
   so the optimizer mostly minimizes CORAL and ignores the labels.
2. Adaptive weights use a numerically stable softmax (subtract the max of
   the detached losses before exponentiating).
3. A warmup phase keeps the per-domain weights uniform for the first
   `dwa_coral_warmup` steps so the classifier sees every domain equally
   before the adaptive weighting kicks in.

Variants (all share the same CORAL/normalization machinery, they only differ
in how the per-domain weights are computed and applied):

* DWA_CORAL_ALIGNONLY
    Classification loss stays an unweighted ERM-style average; adaptive
    weights are applied to the CORAL alignment term only. This avoids
    over-emphasizing high-loss domains in the supervised signal.

* DWA_CORAL_CLIPPED
    Same as DWA_CORAL (weighted classification + weighted alignment) but the
    softmax weights are clamped to [dwa_weight_min, dwa_weight_max] and then
    renormalized so no single domain can dominate.

* DWA_CORAL_EMA
    Same as DWA_CORAL but the weights are computed from an exponential moving
    average of per-domain losses instead of the noisy current-batch loss.

* DWA_CORAL_MIXED_LOSSGAP_ALIGNONLY
    ALIGNONLY-style (unweighted classification loss). The CORAL weights blend
    a uniform vector with a softmax over a (loss, coral-gap) score, so the
    alignment looks at *both* classification difficulty and current covariance
    mismatch. The uniform mix preserves CORAL-like stability.
"""

import torch
import torch.nn.functional as F

from domainbed.algorithms import ERM


ALGORITHMS = [
    "DWA_CORAL",
    "DWA_CORAL_ALIGNONLY",
    "DWA_CORAL_CLIPPED",
    "DWA_CORAL_EMA",
    "DWA_CORAL_MIXED_LOSSGAP_ALIGNONLY",
]


class _DWACoralBase(ERM):
    """Shared machinery for the DWA_CORAL family.

    Subclasses override `_compute_weights` (and optionally the class attribute
    `weight_cls_loss`). Weights returned by `_compute_weights` must be detached
    (they are derived from detached losses) and sum to 1.
    """

    # If True the classification loss is the weighted sum over domains;
    # if False it is the plain (ERM-style) mean over domains.
    weight_cls_loss = True

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(_DWACoralBase, self).__init__(
            input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains
        self.register_buffer(
            "update_count", torch.tensor(0, dtype=torch.long))

    # ----- CORAL helpers ---------------------------------------------------
    @staticmethod
    def _covariance(features):
        n = features.size(0)
        centered = features - features.mean(dim=0, keepdim=True)
        if n <= 1:
            return centered.t().matmul(centered)
        return centered.t().matmul(centered) / (n - 1)

    def _coral_terms(self, features):
        """Per-domain ||C_e - C_bar||_F^2, normalized by 4 * d^2."""
        feature_dim = features[0].shape[1]
        coral_scale = 4.0 * float(feature_dim) * float(feature_dim)

        covariances = torch.stack([self._covariance(z) for z in features])
        mean_covariance = covariances.mean(dim=0)
        return torch.stack([
            ((c_e - mean_covariance) ** 2).sum() / coral_scale
            for c_e in covariances
        ])

    # ----- weighting helpers ----------------------------------------------
    def _softmax_weights(self, detached_losses):
        """Numerically stable softmax(tau * detached_losses)."""
        tau = self.hparams["dwa_coral_tau"]
        logits = tau * (detached_losses - detached_losses.max())
        return F.softmax(logits, dim=0)

    def _in_warmup(self):
        warmup = int(self.hparams.get("dwa_coral_warmup", 0))
        return warmup > 0 and self.update_count.item() < warmup

    @staticmethod
    def _uniform_weights(reference):
        n = reference.shape[0]
        return reference.new_full((n,), 1.0 / n)

    def _compute_weights(self, domain_losses):
        """Return a detached weight vector (sums to 1). Overridden per variant."""
        raise NotImplementedError

    # ----- shared update ---------------------------------------------------
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
        weights = self._compute_weights(domain_losses)

        if self.weight_cls_loss:
            cls_loss = torch.sum(weights * domain_losses)
        else:
            cls_loss = domain_losses.mean()

        coral_terms = self._coral_terms(features)
        coral_loss = torch.sum(weights * coral_terms)

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


class DWA_CORAL(_DWACoralBase):
    """Domain-wise Adaptive CORAL (v2): weighted classification + alignment."""

    weight_cls_loss = True

    def _compute_weights(self, domain_losses):
        detached = domain_losses.detach()
        if self._in_warmup():
            return self._uniform_weights(detached)
        return self._softmax_weights(detached)


class DWA_CORAL_ALIGNONLY(_DWACoralBase):
    """Adaptive weights drive the CORAL alignment only.

    The classification loss is the unweighted mean over source domains, so the
    classifier keeps balanced ERM-like supervision while alignment effort is
    still concentrated on the hard (high-loss) domains.
    """

    weight_cls_loss = False

    def _compute_weights(self, domain_losses):
        return self._softmax_weights(domain_losses.detach())


class DWA_CORAL_CLIPPED(_DWACoralBase):
    """DWA_CORAL with adaptive weights clamped so no domain dominates."""

    weight_cls_loss = True

    def _compute_weights(self, domain_losses):
        detached = domain_losses.detach()
        if self._in_warmup():
            return self._uniform_weights(detached)

        weights = self._softmax_weights(detached)
        w_min = self.hparams["dwa_weight_min"]
        w_max = self.hparams["dwa_weight_max"]
        weights = torch.clamp(weights, min=w_min, max=w_max)
        return weights / weights.sum()


class DWA_CORAL_EMA(_DWACoralBase):
    """DWA_CORAL whose weights come from an EMA of per-domain losses."""

    weight_cls_loss = True

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DWA_CORAL_EMA, self).__init__(
            input_shape, num_classes, num_domains, hparams)
        self.register_buffer("ema_loss", torch.zeros(num_domains))
        self.register_buffer(
            "ema_initialized", torch.tensor(False))

    def _compute_weights(self, domain_losses):
        detached = domain_losses.detach()
        alpha = self.hparams["dwa_ema_alpha"]

        if not bool(self.ema_initialized):
            self.ema_loss.copy_(detached)
            self.ema_initialized.fill_(True)
        else:
            self.ema_loss.mul_(alpha).add_(detached, alpha=1.0 - alpha)

        if self._in_warmup():
            return self._uniform_weights(detached)
        return self._softmax_weights(self.ema_loss)


class DWA_CORAL_MIXED_LOSSGAP_ALIGNONLY(_DWACoralBase):
    """ALIGNONLY classifier + mixed (uniform / loss+coral-gap) CORAL weights.

    The CORAL weights are
        w = (1 - gamma) * uniform + gamma * softmax(tau * score)
    where
        score = alpha * zscore(L_e) + (1 - alpha) * zscore(coral_term_e)
    so the alignment focuses both on high-loss domains and on domains whose
    feature covariance is currently furthest from the mean. gamma controls
    how aggressive the deviation from uniform CORAL is.

    `update()` is overridden because the weights depend on `coral_terms`,
    which the base `_compute_weights(domain_losses)` hook does not receive,
    and because we log per-domain weights and coral terms for diagnostics.
    """

    weight_cls_loss = False  # informational; we override update()

    @staticmethod
    def _zscore(v):
        # +1e-8 keeps the result finite when all entries are identical
        # (then numerator is 0 and the output is 0, not NaN).
        return (v - v.mean()) / (v.std(unbiased=False) + 1e-8)

    def _mixed_weights(self, detached_losses, detached_coral_terms):
        alpha = self.hparams["dwa_score_alpha"]
        gamma = self.hparams["dwa_mix_gamma"]
        tau = self.hparams["dwa_coral_tau"]

        loss_score = self._zscore(detached_losses)
        gap_score = self._zscore(detached_coral_terms)
        score = alpha * loss_score + (1.0 - alpha) * gap_score

        w_adapt = F.softmax(tau * score, dim=0)
        w_uniform = self._uniform_weights(detached_losses)

        weights = (1.0 - gamma) * w_uniform + gamma * w_adapt
        return weights / weights.sum()

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
        coral_terms = self._coral_terms(features)

        weights = self._mixed_weights(
            domain_losses.detach(), coral_terms.detach())

        cls_loss = domain_losses.mean()
        coral_loss = torch.sum(weights * coral_terms)
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

        metrics = {
            "loss": loss.item(),
            "cls_loss": cls_loss.item(),
            "coral_loss": coral_loss.item(),
            "risk_var": risk_var.item(),
            "min_domain_loss": domain_losses.min().item(),
            "max_domain_loss": domain_losses.max().item(),
        }
        for i in range(weights.shape[0]):
            metrics[f"weight_{i}"] = weights[i].item()
            metrics[f"coral_term_{i}"] = coral_terms[i].item()
        return metrics
