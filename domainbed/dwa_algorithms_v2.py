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

* DWA_CORAL_ANCHOR_ALIGNONLY
    ALIGNONLY-style. Each step picks the lowest-loss source domain as the
    "anchor", detaches its covariance, and aligns every other source toward
    detach(C_anchor) instead of toward the mean. The mean-targeted alignment
    used by the other variants is dragged by the abstract (Sketch/Cartoon)
    sources; anchoring on the easiest source preserves the natural-image
    manifold, which targets the Art/Photo test envs specifically.

* DWA_CORAL_ANCHOR_ANNEAL
    DWA_CORAL_ANCHOR_ALIGNONLY with the alignment weight linearly annealed
    from dwa_coral_lambda to dwa_coral_lambda_min over dwa_coral_anneal_steps.
    Strong early alignment builds the invariance the abstract test domains
    need; easing it late frees the classifier to refit natural-image detail,
    pulling Art/Photo back toward the ERM ceiling.
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
    "DWA_CORAL_ANCHOR_ALIGNONLY",
    "DWA_CORAL_ANCHOR_ANNEAL",
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


class DWA_CORAL_ANCHOR_ALIGNONLY(_DWACoralBase):
    """ALIGNONLY classifier + anchor-based CORAL alignment.

    Each step the source domain with the lowest detached loss becomes the
    `anchor`. Its covariance is detached and treated as the alignment target,
    so anchor features are pulled only by classification gradient. Every
    other source contributes
        coral_term_e = ||C_e - detach(C_anchor)||_F^2 / (4 d^2)
    and the non-anchor terms are combined with mixed (uniform / softmax)
    weights:
        w_other = (1 - gamma) * uniform + gamma * softmax(tau * detach(L_other))
        coral_loss = sum_{e != anchor} w_other_e * coral_term_e

    Classification loss stays the unweighted ERM-style mean (ALIGNONLY).

    Motivation: in PACS the lowest-loss source is typically the most natural-
    image-like one (Photo when test=Art, Art when test=Photo). Mean-targeted
    CORAL is dragged by the abstract sources (Sketch/Cartoon); anchoring
    preserves the natural-image manifold and should help Art/Photo
    generalization at a likely small cost on Sketch.
    """

    weight_cls_loss = False  # informational; we override update()

    def _coral_lambda(self):
        """Effective alignment weight at the current step.

        Constant for ANCHOR; subclasses (e.g. ANNEAL) override this to follow
        a schedule. Logged each step as `coral_lambda`.
        """
        return self.hparams["dwa_coral_lambda"]

    def _other_weights(self, detached_losses, anchor_idx):
        """Return a length-n weight vector with 0 at anchor and a normalized
        (uniform / softmax) mix over the n-1 non-anchor positions."""
        n = detached_losses.shape[0]
        weights = detached_losses.new_zeros(n)
        if n <= 1:
            return weights

        mask = torch.ones(n, dtype=torch.bool, device=detached_losses.device)
        mask[anchor_idx] = False
        others = detached_losses[mask]
        m = others.shape[0]

        gamma = self.hparams["dwa_mix_gamma"]
        tau = self.hparams["dwa_coral_tau"]

        w_uniform = others.new_full((m,), 1.0 / m)
        w_adapt = F.softmax(tau * (others - others.max()), dim=0)
        w_others = (1.0 - gamma) * w_uniform + gamma * w_adapt
        w_others = w_others / w_others.sum()

        weights[mask] = w_others
        return weights

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
        detached_losses = domain_losses.detach()
        anchor_idx = int(torch.argmin(detached_losses).item())

        covariances = [self._covariance(z) for z in features]
        c_anchor_detached = covariances[anchor_idx].detach()

        feature_dim = features[0].shape[1]
        coral_scale = 4.0 * float(feature_dim) * float(feature_dim)

        coral_terms_list = []
        for e, c_e in enumerate(covariances):
            if e == anchor_idx:
                # Anchor contributes 0 to alignment; keep entry for logging.
                coral_terms_list.append(domain_losses.new_zeros(()))
            else:
                coral_terms_list.append(
                    ((c_e - c_anchor_detached) ** 2).sum() / coral_scale)
        coral_terms = torch.stack(coral_terms_list)

        weights = self._other_weights(detached_losses, anchor_idx)
        coral_loss = torch.sum(weights * coral_terms)

        cls_loss = domain_losses.mean()
        risk_var = torch.var(domain_losses, unbiased=False)

        coral_lambda = self._coral_lambda()
        loss = (
            cls_loss
            + coral_lambda * coral_loss
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
            "anchor_idx": anchor_idx,
            "coral_lambda": float(coral_lambda),
        }
        for i in range(weights.shape[0]):
            metrics[f"weight_{i}"] = weights[i].item()
            metrics[f"coral_term_{i}"] = coral_terms[i].item()
        return metrics


class DWA_CORAL_ANCHOR_ANNEAL(DWA_CORAL_ANCHOR_ALIGNONLY):
    """Anchor alignment with a linearly annealed alignment weight.

    Identical to DWA_CORAL_ANCHOR_ALIGNONLY except that `dwa_coral_lambda`
    decays linearly from its full value to `dwa_coral_lambda_min` over the
    first `dwa_coral_anneal_steps` steps, then holds:

        lambda(t) = lambda_min
                    + (lambda_0 - lambda_min) * max(0, 1 - t / anneal_steps)

    Rationale: strong early alignment builds the domain-invariant feature
    space that the abstract test domains (Sketch) rely on; easing the
    alignment late lets the classifier refit natural-image detail, recovering
    Art/Photo accuracy toward the ERM ceiling without discarding the
    invariance learned earlier.
    """

    def _coral_lambda(self):
        lambda_0 = self.hparams["dwa_coral_lambda"]
        lambda_min = self.hparams["dwa_coral_lambda_min"]
        anneal_steps = int(self.hparams["dwa_coral_anneal_steps"])

        t = int(self.update_count.item())
        if anneal_steps <= 0:
            frac = 0.0
        else:
            frac = max(0.0, 1.0 - t / float(anneal_steps))
        return lambda_min + (lambda_0 - lambda_min) * frac
