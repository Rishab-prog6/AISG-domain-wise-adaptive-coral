# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Hparam registry for the improved DWA_CORAL and its variants.

Defaults are tuned for the normalized CORAL penalty (||C_e - C_bar||_F^2
divided by 4*d^2), which puts the alignment term on a scale comparable to
cross-entropy.

DWA_CORAL keeps its original v2 defaults so previously reported results stay
reproducible. The three variants (ALIGNONLY / CLIPPED / EMA) use the lighter
defaults requested for the Sketch-vs-Cartoon/Photo trade-off study:
  * dwa_coral_lambda = 0.3   (gentler alignment than DWA_CORAL's 1.0)
  * dwa_coral_beta   = 0.0   (risk-variance penalty off by default)
  * dwa_coral_tau    = 0.5
  * dwa_weight_min   = 0.15  (CLIPPED only)
  * dwa_weight_max   = 0.60  (CLIPPED only)
  * dwa_ema_alpha    = 0.9   (EMA only)
"""

import numpy as np

from domainbed import hparams_registry
from domainbed.lib import misc


DWA_ALGORITHMS = [
    "DWA_CORAL",
    "DWA_CORAL_ALIGNONLY",
    "DWA_CORAL_CLIPPED",
    "DWA_CORAL_EMA",
    "DWA_CORAL_MIXED_LOSSGAP_ALIGNONLY",
]


def add_hparams(algorithm, dataset, _hparam):
    if algorithm not in DWA_ALGORITHMS:
        return

    if algorithm == "DWA_CORAL":
        # Original v2 defaults — kept unchanged for reproducibility.
        _hparam("dwa_coral_lambda", 1.0, lambda r: 10 ** r.uniform(-1, 1))
        _hparam("dwa_coral_beta", 1e-2, lambda r: 10 ** r.uniform(-4, -1))
        _hparam("dwa_coral_tau", 0.5, lambda r: 10 ** r.uniform(-1, 0.5))
        _hparam("dwa_coral_warmup", 100,
                lambda r: int(r.choice([0, 100, 500])))
        return

    if algorithm == "DWA_CORAL_MIXED_LOSSGAP_ALIGNONLY":
        _hparam("dwa_coral_lambda", 0.3,
                lambda r: 10 ** r.uniform(-1.5, 0))
        _hparam("dwa_coral_beta", 0.0,
                lambda r: r.choice([0.0, 1e-3, 1e-2]))
        # Score is z-scored so a slightly larger tau is well-behaved.
        _hparam("dwa_coral_tau", 1.0,
                lambda r: 10 ** r.uniform(-0.5, 0.5))
        _hparam("dwa_score_alpha", 0.5,
                lambda r: r.uniform(0.0, 1.0))
        _hparam("dwa_mix_gamma", 0.5,
                lambda r: r.uniform(0.0, 1.0))
        return

    # Shared defaults for the ALIGNONLY / CLIPPED / EMA variants.
    _hparam("dwa_coral_lambda", 0.3, lambda r: 10 ** r.uniform(-1.5, 0))
    _hparam("dwa_coral_beta", 0.0, lambda r: r.choice([0.0, 1e-3, 1e-2]))
    _hparam("dwa_coral_tau", 0.5, lambda r: 10 ** r.uniform(-1, 0.5))

    # ALIGNONLY uses a plain softmax (no warmup); CLIPPED and EMA keep the
    # DWA_CORAL-style warmup so weights stay uniform for the first steps.
    if algorithm in ("DWA_CORAL_CLIPPED", "DWA_CORAL_EMA"):
        _hparam("dwa_coral_warmup", 100,
                lambda r: int(r.choice([0, 100, 500])))

    if algorithm == "DWA_CORAL_CLIPPED":
        _hparam("dwa_weight_min", 0.15,
                lambda r: r.choice([0.05, 0.10, 0.15]))
        _hparam("dwa_weight_max", 0.60,
                lambda r: r.choice([0.50, 0.60, 0.70]))

    if algorithm == "DWA_CORAL_EMA":
        _hparam("dwa_ema_alpha", 0.9,
                lambda r: r.choice([0.8, 0.9, 0.95, 0.99]))


def _dwa_hparams(algorithm, dataset, random_seed):
    if algorithm not in DWA_ALGORITHMS:
        return hparams_registry._hparams(algorithm, dataset, random_seed)

    hparams = hparams_registry._hparams("ERM", dataset, random_seed)

    def _hparam(name, default_val, random_val_fn):
        assert name not in hparams
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    add_hparams(algorithm, dataset, _hparam)
    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _dwa_hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _dwa_hparams(algorithm, dataset, seed).items()}
