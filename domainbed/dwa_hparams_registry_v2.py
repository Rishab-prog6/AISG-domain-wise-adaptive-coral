# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Hparam registry for the improved DWA_CORAL.

Defaults are retuned for the normalized CORAL penalty (||C_e - C_bar||_F^2
divided by 4*d^2). With this normalization the penalty is on a comparable
scale to cross-entropy, so:
  * dwa_coral_lambda can be ~1.0 (the literature-standard CORAL weight)
  * dwa_coral_beta is small (variance of CE losses across domains is small)
  * dwa_coral_tau is conservative so the softmax doesn't concentrate all
    mass on a single domain early in training
  * dwa_coral_warmup keeps weights uniform for the first 100 steps
"""

import numpy as np

from domainbed import hparams_registry
from domainbed.lib import misc


def add_hparams(algorithm, dataset, _hparam):
    if algorithm != "DWA_CORAL":
        return

    _hparam("dwa_coral_lambda", 1.0, lambda r: 10 ** r.uniform(-1, 1))
    _hparam("dwa_coral_beta", 1e-2, lambda r: 10 ** r.uniform(-4, -1))
    _hparam("dwa_coral_tau", 0.5, lambda r: 10 ** r.uniform(-1, 0.5))
    _hparam("dwa_coral_warmup", 100, lambda r: int(r.choice([0, 100, 500])))


def _dwa_hparams(algorithm, dataset, random_seed):
    if algorithm != "DWA_CORAL":
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
