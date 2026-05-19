# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np

from domainbed import hparams_registry
from domainbed.lib import misc


def add_hparams(algorithm, dataset, _hparam):
    if algorithm != "DWA_CORAL":
        return

    _hparam("dwa_coral_lambda", 1e-3, lambda r: 10 ** r.uniform(-5, -1))
    _hparam("dwa_coral_beta", 1e-1, lambda r: 10 ** r.uniform(-3, 0))
    _hparam("dwa_coral_tau", 1.0, lambda r: 10 ** r.uniform(-1, 1))


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
