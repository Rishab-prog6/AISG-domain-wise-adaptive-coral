# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Entry point for the improved DWA_CORAL.

Wraps domainbed.scripts.train, monkey-patching the algorithm registry and
hparam registry so the trainer picks up dwa_algorithms_v2.DWA_CORAL.
"""

import runpy
import sys

from domainbed import algorithms
from domainbed import dwa_algorithms_v2 as dwa_algorithms
from domainbed import dwa_hparams_registry_v2 as dwa_hparams_registry
from domainbed import hparams_registry


_base_get_algorithm_class = algorithms.get_algorithm_class


def _get_algorithm_class(algorithm_name):
    if algorithm_name in dwa_algorithms.ALGORITHMS:
        return getattr(dwa_algorithms, algorithm_name)
    return _base_get_algorithm_class(algorithm_name)


def _ensure_default_dwa_algorithm():
    has_algorithm_arg = any(
        arg == "--algorithm" or arg.startswith("--algorithm=")
        for arg in sys.argv[1:]
    )
    if not has_algorithm_arg:
        sys.argv.extend(["--algorithm", "DWA_CORAL"])


if __name__ == "__main__":
    for algorithm_name in dwa_algorithms.ALGORITHMS:
        if algorithm_name not in algorithms.ALGORITHMS:
            algorithms.ALGORITHMS.append(algorithm_name)

    algorithms.get_algorithm_class = _get_algorithm_class
    hparams_registry.default_hparams = dwa_hparams_registry.default_hparams
    hparams_registry.random_hparams = dwa_hparams_registry.random_hparams

    _ensure_default_dwa_algorithm()
    runpy.run_module("domainbed.scripts.train", run_name="__main__")
