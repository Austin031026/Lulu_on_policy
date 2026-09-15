"""Lulu / ReN on-policy distillation with recognizable teacher corrections.

Objective helpers load lazily so path discovery and evaluation planning remain
usable without importing torch or any training dependency.
"""
from importlib import import_module

__all__ = [
    'build_cached_target',
    'build_target',
    'forward_kl',
    'probability_mass_at_ids',
    'recognition_weighted_forward_kl',
    'reduce_position_losses',
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module('.objective', __name__), name)
    globals()[name] = value
    return value
