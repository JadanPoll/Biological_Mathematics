"""
Global experiment registry.

All experiments from all tiers are registered here.
Adding a new experiment = adding one Experiment(...) to a list.
No new files, no new scripts, no new boilerplate.

Usage:
    from experiments.registry import REGISTRY, get_experiment
    exp = get_experiment("MM-T0-001")
"""

from experiments.tier0.experiments import TIER0_EXPERIMENTS
from experiments.tier0.info_topology import INFO_TOPOLOGY_EXPERIMENTS

REGISTRY: dict = {}

def _register(experiments):
    for exp in experiments:
        REGISTRY[exp.object_id] = exp

_register(TIER0_EXPERIMENTS)
_register(INFO_TOPOLOGY_EXPERIMENTS)
# Future: _register(TIER1_EXPERIMENTS), _register(TIER2_EXPERIMENTS), ...


def get_experiment(object_id: str):
    if object_id not in REGISTRY:
        raise KeyError(f"Experiment '{object_id}' not found. "
                       f"Known: {sorted(REGISTRY)}")
    return REGISTRY[object_id]


def list_experiments(tier: int = None):
    exps = list(REGISTRY.values())
    if tier is not None:
        exps = [e for e in exps if e.tier == tier]
    return sorted(exps, key=lambda e: e.object_id)
