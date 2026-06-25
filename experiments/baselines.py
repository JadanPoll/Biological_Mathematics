"""
Baseline comparisons — run experiments with signal disabled (sig_weight=0).

Purpose: confirm whether the Walsh signal exchange is helping, hurting, or neutral.
If residuals are the same or better with signal disabled, the signal is noise.

This is the first necessary control before changing the fitness function.
If signal is noise: we save 35% compute and the problem is in the fitness, not the signal.
If signal helps: the current signal design has partial information value worth preserving.
"""

import numpy as np
from experiments.base import run_experiment, poly_eval
from experiments.registry import get_experiment
from toolkit.ga import CoevoConfig
from notebook import db


def run_signal_comparison(object_id: str, gens: int = 400, pop: int = 70,
                          n_seeds: int = 2):
    """
    Run the same experiment with signal enabled vs. disabled.
    Reports which performs better and by how much.
    """
    db.init()
    exp = get_experiment(object_id)

    print(f"\n{'='*60}")
    print(f"Signal enabled vs. disabled: {object_id}")
    print(f"{'='*60}")

    results = {"enabled": [], "disabled": []}

    for seed_offset in range(n_seeds):
        # With signal
        cfg_on = CoevoConfig(**{**exp.cfg.__dict__,
                                "gens": gens, "pop_size": pop,
                                "sig_weight": 0.35,
                                "signal_design": "walsh_landscape",
                                "seed": 42 + seed_offset * 100})
        exp.cfg = cfg_on
        _, r_on = run_experiment(exp, signal_design="walsh_landscape",
                                 seed_offset=seed_offset, verbose=False)
        results["enabled"].append(r_on.residual)

        # Without signal (sig_weight=0 means signal has zero influence on mutation)
        cfg_off = CoevoConfig(**{**exp.cfg.__dict__,
                                 "sig_weight": 0.0,
                                 "signal_design": "walsh_landscape"})
        exp.cfg = cfg_off
        _, r_off = run_experiment(exp, signal_design="walsh_landscape",
                                  seed_offset=seed_offset + 1000, verbose=False)
        results["disabled"].append(r_off.residual)

    mean_on  = np.mean(results["enabled"])
    mean_off = np.mean(results["disabled"])
    delta    = mean_off - mean_on   # positive = signal helps

    print(f"  Signal enabled:  residual = {mean_on:.5f}")
    print(f"  Signal disabled: residual = {mean_off:.5f}")
    print(f"  Delta (disabled - enabled): {delta:+.5f}")

    if abs(delta) < 0.02:
        verdict = "NEUTRAL — signal has negligible effect. Problem is in fitness function."
    elif delta > 0:
        verdict = f"HELPFUL — signal reduces residual by {delta:.3f}."
    else:
        verdict = f"HARMFUL — signal INCREASES residual by {-delta:.3f}. Signal is noise."

    print(f"\n  Verdict: {verdict}")
    return results, verdict


if __name__ == "__main__":
    # Test on the simplest case (identical cos) and the scalar multiple
    for oid in ["MM-T0-001", "MM-T0-003", "MM-T0-009"]:
        run_signal_comparison(oid, gens=400, pop=70, n_seeds=2)
