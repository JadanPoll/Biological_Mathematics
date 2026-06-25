"""
Checkpoint system demo — shows how early termination expands experiment coverage.

Three scenarios on the same compute budget (total_gens = 600):
  Without checkpoints: 1 experiment runs all 600 gens regardless of outcome
  With checkpoints:    same budget covers multiple experiments, each labeled precisely

The four failure modes with their detection generation and cost:
  DEGENERATE              -> detected gen ~15   (2.5% of budget)
  OSCILLATORY_WARNING     -> detected gen ~40   (6.7% of budget)
  TYPE_7                  -> detected gen ~80   (13.3% of budget)
  MISCOORDINATION         -> detected gen ~150  (25.0% of budget)
  CONVERGED               -> runs to gen 600    (100% of budget)

Expected coverage multiplier:
  If 60% of experiments fail (realistic for early taxonomy-building):
    Without checkpoints: 1 experiment per budget unit
    With checkpoints:    ~2-3 experiments per budget unit (redirected early)
"""

import numpy as np
import cma
import math
import time

from toolkit.checkpoint import ExperimentCheckpoint, CheckpointResult
from toolkit.euler_relay import _COS, _SIN, N
from experiments.joint_covariance import make_joint_fit


def _cos2x():
    c = np.zeros(N)
    for k in range(0, N, 2):
        c[k] = ((-1)**(k//2)) * (2.0**k)
    return c


def run_with_checkpoints(fn_a, fn_b, relationship, max_gens=600,
                          sigma0=0.4, seed=42, label=""):
    """
    Run joint CMA-ES with checkpoint monitoring.
    Returns: result dict with failure_mode, final_residual, gens_used, time_s
    """
    t0 = time.perf_counter()
    fit = make_joint_fit(fn_a, fn_b, relationship)
    checker = ExperimentCheckpoint(
        n_genes=N,
        checkpoints=[15, 40, 80, 150, 300],
    )

    np.random.seed(seed)
    target_norm = float(np.linalg.norm(fn_a))
    x0 = np.concatenate([fn_b + np.random.randn(N)*0.3,
                          fn_a + np.random.randn(N)*0.3])

    opts = cma.CMAOptions()
    opts['verbose']  = -9
    opts['seed']     = seed
    opts['maxiter']  = max_gens

    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)

    failure_mode  = None
    early_stop_g  = None
    last_residual = float('inf')

    for gen in range(max_gens):
        solutions = es.ask()
        fitvals   = [-fit(s) for s in solutions]
        es.tell(solutions, fitvals)
        last_residual = min(fitvals)   # fitvals = -fit(s) = loss (positive)

        # Cheap population stats for checkpoint
        pops = np.array(solutions)
        pop_a = pops[:, :N]
        pop_b = pops[:, N:]

        if gen in [15, 40, 80, 150, 300]:
            result = checker.check(gen, pop_a, pop_b, None, None, last_residual)
            if result is not None and result.should_stop:
                failure_mode = result.failure_mode
                early_stop_g = gen
                break
        else:
            checker.check(gen, pop_a, pop_b, None, None, last_residual)

        if es.stop():
            break

    elapsed = time.perf_counter() - t0
    x_best  = np.array(es.result.xbest)
    residual = -fit(x_best)

    return {
        "label":       label,
        "failure_mode": failure_mode or ("CONVERGED" if residual < 0.10 else "PARTIAL"),
        "residual":    residual,
        "gens_used":   early_stop_g or gen,
        "time_s":      elapsed,
        "early_stop":  early_stop_g is not None,
    }


def run_demo():
    cos  = _COS.copy()
    sin  = _SIN.copy()
    cos2 = _cos2x()

    MAX_GENS = 300

    # Six experiments representing a realistic slice of the taxonomy
    experiments = [
        ("cos->sin [derivative, valid]",       sin,  cos,  "derivative"),
        ("cos->2cos [scalar, valid]",           2*cos, cos, "scalar_2"),
        ("cos->cos [identical, valid]",         cos,  cos,  "identical"),
        ("cos->cos2x [Type 7, no path]",        cos2, cos,  "discovery"),
        ("zero init -> cos [degenerate risk]",  cos, np.zeros(N), "identical"),
        ("random A [miscoord risk]",            cos+np.random.randn(N)*2, cos, "derivative"),
    ]

    print("="*72)
    print("CHECKPOINT DEMO: budget comparison")
    print(f"Max gens per experiment (full run): {MAX_GENS}")
    print("="*72)

    results = []
    total_gens_used = 0

    for label, fn_a, fn_b, rel in experiments:
        r = run_with_checkpoints(fn_a, fn_b, rel, max_gens=MAX_GENS,
                                  seed=42, label=label)
        results.append(r)
        total_gens_used += r["gens_used"]

        indicator = "STOP" if r["early_stop"] else "    "
        print(f"  [{indicator}] {label[:38]:<38}  "
              f"gen={r['gens_used']:>4}  res={r['residual']:>8.3f}  "
              f"{r['failure_mode']}")

    total_budget    = MAX_GENS * len(experiments)
    budget_used     = total_gens_used
    efficiency_gain = total_budget / budget_used

    print(f"\n  Total budget (no checkpoints): {total_budget} gens")
    print(f"  Total used  (with checkpoints): {budget_used} gens")
    print(f"  Efficiency multiplier: {efficiency_gain:.2f}×")
    early_stopped = [r for r in results if r['early_stop']]
    if early_stopped:
        avg_early = np.mean([r['gens_used'] for r in early_stopped])
        saved = total_budget - budget_used
        extra = int(saved / avg_early)
        print(f"  Additional experiments possible with saved budget: ~{extra}")
    else:
        print(f"  No early stops — all experiments ran full budget")

    print(f"\n  FAILURE MODE BREAKDOWN:")
    from collections import Counter
    modes = Counter(r['failure_mode'] for r in results)
    for mode, count in sorted(modes.items()):
        print(f"    {mode:<35} n={count}")

    print(f"\n  WHAT EACH EARLY STOP TOLD US:")
    for r in results:
        if r['early_stop']:
            print(f"    {r['label'][:40]}")
            print(f"      -> {r['failure_mode']} at gen {r['gens_used']}")


if __name__ == "__main__":
    run_demo()
