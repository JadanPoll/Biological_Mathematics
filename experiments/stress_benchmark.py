"""
Stress-vs-complexity benchmark.

The core hypothesis test: does the stress of the minimum-effort morphing
path between two mathematical objects correlate with the algebraic
complexity of their relationship?

If YES: we have a working instrument. Stress is informative.
If NO: we get a precise Ramanujan-type failure mode to diagnose.

Five relationship classes, ordered by expected stress:

  Class 0 — IDENTICAL: f(x) = f(x). Stress = 0.
  Class 1 — SCALAR:    f(x) and k*f(x). One bit of information (the scalar).
  Class 2 — PHASE:     cos(x) and cos(x+phi). Simple geometric rotation.
  Class 3 — DERIVATIVE: cos and sin. One algebraic operation (d/dx).
  Class 4 — FREQUENCY: cos(x) and cos(2x). Nonlinear — can't be expressed
                        as simple phase shift or derivative.
  Class 5 — UNRELATED: cos(x) and a random polynomial. No algebraic bridge.

Expected Mendel table result:
  Stress(Class 0) < Stress(Class 1) < Stress(Class 2) < Stress(Class 3)
                  < Stress(Class 4) < Stress(Class 5)

The stress metric: Wasserstein variance of the open-chain relay path
(how far from equal-transport = how far from the geodesic).
"""

import math
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

import cma

from toolkit.euler_relay import _COS, _SIN, N, X, BASIS
from toolkit.stress_metrics import (sliced_wasserstein, population_wasserstein_path,
                                     relay_ricci_curvatures, relay_stress_report)
from toolkit.info_geometry import info_geo_path_report, js_distance
from notebook import db


# ── Function representations in phi_k basis ──────────────────────────────────

def _cos(): return np.array([(-1.)**(k//2) if k%2==0 else 0. for k in range(N)])
def _sin(): return np.array([(-1.)**((k-1)//2) if k%2==1 else 0. for k in range(N)])
def _cos_phase(phi):
    return math.cos(phi)*_cos() - math.sin(phi)*_sin()
def _cos_2x():
    """cos(2x) = 1 - 2x^2 + 2x^4/3 - ... in phi_k basis"""
    c = np.zeros(N)
    for k in range(0, N, 2):
        c[k] = (-1)**(k//2) * (2**k) / math.factorial(k) * math.factorial(k)
    return c / (np.linalg.norm(c) + 1e-10) * np.linalg.norm(_cos())
def _random_poly(seed=42):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(N)
    return c / np.linalg.norm(c) * np.linalg.norm(_cos())


BENCHMARK_PAIRS = [
    {
        "name":        "Identical (cos vs cos)",
        "class":       0,
        "expected":    "zero stress",
        "fn_a":        _cos(),
        "fn_b":        _cos(),
        "relationship":"A = B",
    },
    {
        "name":        "Scalar x2 (cos vs 2*cos)",
        "class":       1,
        "expected":    "very low stress",
        "fn_a":        2.0 * _cos(),
        "fn_b":        _cos(),
        "relationship":"A = 2*B",
    },
    {
        "name":        "Phase shift 45deg",
        "class":       2,
        "expected":    "low stress",
        "fn_a":        _cos_phase(math.pi/4),
        "fn_b":        _cos(),
        "relationship":"A = cos(x + pi/4)",
    },
    {
        "name":        "Phase shift 90deg (derivative)",
        "class":       3,
        "expected":    "medium stress",
        "fn_a":        _sin(),
        "fn_b":        _cos(),
        "relationship":"A = d/dx(B)  [cos -> sin]",
    },
    {
        "name":        "Frequency doubling (cos vs cos(2x))",
        "class":       4,
        "expected":    "high stress",
        "fn_a":        _cos_2x(),
        "fn_b":        _cos(),
        "relationship":"A = cos(2x)  [nonlinear]",
    },
    {
        "name":        "Random polynomial (no relationship)",
        "class":       5,
        "expected":    "maximum stress / fails",
        "fn_a":        _random_poly(42),
        "fn_b":        _cos(),
        "relationship":"None  [negative control]",
    },
]


# ── Minimal relay runner using CMA-ES ─────────────────────────────────────────

def run_relay_cmaes(fn_a: np.ndarray, fn_b: np.ndarray,
                    n_steps: int = 4,
                    sigma0: float = 0.6,
                    maxiter: int = 150,
                    seed: int = 42) -> dict:
    """
    Open-chain relay from fn_b (D-brane 1) to fn_a (D-brane 2).
    CMA-ES optimizes each intermediate.
    Returns populations for stress measurement.
    """
    delta = math.pi / n_steps
    np.random.seed(seed)

    # Initialize: linear interpolation between endpoints
    pops = [fn_b.copy()]
    for k in range(1, n_steps):
        alpha = k / n_steps
        pops.append((1 - alpha) * fn_b + alpha * fn_a
                    + np.random.randn(N) * 0.3)
    pops.append(fn_a.copy())

    # CMA-ES optimize each intermediate
    for it in range(2):
        for k in range(1, n_steps):
            prev, nxt = pops[k-1], pops[k+1]
            def neg_fit(c, p=prev, n=nxt):
                c = np.array(c)
                cost = (np.linalg.norm(c - p)**2 +
                        np.linalg.norm(n - c)**2)
                return float(cost)
            opts = cma.CMAOptions()
            opts['maxiter'] = maxiter
            opts['verbose'] = -9
            opts['seed'] = seed + k + it*100
            es = cma.CMAEvolutionStrategy(pops[k].tolist(), sigma0, opts)
            es.optimize(neg_fit)
            pops[k] = np.array(es.result.xbest)

    # Treat each intermediate as a 1-individual "population" for stress metrics
    pop_list = [p.reshape(1, -1) for p in pops]
    return {"pops": pops, "pop_list": pop_list}


# ── Main benchmark ────────────────────────────────────────────────────────────

def algebraic_consistency(pops: list) -> dict:
    """
    For each consecutive pair in the relay path, find the best algebraic
    transformation from the grammar.  Measure how consistent it is.

    Consistent = same transformation at every step = one algebraic generator.
    Inconsistent = different transformations = no single algebraic relationship.

    This is the REAL stress measure: not how smooth the path is geometrically,
    but how algebraically coherent it is.
    """
    from toolkit.joint_fitness import best_transformation, _GRAMMAR

    transforms = []
    residuals  = []
    for k in range(len(pops) - 1):
        name, res, dl = best_transformation(pops[k+1], pops[k])
        transforms.append(name)
        residuals.append(res)

    # Count unique transformations — more unique = less consistent
    unique_t = len(set(transforms))
    mean_res = float(np.mean(residuals))

    # Entropy of the transformation distribution — higher = more incoherent
    from collections import Counter
    counts = Counter(transforms)
    probs  = np.array(list(counts.values()), dtype=float)
    probs /= probs.sum()
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

    return {
        "transforms":    transforms,
        "residuals":     residuals,
        "unique_count":  unique_t,
        "mean_residual": mean_res,
        "entropy":       entropy,   # 0 = perfectly consistent, higher = chaotic
        "dominant_T":    counts.most_common(1)[0][0],
    }


def run_benchmark(n_steps: int = 4, verbose: bool = True):
    db.init()

    results = []
    for pair in BENCHMARK_PAIRS:
        if verbose:
            print(f"\n{'='*55}")
            print(f"Class {pair['class']}: {pair['name']}")
            print(f"  Relationship: {pair['relationship']}")
            print(f"  Expected:     {pair['expected']}")

        relay = run_relay_cmaes(pair['fn_a'], pair['fn_b'],
                                 n_steps=n_steps, sigma0=0.6, maxiter=120)
        pops = relay['pops']

        # Stress metrics
        wass = population_wasserstein_path(relay['pop_list'], n_proj=20)
        ricci = relay_ricci_curvatures(relay['pop_list'])
        wass_var = float(np.var(wass))
        ricci_var = float(np.var(ricci))

        # Direct coefficient distance along path
        step_dists = [float(np.linalg.norm(pops[k+1] - pops[k]))
                      for k in range(len(pops)-1)]
        dist_var = float(np.var(step_dists))

        # Path smoothness: second derivative of the path (curvature)
        if len(pops) >= 3:
            acc = [np.linalg.norm(pops[k+1] - 2*pops[k] + pops[k-1])
                   for k in range(1, len(pops)-1)]
            path_curvature = float(np.mean(acc))
        else:
            path_curvature = 0.

        alg = algebraic_consistency(relay['pops'])

        result = {
            "name":            pair['name'],
            "class":           pair['class'],
            "wass_variance":   wass_var,
            "ricci_variance":  ricci_var,
            "dist_variance":   dist_var,
            "path_curvature":  path_curvature,
            "wass_steps":      wass,
            "step_dists":      step_dists,
            "alg_entropy":     alg["entropy"],
            "alg_unique":      alg["unique_count"],
            "alg_dominant_T":  alg["dominant_T"],
            "alg_mean_res":    alg["mean_residual"],
        }
        results.append(result)

        if verbose:
            print(f"  Wasserstein variance: {wass_var:.5f}")
            print(f"  Ricci variance:       {ricci_var:.5f}")
            print(f"  Alg entropy:          {alg['entropy']:.4f}  "
                  f"(0=consistent, high=chaotic)")
            print(f"  Dominant transform:   {alg['dominant_T']}")
            print(f"  Transforms per step:  {alg['transforms']}")

    return results


def plot_benchmark(results: list, out: str = "stress_benchmark.png"):
    classes  = [r['class'] for r in results]
    names    = [r['name'].split('(')[0].strip() for r in results]
    wass_var = [r['wass_variance'] for r in results]
    ricci_var= [r['ricci_variance'] for r in results]
    curvature= [r['path_curvature'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Stress-vs-Complexity Benchmark\n"
                 "Core hypothesis: stress of morphing path correlates "
                 "with algebraic relationship complexity",
                 fontsize=11, fontweight='bold')

    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(results)))

    for ax, vals, title in zip(axes,
        [wass_var, ricci_var, curvature],
        ["Wasserstein Variance\n(0 = equal-transport geodesic)",
         "Ricci Variance\n(0 = Ricci soliton)",
         "Path Curvature\n(0 = straight line)"]):

        bars = ax.bar(range(len(results)), vals, color=colors)
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels([f"C{c}" for c in classes], fontsize=10)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Class (0=identical, 5=unrelated)")

        for bar, name in zip(bars, names):
            ax.annotate(name[:14],
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', fontsize=6, rotation=45)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved -> {out}")

    # Print the Mendel table
    print(f"\n{'='*65}")
    print("MENDEL TABLE — stress-vs-complexity calibration")
    print(f"{'='*65}")
    print(f"{'Class':>5} {'Wass Var':>10} {'Ricci Var':>10} "
          f"{'Alg H':>8}  {'Dominant T':<18}  Name")
    print("-"*65)
    for r in results:
        print(f"  {r['class']:>3}  {r['wass_variance']:>10.5f}  "
              f"{r['ricci_variance']:>10.5f}  "
              f"{r.get('alg_entropy',0):>8.4f}  "
              f"{r.get('alg_dominant_T','?'):<18}  {r['name'][:25]}")
    print(f"{'='*65}")

    # Check if hypothesis holds: stress should increase with class
    wass_ordered = all(wass_var[i] <= wass_var[i+1]*1.5
                       for i in range(len(wass_var)-1))
    print(f"\nHypothesis (stress increases with class):")
    print(f"  Wasserstein ordering holds: {wass_ordered}")
    print(f"  If YES: instrument is working. Stress IS informative.")
    print(f"  If NO:  diagnose which class breaks the ordering -> Ramanujan type")


if __name__ == "__main__":
    results = run_benchmark(n_steps=4, verbose=True)
    plot_benchmark(results)
