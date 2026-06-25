"""
Path integral sampler for relay chains.

The core idea from Feynman / string theory:
  Instead of running ONE relay chain to convergence, run MANY short chains
  from random initializations and observe the DISTRIBUTION of where they land.
  That distribution IS the path integral — the set of all paths explored,
  weighted by their action (Walsh spectral distance traveled).

The minimum-action paths dominate. As you run more samples, the distribution
concentrates around the classical trajectory — the mathematical relationship.

Finer relay division (more populations) = more constraints = fewer valid paths
= sharper concentration of the path distribution around the true relationship.
This is the quantum well narrowing: more constraints → discrete spectrum.

Three experiments:
  1. Path distribution:    many short runs, cluster convergence points
  2. Division sweep:       n=4,8,12,16 relay pops, measure path count
  3. Action spectrum:      histogram of total Walsh distance per path

Fractional derivatives:
  The n-step relay chain uses n-th root of the derivative operator.
  d^(1/n)/dx^(1/n) in Fourier space multiplies frequency k by (ik)^(1/n).
  In phi_k basis this is a rotation by π/(2n) per step.
  n=4: full derivative (π/2 each step)
  n=8: half-derivative (π/4 each step)
  n=16: quarter-derivative (π/8 each step)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from toolkit.relay_chain import RelayChainConfig, run as relay_run, _derivative_step
from toolkit.ga import CoevoConfig


# ── Fractional derivative in phi_k basis ─────────────────────────────────────

def fractional_derivative_matrix(n_genes: int, order: float) -> np.ndarray:
    """
    Matrix for d^order/dx^order in the phi_k = x^k/k! basis.

    In Fourier space: multiply frequency k by (ik)^order.
    But in our polynomial basis the exact formula is:
      (d^α/dx^α phi_k)(x) = phi_{k-α}(x)  for integer α
      For non-integer α: use the Riemann-Liouville definition.

    Simple approximation for order = p/q (rational):
      Map each coefficient k to coefficient k - order
      Interpolate between integer indices.

    For our discrete relay chain:
      order = 1/n (n = number of steps for a full derivative)
      The step from A to B multiplies the coefficient vector by this matrix.
    """
    M = np.zeros((n_genes, n_genes))
    for k in range(n_genes):
        # d^order phi_k / dx^order = phi_{k-order} if k >= order, else 0
        # For continuous order: use gamma function
        # M[j, k] = Gamma(k+1) / Gamma(k-order+1) if j = k-order, else 0
        target = k - order
        lo = int(np.floor(target))
        hi = lo + 1
        frac = target - lo
        if 0 <= lo < n_genes:
            coeff_lo = math.gamma(k + 1) / math.gamma(k - order + 1 + (order - int(order)))
            M[lo, k] += (1 - frac) * coeff_lo / math.factorial(lo) if lo >= 0 else 0
        if 0 <= hi < n_genes:
            coeff_hi = math.gamma(k + 1) / math.gamma(k - order + 1 + (order - int(order)))
            M[hi, k] += frac * coeff_hi / math.factorial(hi) if hi >= 0 else 0
    return M


def fractional_derivative_step(ind: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Apply a 1/n_steps derivative to coefficient vector.
    n_steps=1: full derivative (shift left by 1)
    n_steps=2: half-derivative (π/4 rotation in phase space)
    n_steps=4: quarter-derivative (π/8 rotation)

    Approximation: linear interpolation between identity and full derivative.
    Full derivative: a[j] = b[j+1]  (shift left by 1)
    Identity: a[j] = b[j]
    Fractional: a[j] = (1 - 1/n)*b[j] + (1/n)*b[j+1]
    """
    alpha = 1.0 / n_steps
    result = (1 - alpha) * ind.copy()
    shifted = np.append(ind[1:], 0.)
    result += alpha * shifted
    return result


@dataclass
class PathIntegralResult:
    n_paths:           int
    convergence_points: List[np.ndarray]   # best_a from each run
    cycle_residuals:   List[float]
    actions:           List[float]         # total Walsh distance per path
    alternating_scores: List[float]
    min_action_path:   Optional[np.ndarray]
    path_diversity:    float               # std of convergence points (low=concentrated)
    n_relay_pops:      int


def _path_action(result) -> float:
    """Total Walsh spectral distance traveled during the run (the 'action')."""
    if not result.timeseries.get("residuals"):
        return float('inf')
    residuals = [r for _, rlist in result.timeseries["residuals"] for r in rlist]
    return float(sum(residuals)) if residuals else float('inf')


def run_path_integral(n_relay_pops: int = 4,
                       n_paths: int = 20,
                       gens_per_path: int = 300,
                       signal_design: str = "trajectory_ngram",
                       seed_base: int = 0,
                       verbose: bool = True) -> PathIntegralResult:
    """
    Run N_PATHS short relay chains from random initializations.
    Returns the distribution of convergence points — the path integral approximation.

    n_relay_pops: relay chain length (4=full derivative, 8=half, 12=third, 16=quarter)
    n_paths:      number of paths to sample (more = better approximation)
    """
    cfg_base = CoevoConfig(
        n_genes=8, pop_size=60, gens=gens_per_path,
        n_collaborations=1, k_signal_samples=4,
        mut_std=0.05, sig_weight=0.35, delta=0.25,
        coupling_mode="adaptive_kl",
        signal_design=signal_design,
    )

    results_data = []
    if verbose:
        print(f"\nPath integral: n_relay={n_relay_pops}, n_paths={n_paths}, "
              f"gens={gens_per_path}, signal={signal_design}")

    for i in range(n_paths):
        cfg = CoevoConfig(**{**cfg_base.__dict__,
                             "seed": seed_base + i * 17,
                             "gens": gens_per_path})

        # Use fractional derivative step based on n_relay_pops
        # n_relay_pops=4 → full derivative each step
        # n_relay_pops=8 → half-derivative each step
        # etc.
        chain_cfg = RelayChainConfig(
            n_pops=n_relay_pops,
            step_fitness="derivative",
            cfg=cfg,
            cycle_weight=0.20,
        )
        r = relay_run(chain_cfg, log_every=gens_per_path)  # only log final
        results_data.append(r)
        if verbose and (i+1) % 5 == 0:
            print(f"  {i+1}/{n_paths} paths sampled  "
                  f"(cycle_res={r.cycle_residual:.3f}, "
                  f"alt={r.alternating_score:.3f})")

    convergence_points = [r.populations[0] for r in results_data]
    cycle_residuals    = [r.cycle_residual for r in results_data]
    alt_scores         = [r.alternating_score for r in results_data]
    actions            = [_path_action(r) for r in results_data]

    # Minimum-action path
    min_idx = int(np.argmin(actions))

    # Path diversity: how spread are the convergence points?
    stack = np.stack(convergence_points)
    path_diversity = float(stack.std(axis=0).mean())

    return PathIntegralResult(
        n_paths=n_paths,
        convergence_points=convergence_points,
        cycle_residuals=cycle_residuals,
        actions=actions,
        alternating_scores=alt_scores,
        min_action_path=convergence_points[min_idx],
        path_diversity=path_diversity,
        n_relay_pops=n_relay_pops,
    )


def division_sweep(n_relay_values: List[int] = [4, 8, 12, 16],
                   n_paths: int = 15,
                   gens_per_path: int = 250,
                   signal_design: str = "trajectory_ngram"):
    """
    Run path integral at multiple relay chain lengths.
    Measures how path_diversity decreases as n_relay increases.

    Prediction (quantum well analogy):
      More relay populations → tighter betweenness → fewer valid paths
      → lower path_diversity → sharper concentration around true relationship
      This is the discrete energy level selection: finer division = fewer modes survive.
    """
    sweep_results = {}
    for n in n_relay_values:
        r = run_path_integral(n, n_paths, gens_per_path, signal_design, verbose=True)
        sweep_results[n] = r
        print(f"\n  n_relay={n}: diversity={r.path_diversity:.4f}  "
              f"mean_cycle_res={np.mean(r.cycle_residuals):.4f}  "
              f"mean_alt={np.mean(r.alternating_scores):.4f}")

    _plot_division_sweep(sweep_results)
    return sweep_results


def _plot_division_sweep(sweep_results: dict, out="path_integral_sweep.png"):
    n_vals     = sorted(sweep_results)
    diversities = [sweep_results[n].path_diversity for n in n_vals]
    mean_res    = [np.mean(sweep_results[n].cycle_residuals) for n in n_vals]
    mean_alt    = [np.mean(sweep_results[n].alternating_scores) for n in n_vals]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Path integral division sweep — quantum well analogy\n"
                 "More relay populations = tighter constraints = fewer valid paths",
                 fontsize=11, fontweight="bold")

    ax = axes[0]
    ax.plot(n_vals, diversities, "o-", color="steelblue", lw=2)
    ax.set_xlabel("n_relay_populations"); ax.set_ylabel("Path diversity (std)")
    ax.set_title("Path diversity vs. chain length\n(should DECREASE with n)")
    ax.set_xticks(n_vals)

    ax = axes[1]
    ax.plot(n_vals, mean_res, "o-", color="darkorange", lw=2)
    ax.set_xlabel("n_relay_populations"); ax.set_ylabel("Mean cycle residual")
    ax.set_title("Cycle closure quality vs. chain length")
    ax.set_xticks(n_vals)

    ax = axes[2]
    ax.plot(n_vals, mean_alt, "o-", color="seagreen", lw=2)
    ax.set_xlabel("n_relay_populations"); ax.set_ylabel("Mean alternating score")
    ax.set_title("Even/odd alternation vs. chain length\n(spectral fingerprint of rotation)")
    ax.set_xticks(n_vals)

    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"Sweep plot saved -> {out}")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"

    if mode == "sweep":
        division_sweep(n_relay_values=[4, 8, 12], n_paths=10, gens_per_path=200)
    elif mode == "single":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        r = run_path_integral(n_relay_pops=n, n_paths=15, gens_per_path=250)
        print(f"\nMin-action path: {np.round(r.min_action_path, 4)}")
        print(f"Path diversity:  {r.path_diversity:.4f}")
        print(f"Mean alt score:  {np.mean(r.alternating_scores):.4f}")
