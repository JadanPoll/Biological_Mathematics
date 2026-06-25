"""
Euler relay chain + heat kernel comparison.
Run this to see the first experiment where:
  - True intermediates are known analytically at every step
  - The relay chain is attempting to trace the path of e^(i*pi) = -1
  - We compare the GA distribution to the analytical heat kernel on SO(2)
"""

import math
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from toolkit.euler_relay import (run_euler_relay, euler_intermediate,
                                  all_true_intermediates, phase_angle,
                                  euler_step, _COS, _SIN, N)
from toolkit.ga import CoevoConfig


def heat_kernel_SO2(theta: float, t: float, n_terms: int = 30) -> float:
    """
    Heat kernel on the circle S^1 = SO(2).
    K(theta, t) = sum_{n=-inf}^{inf} e^(-n^2 * t) * e^(i*n*theta) / (2*pi)
               = (1/2pi) * (1 + 2*sum_{n=1}^{inf} e^(-n^2*t) * cos(n*theta))

    This is the probability density of being at angle theta after diffusing
    for time t on the circle. For small t: sharp Gaussian near theta=0.
    For large t: approaches uniform distribution.

    In our relay chain: t corresponds to the path length (step_size * n_elapsed).
    """
    s = 1.0
    for n in range(1, n_terms + 1):
        s += 2.0 * math.exp(-n*n*t) * math.cos(n*theta)
    return s / (2.0 * math.pi)


def compare_to_heat_kernel(result, n_steps: int):
    """
    For each relay population, compute:
      - Actual phase angle (extracted from best individual)
      - Expected phase angle (true Euler intermediate)
      - Heat kernel probability at the actual angle, given the path so far

    The heat kernel tells us: given that the relay chain has been running
    for 'time' t (proportional to step number), what is the expected spread
    of valid configurations?

    If the relay populations cluster tightly around the true intermediates:
      heat kernel is sharply peaked -- low spread, high probability
    If they wander:
      heat kernel is diffuse -- high spread, low probability
    """
    delta = math.pi / n_steps
    print(f"\n  HEAT KERNEL COMPARISON (n_steps={n_steps})")
    print(f"  {'Step':>4} {'Expected':>10} {'Actual':>10} {'Angular Err':>12} "
          f"{'K(err,t)':>12} {'Step Res':>10}")
    print(f"  {'-'*62}")

    for k in range(n_steps + 1):
        true_angle  = k * delta
        actual_angle = result.phase_angles[k]
        angle_err   = abs(actual_angle - true_angle)
        step_res    = result.step_residuals[k]

        # Heat kernel: probability of the ANGULAR ERROR under diffusion time t = k*delta^2
        # (diffusion time proportional to number of steps × step size^2)
        t = max(k * delta * delta, 1e-6)
        hk = heat_kernel_SO2(angle_err, t)

        print(f"  {k:>4} {math.degrees(true_angle):>9.1f}° "
              f"{math.degrees(actual_angle):>9.1f}° "
              f"{math.degrees(angle_err):>11.2f}° "
              f"{hk:>12.4f} "
              f"{step_res:>10.4f}")


def run_division_sweep_euler(n_values=(4, 8, 12), gens=600):
    """
    Run Euler relay chain with different numbers of steps.
    The quantum well prediction: as n increases, the path becomes more
    constrained and the step residuals should DECREASE (more precise path).

    This is the proper test because each step uses the EXACT phase rotation
    operator (cos(pi/n)*c + sin(pi/n)*deriv(c)), not a fractional approximation.
    """
    cfg_base = CoevoConfig(
        n_genes=N, pop_size=80, gens=gens,
        n_collaborations=1, k_signal_samples=6,
        mut_std=0.04, sig_weight=0.35, delta=0.25,
        coupling_mode="adaptive_kl",
        signal_design="trajectory_ngram",
        sig_freq=10, elite=2, tourney=5,
    )

    print("="*70)
    print("EULER RELAY CHAIN DIVISION SWEEP")
    print("Path: cos(x) -> cos(x+pi/n) -> ... -> cos(x+pi) = -cos(x)")
    print("Each step: exact phase rotation (not fractional derivative approx)")
    print("="*70)

    results = {}
    for n in n_values:
        print(f"\n--- n_steps = {n}  (delta = {math.degrees(math.pi/n):.1f} deg per step) ---")
        cfg = CoevoConfig(**{**cfg_base.__dict__, "seed": 42 + n})
        r = run_euler_relay(n_steps=n, total_angle=math.pi, cfg=cfg, verbose=True)
        compare_to_heat_kernel(r, n)
        results[n] = r

    # Summary plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Euler relay chain: quantum well division sweep\n"
                 "cos(x) --> -cos(x) through the Euler rotation path",
                 fontsize=11, fontweight="bold")

    n_vals     = sorted(results)
    mean_res   = [results[n].mean_step_res for n in n_vals]
    mean_ang   = [results[n].mean_angle_err for n in n_vals]
    outcomes   = [results[n].outcome for n in n_vals]

    ax = axes[0]
    ax.plot(n_vals, mean_res, "o-", color="steelblue", lw=2, ms=8)
    ax.set_xlabel("n_steps"); ax.set_ylabel("Mean step residual")
    ax.set_title("Step residual vs. chain length\n(should DECREASE: tighter path)")
    ax.set_xticks(n_vals)

    ax = axes[1]
    ax.plot(n_vals, mean_ang, "o-", color="darkorange", lw=2, ms=8)
    ax.set_xlabel("n_steps"); ax.set_ylabel("Mean angular error (rad)")
    ax.set_title("Phase angle error vs. chain length\n(0 = exactly on Euler path)")
    ax.set_xticks(n_vals)

    # True intermediates for n=4 in coefficient space
    ax = axes[2]
    n_show = n_vals[-1]
    ks = np.arange(N)
    w = 0.6 / (n_show + 1)
    true_ints = all_true_intermediates(n_show)
    found_pops = results[n_show].best_pops
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_show + 1))
    for k, (ti, fp, c) in enumerate(zip(true_ints, found_pops, colors)):
        offset = (k - n_show/2) * w
        ax.bar(ks + offset, fp, w, color=c, alpha=0.7, label=f"Found {k}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"Coefficient vectors (n={n_show})")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"k={i}" for i in ks], fontsize=7)
    ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig("euler_sweep.png", dpi=140, bbox_inches="tight")
    plt.show()
    print("\nPlot saved -> euler_sweep.png")
    return results


if __name__ == "__main__":
    run_division_sweep_euler(n_values=[4, 8], gens=500)
