"""
Euler relay chain — path from cos(x) to -cos(x) through the complex exponential.

The mathematical foundation:
  cos(x + theta) = cos(theta)*cos(x) - sin(theta)*sin(x)

  As theta goes from 0 to pi, cos(x + theta) traces a path from cos(x) to -cos(x).
  This path IS the complex exponential e^(i*theta) acting on the function space —
  Euler's formula e^(i*pi) = -1 expressed as an operator on functions, not numbers.

  In phi_k = x^k/k! coefficient space, the step from angle theta to theta + delta is:
    c(theta + delta) = cos(delta) * c(theta) + sin(delta) * d/dx(c(theta))

  Because d/dx maps cos -> -sin -> -cos -> sin -> cos (the rotation generator),
  and applying this rotation continuously gives the path of the complex exponential.

Why this is the right experiment:
  1. TRUE INTERMEDIATES ARE KNOWN ANALYTICALLY at every step.
     At step k of an n-step relay:
       c_k = cos(k*pi/n) * _cos - sin(k*pi/n) * _sin
     This is the exact ground truth. No other relay chain experiment has this.

  2. AS n INCREASES, the path becomes MORE CONSTRAINED.
     With n=4 (pi/4 steps), 4 intermediate populations must all lie on the
     rotation manifold. With n=16 (pi/16 steps), 16 must. The valid paths
     shrink — this IS the quantum well narrowing, properly implemented.

  3. THE ENDPOINT IS FAMOUS.
     The chain goes from cos(x) to cos(x + pi) = -cos(x).
     Euler said: e^(i*pi) = -1. The relay chain is the geometric path
     that makes this true as an operator action, not just a number identity.

  4. THE STEP IS EXACT, NOT FRACTIONAL DERIVATIVE APPROXIMATION.
     Previous relay chains used d/dx as an approximation. This chain uses
     the exact phase rotation step, which has known mathematical properties.
"""

import math
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional

from toolkit.ga import CoevoConfig
from toolkit.signals import get_signal
from toolkit.walsh import (kl_divergence_spectra, spectral_class,
                            effective_dimensionality)
from toolkit.timing import Timer

# ── Coefficient basis ─────────────────────────────────────────────────────────
N = 8
X = np.linspace(-1.5, 1.5, 120)
BASIS = np.stack([X**k / math.factorial(k) for k in range(N)], axis=1)

_COS = np.array([(-1.)**(k//2)   if k%2==0 else 0. for k in range(N)])
_SIN = np.array([(-1.)**((k-1)//2) if k%2==1 else 0. for k in range(N)])


# ── True intermediates (analytical ground truth) ──────────────────────────────

def euler_intermediate(k: int, n_steps: int) -> np.ndarray:
    """
    The exact coefficient vector at step k of an n-step relay from cos to -cos.

    c_k = cos(k*pi/n) * _cos - sin(k*pi/n) * _sin

    k=0:      cos(x)
    k=n/2:   -sin(x)         (if n even; generally: cos(x + pi/2) = -sin(x))
    k=n:    -cos(x)          (Euler's endpoint: cos(x+pi) = -cos(x))

    This IS the path of e^(i*theta) acting on function space.
    """
    theta = k * math.pi / n_steps
    return math.cos(theta) * _COS - math.sin(theta) * _SIN


def all_true_intermediates(n_steps: int) -> List[np.ndarray]:
    """All n_steps+1 true intermediates for the n-step Euler relay."""
    return [euler_intermediate(k, n_steps) for k in range(n_steps + 1)]


# ── Phase rotation step operator ──────────────────────────────────────────────

def euler_step(c: np.ndarray, delta_theta: float) -> np.ndarray:
    """
    Apply a phase rotation of delta_theta to coefficient vector c.

    c(theta + delta) = cos(delta)*c(theta) + sin(delta)*d/dx(c(theta))

    This is exact when c is in the cos/sin subspace.
    For general c it is the leading-order rotation in coefficient space.

    The derivative d/dx in phi_k basis is the shift-left operator:
      (d/dx c)[j] = c[j+1]  (with c[N-1] -> 0)
    """
    deriv_c = np.append(c[1:], 0.)
    return math.cos(delta_theta) * c + math.sin(delta_theta) * deriv_c


def euler_step_fitness(ind: np.ndarray,
                        neighbour_prev: np.ndarray,
                        neighbour_next: np.ndarray,
                        delta_theta: float,
                        cycle_weight: float = 0.2) -> float:
    """
    Fitness of `ind` in the Euler relay chain.

    ind should equal euler_step(neighbour_prev, delta_theta)
    AND euler_step(ind, delta_theta) should equal neighbour_next.
    """
    cost_from_prev = np.linalg.norm(ind - euler_step(neighbour_prev, delta_theta))
    cost_to_next   = np.linalg.norm(neighbour_next - euler_step(ind, delta_theta))
    return -(cost_from_prev + cycle_weight * cost_to_next)


# ── Phase angle extraction ────────────────────────────────────────────────────

def phase_angle(c: np.ndarray) -> float:
    """
    Extract the phase angle of coefficient vector c in the cos/sin subspace.

    Projects c onto _COS and _SIN and computes atan2.
    An exact answer for c in the cos/sin subspace; an approximation otherwise.
    """
    cos_comp = float(np.dot(c, _COS))
    sin_comp = float(-np.dot(c, _SIN))   # note: sin component is stored negatively
    return float(np.arctan2(sin_comp, cos_comp))


# ── Euler relay runner ────────────────────────────────────────────────────────

@dataclass
class EulerRelayResult:
    n_steps:          int
    best_pops:        List[np.ndarray]      # best individual from each population
    true_intermediates: List[np.ndarray]    # analytical ground truth
    step_residuals:   List[float]           # ||best_k - true_k||
    phase_angles:     List[float]           # extracted phase angle per population
    true_angles:      List[float]           # expected phase angles (k*pi/n)
    angle_errors:     List[float]           # |phase_angle - true_angle|
    mean_step_res:    float
    mean_angle_err:   float
    outcome:          str
    timeseries:       dict = field(default_factory=dict)
    timing:           dict = field(default_factory=dict)


def run_euler_relay(n_steps: int = 4,
                     total_angle: float = math.pi,
                     cfg: Optional[CoevoConfig] = None,
                     verbose: bool = True) -> EulerRelayResult:
    """
    Run the Euler relay chain.

    n_steps:     number of relay populations (not counting the endpoints)
                 The chain has n_steps+1 populations: P_0, P_1, ..., P_{n_steps}
                 with P_0 ≈ cos and P_{n_steps} ≈ -cos.
    total_angle: the total phase rotation (pi for Euler, 2*pi for full circle)
    """
    if cfg is None:
        cfg = CoevoConfig(
            n_genes=N, pop_size=80, gens=800,
            n_collaborations=1, k_signal_samples=6,
            mut_std=0.04, sig_weight=0.35, delta=0.25,
            coupling_mode="adaptive_kl",
            signal_design="trajectory_ngram",
            sig_freq=10, elite=2, tourney=5,
        )

    n_pops     = n_steps + 1   # total number of populations in the cycle
    delta      = total_angle / n_steps
    rng        = np.random.default_rng(cfg.seed)
    signal     = get_signal(cfg.signal_design)
    timer      = Timer()
    true_ints  = all_true_intermediates(n_steps)

    # Initialise populations near the true intermediates with noise
    pops = [true_int + rng.standard_normal(N) * 0.5
            for true_int in true_ints]
    pops = [np.array([p[i] for p in pops_row]) if False else p.reshape(N)
            for p in pops]
    # Actually: each population has pop_size individuals
    pops = [rng.standard_normal((cfg.pop_size, N)) * 0.4
            + true_ints[k]
            for k in range(n_pops)]

    sigs      = [np.zeros(2 ** N) for _ in range(n_pops)]
    best_pops = [p[0].copy() for p in pops]

    ts = {"step_residuals": [], "angle_errors": [], "kl": []}

    for g in range(cfg.gens):
        # Update signals
        if g % cfg.sig_freq == 0:
            with timer.measure("signal_compute"):
                for k in range(n_pops):
                    bk = best_pops[k]
                    fa = lambda ind, k=k: euler_step_fitness(
                        ind, best_pops[(k-1)%n_pops],
                        best_pops[(k+1)%n_pops], delta, 0.2)
                    sigs[k] = signal.emit(pops[k], fa, N, rng,
                                          cfg.k_signal_samples, cfg.delta)

        # Evolve each population
        for k in range(n_pops):
            fits = np.array([
                euler_step_fitness(ind, best_pops[(k-1)%n_pops],
                                   best_pops[(k+1)%n_pops], delta, 0.2)
                for ind in pops[k]
            ])

            sig_recv = 0.5 * (sigs[(k-1)%n_pops] + sigs[(k+1)%n_pops])
            me_recv  = signal.receive(sig_recv, N)
            mut_std  = cfg.mut_std * (1.0 + cfg.sig_weight * me_recv)

            elite_idx = np.argsort(fits)[-cfg.elite:]
            new_pop   = np.empty_like(pops[k])
            new_pop[:cfg.elite] = pops[k][elite_idx].copy()
            for j in range(cfg.elite, cfg.pop_size):
                idx = rng.choice(cfg.pop_size, cfg.tourney, replace=False)
                p1  = pops[k][idx[np.argmax(fits[idx])]]
                idx = rng.choice(cfg.pop_size, cfg.tourney, replace=False)
                p2  = pops[k][idx[np.argmax(fits[idx])]]
                mask  = rng.random(N) < 0.5
                child = np.where(mask, p1, p2)
                child += rng.standard_normal(N) * mut_std
                new_pop[j] = child

            with timer.measure("ga_step"):
                pops[k] = new_pop

            fits2 = np.array([
                euler_step_fitness(ind, best_pops[(k-1)%n_pops],
                                   best_pops[(k+1)%n_pops], delta, 0.2)
                for ind in pops[k]
            ])
            best_pops[k] = pops[k][np.argmax(fits2)].copy()

        # Log
        if g % 80 == 0:
            step_res = [float(np.linalg.norm(best_pops[k] - true_ints[k]))
                        for k in range(n_pops)]
            ang_errs = [abs(phase_angle(best_pops[k]) - k*delta)
                        for k in range(n_pops)]
            kl = float(np.mean([kl_divergence_spectra(sigs[k], sigs[(k+1)%n_pops])
                                  for k in range(n_pops)]))
            ts["step_residuals"].append((g, step_res))
            ts["angle_errors"].append((g, ang_errs))
            ts["kl"].append((g, kl))

    # Final diagnostics
    step_residuals = [float(np.linalg.norm(best_pops[k] - true_ints[k]))
                      for k in range(n_pops)]
    phase_angles   = [phase_angle(best_pops[k]) for k in range(n_pops)]
    true_angles    = [k * delta for k in range(n_pops)]
    angle_errors   = [abs(pa - ta) for pa, ta in zip(phase_angles, true_angles)]
    mean_step_res  = float(np.mean(step_residuals))
    mean_angle_err = float(np.mean(angle_errors))

    if mean_step_res < 0.15:
        outcome = "correct"
    elif mean_step_res < 0.40:
        outcome = "partial"
    else:
        outcome = "failed"

    if verbose:
        print(f"\n  n_steps={n_steps}  delta={delta:.4f} rad  outcome={outcome}")
        print(f"  mean_step_residual = {mean_step_res:.5f}  (0 = found true Euler path)")
        print(f"  mean_angle_error   = {mean_angle_err:.5f} rad  (0 = on the rotation manifold)")
        for k in range(n_pops):
            expected_angle_deg = math.degrees(true_angles[k])
            actual_angle_deg   = math.degrees(phase_angles[k])
            print(f"  Pop {k:2d}  expected={expected_angle_deg:6.1f}°  "
                  f"actual={actual_angle_deg:6.1f}°  "
                  f"step_res={step_residuals[k]:.4f}")

    return EulerRelayResult(
        n_steps=n_steps,
        best_pops=best_pops,
        true_intermediates=true_ints,
        step_residuals=step_residuals,
        phase_angles=phase_angles,
        true_angles=true_angles,
        angle_errors=angle_errors,
        mean_step_res=mean_step_res,
        mean_angle_err=mean_angle_err,
        outcome=outcome,
        timeseries=ts,
        timing=timer.to_dict(),
    )
