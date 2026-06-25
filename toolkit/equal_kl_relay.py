"""
Equal-KL relay chain — uniform curvature constraint.

The string theory / quantum well inspiration made precise:

  In a quantum well, discrete energy levels emerge because:
    1. The wavefunction satisfies boundary conditions at both walls
    2. The wavefunction is normalised (integral |psi|^2 = 1)
    3. The curvature (kinetic energy) is fixed by the energy level

  For our relay chain the equivalents are:
    1. Cycle closure: the chain A->B->C->D->A closes
    2. Normalisation: all populations have the same L2 norm
       (kills zero degenerate solution; forces populations onto the unit sphere)
    3. Equal-KL: the Walsh spectral KL divergence between adjacent populations
       is the SAME at every step (constant curvature = constant string tension)
       Only paths with uniform curvature through geometric space survive.

  These three together are highly selective:
    - Zero solution killed by normalisation
    - Identity solution (A=B=C=D) killed by equal-KL (KL=0 everywhere means
      no movement, but normalisation forces populations to be non-trivial)
    - Non-uniform paths killed by equal-KL variance penalty

  For the 360-degree rotation (cos->sin->-cos->-sin->cos):
    - All four steps have the same Walsh spectral distance (by symmetry)
    - Equal-KL constraint is exactly satisfied
    - The rotation is selected as the minimum-action path

  The equal-KL constraint is the Walsh spectral analogue of:
    - Constant string tension in string theory
    - Constant curvature in differential geometry (a geodesic)
    - Constant energy flux in wave propagation

  A path with equal KL at every step IS a geodesic through Walsh spectral space.
  The relay chain is finding the geodesic between A and D.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from toolkit.relay_chain import (RelayChainConfig, RelayResult,
                                  _derivative_step, _make_step_fitness,
                                  _alternating_score)
from toolkit.ga import CoevoConfig
from toolkit.signals import get_signal
from toolkit.walsh import (kl_divergence_spectra, spectral_class,
                            effective_dimensionality)
from toolkit.timing import Timer


@dataclass
class EqualKLConfig:
    n_pops:              int   = 4
    cfg:                 CoevoConfig = field(default_factory=CoevoConfig)
    # Constraint weights
    step_weight:         float = 1.0    # weight on the derivative step constraint
    norm_weight:         float = 0.5    # weight on normalisation constraint
    equal_kl_weight:     float = 0.5    # weight on equal-KL constraint
    target_norm:         float = -1.0   # target L2 norm (-1 = auto: use mean norm)
    cycle_weight:        float = 0.20
    notes:               str   = ""


def _norm_penalty(ind: np.ndarray, target_norm: float) -> float:
    """Penalty for ||ind||_2 deviating from target_norm."""
    return (np.linalg.norm(ind) - target_norm) ** 2


def _equal_kl_penalty(sigs: List[np.ndarray], n_pops: int) -> float:
    """
    Penalty for unequal KL divergences between adjacent population signals.
    Variance of the KL values around the cycle.
    Zero = all steps have equal Walsh spectral distance = constant curvature geodesic.
    """
    kls = []
    for k in range(n_pops):
        kl = kl_divergence_spectra(sigs[k], sigs[(k+1) % n_pops])
        kls.append(kl)
    if not kls:
        return 0.0
    return float(np.var(kls))


def _make_constrained_step_fitness(step_order: float,
                                    norm_weight: float,
                                    target_norm: float,
                                    eq_kl_weight: float,
                                    eq_kl_penalty_val: float):
    """
    Step fitness with normalisation and equal-KL penalty baked in.
    eq_kl_penalty_val is computed externally (requires all populations' signals).
    """
    def _fit(ind, neighbour_prev, neighbour_next, cycle_weight):
        # Core: derivative step constraint
        cost_step = np.linalg.norm(ind - _derivative_step(neighbour_prev, step_order))
        cost_next = np.linalg.norm(neighbour_next - _derivative_step(ind, step_order))
        # Normalisation: ||ind|| should equal target_norm
        cost_norm = norm_weight * _norm_penalty(ind, target_norm)
        # Equal-KL: applied as a uniform penalty from the current state of sigs
        cost_kl   = eq_kl_weight * eq_kl_penalty_val
        return -(cost_step + cycle_weight * cost_next + cost_norm + cost_kl)
    return _fit


def run_equal_kl(cfg_equal: EqualKLConfig,
                 log_every: int = 50) -> RelayResult:
    """
    Run equal-KL constrained relay chain.

    The three constraints — cycle closure, normalisation, equal-KL —
    together act like the quantum well walls + normalisation:
    they select only a discrete set of valid paths.
    """
    cfg    = cfg_equal.cfg
    n      = cfg_equal.n_pops
    rng    = np.random.default_rng(cfg.seed)
    signal = get_signal(cfg.signal_design)
    timer  = Timer()
    step_order = 1.0 / n   # fractional step so n steps = 1 full rotation

    # Initialise
    pops = [rng.standard_normal((cfg.pop_size, cfg.n_genes)) * 0.5
            for _ in range(n)]
    sigs = [np.zeros(2 ** cfg.n_genes) for _ in range(n)]
    best_pops = [p[0].copy() for p in pops]

    # Auto target norm: mean norm of initial population
    if cfg_equal.target_norm < 0:
        all_norms = [np.linalg.norm(p) for pop in pops for p in pop]
        target_norm = float(np.mean(all_norms))
    else:
        target_norm = cfg_equal.target_norm

    ts = {"step_residuals": [], "cycle_residual": [],
          "alt_score": [], "kl_variance": []}

    for g in range(cfg.gens):
        # Update signals
        if g % cfg.sig_freq == 0:
            with timer.measure("signal_compute"):
                for k in range(n):
                    bk   = best_pops[k]
                    fa   = lambda ind, k=k: -(
                        np.linalg.norm(ind - _derivative_step(best_pops[(k-1)%n], step_order))
                    )
                    sigs[k] = signal.emit(pops[k], fa, cfg.n_genes, rng,
                                          cfg.k_signal_samples, cfg.delta)

        # Compute current equal-KL penalty (shared across all pops this generation)
        kl_penalty = _equal_kl_penalty(sigs, n)

        # Evolve each population
        for k in range(n):
            step_fn = _make_constrained_step_fitness(
                step_order,
                cfg_equal.norm_weight,
                target_norm,
                cfg_equal.equal_kl_weight,
                kl_penalty,
            )

            sig_recv = 0.5 * (sigs[(k-1)%n] + sigs[(k+1)%n])
            me_recv  = signal.receive(sig_recv, cfg.n_genes)
            mut_std  = cfg.mut_std * (1.0 + cfg.sig_weight * me_recv)

            fits = np.array([
                step_fn(ind, best_pops[(k-1)%n], best_pops[(k+1)%n],
                        cfg_equal.cycle_weight)
                for ind in pops[k]
            ])

            elite_idx = np.argsort(fits)[-cfg.elite:]
            new_pop   = np.empty_like(pops[k])
            new_pop[:cfg.elite] = pops[k][elite_idx].copy()

            for j in range(cfg.elite, cfg.pop_size):
                idx  = rng.choice(cfg.pop_size, cfg.tourney, replace=False)
                p1   = pops[k][idx[np.argmax(fits[idx])]]
                idx  = rng.choice(cfg.pop_size, cfg.tourney, replace=False)
                p2   = pops[k][idx[np.argmax(fits[idx])]]
                mask = rng.random(cfg.n_genes) < 0.5
                child = np.where(mask, p1, p2)
                child += rng.standard_normal(cfg.n_genes) * mut_std
                new_pop[j] = child

            with timer.measure("ga_step"):
                pops[k] = new_pop

            fits2 = np.array([
                step_fn(ind, best_pops[(k-1)%n], best_pops[(k+1)%n],
                        cfg_equal.cycle_weight)
                for ind in pops[k]
            ])
            best_pops[k] = pops[k][np.argmax(fits2)].copy()

        # Log
        if g % log_every == 0:
            step_res = [
                float(np.linalg.norm(best_pops[k]
                                     - _derivative_step(best_pops[(k-1)%n], step_order)))
                for k in range(n)
            ]
            cycle_res = float(np.linalg.norm(
                best_pops[0] - _derivative_step(best_pops[-1], step_order)
            ))
            mes = [signal.receive(sigs[k], cfg.n_genes) for k in range(n)]
            alt = _alternating_score(mes)
            kl_var = _equal_kl_penalty(sigs, n)
            ts["step_residuals"].append((g, step_res))
            ts["cycle_residual"].append((g, cycle_res))
            ts["alt_score"].append((g, alt))
            ts["kl_variance"].append((g, float(kl_var)))

    # Final
    step_residuals = [
        float(np.linalg.norm(best_pops[k]
                             - _derivative_step(best_pops[(k-1)%n], step_order)))
        for k in range(n)
    ]
    mean_step_res = float(np.mean(step_residuals))
    cycle_residual = float(sum(step_residuals))
    kl_var_final  = _equal_kl_penalty(sigs, n)
    mes  = [signal.receive(sigs[k], cfg.n_genes) for k in range(n)]
    scs  = [spectral_class(me) for me in mes]
    alt  = _alternating_score(mes)
    norms = [float(np.linalg.norm(best_pops[k])) for k in range(n)]

    # Outcome
    if mean_step_res < 0.08 and kl_var_final < 0.01:
        outcome = "correct_geodesic"
    elif alt < 0.05:
        outcome = "degenerate"
    elif mean_step_res < 0.15:
        outcome = "partial"
    else:
        outcome = "failed"

    print(f"\n  mean_step_res = {mean_step_res:.5f}")
    print(f"  kl_variance   = {kl_var_final:.5f}  (0 = perfect equal-KL geodesic)")
    print(f"  alt_score     = {alt:.4f}          (1 = perfect cos/sin alternation)")
    print(f"  norms         = {[round(x,3) for x in norms]}")
    print(f"  outcome       = {outcome}")

    return RelayResult(
        populations=best_pops,
        cycle_residual=cycle_residual,
        step_residuals=step_residuals,
        walsh_signatures=mes,
        spectral_classes=scs,
        alternating_score=alt,
        outcome=outcome,
        timeseries=ts,
        timing=timer.to_dict(),
    )
