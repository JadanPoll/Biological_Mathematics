"""
Coevolutionary GA — clean rewrite with pluggable signals.

Key changes from v1:
  - Cycle consistency removed from optimization (it was fighting correct solutions).
    It is now a POST-HOC diagnostic only — measured at convergence, logged to DB.
  - Signal molecule is now a plugin (WalshLandscape | WalshSolution | RawSparse).
  - N-collaborations: each individual evaluated against N random partners.
  - Population-level signal: signal averaged over K sampled individuals.
  - Adaptive coupling: if KL between spectra is large, reduce signal influence.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from toolkit.signals import SignalMolecule, get_signal
from toolkit.walsh import kl_divergence_spectra, spectral_class, effective_dimensionality
from toolkit.timing import Timer


@dataclass
class CoevoConfig:
    n_genes:            int   = 8
    pop_size:           int   = 80
    gens:               int   = 600
    mut_std:            float = 0.04
    sig_weight:         float = 0.35
    delta:              float = 0.25
    tourney:            int   = 5
    elite:              int   = 2
    sig_freq:           int   = 10
    n_collaborations:   int   = 5
    k_signal_samples:   int   = 8
    coupling_mode:      str   = "adaptive_kl"   # "fixed" | "adaptive_kl"
    signal_design:      str   = "walsh_landscape"
    eff_dim_threshold:  float = 0.05
    seed:               int   = 42


@dataclass
class CoevoResult:
    best_a:           np.ndarray
    best_b:           np.ndarray
    residual:         float
    cycle_residual:   float
    convergence_gen:  Optional[int]
    spectral_class_a: str
    spectral_class_b: str
    eff_dim_a:        int
    eff_dim_b:        int
    walsh_me_a:       np.ndarray
    walsh_me_b:       np.ndarray
    kl_final:         float
    outcome:          str
    failure_mode:     Optional[str]
    timeseries:       dict = field(default_factory=dict)
    timing:           dict = field(default_factory=dict)   # wall-clock cost breakdown


# ── Internal GA helpers ───────────────────────────────────────────────────────

def _tournament(pop, fits, k, rng):
    idx = rng.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax(fits[idx])]].copy()


def _n_collab_fitness(ind, fitness_fn, other_pop, n, rng):
    """Average fitness of `ind` against N random partners. No cycle penalty."""
    idxs = rng.choice(len(other_pop), min(n, len(other_pop)), replace=False)
    return float(np.mean([fitness_fn(ind, other_pop[i]) for i in idxs]))


def _step(pop, fitness_fn, other_pop, cfg: CoevoConfig,
          signal_from_other: np.ndarray, rng, signal: SignalMolecule):

    # Fitness with N-collaborations (no cycle penalty)
    fits = np.array([
        _n_collab_fitness(ind, fitness_fn, other_pop, cfg.n_collaborations, rng)
        for ind in pop
    ])

    # Per-gene mutation std biased by other population's signal
    me_other = signal.receive(signal_from_other, cfg.n_genes)

    # Adaptive coupling: dampen signal influence when populations are spectrally far
    dampen = 1.0
    if cfg.coupling_mode == "adaptive_kl":
        # We don't have own signal here, so use me magnitude as proxy
        # (small me_other = sender near optimum = tight coupling appropriate)
        dampen = float(np.clip(1.0 - me_other.mean(), 0.3, 1.0))

    mut_std = cfg.mut_std * (1.0 + cfg.sig_weight * dampen * me_other)

    # Elitism
    elite_idx = np.argsort(fits)[-cfg.elite:]
    new_pop   = np.empty_like(pop)
    new_pop[:cfg.elite] = pop[elite_idx].copy()

    for j in range(cfg.elite, cfg.pop_size):
        p1   = _tournament(pop, fits, cfg.tourney, rng)
        p2   = _tournament(pop, fits, cfg.tourney, rng)
        mask = rng.random(cfg.n_genes) < 0.5
        child = np.where(mask, p1, p2)
        child += rng.standard_normal(cfg.n_genes) * mut_std
        new_pop[j] = child

    return new_pop, fits


# ── Main entry point ─────────────────────────────────────────────────────────

def run(fitness_a: Callable,   # fitness_a(ind_a, partner_b) -> float
        fitness_b: Callable,   # fitness_b(ind_b, partner_a) -> float
        joint_residual: Callable,  # joint_residual(best_a, best_b) -> float
        cfg: CoevoConfig,
        init_pop_a: Optional[np.ndarray] = None,
        init_pop_b: Optional[np.ndarray] = None,
        log_every: int = 50) -> CoevoResult:

    rng    = np.random.default_rng(cfg.seed)
    signal = get_signal(cfg.signal_design)
    timer  = Timer()

    pop_a = (init_pop_a if init_pop_a is not None
             else rng.standard_normal((cfg.pop_size, cfg.n_genes)) * 0.3)
    pop_b = (init_pop_b if init_pop_b is not None
             else rng.standard_normal((cfg.pop_size, cfg.n_genes)) * 0.3)

    sig_a = np.zeros(2 ** cfg.n_genes)
    sig_b = np.zeros(2 ** cfg.n_genes)

    best_a, best_b = pop_a[0].copy(), pop_b[0].copy()
    convergence_gen = None

    ts = {"fit_a": [], "fit_b": [], "kl": [],
          "eff_dim_a": [], "eff_dim_b": [], "residual": [],
          "walsh_a": [], "walsh_b": []}

    for g in range(cfg.gens):
        with timer.measure("ga_step"):
            pop_a, fits_a = _step(pop_a, fitness_a, pop_b, cfg, sig_b, rng, signal)
            pop_b, fits_b = _step(pop_b, fitness_b, pop_a, cfg, sig_a, rng, signal)

        best_a = pop_a[np.argmax(fits_a)].copy()
        best_b = pop_b[np.argmax(fits_b)].copy()

        if g % cfg.sig_freq == 0:
            fa1 = lambda ind: fitness_a(ind, best_b)
            fb1 = lambda ind: fitness_b(ind, best_a)
            with timer.measure("signal_compute"):
                sig_a = signal.emit(pop_a, fa1, cfg.n_genes, rng,
                                    cfg.k_signal_samples, cfg.delta)
                sig_b = signal.emit(pop_b, fb1, cfg.n_genes, rng,
                                    cfg.k_signal_samples, cfg.delta)

        if g % log_every == 0:
            kl   = kl_divergence_spectra(sig_a, sig_b)
            me_a = signal.receive(sig_a, cfg.n_genes)
            me_b = signal.receive(sig_b, cfg.n_genes)
            res  = joint_residual(best_a, best_b)
            ts["fit_a"].append((g, float(fits_a.max())))
            ts["fit_b"].append((g, float(fits_b.max())))
            ts["kl"].append((g, float(kl)))
            ts["eff_dim_a"].append((g, effective_dimensionality(me_a, cfg.eff_dim_threshold)))
            ts["eff_dim_b"].append((g, effective_dimensionality(me_b, cfg.eff_dim_threshold)))
            ts["residual"].append((g, float(res)))
            # Full Walsh spectra — the training signal for the betweenness model.
            # These trajectories show what a valid/invalid morphing looks like spectrally.
            ts["walsh_a"].append((g, sig_a.copy()))
            ts["walsh_b"].append((g, sig_b.copy()))

            if convergence_gen is None and res < 0.10:
                convergence_gen = g

    # ── Final diagnostics ────────────────────────────────────────────────────
    fits_af = np.array([fitness_a(ind, best_b) for ind in pop_a])
    fits_bf = np.array([fitness_b(ind, best_a) for ind in pop_b])
    best_a  = pop_a[np.argmax(fits_af)].copy()
    best_b  = pop_b[np.argmax(fits_bf)].copy()

    residual = float(joint_residual(best_a, best_b))

    # Cycle residual: diagnostic only — how different are A and B at convergence?
    # For Tier 0 (identical functions) this should be ~0.
    # For Tier 1 (derivative relationship) this tells us about degenerate solutions.
    cycle_residual = float(np.linalg.norm(best_a - best_b)
                           / (np.linalg.norm(best_a) + 1e-8))

    kl_final = float(kl_divergence_spectra(sig_a, sig_b))
    me_a     = signal.receive(sig_a, cfg.n_genes)
    me_b     = signal.receive(sig_b, cfg.n_genes)
    sc_a     = spectral_class(me_a)
    sc_b     = spectral_class(me_b)
    ed_a     = effective_dimensionality(me_a, cfg.eff_dim_threshold)
    ed_b     = effective_dimensionality(me_b, cfg.eff_dim_threshold)

    # Outcome classification
    if residual < 0.10:
        outcome, failure = "correct", None
    elif kl_final < 0.05 and residual > 1.0:
        outcome, failure = "wrong_attractor", "miscoordination"
    elif convergence_gen is None:
        outcome, failure = "failed", "stagnation"
    else:
        outcome, failure = "partial", None

    return CoevoResult(
        best_a=best_a, best_b=best_b,
        residual=residual, cycle_residual=cycle_residual,
        convergence_gen=convergence_gen,
        spectral_class_a=sc_a, spectral_class_b=sc_b,
        eff_dim_a=ed_a, eff_dim_b=ed_b,
        walsh_me_a=me_a, walsh_me_b=me_b,
        kl_final=kl_final,
        outcome=outcome, failure_mode=failure,
        timeseries=ts,
        timing=timer.to_dict(),
    )
