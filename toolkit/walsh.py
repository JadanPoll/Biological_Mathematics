"""
Walsh analysis toolkit.

Three levels of signal:
  1. best_signal     — WHT of fitness landscape around the single best individual
  2. population_signal — WHT averaged across K sampled individuals (distribution-level)
  3. effective_dimensionality — how many Walsh main-effects are significant
                                (proxy for how many algebraic dimensions the
                                 population is actively exploring)
"""

import numpy as np

_EFF_DIM_THRESHOLD = 0.05   # fraction of max main-effect to count as "significant"


def wht(v: np.ndarray) -> np.ndarray:
    """
    Normalised Walsh-Hadamard transform.  len(v) must be a power of 2.
    Index 2^k in the output is the main-effect of gene k (sensitivity of the
    fitness landscape to flipping gene k alone).
    """
    u, h, n = np.array(v, dtype=float), 1, len(v)
    while h < n:
        for i in range(0, n, 2 * h):
            a = u[i:i+h].copy()
            b = u[i+h:i+2*h].copy()
            u[i:i+h]     = a + b
            u[i+h:i+2*h] = a - b
        h *= 2
    return u / n


def main_effects(walsh_spectrum: np.ndarray, n_genes: int) -> np.ndarray:
    """Extract the n_genes main-effect coefficients from a full Walsh spectrum."""
    return walsh_spectrum[1 << np.arange(n_genes)]


def effective_dimensionality(me: np.ndarray,
                             threshold: float = _EFF_DIM_THRESHOLD) -> int:
    """
    Number of genes whose Walsh main-effect magnitude exceeds threshold
    times the maximum main-effect.  Low value = population is exploring
    a low-dimensional slice; high = broad exploration.

    An extradimensional bypass increases this count by adding a new
    intermediate population that opens a previously flat dimension.
    """
    if me.max() < 1e-12:
        return 0
    return int(np.sum(np.abs(me) > threshold * np.abs(me).max()))


def landscape_signal(best: np.ndarray, fitness_fn, delta: float,
                     n_genes: int) -> np.ndarray:
    """
    WHT of fitness values at all 2^n_genes corners of a ±delta hypercube
    centred on `best`.  Returns full spectrum (length 2^n_genes).
    """
    n_corners = 2 ** n_genes
    vals = np.empty(n_corners)
    for i in range(n_corners):
        perturb = np.array([(delta if (i >> k) & 1 else -delta)
                            for k in range(n_genes)])
        vals[i] = fitness_fn(best + perturb)
    return wht(vals)


def population_signal(pop: np.ndarray, fitness_fn, delta: float,
                      n_genes: int, k_samples: int = 8,
                      rng: np.random.Generator = None) -> np.ndarray:
    """
    Population-level signal: average Walsh spectrum over K individuals
    sampled from `pop`.  Encodes the DISTRIBUTION of the population's
    landscape structure, not just the peak.

    This is the N-collaborations analogue for signal molecules:
    information about the shape of the whole population, not one individual.
    """
    if rng is None:
        rng = np.random.default_rng()
    k = min(k_samples, len(pop))
    idxs = rng.choice(len(pop), k, replace=False)
    spectra = np.stack([landscape_signal(pop[i], fitness_fn, delta, n_genes)
                        for i in idxs])
    return spectra.mean(axis=0)


def kl_divergence_spectra(spec_a: np.ndarray, spec_b: np.ndarray,
                           eps: float = 1e-10) -> float:
    """
    KL divergence between two Walsh spectra treated as (unnormalised) distributions.
    Used as adaptive coupling-strength signal: small KL = tight coupling OK.
    """
    a = np.abs(spec_a) + eps
    b = np.abs(spec_b) + eps
    a /= a.sum();  b /= b.sum()
    return float(np.sum(a * np.log(a / b)))


def spectral_class(walsh_me: np.ndarray) -> str:
    """
    Classify the Walsh main-effect profile into one of four categories.
    This is the 'ratio' we record — the Mendel 3:1 equivalent.
    """
    if walsh_me.max() < 1e-10:
        return "flat"
    me = np.abs(walsh_me) / np.abs(walsh_me).max()
    # Check monotone decay (smooth landscape)
    diffs = np.diff(me)
    if np.all(diffs <= 0.01):
        return "smooth_decay"
    # Check oscillatory (alternating sign → typical of cos/sin structure)
    signs = np.sign(walsh_me[walsh_me != 0])
    if len(signs) > 2 and not np.all(signs == signs[0]):
        return "oscillatory"
    # Non-monotone without obvious oscillation
    if np.any(diffs > 0.05):
        return "nonmonotone"
    return "flat"
