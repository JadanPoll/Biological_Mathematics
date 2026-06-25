"""
Stress metrics — the mathematical toolkit for measuring morphing difficulty.

Fixes mistakes identified from research:
  1. KL divergence replaced by Jensen-Shannon (symmetric, bounded, true metric)
  2. Wasserstein distance implemented via sliced approximation (no POT needed)
  3. Ollivier-Ricci curvature for relay chain path geometry
  4. Hausdorff dimension estimation for path roughness detection
  5. Fisher Information approximation for natural gradient signal

Theoretical grounding:
  - JSD: symmetric KL, bounded in [0,1], square root is a metric (Endres & Schindelin 2003)
  - Sliced Wasserstein: O(n*d*log(n)), unbiased estimator of Wasserstein-2 (Rabin 2012)
  - Ollivier-Ricci: measures curvature via optimal transport between neighbour
    distributions; equal curvature on all edges = geodesic (Ollivier 2009)
  - Hausdorff dim: box-counting dimension of path in coefficient space; > 1 = fractal
  - Fisher approximation: empirical Fisher of population as natural gradient proxy

These replace the broken tools:
  - walsh.kl_divergence_spectra  →  stress_metrics.jsd_spectra (symmetric)
  - relay chain plain GA          →  euler_relay with CMA-ES step
  - betweenness score             →  ollivier_ricci_path_curvature
"""

import math
import numpy as np
from typing import List, Optional
import scipy.stats as stats
import scipy.spatial.distance as spdist


# ── Jensen-Shannon Divergence (correct symmetric information distance) ────────

def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Jensen-Shannon Divergence between two non-negative vectors (treated as
    unnormalised probability distributions).

    Properties (unlike KL):
      - Symmetric: JSD(p,q) = JSD(q,p)
      - Bounded:   JSD in [0, log(2)]  (or [0,1] with log base 2)
      - sqrt(JSD) is a metric (satisfies triangle inequality)
      - Works when either distribution has zeros

    This is the correct replacement for kl_divergence_spectra.
    """
    p = np.abs(p) + eps;  p /= p.sum()
    q = np.abs(q) + eps;  q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def jsd_metric(p: np.ndarray, q: np.ndarray) -> float:
    """
    sqrt(JSD) — this IS a metric. Use when triangle inequality matters.
    E.g., for computing path lengths as sums of pairwise distances.
    """
    return float(math.sqrt(max(jsd(p, q), 0.0)))


# ── Sliced Wasserstein Distance (no POT needed) ───────────────────────────────

def sliced_wasserstein(pop_a: np.ndarray, pop_b: np.ndarray,
                        n_projections: int = 50,
                        rng: Optional[np.random.Generator] = None) -> float:
    """
    Sliced Wasserstein-2 distance between two populations (sets of points).

    Approximates the true Wasserstein-2 distance by averaging the 1D
    Wasserstein distance over random linear projections.

    Complexity: O(n_projections * n * log(n))  vs O(n^3) for exact OT

    This is the Earth Mover's Distance between two GA populations.
    Low value = populations are geometrically close (similar morphing).
    High value = populations are far apart (large morphing effort required).

    Reference: Rabin et al. (2012), "Wasserstein Barycenter and its Application"
    """
    if rng is None:
        rng = np.random.default_rng()

    n_genes = pop_a.shape[1]
    directions = rng.standard_normal((n_projections, n_genes))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    total = 0.0
    for d in directions:
        proj_a = pop_a @ d
        proj_b = pop_b @ d
        # 1D Wasserstein = mean absolute difference of sorted distributions
        total += stats.wasserstein_distance(proj_a, proj_b)

    return float(total / n_projections)


def population_wasserstein_path(populations: List[np.ndarray],
                                 n_proj: int = 30) -> List[float]:
    """
    Compute Wasserstein distances between adjacent relay chain populations.
    Returns a list of n_pops-1 distances.

    Equal Wasserstein distances at every step = uniform curvature = geodesic.
    Variance of these distances = how far from a geodesic the path is.
    """
    distances = []
    for k in range(len(populations) - 1):
        # Expand singletons to populations for sliced Wasserstein
        if populations[k].ndim == 1:
            pa = populations[k].reshape(1, -1)
            pb = populations[k+1].reshape(1, -1)
            d = float(np.linalg.norm(populations[k] - populations[k+1]))
        else:
            d = sliced_wasserstein(populations[k], populations[k+1], n_proj)
        distances.append(d)
    return distances


# ── Ollivier-Ricci Curvature (geometry of the path) ──────────────────────────

def ollivier_ricci_edge(pop_a: np.ndarray, pop_b: np.ndarray,
                         alpha: float = 0.5,
                         n_proj: int = 20) -> float:
    """
    Ollivier-Ricci curvature of the edge between population A and population B.

    kappa(A, B) = 1 - W_1(m_A, m_B) / d(A, B)

    where m_A, m_B are the "neighbourhood distributions" (lazy random walk from
    the population centroids), and d(A, B) is the Wasserstein distance.

    Properties:
      kappa > 0: edge is on a positively curved manifold (like a sphere)
                  implies populations are "converging" toward each other
      kappa = 0: flat geometry (Euclidean space)
      kappa < 0: negatively curved (hyperbolic) — populations "diverging"

    For a geodesic relay chain: all edges should have equal curvature.
    Equal curvature = the path is a Ricci soliton (stable under Ricci flow).

    Reference: Ollivier (2009), "Ricci curvature of Markov chains on metric spaces"
    """
    centroid_a = np.mean(pop_a, axis=0) if pop_a.ndim > 1 else pop_a
    centroid_b = np.mean(pop_b, axis=0) if pop_b.ndim > 1 else pop_b

    d_ab = float(np.linalg.norm(centroid_a - centroid_b))
    if d_ab < 1e-10:
        return 0.0   # degenerate: populations at same point

    # Neighbourhood distributions: Gaussian blobs centred at each population
    # (approximation to the lazy random walk neighbourhood)
    spread = max(d_ab * 0.1, 0.01)
    rng = np.random.default_rng(0)
    n_samples = 50

    if pop_a.ndim > 1:
        m_a = pop_a[rng.choice(len(pop_a), n_samples, replace=True)]
        m_b = pop_b[rng.choice(len(pop_b), n_samples, replace=True)]
    else:
        m_a = pop_a + rng.standard_normal((n_samples, len(pop_a))) * spread
        m_b = pop_b + rng.standard_normal((n_samples, len(pop_b))) * spread

    w1_mb_ma = sliced_wasserstein(m_a, m_b, n_proj, rng)
    kappa = 1.0 - w1_mb_ma / d_ab
    return float(kappa)


def relay_ricci_curvatures(populations: List[np.ndarray]) -> List[float]:
    """
    Compute Ollivier-Ricci curvature for each edge in the relay chain.
    For a geodesic: all curvatures should be equal.
    Variance of curvatures = how far from geodesic the path is.
    """
    curvatures = []
    for k in range(len(populations) - 1):
        pa = populations[k].reshape(1,-1) if populations[k].ndim==1 else populations[k]
        pb = populations[k+1].reshape(1,-1) if populations[k+1].ndim==1 else populations[k+1]
        curvatures.append(ollivier_ricci_edge(pa, pb))
    return curvatures


# ── Hausdorff Dimension Estimation (path roughness) ──────────────────────────

def hausdorff_dimension_path(path_points: List[np.ndarray],
                               min_scale: float = 0.01,
                               max_scale: float = 1.0,
                               n_scales: int = 20) -> float:
    """
    Estimate the Hausdorff (box-counting) dimension of a path in coefficient space.

    For a smooth curve: dimension ≈ 1.0
    For a fractal path:  dimension > 1.0 (approaches 2.0 for space-filling)

    If dimension > 1.0, the relay chain is finding fractal paths — meaning
    finer discretization will NOT converge (more relay pops = more, not fewer,
    valid paths). This is the "Weierstrass function" failure mode.

    Method: box-counting. Count how many boxes of size ε cover the path.
    D = -lim(ε→0) log(N(ε)) / log(ε)
    """
    if len(path_points) < 3:
        return 1.0

    points = np.stack([p.flatten() for p in path_points])
    scales = np.logspace(math.log10(min_scale), math.log10(max_scale), n_scales)
    counts = []

    for scale in scales:
        # Count occupied boxes at this scale
        indices = np.floor(points / scale).astype(int)
        n_boxes = len(set(map(tuple, indices)))
        counts.append(n_boxes)

    # Linear regression of log(count) vs log(1/scale)
    log_inv_scales = np.log(1.0 / scales)
    log_counts = np.log(np.array(counts, dtype=float) + 1)

    # Dimension = slope of log(N) vs log(1/ε)
    coeffs = np.polyfit(log_inv_scales, log_counts, 1)
    return float(max(0.5, min(coeffs[0], 3.0)))   # clamp to sensible range


# ── Fisher Information Approximation ─────────────────────────────────────────

def empirical_fisher(population: np.ndarray, fitness_fn,
                      eps: float = 1e-4) -> np.ndarray:
    """
    Empirical approximation to the Fisher Information Matrix for the
    current population treated as samples from a Gaussian.

    The natural gradient uses the inverse Fisher matrix to correct the
    gradient direction, following the steepest descent in distribution space
    rather than parameter space.

    This is what CMA-ES computes implicitly through covariance adaptation.

    For a Gaussian population: F ≈ Cov(population)^{-1}
    The natural gradient g_nat = F^{-1} * g_ordinary

    Returns the covariance matrix (proxy for Fisher^{-1}).
    """
    return np.cov(population.T)


def natural_gradient_correction(gradient: np.ndarray,
                                  population: np.ndarray) -> np.ndarray:
    """
    Correct a raw gradient by the empirical Fisher matrix (CMA-ES style).
    Returns the natural gradient direction.

    This is the geodesic direction in the Fisher Information geometry —
    the "true" steepest descent that respects the curvature of distribution space.
    """
    cov = empirical_fisher(population, None)
    try:
        cov_inv = np.linalg.pinv(cov + 1e-6 * np.eye(len(cov)))
        return cov_inv @ gradient
    except np.linalg.LinAlgError:
        return gradient


# ── Unified stress report ─────────────────────────────────────────────────────

def relay_stress_report(populations: List[np.ndarray],
                         signals: List[np.ndarray],
                         true_intermediates: Optional[List[np.ndarray]] = None,
                         verbose: bool = True) -> dict:
    """
    Compute all stress metrics for a relay chain.
    Returns dict with metrics needed for the Mendel notebook.
    """
    wass_dists  = population_wasserstein_path(populations)
    ricci_curv  = relay_ricci_curvatures(populations)
    jsd_dists   = [jsd_metric(signals[k], signals[(k+1)%len(signals)])
                   for k in range(len(signals))]

    path_points = [np.mean(p, axis=0) if p.ndim > 1 else p for p in populations]
    h_dim = hausdorff_dimension_path(path_points)

    wass_var    = float(np.var(wass_dists)) if wass_dists else 0.
    ricci_var   = float(np.var(ricci_curv)) if ricci_curv else 0.
    jsd_var     = float(np.var(jsd_dists))  if jsd_dists  else 0.

    if true_intermediates:
        dist_to_true = [float(np.linalg.norm(
            (np.mean(populations[k], axis=0) if populations[k].ndim > 1
             else populations[k]) - true_intermediates[k]))
            for k in range(min(len(populations), len(true_intermediates)))]
    else:
        dist_to_true = []

    report = {
        "wasserstein_per_step":  wass_dists,
        "wasserstein_variance":  wass_var,    # 0 = equal-transport geodesic
        "ricci_curvature":       ricci_curv,
        "ricci_variance":        ricci_var,   # 0 = Ricci soliton (true geodesic)
        "jsd_per_step":          jsd_dists,
        "jsd_variance":          jsd_var,
        "hausdorff_dimension":   h_dim,       # 1.0 = smooth path, >1 = fractal
        "dist_to_true":          dist_to_true,
        "is_geodesic":           wass_var < 0.01 and ricci_var < 0.01,
        "is_fractal":            h_dim > 1.2,
    }

    if verbose:
        print(f"\n  RELAY CHAIN STRESS REPORT")
        print(f"  Wasserstein per step: {[round(x,4) for x in wass_dists]}")
        print(f"  Wasserstein variance: {wass_var:.5f}  (0 = equal-transport geodesic)")
        print(f"  Ricci curvatures:     {[round(x,4) for x in ricci_curv]}")
        print(f"  Ricci variance:       {ricci_var:.5f}  (0 = Ricci soliton)")
        print(f"  Hausdorff dimension:  {h_dim:.4f}  (1.0=smooth, >1.2=fractal warning)")
        print(f"  Is geodesic:          {report['is_geodesic']}")
        print(f"  Is fractal path:      {report['is_fractal']}")
        if dist_to_true:
            print(f"  Dist to true interms: {[round(x,4) for x in dist_to_true]}")

    return report
