"""
Information geometry utilities — minimal implementation without geomstats.
(geomstats incompatible with Python 3.14 / numpy 2.x)

Implements the core Fisher information geometry operations needed for our system:
  - Fisher-Rao metric between two probability distributions
  - Geodesic distance on statistical manifolds
  - Natural gradient direction (inverse Fisher times ordinary gradient)
  - Exponential and log maps on the hypersphere (population distributions)

Connection to the Ramanujan partition problem:
  Just as the partition generating function lifted from discrete integer counts
  to a smooth analytic function in the complex disk, the Fisher information
  metric lifts from a discrete population of coefficient vectors to a smooth
  Riemannian manifold of probability distributions.
  The geodesics in this manifold correspond to minimum-effort paths
  between population states — the "Rademacher exact formula" analog for
  our system.

Connection to the relay chain:
  The Wasserstein distance measures effort in physical space (how far to
  move the mass). The Fisher-Rao distance measures effort in distribution
  space (how distinguishable are the two population states). For Gaussian
  populations, these are related but different.
  A population near the true Euler intermediate has low Fisher-Rao distance
  from the ideal distribution. A degenerate population has high distance.
"""

import math
import numpy as np
from scipy.special import rel_entr
from scipy.stats import entropy as scipy_entropy


# ── Fisher-Rao distance for Gaussian populations ─────────────────────────────

def gaussian_fisher_rao(mu1: np.ndarray, sigma1: float,
                         mu2: np.ndarray, sigma2: float) -> float:
    """
    Fisher-Rao geodesic distance between two univariate Gaussian distributions
    extended to multivariate case via the trace of the Fisher matrix.

    For univariate Gaussians N(mu, sigma^2):
    d_FR = sqrt(2) * |log(sigma1/sigma2)| + ||mu1-mu2|| / sqrt(sigma1*sigma2)

    This is the geodesic distance on the statistical manifold of Gaussians.
    Compare to Wasserstein distance which measures mass transport in parameter space.
    """
    if abs(sigma1) < 1e-10 or abs(sigma2) < 1e-10:
        return float(np.linalg.norm(mu1 - mu2))
    log_term = math.sqrt(2) * abs(math.log(sigma1 / sigma2))
    mean_term = float(np.linalg.norm(mu1 - mu2)) / math.sqrt(sigma1 * sigma2)
    return log_term + mean_term


def population_fisher_rao(pop_a: np.ndarray, pop_b: np.ndarray) -> float:
    """
    Approximate Fisher-Rao distance between two populations treated as
    Gaussian distributions.
    """
    mu_a = np.mean(pop_a, axis=0)
    mu_b = np.mean(pop_b, axis=0)
    # Scalar sigma from mean trace of covariance
    sig_a = float(np.sqrt(np.mean(np.var(pop_a, axis=0))))
    sig_b = float(np.sqrt(np.mean(np.var(pop_b, axis=0))))
    return gaussian_fisher_rao(mu_a, max(sig_a, 1e-6), mu_b, max(sig_b, 1e-6))


# ── Jensen-Shannon distance (square root of JSD — a true metric) ─────────────

def js_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Jensen-Shannon distance = sqrt(JSD).
    This IS a metric (satisfies triangle inequality), unlike KL.
    Bounded in [0, 1] when using log base 2.
    """
    p = np.abs(p) + eps;  p /= p.sum()
    q = np.abs(q) + eps;  q /= q.sum()
    m = 0.5 * (p + q)
    jsd = 0.5 * float(np.sum(rel_entr(p, m))) + 0.5 * float(np.sum(rel_entr(q, m)))
    return float(math.sqrt(max(jsd, 0.0)))


# ── Natural gradient ──────────────────────────────────────────────────────────

def natural_gradient(gradient: np.ndarray, population: np.ndarray,
                      damping: float = 1e-4) -> np.ndarray:
    """
    Compute the natural gradient by pre-conditioning with the empirical
    Fisher information matrix.

    Natural gradient = F^{-1} * gradient

    where F = Cov(population) is the empirical Fisher approximation.
    This is what CMA-ES implicitly computes.

    The natural gradient follows the geodesic in distribution space —
    it's the steepest descent direction that respects the Riemannian
    geometry of the population distribution.
    """
    cov = np.cov(population.T) + damping * np.eye(population.shape[1])
    try:
        return np.linalg.solve(cov, gradient)
    except np.linalg.LinAlgError:
        return gradient


# ── Hypersphere (for normalised populations) ──────────────────────────────────

def hypersphere_log_map(x: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    Logarithmic map on the unit hypersphere S^{n-1}.
    Maps point x back to the tangent space at base.

    This is the 'inverse exponential map' — given where you are (x)
    and where you started (base), it tells you the tangent vector
    you were travelling along.

    For our relay chain: if adjacent populations are unit-normalised,
    this gives the 'velocity' of the morphing path at each step.
    """
    x_n = x / (np.linalg.norm(x) + 1e-10)
    b_n = base / (np.linalg.norm(base) + 1e-10)
    dot = float(np.clip(np.dot(x_n, b_n), -1.0, 1.0))
    angle = math.acos(dot)
    if angle < 1e-10:
        return np.zeros_like(x)
    v = x_n - dot * b_n
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10:
        return np.zeros_like(x)
    return angle * v / v_norm


def hypersphere_geodesic_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Geodesic (great circle) distance between two points on the unit hypersphere.
    """
    x_n = x / (np.linalg.norm(x) + 1e-10)
    y_n = y / (np.linalg.norm(y) + 1e-10)
    dot = float(np.clip(np.dot(x_n, y_n), -1.0, 1.0))
    return float(math.acos(dot))


# ── Information geometry path quality ────────────────────────────────────────

def info_geo_path_report(populations: list, signals: list = None,
                          verbose: bool = True) -> dict:
    """
    Compute information geometry diagnostics for a relay chain path.

    For each adjacent pair of populations:
      - Fisher-Rao distance (effort in distribution space)
      - Hypersphere geodesic distance (effort on the unit sphere)
      - JS distance (symmetric information distance)

    A geodesic path in information space has equal Fisher-Rao distances
    at every step — analogous to equal Wasserstein distances for OT geodesics.
    """
    n = len(populations)
    fisher_dists = []
    geo_dists    = []
    js_dists     = []

    for k in range(n - 1):
        pa = populations[k]
        pb = populations[k + 1]

        if pa.ndim > 1:
            fr = population_fisher_rao(pa, pb)
        else:
            fr = gaussian_fisher_rao(pa, 1.0, pb, 1.0)
        fisher_dists.append(fr)

        # Geodesic distance between centroids on hypersphere
        ca = np.mean(pa, axis=0) if pa.ndim > 1 else pa
        cb = np.mean(pb, axis=0) if pb.ndim > 1 else pb
        geo_dists.append(hypersphere_geodesic_distance(ca, cb))

    if signals is not None:
        for k in range(len(signals) - 1):
            js_dists.append(js_distance(signals[k], signals[(k + 1) % len(signals)]))

    report = {
        "fisher_rao_per_step":  fisher_dists,
        "fisher_rao_variance":  float(np.var(fisher_dists)) if fisher_dists else 0.,
        "geo_dist_per_step":    geo_dists,
        "geo_dist_variance":    float(np.var(geo_dists)) if geo_dists else 0.,
        "js_dist_per_step":     js_dists,
        "js_dist_variance":     float(np.var(js_dists)) if js_dists else 0.,
        "is_info_geodesic":     (float(np.var(fisher_dists)) < 0.01
                                 if fisher_dists else False),
    }

    if verbose:
        print("\n  INFORMATION GEOMETRY PATH REPORT")
        print(f"  Fisher-Rao per step: {[round(x,4) for x in fisher_dists]}")
        print(f"  Fisher-Rao variance: {report['fisher_rao_variance']:.5f}  "
              f"(0 = info-geometric geodesic)")
        print(f"  Geodesic dist:       {[round(x,4) for x in geo_dists]}")
        print(f"  Geodesic variance:   {report['geo_dist_variance']:.5f}")
        if js_dists:
            print(f"  JS dist per step:    {[round(x,4) for x in js_dists]}")
        print(f"  Is info-geodesic:    {report['is_info_geodesic']}")

    return report
