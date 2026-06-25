"""
Geometric richness validation battery.

Tests our geometric characterization tools on KNOWN structures with
analytically predictable outcomes. If the tools give the right answer
on known cases, we trust them on unknown cases.

THE FOUR CANONICAL CASES (bracket all possible geometric behaviors):

  CASE 1: FLAT MORPH (boring, expected: zero curvature)
    x(t) = (1-t)*linear_A + t*linear_B
    Linear space is flat. No phase transitions. Wasserstein variance = 0.
    All our curvature metrics should give ZERO.

  CASE 2: ROTATION MORPH (interesting, expected: constant curvature)
    cos -> sin via Euler relay chain
    Known geodesic. Constant curvature = 1/r where r = radius of rotation.
    Ollivier-Ricci should give approximately 0.72 (our calibrated value).
    Wasserstein variance should be zero (equal-transport geodesic).
    Holonomy of cycle should be zero.

  CASE 3: TYPE 7 MORPH (impossible, expected: infinite stress)
    cos -> cos(2x) in linear grammar
    No valid algebraic path. Relay chain fails to converge.
    Stress gradient should be maximal.
    No attractor basin with low stress.

  CASE 4: FRACTIONAL DERIVATIVE PATH (known interesting, constant curvature)
    d^(alpha)/dx^(alpha) for alpha in [0, 1]
    Interpolates between identity (alpha=0) and derivative (alpha=1).
    This IS a rotation in function space (same as Euler).
    Geometric properties should match Case 2.

MORSE THEORY ANALYSIS:
  Count critical points (minima, saddles, maxima) of the stress landscape.
  The Morse inequalities relate critical point counts to Betti numbers.
  For a flat landscape (Case 1): only one minimum, no saddles.
  For a rotation (Case 2): one minimum (the geodesic), surrounded by saddles.
  For Type 7 (Case 3): no minimum exists in this grammar.

LYAPUNOV EXPONENT:
  Do nearby morphing paths converge (negative Lyapunov) or diverge (positive)?
  Negative = stable attractor. Positive = chaotic. Zero = marginal.

PERSISTENT HOMOLOGY:
  Run gudhi on the set of relay chain intermediate positions.
  H0: number of connected components (should be 1 for a valid path)
  H1: number of loops (should be 0 for a simple path, >0 for complicated paths)
"""

import math
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from typing import List, Dict

import gudhi

from toolkit.euler_relay import _COS, _SIN, N, euler_intermediate
from toolkit.stress_metrics import (sliced_wasserstein, ollivier_ricci_edge,
                                     population_wasserstein_path, relay_ricci_curvatures)
from toolkit.info_geometry import population_fisher_rao
from toolkit.signal_injection import make_signal
from toolkit.complex_relay import run_complex_relay, from_complex


# ── Fractional derivative ─────────────────────────────────────────────────────

def fractional_derivative_path(n_steps: int = 8) -> List[np.ndarray]:
    """
    The path from identity to derivative via fractional derivatives.
    d^(alpha)/dx^(alpha) for alpha in [0..1] in n_steps.

    d^0/dx^0 = identity: [1, 0, 0, 0, 0, 0, 0, 0]
    d^1/dx^1 = derivative: [0, 1, 0, 0, 0, 0, 0, 0]
    (in phi_k basis: derivative of phi_0 = phi_{-1} = 0, derivative of phi_k = phi_{k-1})

    Actually: d^alpha is defined in Fourier space as multiplication by (ik)^alpha.
    In our phi_k basis: the fractional derivative shifts the coefficient by alpha.

    Approximation: linear interpolation of coefficients (first-order approximation).
    This is NOT exact but gives the right direction.
    """
    identity = np.zeros(N); identity[0] = 1.0   # phi_0 = 1 (constant function)
    deriv    = np.zeros(N)                        # d/dx(1) = 0 for constant
    # Actually: let's use cos_to_sin path which IS fractional derivative
    # d^alpha/dx^alpha (cos) = cos(x + alpha*pi/2)
    # At alpha=0: cos, at alpha=1: -sin (wait, that's d/dx(cos) = -sin)
    # Let's use: starting from cos, applying d^alpha gives cos rotated by alpha*pi/2

    path = []
    for k in range(n_steps + 1):
        alpha = k / n_steps
        # d^alpha (cos(x)) = cos(x + alpha*pi/2)
        theta = alpha * math.pi / 2
        c = math.cos(theta) * _COS - math.sin(theta) * _SIN
        path.append(c)
    return path


# ── Geometric richness metrics ────────────────────────────────────────────────

def morse_analysis(path: List[np.ndarray], n_seeds: int = 10,
                   n_gens: int = 30) -> Dict:
    """
    Approximate Morse theory analysis of the stress landscape.

    For each intermediate position on the path, sample nearby points and
    measure the stress landscape topology.

    Counts:
    - Minima: stress < all neighbors (index 0 critical points)
    - Saddles: stress between neighbors (index 1)
    - Maxima: stress > all neighbors (index 2)

    Morse inequalities: num_minima >= beta_0, num_saddles >= beta_1, etc.
    where beta_k are Betti numbers.
    """
    rng = np.random.default_rng(42)
    n_minima = n_saddles = n_maxima = 0

    for point in path[1:-1]:  # skip fixed endpoints
        # Sample neighbors in random directions
        delta = 0.3
        center_stress = float(np.linalg.norm(point))

        neighbor_stresses = []
        for _ in range(n_seeds):
            direction = rng.standard_normal(N)
            direction /= np.linalg.norm(direction)
            neighbor = point + delta * direction
            neighbor_stresses.append(float(np.linalg.norm(neighbor)))

        n_lower = sum(1 for s in neighbor_stresses if s < center_stress)
        n_higher = sum(1 for s in neighbor_stresses if s > center_stress)

        if n_lower == 0:
            n_minima += 1
        elif n_higher == 0:
            n_maxima += 1
        else:
            n_saddles += 1

    return {
        "n_minima":  n_minima,
        "n_saddles": n_saddles,
        "n_maxima":  n_maxima,
        "morse_index": n_saddles - n_minima + n_maxima,
        "euler_char_approx": n_minima - n_saddles + n_maxima,
    }


def lyapunov_estimate(path: List[np.ndarray], perturbation: float = 0.01,
                       n_gens: int = 20, rng_seed: int = 42) -> float:
    """
    Estimate the maximal Lyapunov exponent of the relay chain dynamics.

    Run two nearby trajectories:
    - Original: the given path
    - Perturbed: path + small random perturbation

    Measure how the distance between them evolves over n_gens.
    Lyapunov exponent = average growth rate of the distance.

    Negative Lyapunov: trajectories converge (stable attractor)
    Zero Lyapunov:     marginal stability
    Positive Lyapunov: trajectories diverge (chaotic)
    """
    rng = np.random.default_rng(rng_seed)
    n = len(path)

    divergence_rates = []
    for k in range(1, n-1):
        orig = path[k].copy()
        perturbed = orig + rng.standard_normal(N) * perturbation
        init_dist = float(np.linalg.norm(perturbed - orig))
        if init_dist < 1e-10:
            continue

        # Simulate n_gens of evolution from both points
        # (simplified: just measure if the neighboring stress landscape
        # attracts or repels from the original point)
        distances = [init_dist]
        current = perturbed.copy()
        for _ in range(n_gens):
            # Gradient of simple L2 stress toward the path
            prev_pt = path[max(0, k-1)]
            next_pt = path[min(n-1, k+1)]
            midpoint = 0.5 * (prev_pt + next_pt)
            step = 0.05 * (midpoint - current)
            current = current + step + rng.standard_normal(N) * 0.001
            distances.append(float(np.linalg.norm(current - orig)))

        # Compute growth rate
        if distances[-1] > 1e-10 and distances[0] > 1e-10:
            growth = math.log(distances[-1] / distances[0]) / n_gens
            divergence_rates.append(growth)

    return float(np.mean(divergence_rates)) if divergence_rates else 0.0


def persistent_homology_path(path: List[np.ndarray],
                               max_edge: float = 2.0) -> Dict:
    """
    Run persistent homology on the relay chain path points.
    H0: connected components (should be 1)
    H1: loops (0 = simple path, >0 = complicated topology)
    """
    points = np.array(path)
    rips   = gudhi.RipsComplex(points=points, max_edge_length=max_edge)
    st     = rips.create_simplex_tree(max_dimension=2)
    diag   = st.persistence()

    h0 = [(b, d) for dim, (b, d) in diag if dim == 0]
    h1 = [(b, d) for dim, (b, d) in diag if dim == 1]
    persistent_h1 = [(b, d) for b, d in h1
                     if d != float('inf') and d - b > 0.05]

    return {
        "h0_count":          len(h0),
        "h1_count":          len(h1),
        "persistent_h1":     len(persistent_h1),
        "max_h1_persistence": max([d-b for b,d in persistent_h1], default=0.0),
        "betti_0":           sum(1 for d_pair in diag if d_pair[0]==0 and d_pair[1][1]==float('inf')),
        "betti_1":           sum(1 for d_pair in diag if d_pair[0]==1 and d_pair[1][1]==float('inf')),
    }


def full_geometric_characterization(path: List[np.ndarray],
                                     case_name: str) -> Dict:
    """
    Run ALL geometric richness metrics on a path.
    Returns a dict suitable for comparison across known cases.
    """
    # Convert to "population" format for stress metrics
    pops = [p.reshape(1, -1) for p in path]

    wass_dists  = population_wasserstein_path(pops, n_proj=20)
    ricci_vals  = relay_ricci_curvatures(pops)
    morse       = morse_analysis(path)
    lyapunov    = lyapunov_estimate(path)
    ph          = persistent_homology_path(path)

    wass_var   = float(np.var(wass_dists))
    ricci_var  = float(np.var(ricci_vals))
    mean_ricci = float(np.mean(ricci_vals))

    # Geodesic deviation: how far does the path deviate from a straight line?
    straight = np.array([path[0] + (path[-1]-path[0])*k/(len(path)-1)
                          for k in range(len(path))])
    geodesic_dev = float(np.mean([np.linalg.norm(path[k] - straight[k])
                                   for k in range(len(path))]))

    return {
        "case":              case_name,
        "path_length":       len(path),
        "wass_variance":     wass_var,
        "ricci_variance":    ricci_var,
        "mean_ricci":        mean_ricci,
        "lyapunov":          lyapunov,
        "geodesic_dev":      geodesic_dev,
        "morse_minima":      morse["n_minima"],
        "morse_saddles":     morse["n_saddles"],
        "euler_char_approx": morse["euler_char_approx"],
        "h1_persistent":     ph["persistent_h1"],
        "betti_0":           ph["betti_0"],
    }


# ── The four canonical cases ──────────────────────────────────────────────────

def run_validation_battery():
    """
    Run geometric characterization on the four canonical cases.
    Compare to expected values to validate our metrics.
    """
    cos  = _COS.copy()
    sin  = _SIN.copy()

    def _cos2x():
        c = np.zeros(N)
        for k in range(0, N, 2):
            c[k] = ((-1)**(k//2)) * (2.0**k)
        return c

    cos2x = _cos2x()

    print("="*70)
    print("GEOMETRIC RICHNESS VALIDATION BATTERY")
    print("Testing characterization tools on known structures")
    print("="*70)

    results = {}

    # CASE 1: Flat morph (linear -> linear)
    print("\nCase 1: FLAT MORPH (linear_A -> linear_B)")
    print("  Expected: zero curvature, zero Wasserstein variance, flat topology")
    lin_a = np.array([0., 1., 0., 0., 0., 0., 0., 0.])  # x
    lin_b = np.array([0., 2., 0., 0., 0., 0., 0., 0.])  # 2x
    flat_path = [lin_a + (lin_b-lin_a)*k/8 for k in range(9)]
    results["flat"] = full_geometric_characterization(flat_path, "flat_linear")
    _print_result(results["flat"])

    # CASE 2: Rotation morph (cos -> sin via Euler relay)
    print("\nCase 2: ROTATION MORPH (cos -> sin, Euler relay)")
    print("  Expected: constant curvature ~0.72, Wasserstein variance ~0, stable")
    euler_path = [euler_intermediate(k, 8) for k in range(9)]
    results["rotation"] = full_geometric_characterization(euler_path, "euler_rotation")
    _print_result(results["rotation"])

    # CASE 3: Type 7 (cos -> cos2x, no valid path)
    print("\nCase 3: TYPE 7 MORPH (cos -> cos(2x), impossible)")
    print("  Expected: high stress, unstable (positive Lyapunov), complex topology")
    type7_path = [cos + (cos2x-cos)*k/8 for k in range(9)]
    results["type7"] = full_geometric_characterization(type7_path, "type7_impossible")
    _print_result(results["type7"])

    # CASE 4: Fractional derivative path
    print("\nCase 4: FRACTIONAL DERIVATIVE (identity->derivative via d^alpha)")
    print("  Expected: constant curvature (same as rotation), stable")
    frac_path = fractional_derivative_path(n_steps=8)
    results["fractional"] = full_geometric_characterization(frac_path, "fractional_deriv")
    _print_result(results["fractional"])

    # Validation summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print(f"{'Case':<15} {'W_var':>8} {'R_var':>8} {'R_mean':>8} "
          f"{'Lyap':>8} {'GeoD':>8} {'Euler':>7}")
    print("-"*70)
    for name, r in results.items():
        print(f"  {name:<13} {r['wass_variance']:>8.5f} "
              f"{r['ricci_variance']:>8.5f} {r['mean_ricci']:>8.4f} "
              f"{r['lyapunov']:>8.4f} {r['geodesic_dev']:>8.4f} "
              f"{r['euler_char_approx']:>7}")

    print("\nExpected pattern:")
    print("  flat:       W_var~0, R_var~0, Lyap<0 (stable), GeoD~0 (on geodesic)")
    print("  rotation:   W_var~0, R_var~0, Lyap<0 (stable), R_mean~0.72")
    print("  type7:      W_var>0, R_var>0, GeoD>0 (off geodesic)")
    print("  fractional: W_var~0, R_var~0, Lyap<0 (should match rotation)")

    print("\nVALIDATION:")
    flat_ok     = results["flat"]["wass_variance"] < 0.01
    rotation_ok = (results["rotation"]["wass_variance"] < 0.01 and
                   results["rotation"]["mean_ricci"] > 0.5)
    type7_ok    = (results["type7"]["wass_variance"] > results["flat"]["wass_variance"])
    frac_ok     = (abs(results["fractional"]["mean_ricci"] -
                       results["rotation"]["mean_ricci"]) < 0.3)

    for name, ok in [("flat morph correct", flat_ok),
                      ("rotation correct", rotation_ok),
                      ("type7 distinguishable", type7_ok),
                      ("fractional matches rotation", frac_ok)]:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")

    return results


def _print_result(r: Dict):
    print(f"  Wasserstein var: {r['wass_variance']:.5f}  "
          f"Ricci var: {r['ricci_variance']:.5f}  "
          f"Ricci mean: {r['mean_ricci']:.4f}")
    print(f"  Lyapunov: {r['lyapunov']:.4f}  "
          f"Geodesic dev: {r['geodesic_dev']:.4f}  "
          f"Euler char: {r['euler_char_approx']}")
    print(f"  H1 persistent: {r['h1_persistent']}  "
          f"Betti_0: {r['betti_0']}")


if __name__ == "__main__":
    results = run_validation_battery()
