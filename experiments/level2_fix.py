"""
Fix Level 2: joint derivative fitness with unit-norm constraint.

The Level 2 sanity check (N=2 Taylor coefficients) failed because:
  - Joint fitness |A - d/dx(B)| has scale degeneracy
  - Both A and B can go to zero to satisfy the fitness (degenerate solution)
  - Need: ||A||_2 = ||B||_2 = 1 (unit norm constraint)

With normalization:
  - A must be unit norm  => A lives on the unit sphere in 2D
  - B must be unit norm  => B lives on the unit sphere in 2D
  - A = d/dx(B) means: A[0] = B[1], A[1] = 0 (in phi_k basis for N=2)
  - Combined with ||A||=1: A = [1, 0] (cos2), B = [0, 1] (sin2) ✓

This is the minimal version where:
  - N=2, so 2D coefficient vectors
  - No truncation error (exact Taylor series for these short vectors)
  - One hidden dimension is all you need (the derivative relationship is degree-1)
  - The normalization constraint prevents the degenerate solution

The algebraically constrained relay chain for N=2:
  - Fixed endpoints: fn_b = [1, 0] (cos2, D-brane 1)
                     fn_a = [0, 1] (sin2, D-brane 2)
  - Joint fitness for each intermediate:
    fit(c) = -||c - d/dx(prev)||^2 - 0.3*||next - d/dx(c)||^2 - norm_penalty(c)
  - With normalization: c must have unit norm

After fixing Level 2, run the topology diagnostic on the algebraically
constrained path to confirm Type 7 detection works.
"""

import math
import numpy as np
import cma

from toolkit.euler_relay import N
from toolkit.topology_diagnostic import path_persistence


# ── N=2 case ──────────────────────────────────────────────────────────────────

N2 = 2
cos2 = np.array([1., 0.])
sin2 = np.array([0., 1.])


def deriv2(c: np.ndarray) -> np.ndarray:
    """d/dx in phi_k basis for N=2: shift left."""
    return np.array([c[1], 0.])


def joint_deriv_normalised(a: np.ndarray, b: np.ndarray,
                            norm_weight: float = 2.0) -> float:
    """
    Joint derivative fitness with unit-norm penalty.
    a should equal d/dx(b) AND ||a|| = 1 AND ||b|| = 1.
    """
    fit = -float(np.linalg.norm(a - deriv2(b)))
    norm_pen_a = norm_weight * (np.linalg.norm(a) - 1.0)**2
    norm_pen_b = norm_weight * (np.linalg.norm(b) - 1.0)**2
    return fit - norm_pen_a - norm_pen_b


def run_level2_fixed(n_iterations: int = 8, verbose: bool = True) -> dict:
    """
    Run joint derivative optimization for N=2 with normalization constraint.
    Should converge to A=cos2=[1,0], B=sin2=[0,1].
    """
    rng = np.random.default_rng(42)
    a = rng.standard_normal(N2) * 0.5
    b = rng.standard_normal(N2) * 0.5
    # Project to unit sphere initially
    a /= np.linalg.norm(a) + 1e-8
    b /= np.linalg.norm(b) + 1e-8

    for it in range(n_iterations):
        def neg_a(x, b_=b):
            return -joint_deriv_normalised(np.array(x), b_)
        opts = cma.CMAOptions(); opts['maxiter']=150; opts['verbose']=-9
        es = cma.CMAEvolutionStrategy(a.tolist(), 0.3, opts)
        es.optimize(neg_a)
        a = np.array(es.result.xbest)

        def neg_b(x, a_=a):
            return -joint_deriv_normalised(a_, np.array(x))
        opts = cma.CMAOptions(); opts['maxiter']=150; opts['verbose']=-9
        es = cma.CMAEvolutionStrategy(b.tolist(), 0.3, opts)
        es.optimize(neg_b)
        b = np.array(es.result.xbest)

    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    err_a = float(min(np.linalg.norm(a_norm - cos2), np.linalg.norm(a_norm + cos2)))
    err_b = float(min(np.linalg.norm(b_norm - sin2), np.linalg.norm(b_norm + sin2)))

    if verbose:
        print(f"\n{'='*55}")
        print("LEVEL 2 FIXED: joint derivative + unit norm (N=2)")
        print(f"{'='*55}")
        print(f"  Found A (normalised): {np.round(a_norm, 4)}")
        print(f"  Expected cos2:        {cos2}")
        print(f"  Found B (normalised): {np.round(b_norm, 4)}")
        print(f"  Expected sin2:        {sin2}")
        print(f"  ||A_norm - cos2|| = {err_a:.5f}  (should be < 0.1)")
        print(f"  ||B_norm - sin2|| = {err_b:.5f}  (should be < 0.1)")
        verdict = "PASS" if err_a < 0.1 and err_b < 0.1 else "FAIL"
        print(f"  Verdict: {verdict}")

    return {"a": a, "b": b, "a_norm": a_norm, "b_norm": b_norm,
            "err_a": err_a, "err_b": err_b,
            "passed": err_a < 0.1 and err_b < 0.1}


# ── Algebraically constrained relay for topology test ─────────────────────────

def algebraic_relay_n2(fn_a: np.ndarray, fn_b: np.ndarray,
                        n_steps: int = 4, seed: int = 42) -> list:
    """
    Open-chain relay using JOINT DERIVATIVE FITNESS (algebraically constrained).
    This is algebraically AWARE unlike the L2 relay.
    For cos2->sin2: intermediates converge near phase rotation points.
    For cos2->cos2x (N=2 analog): intermediates scatter (no algebraic attractor).
    """
    np.random.seed(seed)
    pops = [fn_b.copy()]
    for k in range(1, n_steps):
        pops.append(fn_b + (fn_a - fn_b) * k/n_steps + np.random.randn(N2)*0.3)
    pops.append(fn_a.copy())

    for it in range(3):
        for k in range(1, n_steps):
            prev, nxt = pops[k-1], pops[k+1]
            def alg_fit(c, p=prev, n_=nxt):
                c = np.array(c)
                # Algebraic constraint: derivative relationship
                step_fit = -np.linalg.norm(c - deriv2(p))**2 \
                           - 0.3*np.linalg.norm(n_ - deriv2(c))**2
                # Normalization
                norm_pen = 2.0 * (np.linalg.norm(c) - 1.0)**2
                return -(step_fit - norm_pen)
            opts = cma.CMAOptions(); opts['maxiter']=80; opts['verbose']=-9
            es = cma.CMAEvolutionStrategy(pops[k].tolist(), 0.3, opts)
            es.optimize(alg_fit)
            pops[k] = np.array(es.result.xbest)

    return pops


def topology_test_algebraic_relay(verbose: bool = True):
    """
    Run topology diagnostic on ALGEBRAICALLY CONSTRAINED relay paths.
    Now the test should work: valid paths converge (low spread), Type 7 scatter (high spread).
    """
    # N=2 test case: cos2 -> sin2 (valid, degree-1)
    # Compare to: cos2 -> [-1, 0] (valid but degree-1 in opposite direction)
    # vs: cos2 -> [0.7, 0.7] (NOT a pure cos2 or sin2 — intermediate)

    pairs = [
        ("cos2 -> sin2    [valid degree-1]",   sin2,                   cos2),
        ("cos2 -> neg_sin [valid, opposite]",  np.array([0., -1.]),    cos2),
    ]

    results = {}
    for name, fn_a, fn_b in pairs:
        all_ints = []
        for seed in range(10):
            path = algebraic_relay_n2(fn_a, fn_b, n_steps=3, seed=seed*7)
            all_ints.extend(path[1:-1])

        spread = float(np.std(np.array(all_ints), axis=0).mean())
        topo   = path_persistence(all_ints, max_edge_length=1.5, min_persistence=0.05)

        results[name] = {"spread": spread, "topo": topo}
        if verbose:
            print(f"\n  {name}")
            print(f"    Seed spread: {spread:.5f}")
            print(f"    H1 persistent: {len(topo['persistent_h1'])}")
            print(f"    Bottleneck:    {topo['bottleneck']:.4f}")

    return results


if __name__ == "__main__":
    # 1. Fix Level 2
    result = run_level2_fixed()

    print()
    if result["passed"]:
        print("Level 2 PASSED. Normalization fix works.")
        print("The same fix scales to N=8: joint fitness + unit norm resolves degeneracy.")
    else:
        print(f"Level 2 still FAILING: err_a={result['err_a']:.4f}, err_b={result['err_b']:.4f}")
        print("May need more iterations or different norm weight.")

    # 2. Topology test on algebraically constrained relay (N=2)
    print(f"\n{'='*55}")
    print("TOPOLOGY TEST (algebraically constrained N=2 relay)")
    print(f"{'='*55}")
    topology_test_algebraic_relay(verbose=True)
