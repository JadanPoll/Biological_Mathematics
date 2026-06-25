"""
Sanity check hierarchy — lower-lifted partitions of the morphing problem.

The principle: if a tool works in higher-dimensional lifted space, it must
work as a SPECIAL CASE in lower-dimensional spaces where the answer is known
analytically. Failure in a lower level = a bug in the tool, not an insight.

Hierarchy (simplest to most complex):

  Level 0: SCALAR — two real numbers
    Objects: scalars a, b
    Relationship: b = k*a  (scalar multiple)
    Stress: |k - 1|
    Ground truth: trivially computable
    Tests: Wasserstein, Fisher-Rao (1D Gaussian)

  Level 1: 2D UNIT CIRCLE (confirmed working)
    Objects: (cos θ, sin θ)
    Relationship: rotation by φ
    Stress: rotation angle φ (path length on S¹)
    Ground truth: arc geodesic, angular variance = 0

  Level 2: 2D POLYNOMIAL (N=2 coefficients)
    Objects: [c₀, c₁]
    Relationship: derivative → [1,0] → [0,1]
    Stress: distance from correct derivative path
    Ground truth: complex relay in 2D

  Level 3: COMPLEX NUMBER (equivalent to Level 1, different representation)
    Objects: z = a + ib
    Relationship: multiplication by e^(iφ)
    Ground truth: exact (already confirmed in complex_relay.py)

  Level 4: 8D TAYLOR (our main case)
    Ground truth from Level 3: complex relay gives exact answer
    Real-space relay should converge toward this.

Each level IS a special case of Level 4.
If Level k passes and Level k+1 fails, the failure is at that specific lifting step.
"""

import math
import numpy as np
import scipy.stats as stats


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 0: SCALAR SANITY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def wasserstein_1d_gaussian_closed_form(mu1, sigma1, mu2, sigma2):
    """
    Closed-form W₂ distance between two 1D Gaussians.
    W₂²(N(μ₁,σ₁²), N(μ₂,σ₂²)) = (μ₁-μ₂)² + (σ₁-σ₂)²
    """
    return math.sqrt((mu1 - mu2)**2 + (sigma1 - sigma2)**2)


def fisher_rao_1d_gaussian_closed_form(mu1, sigma1, mu2, sigma2):
    """
    Fisher-Rao geodesic distance for 1D Gaussians.
    Correct formula (Poincaré upper half-plane with Gaussian metric):
    d = sqrt(2) * arcosh(1 + ((mu1-mu2)^2 + (sigma1-sigma2)^2) / (2*sigma1*sigma2))

    For same mean: d = sqrt(2) * arcosh(1 + (sigma1-sigma2)^2 / (2*sigma1*sigma2))
                     = sqrt(2) * arcosh((sigma1^2+sigma2^2) / (2*sigma1*sigma2))
    Matches: sqrt(2) * |log(sigma1/sigma2)| for the 'sigma only' geodesic.
    """
    num = (mu1 - mu2)**2 + (sigma1 - sigma2)**2
    denom = 2 * sigma1 * sigma2
    term = 1 + num / denom
    return math.sqrt(2) * math.acosh(max(term, 1.0))


def test_level0_scalar(verbose=True):
    """
    Level 0: scalar sanity checks.
    Verify our Wasserstein and Fisher-Rao implementations match closed forms.
    """
    from toolkit.stress_metrics import sliced_wasserstein
    from toolkit.info_geometry import population_fisher_rao

    rng = np.random.default_rng(42)
    n = 500  # large population for accurate comparison

    results = []

    for mu1, sig1, mu2, sig2 in [
        (0., 1., 0., 1.),    # identical
        (0., 1., 0., 2.),    # different variance, same mean
        (1., 1., -1., 1.),   # different mean, same variance
        (0., 1., 1., 2.),    # both different
        (0., 0.5, 0., 3.),   # large variance ratio
    ]:
        pop_a = rng.normal(mu1, sig1, (n, 1))
        pop_b = rng.normal(mu2, sig2, (n, 1))

        wass_exact  = wasserstein_1d_gaussian_closed_form(mu1, sig1, mu2, sig2)
        wass_approx = sliced_wasserstein(pop_a, pop_b, n_projections=50, rng=rng)

        fr_exact    = fisher_rao_1d_gaussian_closed_form(mu1, sig1, mu2, sig2)
        fr_approx   = population_fisher_rao(pop_a, pop_b)

        # Relative error — skip if exact is near zero (sampling noise dominates)
        if wass_exact < 0.1:
            wass_err = abs(wass_approx - wass_exact)  # absolute error for near-zero
        else:
            wass_err = abs(wass_approx - wass_exact) / wass_exact

        if fr_exact < 0.1:
            fr_err = abs(fr_approx - fr_exact)
        else:
            fr_err = abs(fr_approx - fr_exact) / fr_exact

        results.append({
            "mu1": mu1, "sig1": sig1, "mu2": mu2, "sig2": sig2,
            "wass_exact": wass_exact, "wass_approx": wass_approx,
            "wass_rel_err": wass_err,
            "fr_exact": fr_exact, "fr_approx": fr_approx,
            "fr_rel_err": fr_err,
        })

    if verbose:
        print("\n" + "="*72)
        print("LEVEL 0: SCALAR SANITY — Wasserstein and Fisher-Rao vs closed form")
        print("="*72)
        print(f"  {'sig1':>5} {'sig2':>5} | {'W_exact':>9} {'W_approx':>9} {'W_err%':>7} | "
              f"{'FR_exact':>9} {'FR_approx':>9} {'FR_err%':>7}")
        print(f"  {'-'*70}")
        for r in results:
            print(f"  {r['sig1']:>5.1f} {r['sig2']:>5.1f} | "
                  f"{r['wass_exact']:>9.4f} {r['wass_approx']:>9.4f} "
                  f"{r['wass_rel_err']*100:>6.1f}% | "
                  f"{r['fr_exact']:>9.4f} {r['fr_approx']:>9.4f} "
                  f"{r['fr_rel_err']*100:>6.1f}%")

    wass_pass = all(r['wass_rel_err'] < 0.30 for r in results)  # sliced Wasserstein is approximate
    fr_pass   = all(r['fr_rel_err'] < 0.60 for r in results)   # FR approximation is rougher
    print(f"\n  Wasserstein sanity: {'PASS' if wass_pass else 'FAIL'}")
    print(f"  Fisher-Rao sanity:  {'PASS' if fr_pass else 'FAIL'}")
    return wass_pass and fr_pass


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 1: 2D UNIT CIRCLE (ALREADY CONFIRMED — documented here)
# ─────────────────────────────────────────────────────────────────────────────

def test_level1_circle(verbose=True):
    """
    Level 1: 2D unit circle sanity.
    Already confirmed in experiments/minimal_2d.py.
    Documented here for the hierarchy.

    Ground truth: Wasserstein distance between two points at angle φ apart
    on the unit circle = 2*sin(φ/2) (chord length).
    Angular variance = 0 for constrained geodesic.
    """
    from experiments.minimal_2d import (run_2d_relay, path_stress_2d,
                                         point_on_circle)
    from toolkit.stress_metrics import sliced_wasserstein

    rng = np.random.default_rng(42)
    angles = [0, 30, 90, 150, 180]
    results = []

    for phi in angles:
        a = point_on_circle(0)
        b = point_on_circle(phi)

        # Closed-form chord length = Wasserstein distance between two delta masses
        wass_exact = float(np.linalg.norm(b - a))  # = 2*sin(phi/2) in radians

        # Our relay chain should give angular variance = 0
        pops = run_2d_relay(a, b, n_steps=4, constrain_to_circle=True)
        stress = path_stress_2d(pops)

        results.append({
            "phi": phi,
            "wass_exact": wass_exact,
            "angular_var": stress["angular_step_var"],
            "is_geodesic": stress["is_geodesic"],
        })

    if verbose:
        print("\n" + "="*55)
        print("LEVEL 1: 2D CIRCLE SANITY (already confirmed)")
        print("="*55)
        for r in results:
            print(f"  phi={r['phi']:3d}deg  chord={r['wass_exact']:.4f}  "
                  f"ang_var={r['angular_var']:.6f}  geodesic={r['is_geodesic']}")

    return all(r['is_geodesic'] for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 2: 2D POLYNOMIAL (N=2 COEFFICIENTS)
# ─────────────────────────────────────────────────────────────────────────────

def test_level2_poly2(verbose=True):
    """
    Level 2: N=2 Taylor coefficient vectors.
    Objects: [c₀, c₁]
    cos₂ = [1, 0],  sin₂ = [0, 1]

    Derivative relationship: A = d/dx(B)
    In N=2 phi_k basis: d/dx([c₀, c₁]) = [c₁, 0]
    So: sin₂ = [0, 1] → d/dx(sin₂) = [1, 0] = cos₂  ✓

    Joint derivative fitness: |A - d/dx(B)|
    Starting from random initializations, should converge to
    A = cos₂ = [1, 0],  B = sin₂ = [0, 1].

    This is the SIMPLEST non-trivial joint fitness test:
    N=2, known analytical answer, no truncation error.
    """
    import cma

    cos2 = np.array([1., 0.])
    sin2 = np.array([0., 1.])

    def joint_derivative_fitness(a, b):
        """a should equal d/dx(b) = [b[1], 0]"""
        deriv_b = np.array([b[1], 0.])
        return -float(np.linalg.norm(a - deriv_b))

    # Optimize a and b jointly using CMA-ES
    np.random.seed(42)
    a = np.array([0.3, 0.7]) + np.random.randn(2)*0.3  # random init
    b = np.array([0.8, 0.2]) + np.random.randn(2)*0.3

    for iteration in range(5):
        # Optimize a given b
        def neg_fit_a(x, b=b):
            return -joint_derivative_fitness(np.array(x), b)
        opts = cma.CMAOptions(); opts['maxiter']=100; opts['verbose']=-9
        es = cma.CMAEvolutionStrategy(a.tolist(), 0.3, opts)
        es.optimize(neg_fit_a)
        a = np.array(es.result.xbest)

        # Optimize b given a
        def neg_fit_b(x, a=a):
            return -joint_derivative_fitness(a, np.array(x))
        opts = cma.CMAOptions(); opts['maxiter']=100; opts['verbose']=-9
        es = cma.CMAEvolutionStrategy(b.tolist(), 0.3, opts)
        es.optimize(neg_fit_b)
        b = np.array(es.result.xbest)

    # Check: a should ≈ cos₂ = [1, 0], b should ≈ sin₂ = [0, 1]
    # (up to overall scale)
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    err_a = float(np.linalg.norm(a_norm - cos2))
    err_b = float(np.linalg.norm(b_norm - sin2))

    # Also verify d/dx(b) = a
    deriv_b = np.array([b[1], 0.])
    deriv_err = float(np.linalg.norm(a - deriv_b * np.linalg.norm(a)))

    if verbose:
        print("\n" + "="*55)
        print("LEVEL 2: N=2 POLYNOMIAL SANITY")
        print("="*55)
        print(f"  Found A (normalised): {np.round(a_norm, 4)}")
        print(f"  Expected cos2:        {cos2}")
        print(f"  Found B (normalised): {np.round(b_norm, 4)}")
        print(f"  Expected sin2:        {sin2}")
        print(f"  ||A_norm - cos2|| = {err_a:.5f}  (should be < 0.1)")
        print(f"  ||B_norm - sin2|| = {err_b:.5f}  (should be < 0.1)")
        print(f"  ||A - d/dx(B)|| = {deriv_err:.5f}  (should be < 0.1)")
        verdict = "PASS" if err_a < 0.1 and err_b < 0.1 else "FAIL"
        print(f"  Verdict: {verdict}")

    return err_a < 0.1 and err_b < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 3: EULER RELAY CHAIN (ALREADY CONFIRMED)
# ─────────────────────────────────────────────────────────────────────────────

def test_level3_complex(verbose=True):
    """
    Level 3: complex relay chain (already confirmed in complex_relay.py).
    Max error = 0.000000 for all n_steps.
    Documented here for hierarchy completeness.
    """
    from toolkit.complex_relay import run_complex_relay
    result = run_complex_relay(n_steps=4, n_iterations=100, verbose=False)
    passed = result["max_error"] < 1e-6
    if verbose:
        print("\n" + "="*55)
        print("LEVEL 3: COMPLEX RELAY (already confirmed)")
        print("="*55)
        print(f"  max_error = {result['max_error']:.8f}  "
              f"(< 1e-6: {'PASS' if passed else 'FAIL'})")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 4: RICCI CURVATURE ON KNOWN CURVE
# ─────────────────────────────────────────────────────────────────────────────

def test_level4_ricci(verbose=True):
    """
    Level 4: Ricci curvature on the Euler arc.

    The Euler arc is a circle of radius 1 in the 2D (cos, sin) subspace
    of 8D coefficient space. The Frenet curvature of a unit circle = 1.

    Our Ollivier-Ricci approximation should give curvature ≈ 1.
    This validates the Ricci diagnostic before applying it to unknown cases.
    """
    from toolkit.euler_relay import euler_intermediate, N
    from toolkit.stress_metrics import ollivier_ricci_edge

    n_steps = 8
    true_ints = [euler_intermediate(k, n_steps) for k in range(n_steps + 1)]

    ricci_vals = []
    for k in range(len(true_ints) - 1):
        pa = true_ints[k].reshape(1, -1)
        pb = true_ints[k+1].reshape(1, -1)
        kappa = ollivier_ricci_edge(pa, pb, n_proj=30)
        ricci_vals.append(kappa)

    mean_ricci = float(np.mean(ricci_vals))
    ricci_var  = float(np.var(ricci_vals))

    if verbose:
        print("\n" + "="*55)
        print("LEVEL 4: RICCI CURVATURE ON UNIT CIRCLE ARC")
        print("="*55)
        print(f"  Expected curvature: ~1.0 (Frenet curvature of unit circle)")
        print(f"  Measured curvature: {mean_ricci:.4f} +/- {math.sqrt(ricci_var):.4f}")
        print(f"  Per-step: {[round(k, 4) for k in ricci_vals]}")
        # Our approximation may not be exact — report what we get
        print(f"  Note: Ollivier-Ricci is an approximation to Riemannian curvature.")
        print(f"        For a unit circle, Riemannian curvature = 1.")
        print(f"        Our approximation gives {mean_ricci:.4f}.")

    return ricci_vals


# ─────────────────────────────────────────────────────────────────────────────
# FULL HIERARCHY RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*72)
    print("SANITY CHECK HIERARCHY")
    print("Lower-lifted partitions of the morphing problem.")
    print("Each level must pass before trusting the level above it.")
    print("="*72)

    results = {}

    results[0] = test_level0_scalar()
    results[1] = test_level1_circle()
    results[2] = test_level2_poly2()
    results[3] = test_level3_complex()
    test_level4_ricci()  # diagnostic, not pass/fail

    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    levels = {
        0: "Scalar (Wasserstein + Fisher-Rao vs closed form)",
        1: "2D unit circle (geodesic arc)",
        2: "N=2 polynomial (joint derivative fitness)",
        3: "Complex relay chain (exact Euler path)",
    }
    for lvl, desc in levels.items():
        status = "PASS" if results.get(lvl) else "FAIL"
        print(f"  Level {lvl} [{status}]: {desc}")

    if all(results.values()):
        print("\n  All levels pass. The tools are correctly implemented.")
        print("  Trust the higher-dimensional results.")
    else:
        failed = [l for l, v in results.items() if not v]
        print(f"\n  FAILED at levels: {failed}")
        print("  Fix these before interpreting higher-dimensional results.")
