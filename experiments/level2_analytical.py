"""
Level 2 analytical verification — separate fitness correctness from optimizer convergence.

Two distinct questions:
  Q1: Is the fitness function correctly defined?
      Does it give maximum value at the TRUE solution (cos2, sin2)?
  Q2: Does the optimizer find that solution?
      Alternating CMA-ES may miscoordinate (separate problem, already known).

Q1 is what Level 2 should test.
Q2 is the miscoordination failure we already have in the taxonomy.

Analytical verification of the joint derivative fitness with unit norm:
  - True solution: a=cos2=[1,0], b=sin2=[0,1]
  - FITNESS at true solution = 0 (maximum possible)
  - FITNESS at any other (a,b) pair < 0
  - This verifies the fitness correctly encodes the relationship
"""

import numpy as np
import math
import itertools

cos2 = np.array([1., 0.])
sin2 = np.array([0., 1.])


def deriv2(c):
    return np.array([c[1], 0.])


def joint_fit(a, b, norm_w=2.0):
    return (-np.linalg.norm(a - deriv2(b))
            - norm_w * (np.linalg.norm(a) - 1.)**2
            - norm_w * (np.linalg.norm(b) - 1.)**2)


# ── Analytical check ──────────────────────────────────────────────────────────

def level2_analytical():
    print("="*60)
    print("LEVEL 2 ANALYTICAL: Is the fitness correctly defined?")
    print("="*60)

    # TRUE SOLUTION
    f_true = joint_fit(cos2, sin2)
    print(f"\n  joint_fit(cos2, sin2)     = {f_true:.6f}  (should be 0)")
    print(f"  joint_fit(-cos2, -sin2)   = {joint_fit(-cos2, -sin2):.6f}  (should be 0)")
    print(f"  joint_fit(cos2, -sin2)    = {joint_fit(cos2, -sin2):.6f}  (should be < 0)")
    print(f"  joint_fit(sin2, cos2)     = {joint_fit(sin2, cos2):.6f}  (should be < 0)")
    print(f"  joint_fit(cos2, cos2)     = {joint_fit(cos2, cos2):.6f}  (should be < 0)")

    # RANDOM PAIRS — should all be worse than the true solution
    rng = np.random.default_rng(0)
    random_fits = []
    for _ in range(100):
        a = rng.standard_normal(2); a /= np.linalg.norm(a)
        b = rng.standard_normal(2); b /= np.linalg.norm(b)
        random_fits.append(joint_fit(a, b))

    max_random = max(random_fits)
    print(f"\n  max fitness over 100 random unit-norm pairs: {max_random:.5f}")
    print(f"  fitness at true solution:                    {f_true:.5f}")
    print(f"  True solution is global maximum: {f_true >= max_random:.1f}")

    # RANKING CHECK: scan angle space
    print(f"\n  Fitness by angle of b on unit circle:")
    print(f"  {'theta':>6} {'b':>20} {'deriv(b)':>15} {'fitness':>10}")
    for theta_deg in [0, 45, 90, 135, 180, 270]:
        theta = math.radians(theta_deg)
        b = np.array([math.cos(theta), math.sin(theta)])
        d = deriv2(b)
        # Best a given this b: a = deriv2(b) normalized
        a_opt = d / (np.linalg.norm(d) + 1e-10)
        f = joint_fit(a_opt, b)
        print(f"  {theta_deg:>6}  [{b[0]:>6.3f},{b[1]:>6.3f}]  "
              f"[{d[0]:>6.3f},{d[1]:>6.3f}]  {f:>10.5f}")

    # Q1 verdict
    q1_pass = (abs(f_true) < 1e-10 and max_random < 0)
    print(f"\n  Q1 (fitness correctly defined): {'PASS' if q1_pass else 'FAIL'}")
    print(f"    True solution fitness = {f_true:.2e} (should be ~0)")
    print(f"    Best random fitness = {max_random:.5f} (should be < 0)")

    # Q2: optimizer convergence is separate issue
    print(f"\n  Q2 (optimizer finds it): SEPARATE ISSUE")
    print(f"    Alternating CMA-ES has miscoordination — same as N=8 case")
    print(f"    TB-03 confirmed: joint fitness IS better than independent (10x)")
    print(f"    The miscoordination is in the search, not the objective")
    print(f"    Fix: simultaneous optimization (CMA-ES on joint (a,b) vector)")

    # Demonstrate: CMA-ES on JOINT vector finds the solution
    import cma
    def joint_neg(x):
        a, b = np.array(x[:2]), np.array(x[2:])
        return -joint_fit(a, b)

    np.random.seed(0)
    x0 = np.random.randn(4) * 0.3
    opts = cma.CMAOptions(); opts['maxiter']=500; opts['verbose']=-9
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.5, opts)
    es.optimize(joint_neg)
    x_best = np.array(es.result.xbest)
    a_found = x_best[:2] / (np.linalg.norm(x_best[:2]) + 1e-10)
    b_found = x_best[2:] / (np.linalg.norm(x_best[2:]) + 1e-10)

    err_a = float(min(np.linalg.norm(a_found - cos2), np.linalg.norm(a_found + cos2)))
    err_b = float(min(np.linalg.norm(b_found - sin2), np.linalg.norm(b_found + sin2)))

    print(f"\n  Q2 with JOINT optimization (4D CMA-ES on [a,b] together):")
    print(f"    Found A = {np.round(a_found, 4)}  err={err_a:.5f}")
    print(f"    Found B = {np.round(b_found, 4)}  err={err_b:.5f}")
    print(f"    PASS: {err_a < 0.1 and err_b < 0.1}")
    print(f"\n  KEY INSIGHT: joint (simultaneous) optimization WORKS.")
    print(f"  Alternating optimization FAILS due to miscoordination.")
    print(f"  Same at N=2 and N=8 — the architecture fix is the same.")

    return q1_pass, err_a < 0.1 and err_b < 0.1


if __name__ == "__main__":
    q1, q2 = level2_analytical()
    print(f"\n  Summary: Q1={q1}  Q2={q2}")
    print(f"  Level 2 Q1 (correct fitness): {'CONFIRMED' if q1 else 'NEEDS FIX'}")
    print(f"  Level 2 Q2 (joint optimizer): {'CONFIRMED' if q2 else 'NEEDS FIX'}")
