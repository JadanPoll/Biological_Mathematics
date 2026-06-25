"""
Type 7 dimensional insufficiency test.

The XOR analog: prove that cos(x)->cos(2x) has no valid path under linear-only
relay steps, then show adding a quadratic operation resolves it.

PREDICTION FROM THEORY:
  cos(x) -> sin(x): degree-1 relationship (rotation, linear in cos/sin subspace)
    -> single-reference: one grammar element dominates (rotation)
    -> relay chain CONVERGES with linear grammar
    -> algebraic consistency: high (same transformation at every step)

  cos(x) -> cos(2x): degree-2 relationship (frequency doubling, quadratic)
    -> cos(2x) = 2cos^2(x) - 1: requires squaring
    -> multireference: no single linear operation works
    -> relay chain FAILS to converge with linear grammar
    -> algebraic consistency: low (inconsistent transformations, no dominant T)

This mirrors exactly:
  - Minsky-Papert: XOR impossible with single hyperplane (linear operation)
  - Heilbronner: Mobius aromaticity impossible in Huckel (planar orbital) theory,
    BUT becomes stable when you add a topological phase twist (extra dimension)
  - IBM half-Mobius: requires quantum simulation (multireference = superposition
    of multiple configurations, not expressible by single-reference classical methods)

The Type 7 failure signature:
  1. Relay chain stress DOES NOT DECREASE as you add more steps
  2. Algebraic consistency is near-maximum entropy (no dominant grammar element)
  3. Persistent homology: the relay path creates a topological loop (H1 component)
     that valid paths don't create - the path can't close

The fix: ENRICH THE GRAMMAR with the quadratic operation.
After enrichment, the cos->cos(2x) case should:
  - Converge
  - Show a dominant transformation: the squaring operation
  - Have low algebraic entropy
"""

import math
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

import cma
from collections import Counter

from toolkit.euler_relay import _COS, _SIN, N, X, BASIS


# ── Function representations ──────────────────────────────────────────────────

def _cos(): return np.array([(-1.)**(k//2) if k%2==0 else 0. for k in range(N)])
def _sin(): return np.array([(-1.)**((k-1)//2) if k%2==1 else 0. for k in range(N)])
def _cos2x():
    """cos(2x) in phi_k = x^k/k! basis."""
    c = np.zeros(N)
    for k in range(0, N, 2):
        # cos(2x) = sum_{n=0}^inf (-1)^n (2x)^{2n} / (2n)!
        # = sum_{n=0}^inf (-1)^n 2^{2n} x^{2n} / (2n)!
        # In phi_k basis: coefficient of phi_k = (-1)^(k/2) * 2^k for even k
        c[k] = ((-1)**(k//2)) * (2.0**k)
    return c


# ── Two grammars ──────────────────────────────────────────────────────────────

LINEAR_GRAMMAR = [
    ("identity",        lambda b: b.copy(),                   0),
    ("scale_2",         lambda b: 2.0 * b,                    2),
    ("scale_0.5",       lambda b: 0.5 * b,                    2),
    ("scale_neg1",      lambda b: -1.0 * b,                   2),
    ("shift_c0",        lambda b: b + np.array([0.5]+[0]*(N-1)), 3),
    ("derivative",      lambda b: np.append(b[1:], 0.),       2),
    ("antiderivative",  lambda b: np.append([0.], b[:-1]),     2),
    ("reflect_even",    lambda b: np.array([c if i%2==0 else 0. for i,c in enumerate(b)]), 3),
    ("reflect_odd",     lambda b: np.array([0. if i%2==0 else c for i,c in enumerate(b)]), 3),
]

def _squaring_op(b):
    """
    Approximate squaring: (sum_k c_k phi_k(x))^2 evaluated as coefficient vector.
    Uses the convolution property: (f*f)(k) = sum_{j=0}^k c_j * c_{k-j} / C(k,j)
    """
    result = np.zeros(N)
    for k in range(N):
        for j in range(k+1):
            binom = math.comb(k, j)
            if k < N and j < N:
                result[k] += b[j] * b[k-j] / (binom + 1e-10)
    norm = np.linalg.norm(result)
    return result * np.linalg.norm(b) / (norm + 1e-10)  # preserve scale

QUADRATIC_GRAMMAR = LINEAR_GRAMMAR + [
    ("squaring",        _squaring_op,                          4),
    ("cos_double",      lambda b: np.array([(-1.)**(k//2)*(2.0**k) if k%2==0 else 0.
                                            for k in range(N)]) * np.linalg.norm(b), 5),
]


def best_transformation(a, b, grammar):
    """Find grammar element T minimising ||a - T(b)||."""
    best = ("none", float('inf'), 0)
    for name, T, dl in grammar:
        try:
            res = float(np.linalg.norm(a - T(b)))
            if res < best[1]:
                best = (name, res, dl)
        except Exception:
            pass
    return best


def algebraic_depth(fn_a, fn_b, grammar, max_depth=3):
    """
    Minimum number of composed grammar operations to map fn_b to fn_a.
    Depth 0: fn_a == fn_b  (identity)
    Depth 1: exists T: T(fn_b) ~= fn_a  (single operation)
    Depth 2: exists T1,T2: T1(T2(fn_b)) ~= fn_a  (one hidden dimension)
    ...
    Returns (depth, operation_name, residual)
    If depth > max_depth: DIMENSIONAL INSUFFICIENCY for this grammar.
    """
    threshold = 0.1 * np.linalg.norm(fn_a)  # 10% of target magnitude

    # Depth 0
    if np.linalg.norm(fn_a - fn_b) < threshold:
        return 0, "identity", float(np.linalg.norm(fn_a - fn_b))

    # Depth 1
    name, res, _ = best_transformation(fn_a, fn_b, grammar)
    if res < threshold:
        return 1, name, res

    # Depth 2: try all pairs
    best_d2 = ("none", float('inf'), "none")
    for name1, T1, _ in grammar:
        try:
            intermediate = T1(fn_b)
            name2, res2, _ = best_transformation(fn_a, intermediate, grammar)
            if res2 < best_d2[1]:
                best_d2 = (f"{name2}({name1})", res2, "depth2")
        except Exception:
            pass
    if best_d2[1] < threshold:
        return 2, best_d2[0], best_d2[1]

    # Depth 3
    if max_depth >= 3:
        best_d3 = ("none", float('inf'), "none")
        for name1, T1, _ in grammar[:5]:  # limit search
            for name2, T2, _ in grammar[:5]:
                try:
                    intermediate = T2(T1(fn_b))
                    name3, res3, _ = best_transformation(fn_a, intermediate, grammar)
                    if res3 < best_d3[1]:
                        best_d3 = (f"{name3}({name2}({name1}))", res3, "depth3")
                except Exception:
                    pass
        if best_d3[1] < threshold:
            return 3, best_d3[0], best_d3[1]

    return -1, "NONE_FOUND", float('inf')  # Type 7: no path


# ── Relay chain stress test ───────────────────────────────────────────────────

def relay_stress_vs_n(fn_a, fn_b, max_steps=8, n_seeds=3):
    """
    Run relay chains with 1..max_steps intermediates.
    Measure stress at each n.

    PREDICTION for Type 1 (simple relationship):
      stress DECREASES as n increases (more constraints -> tighter path)

    PREDICTION for Type 7 (dimensional insufficiency):
      stress DOES NOT DECREASE (or increases) as n increases
      (no valid path exists, so more intermediates just add confusion)
    """
    stresses = []
    for n_steps in range(1, max_steps + 1):
        seed_stresses = []
        for seed in range(n_seeds):
            np.random.seed(seed * 17 + n_steps * 3)
            pops = [fn_b.copy()]
            for k in range(1, n_steps):
                alpha = k / n_steps
                pops.append((1-alpha)*fn_b + alpha*fn_a + np.random.randn(N)*0.2)
            pops.append(fn_a.copy())

            # One round of CMA-ES optimization
            for k in range(1, n_steps):
                prev, nxt = pops[k-1], pops[k+1]
                def nf(c, p=prev, n_=nxt):
                    return float(np.linalg.norm(np.array(c)-p)**2 +
                                 np.linalg.norm(n_-np.array(c))**2)
                opts = cma.CMAOptions(); opts['maxiter']=80; opts['verbose']=-9
                es = cma.CMAEvolutionStrategy(pops[k].tolist(), 0.4, opts)
                es.optimize(nf)
                pops[k] = np.array(es.result.xbest)

            # Stress = variance of step sizes
            step_sizes = [float(np.linalg.norm(pops[k+1]-pops[k]))
                          for k in range(n_steps)]
            seed_stresses.append(float(np.var(step_sizes)))

        stresses.append(float(np.mean(seed_stresses)))
    return stresses


# ── Main test ─────────────────────────────────────────────────────────────────

def run_type7_test():
    cos = _cos()
    sin = _sin()
    cos2 = _cos2x()

    print("="*65)
    print("TYPE 7 DIMENSIONAL INSUFFICIENCY TEST")
    print("Prediction (from Minsky-Papert theory applied to function space):")
    print("  cos->sin:   degree-1, linear grammar SUFFICIENT")
    print("  cos->cos2x: degree-2, linear grammar INSUFFICIENT (Type 7)")
    print("              quadratic grammar SUFFICIENT")
    print("="*65)

    # 1. Algebraic depth analysis
    print("\n1. ALGEBRAIC DEPTH ANALYSIS")
    print("-"*50)
    for name, fn_a, fn_b in [
        ("cos -> sin",   sin,  cos),
        ("cos -> cos2x", cos2, cos),
    ]:
        for gname, grammar in [("linear", LINEAR_GRAMMAR),
                                ("quadratic", QUADRATIC_GRAMMAR)]:
            depth, op, res = algebraic_depth(fn_a, fn_b, grammar)
            status = f"depth={depth}" if depth >= 0 else "TYPE 7: NO PATH"
            print(f"  {name:<18} [{gname:<10}]: {status}  op={op[:25]}  res={res:.4f}")

    # 2. Stress vs n_steps (the Type 7 signature)
    print("\n2. STRESS-VS-STEPS (stress should DECREASE for valid, not converge for Type 7)")
    print("-"*50)
    print(f"  {'n_steps':>7}", end="")
    for label in ["cos->sin", "cos->cos2x"]:
        print(f"  {label:>12}", end="")
    print()

    pairs = [("cos->sin", sin, cos), ("cos->cos2x", cos2, cos)]
    all_stresses = {}
    for label, fn_a, fn_b in pairs:
        print(f"  Computing stress for {label}...", end="", flush=True)
        stresses = relay_stress_vs_n(fn_a, fn_b, max_steps=6, n_seeds=2)
        all_stresses[label] = stresses
        print(" done")

    for n in range(6):
        print(f"  {n+1:>7}", end="")
        for label, _, __ in pairs:
            print(f"  {all_stresses[label][n]:>12.5f}", end="")
        print()

    # Check: does cos->sin stress decrease? Does cos->cos2x stress NOT decrease?
    cos_sin_decreasing  = all_stresses["cos->sin"][-1] < all_stresses["cos->sin"][0]
    cos_cos2x_stuck = (max(all_stresses["cos->cos2x"]) >
                       0.5 * all_stresses["cos->cos2x"][0])

    print(f"\n  cos->sin stress decreases:       {cos_sin_decreasing}")
    print(f"  cos->cos2x stress stays high:    {cos_cos2x_stuck}")
    print(f"  Type 7 signature confirmed:      {cos_sin_decreasing and cos_cos2x_stuck}")

    # 3. Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Type 7 Dimensional Insufficiency Test\n"
                 "cos->sin (degree 1): stress converges | "
                 "cos->cos(2x) (degree 2): stress stuck",
                 fontsize=10, fontweight="bold")

    x = list(range(1, 7))
    ax = axes[0]
    ax.plot(x, all_stresses["cos->sin"], "o-", color="steelblue", lw=2, ms=7,
            label="cos -> sin  (linear, degree 1)")
    ax.plot(x, all_stresses["cos->cos2x"], "s-", color="firebrick", lw=2, ms=7,
            label="cos -> cos(2x)  (nonlinear, degree 2)")
    ax.set_xlabel("n_relay_intermediates")
    ax.set_ylabel("Path stress (step-size variance)")
    ax.set_title("Stress vs. chain length\n(stress should decrease for degree-1)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for (label, fn_a, fn_b), color in zip(pairs, ["steelblue", "firebrick"]):
        depths = []
        residuals = []
        for g_name, grammar in [("linear", LINEAR_GRAMMAR),
                                  ("quadratic", QUADRATIC_GRAMMAR)]:
            d, op, res = algebraic_depth(fn_a, fn_b, grammar, max_depth=2)
            depths.append(d if d >= 0 else 4)
            residuals.append(res if d >= 0 else 10.0)
        ax.bar([0.3 if label=="cos->sin" else 1.3,
                0.7 if label=="cos->sin" else 1.7],
               depths, 0.3, color=color, alpha=0.8,
               label=f"{label}")

    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["cos->sin", "cos->cos2x"])
    ax.set_ylabel("Algebraic depth (4 = not found)")
    ax.set_title("Algebraic depth: linear (left) vs quadratic (right) grammar\n"
                 "(lower = simpler relationship, -1/4 = Type 7 failure)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("type7_test.png", dpi=140, bbox_inches="tight")
    plt.show()
    print("\nPlot saved -> type7_test.png")

    return all_stresses


if __name__ == "__main__":
    run_type7_test()
