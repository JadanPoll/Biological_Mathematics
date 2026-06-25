"""
Joint CMA-ES covariance analysis.

The core problem:
  - Alternating optimization: miscoordination, may not converge to joint optimum
  - Joint (simultaneous) optimization: correct, but 2N-dimensional CMA-ES
  - We need: cheap structural validation that detects the algebraic relationship

The insight:
  CMA-ES learns a covariance matrix over the 2N joint space.
  This matrix has four N×N blocks:
    [ Cov(A,A)  Cov(A,B) ]
    [ Cov(B,A)  Cov(B,B) ]

  The OFF-DIAGONAL BLOCK Cov(A,B) encodes how A and B SHOULD co-vary
  to maintain the algebraic relationship. This is the learned signal molecule.

  For derivative relationship (A = d/dx(B)):
    A[j] = B[j+1]  =>  Cov(A,B)[j, j+1] should be large (positive correlation)
    The cross-block should have a strong sub-diagonal structure.

  For scalar multiple (A = k*B):
    A[j] = k*B[j]  =>  Cov(A,B)[j, j] should be large (diagonal structure)
    The cross-block should be approximately diagonal.

  For Type 7 (no algebraic path):
    No consistent A-B correlation => cross-block near-zero or noisy
    Structure-free matrix.

This is the CHEAP structural fingerprint:
  - Run joint CMA-ES briefly (50-100 iterations, not to convergence)
  - Extract the A-B covariance block
  - The block structure IS the algebraic relationship, whether or not
    the optimizer has converged to the correct (A, B) values

Why this is cheaper than full convergence:
  - We only need the covariance STRUCTURE, not the exact solution
  - Structure emerges early in CMA-ES training
  - O(N^2) storage, O(N^2 * pop_size) computation per generation
  - For N=8: 64 covariance entries, manageable
"""

import math
import numpy as np
import cma
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from toolkit.euler_relay import _COS, _SIN, N, BASIS
from toolkit.joint_fitness import make_pair_fitness, _GRAMMAR, best_transformation


# ── Function representations ──────────────────────────────────────────────────

def _cos(): return np.array([(-1.)**(k//2) if k%2==0 else 0. for k in range(N)])
def _sin(): return np.array([(-1.)**((k-1)//2) if k%2==1 else 0. for k in range(N)])
def _cos2x():
    c = np.zeros(N)
    for k in range(0, N, 2):
        c[k] = ((-1)**(k//2)) * (2.0**k)
    return c
def _lin(): return np.array([0., 1., 0., 0., 0., 0., 0., 0.])
def _rand(seed=42):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(N); return c / np.linalg.norm(c)


# ── Joint fitness functions ───────────────────────────────────────────────────

L1_W = 1e-5
X    = np.linspace(-1.5, 1.5, 100)
_BASIS = np.stack([X**k / math.factorial(k) for k in range(N)], axis=1)

def poly_eval(c):
    return _BASIS @ c

def make_joint_fit(fn_a, fn_b, relationship="derivative",
                    norm_weight=3.0):
    """
    Joint fitness for [a, b] 2N vector WITH NORMALIZATION CONSTRAINT.

    Normalization prevents the degenerate zero solution:
      Without it: A=B=0 satisfies any relationship trivially
      With it:    both A and B must have the same norm as the targets

    The target norm is set to ||fn_a|| so the optimizer is pushed
    toward solutions of the right scale.
    """
    target_norm_a = float(np.linalg.norm(fn_a))
    target_norm_b = float(np.linalg.norm(fn_b))

    def norm_pen(a, b):
        return (norm_weight * (np.linalg.norm(a) - target_norm_a)**2 +
                norm_weight * (np.linalg.norm(b) - target_norm_b)**2)

    if relationship == "derivative":
        def fit(x):
            a, b = np.array(x[:N]), np.array(x[N:])
            deriv_b = np.append(b[1:], 0.)
            return -float(np.linalg.norm(a - deriv_b)) - norm_pen(a, b)

    elif relationship == "scalar_2":
        def fit(x):
            a, b = np.array(x[:N]), np.array(x[N:])
            return -float(np.linalg.norm(a - 2.0 * b)) - norm_pen(a, b)

    elif relationship == "identical":
        def fit(x):
            a, b = np.array(x[:N]), np.array(x[N:])
            return -float(np.linalg.norm(a - b)) - norm_pen(a, b)

    elif relationship == "discovery":
        def fit(x):
            a, b = np.array(x[:N]), np.array(x[N:])
            _, res, _ = best_transformation(a, b, _GRAMMAR)
            return -float(res) - norm_pen(a, b)
    else:
        raise ValueError(f"Unknown relationship: {relationship}")

    return fit


# ── Joint CMA-ES with covariance extraction ───────────────────────────────────

def run_joint_cmaes(fn_a, fn_b, relationship="derivative",
                    n_generations=150, sigma0=0.4, seed=42,
                    verbose=True):
    """
    Run joint CMA-ES on the 2N-dimensional [a, b] vector.
    Extract the A-B cross-covariance block after n_generations.

    Returns:
      best_a, best_b: found solutions
      cross_cov:      N×N A-B covariance block (the learned signal molecule)
      residual:       fitness at convergence
      convergence:    generation at which residual < threshold
    """
    fit = make_joint_fit(fn_a, fn_b, relationship)

    np.random.seed(seed)
    x0 = np.concatenate([fn_b + np.random.randn(N)*0.3,
                          fn_a + np.random.randn(N)*0.3])

    opts = cma.CMAOptions()
    opts['maxiter']  = n_generations
    opts['verbose']  = -9
    opts['seed']     = seed
    opts['popsize']  = 4 + int(3 * math.log(2*N))  # standard CMA-ES popsize

    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
    convergence_gen = None

    for gen in range(n_generations):
        solutions = es.ask()
        fitvals   = [-fit(s) for s in solutions]   # CMA-ES minimizes
        es.tell(solutions, fitvals)

        best_fit = -min(fitvals)
        if convergence_gen is None and best_fit > -0.10:
            convergence_gen = gen

        if es.stop():
            break

    x_best = np.array(es.result.xbest)
    best_a = x_best[:N]
    best_b = x_best[N:]
    residual = -fit(x_best)

    # Extract covariance matrix and pull out A-B cross block
    # CMA-ES covariance C is available as es.result.... or from the object
    try:
        C = es.sm.C if hasattr(es, 'sm') else np.eye(2*N)
        # Normalize to correlation matrix
        std = np.sqrt(np.diag(C)) + 1e-10
        corr = C / np.outer(std, std)
        cross_cov = corr[:N, N:]  # A-B cross-correlation block
    except Exception:
        # Fallback: estimate covariance from final population
        final_pop  = np.array(es.ask())
        cross_cov  = np.cov(final_pop.T)[:N, N:]

    if verbose:
        print(f"  residual={residual:.5f}  conv_gen={convergence_gen}  "
              f"||A-fn_a||={np.linalg.norm(best_a-fn_a):.4f}  "
              f"||B-fn_b||={np.linalg.norm(best_b-fn_b):.4f}")

    return {
        "best_a":       best_a,
        "best_b":       best_b,
        "residual":     residual,
        "cross_cov":    cross_cov,
        "convergence":  convergence_gen,
    }


# ── Main comparison ───────────────────────────────────────────────────────────

def run_comparison(n_generations=200, verbose=True):
    """
    Run joint CMA-ES on four relationship types.
    Compare: does it converge? What does the A-B covariance block look like?
    """
    cos, sin, cos2, lin, rand = _cos(), _sin(), _cos2x(), _lin(), _rand()

    experiments = [
        ("cos -> sin  [derivative]",      sin,  cos,  "derivative"),
        ("cos -> 2cos [scalar_2]",         2*cos, cos,  "scalar_2"),
        ("cos -> cos  [identical]",        cos,  cos,  "identical"),
        ("cos -> cos2x [Type 7 in linear]",cos2, cos,  "discovery"),
    ]

    print("="*65)
    print("JOINT CMA-ES COMPARISON")
    print("Core problem: alternating miscoordinates; joint solves it but costs 2N dims")
    print("Cheap structural check: A-B covariance block after partial convergence")
    print("="*65)

    results = {}
    for name, fn_a, fn_b, rel in experiments:
        print(f"\n  {name}")
        r = run_joint_cmaes(fn_a, fn_b, relationship=rel,
                            n_generations=n_generations, verbose=verbose)
        results[name] = r

    # Plot cross-covariance blocks
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("A-B cross-covariance blocks from joint CMA-ES\n"
                 "Structure reveals the algebraic relationship\n"
                 "Derivative: sub-diagonal | Scalar: diagonal | Type 7: noisy",
                 fontsize=10, fontweight="bold")

    cmap = "RdBu_r"
    for ax, (name, r) in zip(axes.flatten(), results.items()):
        cc = r["cross_cov"]
        vmax = max(abs(cc).max(), 0.1)
        im = ax.imshow(cc, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{name[:40]}\nresidual={r['residual']:.4f}", fontsize=8)
        ax.set_xlabel("B component"); ax.set_ylabel("A component")
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig("joint_covariance.png", dpi=140, bbox_inches="tight")
    plt.show()
    print("\nPlot saved -> joint_covariance.png")

    # Print structural summary
    print("\n" + "="*65)
    print("STRUCTURAL SUMMARY")
    print(f"{'Experiment':<35} {'Residual':>9} {'Conv':>6} {'Cross-cov max':>13}")
    print("-"*65)
    for name, r in results.items():
        conv = str(r['convergence']) if r['convergence'] else "—"
        print(f"  {name:<33} {r['residual']:>9.5f} {conv:>6}  "
              f"{abs(r['cross_cov']).max():>9.5f}")

    return results


if __name__ == "__main__":
    results = run_comparison(n_generations=300, verbose=True)
