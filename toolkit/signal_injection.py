"""
Artificial signal molecule injection — designed signals with known structure.

Pharmacology analogy:
  Instead of waiting for cells to produce natural ligands, pharmacologists inject
  designed molecules to study receptor-response relationships.

  Agonist:        molecule that mimics natural ligand, triggers full response
  Antagonist:     molecule that blocks receptor without triggering response
  Partial agonist: triggers weaker response than natural ligand
  Inverse agonist: triggers opposite response

For our system:
  Natural signal:    Walsh spectrum of population B's fitness landscape (what emerges)
  Artificial signal: designed Walsh pattern with specific structure (what we inject)

Key experiments this enables:
  1. OPTIMAL SIGNAL INJECTION
     Compute analytically what the signal SHOULD look like for a known relationship.
     Inject it and measure if convergence speed increases.
     If yes: the signal design correctly encodes the relationship.
     If no: the signal design is wrong, or A cannot decode this signal structure.

  2. PHARMACOPHORE IDENTIFICATION
     Systematically zero out different Walsh coefficient positions.
     Measure which positions are NECESSARY for effective steering.
     The necessary positions = the pharmacophore.

  3. LIBRARY MATCHING
     Pre-compute signals for all known Tier 0 relationships.
     For an unknown pair, compare the natural emerging signal against the library.
     The closest match predicts the algebraic relationship.

  4. AGONIST vs. ANTAGONIST
     Inject signals that would correspond to CORRECT vs. WRONG relationships.
     Measure: does the wrong signal steer toward a wrong attractor?
     Quantifies: how much does signal structure determine the attractor?

  5. DOSE-RESPONSE
     Inject partial signals (1/4, 1/2, 3/4 of the optimal pattern).
     Measure convergence quality as a function of signal completeness.
     The dose-response curve = the information content required.

This is MUCH cheaper than full coevolution experiments:
  Full experiment: 300 generations * population evaluation = thousands of fitness calls
  Signal injection test: inject at gen 0, run 30-50 generations, measure response
  Cost ratio: ~10x faster per hypothesis tested
"""

import math
import numpy as np
from typing import Optional, Dict, List

from toolkit.walsh import wht, main_effects
from toolkit.euler_relay import _COS, _SIN, N


# ── Analytically optimal signals for known relationships ─────────────────────

def optimal_signal_derivative(fn_b: np.ndarray, signal_type: str = "walsh_solution") -> np.ndarray:
    """
    The analytically optimal signal from population B for the derivative relationship.

    For A = d/dx(B):
    - B emits a signal encoding "my derivative looks like THIS pattern"
    - A receives it and should steer toward being that derivative

    WalshSolution design:
      The signal is WHT(d/dx(fn_b)) — the Walsh spectrum of B's DERIVATIVE.
      This directly tells A: "move toward this pattern."

    WalshLandscape design:
      The signal is WHT of fitness values at corners of hypercube around d/dx(fn_b).
      Encodes gradient structure of the landscape around the correct A.
    """
    deriv_b = np.append(fn_b[1:], 0.)   # d/dx in phi_k basis = shift left

    # All signals must be length 2^N so main_effects can extract indices 2^k
    SIG_LEN = 2 ** N   # = 256 for N=8

    if signal_type == "walsh_solution":
        padded = np.zeros(SIG_LEN)
        padded[:N] = deriv_b
        return wht(padded)

    elif signal_type == "raw_sparse":
        sparse = np.zeros(N)
        top_idx = np.argsort(np.abs(deriv_b))[-4:]
        sparse[top_idx] = deriv_b[top_idx]
        out = np.zeros(SIG_LEN); out[:N] = sparse
        return wht(out)

    elif signal_type == "directional":
        direction = deriv_b - fn_b
        out = np.zeros(SIG_LEN); out[:N] = direction
        return wht(out)

    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


def optimal_signal_scalar(fn_b: np.ndarray, k: float = 2.0) -> np.ndarray:
    """Optimal signal from B for the scalar-k relationship A = k*B."""
    SIG_LEN = 2 ** N
    out = np.zeros(SIG_LEN); out[:N] = k * fn_b
    return wht(out)


def optimal_signal_identical(fn_b: np.ndarray) -> np.ndarray:
    """Optimal signal for identical relationship A = B."""
    SIG_LEN = 2 ** N
    out = np.zeros(SIG_LEN); out[:N] = fn_b.copy()
    return wht(out)


# ── Signal library for known relationships ────────────────────────────────────

def build_signal_library(n_coeffs: int = N) -> Dict[str, Dict]:
    """
    Pre-compute optimal signals for all Tier 0 algebraic relationships.

    Library structure: {relationship_name: {signal_type: signal_array}}
    Used for library matching: compare natural signal to library, predict relationship.
    """
    cos = np.array([(-1.)**(k//2) if k%2==0 else 0. for k in range(n_coeffs)])
    sin = np.array([(-1.)**((k-1)//2) if k%2==1 else 0. for k in range(n_coeffs)])

    library = {}

    for rel_name, fn_b, signal_fn in [
        ("derivative",  cos, lambda b: optimal_signal_derivative(b, "walsh_solution")),
        ("scalar_2",    cos, lambda b: optimal_signal_scalar(b, 2.0)),
        ("scalar_0.5",  cos, lambda b: optimal_signal_scalar(b, 0.5)),
        ("identical",   cos, lambda b: optimal_signal_identical(b)),
        ("antideriv",   sin, lambda b: optimal_signal_derivative(b, "walsh_solution")),
    ]:
        sig = signal_fn(fn_b)
        library[rel_name] = {
            "signal":   sig,
            "fn_b":     fn_b.copy(),
            "me":       main_effects(sig, n_coeffs),
        }

    return library


def match_to_library(natural_signal: np.ndarray,
                      library: Dict[str, Dict],
                      n_coeffs: int = N) -> List[tuple]:
    """
    Compare a natural population signal against the library of known signals.
    Returns: sorted list of (relationship_name, similarity_score) — highest first.

    This is the "virtual screening" step: before running a full experiment,
    predict which algebraic relationship the natural signal corresponds to.
    Cost: O(|library|) comparisons, essentially free.
    """
    nat_me = main_effects(natural_signal, n_coeffs)
    nat_me_norm = nat_me / (np.linalg.norm(nat_me) + 1e-10)

    similarities = []
    for rel_name, lib_entry in library.items():
        lib_me = lib_entry["me"]
        lib_me_norm = lib_me / (np.linalg.norm(lib_me) + 1e-10)
        sim = float(np.dot(nat_me_norm, lib_me_norm))
        similarities.append((rel_name, sim))

    return sorted(similarities, key=lambda x: -x[1])


# ── Pharmacophore identification ──────────────────────────────────────────────

def pharmacophore_scan(fn_a_target: np.ndarray,
                        fn_b: np.ndarray,
                        base_signal: np.ndarray,
                        n_genes: int = N,
                        n_test_gens: int = 40) -> Dict:
    """
    Systematically zero out individual Walsh coefficient positions and measure
    how much each position contributes to effective steering.

    The positions that, when zeroed, MOST REDUCE the steering effectiveness
    = the pharmacophore = the necessary structural features of the signal.

    This is a knockdown screen: which positions are essential for activity?
    """
    import cma

    def quick_convergence_score(signal_override: np.ndarray) -> float:
        """
        Inject a signal, run n_test_gens, measure how close population A gets
        to fn_a_target. Higher = better signal.
        """
        # Start population A near fn_b (wrong starting point) and inject signal
        rng = np.random.default_rng(42)
        pop_a = rng.standard_normal((30, n_genes)) * 0.3 + fn_b

        # The signal modulates mutation variance
        me = main_effects(signal_override, n_genes)
        me = np.abs(me) / (np.abs(me).max() + 1e-10)
        mut_std = 0.04 * (1.0 + 0.35 * me)

        target_data = fn_a_target  # what A should converge to

        def fit_a(c):
            return -float(np.linalg.norm(c - target_data)**2)

        for _ in range(n_test_gens):
            fits = np.array([fit_a(ind) for ind in pop_a])
            elite = pop_a[np.argsort(fits)[-3:]].copy()
            new_pop = np.empty_like(pop_a)
            new_pop[:3] = elite
            for j in range(3, len(pop_a)):
                idx = rng.choice(len(pop_a), 4, replace=False)
                p1 = pop_a[idx[np.argmax(fits[idx])]]
                child = p1 + rng.standard_normal(n_genes) * mut_std
                new_pop[j] = child
            pop_a = new_pop

        best = pop_a[np.argmax([fit_a(ind) for ind in pop_a])]
        return -float(np.linalg.norm(best - target_data))  # higher = closer to target

    # Baseline: score with full signal
    baseline_score = quick_convergence_score(base_signal)

    # Knockdown scan: zero out each position
    knockdown_effects = {}
    signal_len = len(base_signal)
    key_positions = [1 << k for k in range(n_genes)]  # main-effect positions

    for pos_idx, pos in enumerate(key_positions):
        if pos >= signal_len:
            continue
        masked = base_signal.copy()
        masked[pos] = 0.0   # zero out this position
        score = quick_convergence_score(masked)
        importance = baseline_score - score  # how much did we lose?
        knockdown_effects[f"gene_{pos_idx}(W[2^{pos_idx}])"] = {
            "position": pos,
            "score_with":    baseline_score,
            "score_without": score,
            "importance":    importance,   # higher = this position is essential
        }

    # Sort by importance
    pharmacophore = sorted(knockdown_effects.items(),
                            key=lambda x: -x[1]["importance"])

    return {
        "baseline_score": baseline_score,
        "knockdown_effects": dict(pharmacophore),
        "essential_positions": [k for k, v in pharmacophore
                                  if v["importance"] > 0.05 * abs(baseline_score)],
    }


# ── Injection experiment runner ───────────────────────────────────────────────

def run_injection_experiment(fn_a_target: np.ndarray,
                              fn_b_start: np.ndarray,
                              signals_to_test: Dict[str, np.ndarray],
                              n_gens: int = 80,
                              pop_size: int = 40,
                              verbose: bool = True) -> Dict:
    """
    Run the same population under different injected signals and measure
    how each signal affects convergence toward fn_a_target.

    This is the "agonist/antagonist screen": which signal designs
    are agonists (steer toward target) vs. antagonists (steer away)?

    Returns dose-response data for each signal.
    """
    rng = np.random.default_rng(42)
    results = {}

    def fit_a(c, target=fn_a_target):
        return -float(np.linalg.norm(c - target)**2)

    for sig_name, signal in signals_to_test.items():
        # Reset population to same starting point
        pop_a = rng.standard_normal((pop_size, N)) * 0.3 + fn_b_start

        # Extract mutation bias from signal
        me = main_effects(signal, N)
        me = np.abs(me) / (np.abs(me).max() + 1e-10)
        mut_std = 0.04 * (1.0 + 0.35 * me)

        trajectory = []
        for g in range(n_gens):
            fits = np.array([fit_a(ind) for ind in pop_a])
            elite_idx = np.argsort(fits)[-3:]
            new_pop = np.empty_like(pop_a)
            new_pop[:3] = pop_a[elite_idx].copy()
            for j in range(3, pop_size):
                idx = rng.choice(pop_size, 4, replace=False)
                p1 = pop_a[idx[np.argmax(fits[idx])]]
                child = p1 + rng.standard_normal(N) * mut_std
                new_pop[j] = child
            pop_a = new_pop

            best_fit = fits.max()
            trajectory.append(float(-best_fit))  # residual

        final_best = pop_a[np.argmax([fit_a(ind) for ind in pop_a])]
        final_residual = float(np.linalg.norm(final_best - fn_a_target))

        results[sig_name] = {
            "trajectory":      trajectory,
            "final_residual":  final_residual,
            "initial_residual": trajectory[0],
            "improvement":     trajectory[0] - final_residual,
            "signal_me":       me.tolist(),
        }

        if verbose:
            print(f"  {sig_name:<25}: final_res={final_residual:.4f}  "
                  f"improvement={trajectory[0]-final_residual:.4f}")

    return results


def make_signal(pattern: str, strength: float = 1.0) -> np.ndarray:
    """
    Create an artificial signal molecule with a specific Walsh coefficient pattern.
    Patterns: 'linearity','oscillation','oscillation_even','oscillation_odd',
              'scale_growth','derivative','uniform','zero'
    """
    SIG_LEN = 2 ** N
    sig = np.zeros(SIG_LEN)
    me_positions = [1 << k for k in range(N)]

    weight_map = {
        'linearity':       np.array([3., 2., 1., .5, .2, .1, .05, .02]),
        'oscillation':     np.array([1.,-1., 1.,-1., 1.,-1., 1.,-1.]),
        'oscillation_even':np.array([1., 0., 1., 0., 1., 0., 1., 0.]),
        'oscillation_odd': np.array([0., 1., 0., 1., 0., 1., 0., 1.]),
        'scale_growth':    np.array([.02,.05,.1,.2,.5, 1., 2., 3.]),
        'derivative':      np.array([0., 1., 0.,-1., 0., 1., 0.,-1.]),
        'uniform':         np.ones(N),
        'zero':            np.zeros(N),
    }

    if pattern not in weight_map:
        raise ValueError(f"Unknown pattern '{pattern}'. Choose from: {list(weight_map)}")

    weights = weight_map[pattern]
    for k, w in enumerate(weights):
        if me_positions[k] < SIG_LEN:
            sig[me_positions[k]] = strength * w

    return sig


def mix_signals(s1: np.ndarray, s2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Linear mixture of two signals."""
    return alpha * s1 + (1 - alpha) * s2


if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt

    cos = _COS.copy()
    sin = _SIN.copy()

    print("="*60)
    print("ARTIFICIAL SIGNAL INJECTION EXPERIMENT")
    print("Target: steer population toward sin (derivative of cos)")
    print("="*60)

    # Build signal library
    library = build_signal_library()
    print(f"\nSignal library built: {list(library.keys())}")

    # Compute optimal signal for derivative relationship
    opt_sig_ws  = optimal_signal_derivative(cos, "walsh_solution")
    opt_sig_dir = optimal_signal_derivative(cos, "directional")
    opt_sig_sp  = optimal_signal_derivative(cos, "raw_sparse")

    # Also build a WRONG signal (for the scalar relationship, not derivative)
    wrong_sig   = optimal_signal_scalar(cos, 2.0)

    # Null signal
    null_sig    = np.zeros(2**N)

    signals = {
        "null (no injection)":          null_sig,
        "optimal_walsh_solution":        opt_sig_ws,
        "optimal_directional":           opt_sig_dir,
        "optimal_raw_sparse":            opt_sig_sp,
        "WRONG (scalar, antagonist)":   wrong_sig,
    }

    print(f"\nRunning injection experiment:")
    print(f"Start: population near cos. Target: sin (derivative of cos).")
    print(f"{'Signal':<30} {'Final residual':>15} {'Improvement':>12}")
    print("-"*60)

    results = run_injection_experiment(sin, cos, signals, n_gens=80, verbose=True)

    # Library matching test
    print(f"\nLIBRARY MATCHING TEST:")
    print(f"Can we predict 'derivative' from the optimal signal?")
    matches = match_to_library(opt_sig_ws, library)
    for rel_name, sim in matches[:3]:
        print(f"  {rel_name:<20}: similarity = {sim:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"null (no injection)": "gray",
               "optimal_walsh_solution": "steelblue",
               "optimal_directional": "darkorange",
               "optimal_raw_sparse": "seagreen",
               "WRONG (scalar, antagonist)": "firebrick"}

    for sig_name, res in results.items():
        ax.plot(res["trajectory"], label=sig_name,
                color=colors.get(sig_name, "black"), lw=1.8)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Residual ||pop_best - sin||")
    ax.set_title("Artificial signal injection: which signal steers toward sin?\n"
                 "(lower = better; null baseline vs. designed signals)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("signal_injection.png", dpi=140, bbox_inches="tight")
    plt.show()
    print("\nPlot saved -> signal_injection.png")
