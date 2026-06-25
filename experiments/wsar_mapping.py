"""
Walsh Signal Activity Relationship (WSAR) mapping.

Analogous to Structure-Activity Relationship (SAR) in drug discovery:
  - SAR: vary molecular structure, measure biological activity
  - WSAR: vary Walsh coefficient pattern in signal molecule, measure
    which algebraic relationship it steers toward

Goal: build a MAP of signal-molecule space.
Each point in this space is a specific Walsh coefficient pattern.
For each point, measure:
  - Which algebraic relationship does this signal steer toward?
  - How strongly?
  - Can a MIXTURE of signals encode multiple relationships simultaneously?

Three signal families to explore:

  LINEARITY SIGNAL: encodes "I am a linear / polynomial object"
    Hypothesis: large main-effects at LOW-ORDER even-index genes (k=0, 2)
    Prediction: steers A toward functions that are smooth and polynomial

  OSCILLATION SIGNAL: encodes "I am an oscillatory object"
    Hypothesis: large main-effects at ALTERNATING even/odd genes
    Prediction: steers A toward functions that oscillate (cos/sin-like)

  SCALE GROWTH SIGNAL: encodes "my coefficients grow in magnitude with index"
    Hypothesis: main-effects INCREASING with gene index (k=6, 7 dominate)
    Prediction: steers A toward functions with growing high-order terms (like cos(2x))

  JOINT SIGNAL: mixture encoding "I am oscillatory AND growing in scale"
    Hypothesis: this corresponds to a specific algebraic relationship type
    Prediction: steers toward a specific higher-complexity function class
"""

import math
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from toolkit.walsh import wht, main_effects
from toolkit.euler_relay import _COS, _SIN, N


# ── Canonical signal molecule designs ─────────────────────────────────────────

def make_signal(pattern: str, strength: float = 1.0) -> np.ndarray:
    """
    Create an artificial signal molecule with a specific Walsh coefficient pattern.

    pattern options:
      'linearity':       concentrated at low-order even indices (k=0, 2)
      'oscillation':     alternating large/small at even/odd indices
      'scale_growth':    increasing toward high-order indices
      'derivative':      odd-index dominated (shift structure)
      'uniform':         all main-effects equal (no structural preference)
      'zero':            null signal (no information)

    The signal is in the Walsh spectral domain (length 2^N).
    Main-effect positions: indices 1, 2, 4, 8, 16, 32, 64, 128 (= 2^k for k=0..7)
    """
    sig = np.zeros(2 ** N)
    me_positions = [1 << k for k in range(N)]  # 2^k positions

    if pattern == 'linearity':
        # Strong signal at low-order polynomial genes (k=0, 1, 2)
        # "I am smooth and low-degree"
        weights = np.array([3.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02])

    elif pattern == 'oscillation':
        # Alternating large/small at even/odd genes
        # "I alternate between positive and negative — I am oscillatory"
        weights = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

    elif pattern == 'oscillation_even':
        # Strong only at even genes (cos-like: even indices dominate)
        weights = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    elif pattern == 'oscillation_odd':
        # Strong only at odd genes (sin-like: odd indices dominate)
        weights = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    elif pattern == 'scale_growth':
        # Increasing toward high-order indices
        # "My high-order coefficients are large — I am a high-frequency function"
        weights = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0])

    elif pattern == 'derivative':
        # Odd-index dominated with shift structure
        # "I am the derivative of a cos-like object — you should be sin-like"
        weights = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0])

    elif pattern == 'uniform':
        weights = np.ones(N)

    elif pattern == 'zero':
        return sig

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    for k, w in enumerate(weights):
        if me_positions[k] < len(sig):
            sig[me_positions[k]] = strength * w

    return sig


def mix_signals(sig1: np.ndarray, sig2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create a joint signal: alpha * sig1 + (1-alpha) * sig2.
    Tests whether a mixture encodes a combination of algebraic properties.
    """
    return alpha * sig1 + (1 - alpha) * sig2


# ── WSAR mapping experiment ───────────────────────────────────────────────────

def run_wsar_mapping(n_gens: int = 60, pop_size: int = 40) -> dict:
    """
    Run the WSAR mapping experiment.

    For each signal pattern, test against each of four target relationships:
      - derivative (cos -> sin)
      - scalar_2 (cos -> 2*cos)
      - identical (cos -> cos)
      - random (cos -> random)

    Measure: which signal pattern most effectively steers toward each target?
    Build the activity matrix: rows = signals, columns = targets.
    """
    cos = _COS.copy()
    sin = _SIN.copy()
    rng = np.random.default_rng(42)

    def _cos2x():
        c = np.zeros(N)
        for k in range(0, N, 2):
            c[k] = ((-1)**(k//2)) * (2.0**k)
        return c

    targets = {
        "derivative (->sin)":    sin,
        "scalar_2 (->2cos)":     2 * cos,
        "identical (->cos)":     cos,
        "freq_double (->cos2x)": _cos2x(),
    }

    signal_patterns = [
        "zero",
        "linearity",
        "oscillation",
        "oscillation_even",
        "oscillation_odd",
        "scale_growth",
        "derivative",
        "uniform",
    ]

    # Also add joint (mixed) signals
    joint_signals = {
        "joint_osc+deriv": mix_signals(make_signal("oscillation"), make_signal("derivative"), 0.5),
        "joint_osc+scale": mix_signals(make_signal("oscillation_even"), make_signal("scale_growth"), 0.5),
    }

    def evaluate_signal(signal: np.ndarray, target: np.ndarray) -> float:
        """
        Run population A from cos toward target, steered by signal.
        Returns: improvement = initial_residual - final_residual (higher = better).
        """
        pop_a = rng.standard_normal((pop_size, N)) * 0.2 + cos

        me = main_effects(signal, N)
        me = np.abs(me) / (np.abs(me).max() + 1e-10)
        mut_std = 0.04 * (1.0 + 0.35 * me)

        def fit(c):
            return -float(np.linalg.norm(c - target)**2)

        initial_res = float(np.linalg.norm(pop_a[0] - target))

        for _ in range(n_gens):
            fits = np.array([fit(ind) for ind in pop_a])
            elite = pop_a[np.argsort(fits)[-3:]].copy()
            new_pop = np.empty_like(pop_a)
            new_pop[:3] = elite
            for j in range(3, pop_size):
                idx = rng.choice(pop_size, 4, replace=False)
                p1 = pop_a[idx[np.argmax(fits[idx])]]
                child = p1 + rng.standard_normal(N) * mut_std
                new_pop[j] = child
            pop_a = new_pop

        best = pop_a[np.argmax([fit(ind) for ind in pop_a])]
        final_res = float(np.linalg.norm(best - target))
        return initial_res - final_res

    # Build activity matrix
    print("="*70)
    print("WSAR MAPPING: signal pattern x target relationship")
    print("Each cell = improvement in convergence (higher = signal is agonist)")
    print("="*70)

    all_signals = {p: make_signal(p) for p in signal_patterns}
    all_signals.update(joint_signals)

    header = f"{'Signal':<28}" + "".join(f"{t[:12]:>14}" for t in targets)
    print(header)
    print("-"*70)

    activity_matrix = {}
    for sig_name, sig in all_signals.items():
        row = {}
        line = f"  {sig_name:<26}"
        for tgt_name, target in targets.items():
            activity = evaluate_signal(sig, target)
            row[tgt_name] = activity
            line += f"  {activity:>12.4f}"
        activity_matrix[sig_name] = row
        print(line)

    print(f"\nINTERPRETATION:")
    print("- Row with highest value for each column = best agonist for that relationship")
    print("- Diagonal pattern (high only on target match) = specific signal")
    print("- Flat row = signal has no algebraic specificity")
    print("- Joint signals: does mixing capture both targets partially?")

    return activity_matrix


def plot_activity_matrix(matrix: dict, targets: list, out: str = "wsar_matrix.png"):
    signals = list(matrix.keys())
    data = np.array([[matrix[s][t] for t in targets] for s in signals])

    # Normalize each column to [0,1] for better visualization
    col_mins = data.min(axis=0)
    col_maxs = data.max(axis=0)
    data_norm = (data - col_mins) / (col_maxs - col_mins + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([t[:12] for t in targets], rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(signals)))
    ax.set_yticklabels(signals, fontsize=9)
    ax.set_title("WSAR Activity Matrix (normalised per target)\n"
                 "Green = strong agonist | Red = weak / antagonist",
                 fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Normalised activity')
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.show()
    print(f"Plot saved -> {out}")


if __name__ == "__main__":
    targets = ["derivative (->sin)", "scalar_2 (->2cos)",
               "identical (->cos)", "freq_double (->cos2x)"]
    matrix = run_wsar_mapping(n_gens=60)
    plot_activity_matrix(matrix, targets)
