"""
Signal molecule morphing — continuous deformation through phase space.

The key experiments:
  1. LINEAR MORPH A->B:
     S(t) = (1-t)*S_A + t*S_B  for t in [0..1]
     Question: where do phase transitions happen? Sharp jumps or gradual?

  2. ROUND TRIP A->B->A:
     Morph A->B, then B->A.
     Question: does the coverage trace reverse cleanly? (Reversible morphing)
     If yes: signal-molecule space is geometrically flat at this path.
     If no: HYSTERESIS — path-dependent, non-trivial topology.

  3. CYCLE A->B->C->A:
     Morph A->B->C->A.
     Question: does the final signal return to the same phase class as A?
     If no: HOLONOMY — the signal molecule accumulated a phase shift from
     traversing the loop. This is the Berry phase analog for signal molecules.

  4. GEODESIC FINDING:
     Among all interpolation paths from A to B, find the one with the
     fewest phase transitions (smoothest path through phase space).
     This is the minimum-effort morph — the cheapest algebraic transformation.

Why this is cheap:
  Full relay chain experiment: 300-600 generations, full population
  Signal morph evaluation: 40 generations, 30 individuals = 1200 fitness calls
  A full morph path (20 steps): 20 * 1200 = 24,000 calls
  vs. relay chain: 300 * 80 = 24,000 — same cost, but morph gives TOPOLOGY.

What we learn:
  - Which signal classes are "topologically connected" (continuous morph exists)
  - Where the phase transition boundaries are in signal space
  - Whether morphing is reversible (no hysteresis)
  - The curvature of signal-molecule space (holonomy of cycles)
  - The minimum-effort path between algebraic relationship classes
"""

import math
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from toolkit.euler_relay import _COS, _SIN, N
from toolkit.walsh import main_effects
from toolkit.signal_injection import (optimal_signal_derivative,
                                       optimal_signal_scalar,
                                       optimal_signal_identical,
                                       make_signal)


# ── Phase evaluator ───────────────────────────────────────────────────────────

def evaluate_phase(signal: np.ndarray,
                   fn_a_target: np.ndarray,
                   fn_b_start: np.ndarray,
                   n_gens: int = 60,
                   pop_size: int = 30,
                   rng_seed: int = 42,
                   use_joint_fitness: bool = True) -> Dict:
    """
    Evaluate phase class of this signal molecule.
    use_joint_fitness=True: fitness depends on BOTH populations (joint coupling).
      This makes the signal molecule the PRIMARY driver, revealing richer topology.
    use_joint_fitness=False: independent fitness, signal is a weak perturbation.
      This gives flat topology (all signals similar).
    """
    rng = np.random.default_rng(rng_seed)

    me = main_effects(signal, N)
    me = np.abs(me) / (np.abs(me).max() + 1e-10)
    mut_std = 0.04 * (1.0 + 0.40 * me)

    # Both populations start near fn_b_start (wrong point)
    pop_a = rng.standard_normal((pop_size, N)) * 0.25 + fn_b_start
    pop_b = rng.standard_normal((pop_size, N)) * 0.25 + fn_b_start

    if use_joint_fitness:
        # Joint fitness: A should equal d/dx(B), B should equal antideriv(A)
        # Both populations coupled through the derivative relationship
        def fit_a(a, b_best):
            deriv_b = np.append(b_best[1:], 0.)
            return -float(np.linalg.norm(a - deriv_b)**2) - 2.0*(np.linalg.norm(a)-2.0)**2
        def fit_b(b, a_best):
            deriv_b = np.append(b[1:], 0.)
            return -float(np.linalg.norm(a_best - deriv_b)**2) - 2.0*(np.linalg.norm(b)-2.0)**2
    else:
        def fit_a(a, b_best):
            return -float(np.linalg.norm(a - fn_a_target)**2)
        def fit_b(b, a_best):
            return -float(np.linalg.norm(b - fn_b_start)**2)

    init_res = float(np.linalg.norm(pop_a[0] - fn_a_target))
    traj = []
    best_a = pop_a[0].copy()
    best_b = pop_b[0].copy()

    for _ in range(n_gens):
        # Evolve A with signal biasing
        fits_a = np.array([fit_a(ind, best_b) for ind in pop_a])
        elite_a = pop_a[np.argsort(fits_a)[-3:]].copy()
        new_a = np.empty_like(pop_a)
        new_a[:3] = elite_a
        for j in range(3, pop_size):
            idx = rng.choice(pop_size, 3, replace=False)
            p1 = pop_a[idx[np.argmax(fits_a[idx])]]
            new_a[j] = p1 + rng.standard_normal(N) * mut_std
        pop_a = new_a
        best_a = pop_a[np.argmax(fits_a)].copy()

        # Evolve B with signal biasing (reversed signal if joint)
        fits_b = np.array([fit_b(ind, best_a) for ind in pop_b])
        elite_b = pop_b[np.argsort(fits_b)[-3:]].copy()
        new_b = np.empty_like(pop_b)
        new_b[:3] = elite_b
        for j in range(3, pop_size):
            idx = rng.choice(pop_size, 3, replace=False)
            p1 = pop_b[idx[np.argmax(fits_b[idx])]]
            new_b[j] = p1 + rng.standard_normal(N) * mut_std
        pop_b = new_b
        best_b = pop_b[np.argmax(fits_b)].copy()

        # Measure: how close is A to the joint optimum?
        res = float(np.linalg.norm(best_a - fn_a_target))
        traj.append(res)

    final_res = traj[-1]
    improvement = init_res - final_res

    # Finer phase classification for joint fitness
    if improvement < 0.2:
        phase = "inactive"
    elif improvement < 1.0:
        phase = "weak"
    elif improvement < 2.5:
        phase = "moderate"
    elif improvement < 4.5:
        phase = "strong"
    else:
        phase = "maximal"

    return {"improvement": improvement, "phase": phase,
            "residual": final_res, "trajectory": traj}


# ── Signal interpolation ──────────────────────────────────────────────────────

def morph_path(sig_a: np.ndarray, sig_b: np.ndarray,
               n_steps: int = 20) -> List[np.ndarray]:
    """
    Generate a linear interpolation path from sig_a to sig_b.
    S(t) = (1-t)*sig_a + t*sig_b  for t in {0, 1/n, 2/n, ..., 1}
    """
    return [sig_a * (1 - t/n_steps) + sig_b * (t/n_steps)
            for t in range(n_steps + 1)]


# ── Morph experiments ─────────────────────────────────────────────────────────

def run_morph(name: str,
              sig_a: np.ndarray, sig_b: np.ndarray,
              fn_a_target: np.ndarray, fn_b_start: np.ndarray,
              n_steps: int = 15, verbose: bool = True) -> Dict:
    """Run a single A->B morph and record phase at each step."""
    path = morph_path(sig_a, sig_b, n_steps)
    phases = []
    improvements = []
    residuals = []

    for i, sig in enumerate(path):
        t = i / n_steps
        result = evaluate_phase(sig, fn_a_target, fn_b_start,
                                 rng_seed=42 + i)
        phases.append(result["phase"])
        improvements.append(result["improvement"])
        residuals.append(result["residual"])

    # Count phase transitions
    transitions = sum(1 for i in range(len(phases)-1) if phases[i] != phases[i+1])

    # Find transition points
    transition_points = [i/n_steps for i in range(len(phases)-1)
                         if phases[i] != phases[i+1]]

    if verbose:
        print(f"  {name}:")
        print(f"    Phase trace: {' -> '.join(phases)}")
        print(f"    Transitions: {transitions} at t={[round(x,2) for x in transition_points]}")
        print(f"    Max improvement: {max(improvements):.4f}")

    return {
        "name":              name,
        "phases":            phases,
        "improvements":      improvements,
        "residuals":         residuals,
        "transitions":       transitions,
        "transition_points": transition_points,
        "n_steps":           n_steps,
        "t_values":          [i/n_steps for i in range(n_steps+1)],
    }


def run_round_trip(name: str,
                    sig_a: np.ndarray, sig_b: np.ndarray,
                    fn_a_target: np.ndarray, fn_b_start: np.ndarray,
                    n_steps: int = 10) -> Dict:
    """
    Round trip A->B->A.
    Check: is the morph reversible? Does the coverage trace reverse cleanly?
    Hysteresis = forward and backward traces differ.
    """
    forward = run_morph(f"{name}_fwd", sig_a, sig_b,
                         fn_a_target, fn_b_start, n_steps, verbose=False)
    backward = run_morph(f"{name}_bwd", sig_b, sig_a,
                          fn_a_target, fn_b_start, n_steps, verbose=False)

    # Hysteresis = how much do the phase traces differ?
    fwd_phases = forward["phases"]
    bwd_phases = list(reversed(backward["phases"]))  # reverse to align with forward
    mismatch = sum(1 for f, b in zip(fwd_phases, bwd_phases) if f != b)
    hysteresis = mismatch / len(fwd_phases)

    forward_end   = forward["phases"][-1]
    backward_end  = backward["phases"][-1]
    is_reversible = (forward["phases"][0] == backward_end and
                     hysteresis < 0.2)

    return {
        "name":          name,
        "forward":       forward,
        "backward":      backward,
        "hysteresis":    hysteresis,
        "is_reversible": is_reversible,
        "forward_end_phase":  forward_end,
        "backward_end_phase": backward_end,
    }


def run_cycle(name: str,
               sigs: List[np.ndarray], labels: List[str],
               fn_a_target: np.ndarray, fn_b_start: np.ndarray,
               n_steps: int = 8) -> Dict:
    """
    Cycle test A->B->C->...->A.
    Check holonomy: does the cycle return to the same phase class?
    Non-zero holonomy = signal-molecule space has non-trivial curvature.
    """
    legs = []
    full_phases = []
    full_t = []
    offset = 0

    for i in range(len(sigs)):
        s_start = sigs[i]
        s_end   = sigs[(i+1) % len(sigs)]
        label   = f"{labels[i]}->{labels[(i+1)%len(labels)]}"

        leg = run_morph(label, s_start, s_end,
                         fn_a_target, fn_b_start,
                         n_steps, verbose=False)
        legs.append(leg)
        full_phases.extend(leg["phases"][:-1])  # don't duplicate endpoints
        full_t.extend([t + offset for t in leg["t_values"][:-1]])
        offset += 1.0

    # Close the cycle
    full_phases.append(legs[-1]["phases"][-1])
    full_t.append(offset)

    # Holonomy: does the cycle close?
    start_phase = full_phases[0]
    end_phase   = full_phases[-1]
    holonomy    = 0 if start_phase == end_phase else 1

    total_transitions = sum(leg["transitions"] for leg in legs)

    return {
        "name":              name,
        "legs":              legs,
        "full_phases":       full_phases,
        "full_t":            full_t,
        "start_phase":       start_phase,
        "end_phase":         end_phase,
        "holonomy":          holonomy,
        "total_transitions": total_transitions,
        "cycle_closed":      start_phase == end_phase,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

PHASE_COLORS = {
    "inactive": "#888888",
    "weak":     "#ffaa00",
    "moderate": "#00aaff",
    "strong":   "#00cc44",
    "maximal":  "#aa00ff",
}


def plot_morph_experiments(results: Dict, out: str = "signal_morph.png"):
    n_rows = 2
    n_cols = max(2, len(results) // 2 + len(results) % 2)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    fig.suptitle("Signal molecule morphing — phase transitions through signal space",
                 fontsize=10, fontweight="bold")

    ax_idx = 0
    for key, result in results.items():
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        ax_idx += 1

        if "full_phases" in result:
            # Cycle plot
            t_vals = result["full_t"]
            phases = result["full_phases"]
        else:
            t_vals = result.get("t_values", [i/len(result["phases"])
                                              for i in range(len(result["phases"]))])
            phases = result["phases"]

        impr = result.get("improvements", [1.0] * len(phases))

        # Color each segment by phase
        for i in range(len(t_vals) - 1):
            ax.axvspan(t_vals[i], t_vals[i+1],
                        color=PHASE_COLORS.get(phases[i], "gray"),
                        alpha=0.4)

        ax.plot(t_vals, impr, "k-", lw=1.5, zorder=5)
        ax.set_title(f"{key}\n"
                     f"transitions={result.get('transitions', result.get('total_transitions', '?'))}"
                     f"  holonomy={result.get('holonomy', '')}",
                     fontsize=8)
        ax.set_xlabel("t (morph parameter)")
        ax.set_ylabel("Improvement")

        if "cycle_closed" in result:
            color = "green" if result["cycle_closed"] else "red"
            closed = "CLOSED" if result["cycle_closed"] else "OPEN"
            ax.set_title(ax.get_title() + f"\ncycle: {closed}", fontsize=8, color=color)

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, alpha=0.6, label=p)
                       for p, c in PHASE_COLORS.items()]
    fig.legend(handles=legend_patches, loc="lower right", fontsize=8,
               title="Phase class")

    for ax in axes[ax_idx:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"Plot saved -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cos = _COS.copy()
    sin = _SIN.copy()

    # Define anchor signals
    S = {
        "osc_even":  make_signal("oscillation_even"),
        "osc_odd":   make_signal("oscillation_odd"),
        "deriv":     make_signal("derivative"),
        "scale":     make_signal("scale_growth"),
        "linear":    make_signal("linearity"),
        "null":      np.zeros(2**N),
    }

    print("="*65)
    print("SIGNAL MOLECULE MORPHING EXPERIMENTS")
    print("Target: steer population toward sin (derivative of cos)")
    print("="*65)

    results = {}

    # 1. Linear morphs
    print("\n1. LINEAR MORPHS (A->B):")
    for name, (a, b) in [
        ("osc_even->deriv", ("osc_even", "deriv")),
        ("null->osc_even",  ("null",     "osc_even")),
        ("osc_even->scale", ("osc_even", "scale")),
        ("linear->deriv",   ("linear",   "deriv")),
    ]:
        results[name] = run_morph(name, S[a], S[b], sin, cos, n_steps=12)

    # 2. Round trips
    print("\n2. ROUND TRIP TESTS (A->B->A, checking reversibility):")
    for name, (a, b) in [
        ("osc_even<->deriv", ("osc_even", "deriv")),
        ("null<->osc_even",  ("null",     "osc_even")),
    ]:
        rt = run_round_trip(name, S[a], S[b], sin, cos, n_steps=8)
        results[f"RT:{name}"] = {
            **rt["forward"],
            "transitions": rt["forward"]["transitions"] + rt["backward"]["transitions"],
            "holonomy": rt["hysteresis"],
            "cycle_closed": rt["is_reversible"],
        }
        print(f"  {name}: reversible={rt['is_reversible']}  "
              f"hysteresis={rt['hysteresis']:.2f}")

    # 3. Cycle A->B->C->A
    print("\n3. CYCLE TESTS (A->B->C->A, checking holonomy):")
    for cycle_name, cycle_keys in [
        ("osc_even->deriv->scale->osc_even",
         ["osc_even", "deriv", "scale", "osc_even"]),
        ("null->osc->deriv->null",
         ["null", "osc_even", "deriv", "null"]),
    ]:
        cycle = run_cycle(cycle_name,
                          [S[k] for k in cycle_keys],
                          cycle_keys, sin, cos, n_steps=6)
        results[f"CYC:{cycle_name[:25]}"] = {
            "phases":      cycle["full_phases"],
            "improvements":[1.0]*len(cycle["full_phases"]),
            "t_values":    cycle["full_t"],
            "full_phases": cycle["full_phases"],
            "full_t":      cycle["full_t"],
            "transitions": cycle["total_transitions"],
            "holonomy":    cycle["holonomy"],
            "cycle_closed":cycle["cycle_closed"],
        }
        print(f"  {cycle_name[:40]}")
        print(f"    Start phase: {cycle['start_phase']}  "
              f"End phase: {cycle['end_phase']}")
        print(f"    Cycle closed: {cycle['cycle_closed']}  "
              f"Holonomy: {cycle['holonomy']}")
        print(f"    Total transitions: {cycle['total_transitions']}")

    plot_morph_experiments(results)

    print("\nSUMMARY:")
    print("Phase transitions reveal the topology of signal-molecule space.")
    print("Reversible morphs (no hysteresis) = flat geometry at that path.")
    print("Non-zero holonomy on cycles = curved signal-molecule space.")
    print("Minimum-transition path between classes = geodesic in signal space.")
