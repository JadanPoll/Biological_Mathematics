"""
Algebraic Origami Explorer — autonomous taxonomy filling.

Fourier's motivation: solve a specific problem (heat flow), discover a
general principle (decomposition into sinusoids) as a side effect.

Rothemund's motivation: make a smiley face out of DNA (2006), discover a
general principle (any 2D shape is foldable) as a side effect.

Our motivation: probe whether algebraic morphing reveals structure.

This module runs autonomously, storing outputs that are "interesting" by
entropy, curvature, and stress criteria — without knowing in advance what
interesting means.

KEY EXPERIMENTS this enables:

1. SIMPLE ALGEBRAIC MORPHING (linear chains)
   Linear functions form a VECTOR SPACE (convex set).
   Any convex combination of linear functions is linear.
   => The morph L1->L2->L3->L4 should be a straight line with NO phase transitions.
   Prediction: much faster convergence than rotation morphing (30-50 gens vs 300+).

2. UNMORPHING (complex -> simple)
   Given a complex function, can we find the simplest algebraic structure?
   This is LOSSY COMPRESSION — the inverse of the relay chain.
   Project to low-frequency Walsh terms first (Fourier warmup).
   Then observe what algebraic class the projection belongs to.

3. TESSELLATED MORPHING (AM modulation structure)
   Low-frequency Walsh terms (W[1],W[2],W[4]) = carrier wave
   High-frequency Walsh terms (W[32],W[64],W[128]) = modulation
   Can a signal encode structure at BOTH frequency scales independently?
   DNA analog: primary sequence (high-frequency) on double helix (low-frequency)
   on supercoiling (very-low-frequency).

4. REPLAY AND MUTATION
   Record a successful joint signal from a converged run.
   Replay it to a NEW population.
   Observe: determinism (same convergence) vs. ambiguity (probability distribution).
   Cross two replays (linearity x oscillation): what joint structure emerges?

5. SIGNAL ROBUSTNESS
   Same signal -> different receiver types.
   Does the linearity signal steer a linear population? (No — receiver already there)
   Does the linearity signal steer an oscillatory population? (Yes — mismatch = information)
   Receptor specificity: the signal's effectiveness depends on the mismatch.

6. FOURIER WARMUP -> PULLBACK (P-code lifting)
   Warm up in low-frequency Fourier space (cheap, 3-4 terms, 20-30 gens)
   Lock the carrier wave (the coarse algebraic type)
   Then pull back to full Walsh space for fine structure
   Prediction: dramatically faster convergence than cold-start full Walsh.

AUTOMATED TAXONOMY FILLING CRITERIA:
   Run continuously. Store if:
   - Phase entropy > threshold (many different phase classes reached = interesting topology)
   - Curvature variance > threshold (non-uniform morphing = curved geometry)
   - Stress gradient > threshold (sharp phase transition = boundary of algebraic class)
   - Surprise score > threshold (outcome differs significantly from prediction)
"""

import math
import time
import json
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from toolkit.euler_relay import _COS, _SIN, N
from toolkit.walsh import wht, main_effects
from toolkit.signal_injection import make_signal


# ── Interest metrics ──────────────────────────────────────────────────────────

def phase_entropy(improvements: List[float], n_buckets: int = 5) -> float:
    """
    Shannon entropy of the phase distribution along a morph path.
    High entropy = many different phase classes visited = topologically rich.
    Low entropy = stuck in one phase = boring morph.
    Fourier: heat flow explores many temperature states = high entropy initially.
    """
    if not improvements:
        return 0.0
    hist, _ = np.histogram(improvements, bins=n_buckets, density=False)
    total = sum(hist)
    if total == 0:
        return 0.0
    probs = [h/total for h in hist if h > 0]
    return float(-sum(p * math.log(p) for p in probs))


def curvature_variance(improvements: List[float]) -> float:
    """
    Variance of the second derivative of the improvement curve.
    High variance = non-uniform curvature = interesting geometry.
    Low variance = straight-line morph = flat space.
    DNA: the helix has CONSTANT curvature, so variance = 0.
    A twisted helix on a helix has VARIABLE curvature = high variance.
    """
    if len(improvements) < 3:
        return 0.0
    arr = np.array(improvements)
    d2 = np.diff(arr, n=2)
    return float(np.var(d2))


def stress_gradient(improvements: List[float]) -> float:
    """
    Maximum absolute change in improvement between adjacent steps.
    High = sharp phase transition (interesting boundary).
    Low = smooth morph (no sharp features).
    """
    if len(improvements) < 2:
        return 0.0
    diffs = [abs(improvements[i+1] - improvements[i]) for i in range(len(improvements)-1)]
    return float(max(diffs))


def surprise_score(improvements: List[float], expected_monotone: bool = True) -> float:
    """
    How much does the actual improvement curve deviate from naive expectation?
    If expected_monotone=True: expected = linear increase from start to max.
    Surprise = deviation from this naive model.
    """
    if len(improvements) < 2:
        return 0.0
    arr = np.array(improvements)
    n = len(arr)
    naive = np.linspace(arr[0], arr[-1], n)
    return float(np.mean(np.abs(arr - naive)))


def interest_score(improvements: List[float], weights: dict = None) -> float:
    """
    Combined interest score for automated taxonomy filing.
    High = worth storing and investigating further.
    """
    w = weights or {"entropy": 1.0, "curvature": 2.0,
                    "stress": 1.5, "surprise": 1.0}
    return (w["entropy"]   * phase_entropy(improvements) +
            w["curvature"] * curvature_variance(improvements) * 10 +
            w["stress"]    * stress_gradient(improvements) +
            w["surprise"]  * surprise_score(improvements))


def topological_protection_index(forward_impr: float,
                                   reverse_impr: float) -> float:
    """
    Asymmetry between forward morphing (A->B) and reverse (B->A).

    High TPI = topologically protected complex state:
      - Easy to build (forward_impr high)
      - Hard to decompose (reverse_impr low)
      - Like a prime product: easy to multiply, hard to factor

    TPI = 1: symmetric, no protection (both directions equally hard/easy)
    TPI > 2: asymmetric, complex state has some protection
    TPI >> 1: strongly protected (like RSA — the activation barrier is real)
    TPI < 1: reverse is EASIER than forward (downhill decomposition)

    Connection to physics:
      TPI encodes the activation energy asymmetry.
      Chemical reactions: TPI = exp(ΔG_activation / kT)
      Prime factorization: TPI ~ exp(sqrt(n)) for an n-bit number
      Algebraic relationships: TPI ~ exp(algebraic_depth)
    """
    if reverse_impr < 1e-3:
        return 10.0   # perfectly protected (reverse makes no progress)
    return float(min(forward_impr / (reverse_impr + 1e-3), 10.0))


# ── Function library ──────────────────────────────────────────────────────────

def build_function_library() -> Dict[str, np.ndarray]:
    """
    The periodic table of algebraic structures to explore.
    Organized by algebraic type and complexity.
    """
    X = np.linspace(-1.5, 1.5, 120)

    lib = {}

    # Linear family (vector space — should morph trivially fast)
    for slope in [0.5, 1.0, 2.0, -1.0]:
        for intercept in [0.0, 0.5, -0.5]:
            c = np.zeros(N)
            c[0] = intercept; c[1] = slope
            lib[f"linear_{slope}_{intercept}"] = c

    # Oscillatory family
    lib["cos_1x"] = np.array([(-1.)**(k//2) if k%2==0 else 0. for k in range(N)])
    lib["sin_1x"] = np.array([(-1.)**((k-1)//2) if k%2==1 else 0. for k in range(N)])

    # Phase-shifted oscillation
    for phi in [30, 45, 60, 90, 120, 135, 150]:
        phi_r = math.radians(phi)
        c = math.cos(phi_r) * lib["cos_1x"] - math.sin(phi_r) * lib["sin_1x"]
        lib[f"cos_phase_{phi}"] = c

    # Polynomial family
    for deg in [2, 3, 4]:
        c = np.zeros(N); c[deg] = 1.0
        lib[f"poly_deg{deg}"] = c

    # Mixed: linear + oscillatory
    for lin_w, osc_w in [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]:
        c = lin_w * np.array([0,1,0,0,0,0,0,0], dtype=float) + osc_w * lib["cos_1x"]
        c /= (np.linalg.norm(c) + 1e-10) * 2
        lib[f"mixed_lin{lin_w}_osc{osc_w}"] = c

    return lib


# ── Morph with interest tracking ──────────────────────────────────────────────

def _single_direction_morph(fn_start: np.ndarray, fn_end: np.ndarray,
                              signal: np.ndarray, n_steps: int, n_gens: int,
                              pop_size: int, rng_seed: int) -> Dict:
    """Internal: one-direction morph. Used by morph_and_score."""
    rng = np.random.default_rng(rng_seed)
    me = main_effects(signal, N)
    me = np.abs(me) / (np.abs(me).max() + 1e-10)
    mut_std = 0.04 * (1.0 + 0.40 * me)
    improvements = []
    for step in range(n_steps + 1):
        t = step / n_steps
        target = (1 - t) * fn_start + t * fn_end
        init_res = float(np.linalg.norm(fn_start - target))
        pop = rng.standard_normal((pop_size, N)) * 0.2 + fn_start
        for _ in range(n_gens):
            fits = np.array([-float(np.linalg.norm(ind - target)**2) for ind in pop])
            elite = pop[np.argsort(fits)[-2:]].copy()
            new_pop = np.empty_like(pop)
            new_pop[:2] = elite
            for j in range(2, pop_size):
                idx = rng.choice(pop_size, 3, replace=False)
                p1 = pop[idx[np.argmax(fits[idx])]]
                new_pop[j] = p1 + rng.standard_normal(N) * mut_std
            pop = new_pop
        best = pop[np.argmax([-float(np.linalg.norm(ind - target)**2) for ind in pop])]
        final_res = float(np.linalg.norm(best - target))
        improvements.append(max(0.0, init_res - final_res))
    return improvements


def morph_and_score(fn_start: np.ndarray, fn_end: np.ndarray,
                     signal: np.ndarray,
                     n_steps: int = 12, n_gens: int = 30,
                     pop_size: int = 20, rng_seed: int = 42) -> Dict:
    """
    Morph fn_start toward fn_end AND reverse (fn_end toward fn_start).
    Records:
      - forward improvements (A->B): building the complex state
      - reverse improvements (B->A): decomposing back (the activation energy)
      - topological_protection_index: forward/reverse asymmetry

    High TPI = topologically protected state (easy to build, hard to decompose).
    Low TPI = symmetric (equally easy/hard in both directions).
    TPI < 1 = naturally decomposes (thermodynamically downhill in reverse).

    Analogy: TPI ~ 1 for scalar multiple (symmetric, no protection).
             TPI >> 1 for derivative relationship (non-trivially stable).
             TPI -> inf for Type 7 (perfectly protected against reverse).
    """
    # Forward: A -> B (building the complex state)
    fwd = _single_direction_morph(fn_start, fn_end, signal,
                                    n_steps, n_gens, pop_size, rng_seed)
    # Reverse: B -> A (decomposing — the activation energy barrier)
    rev = _single_direction_morph(fn_end, fn_start, signal,
                                    n_steps, n_gens, pop_size, rng_seed + 7777)

    fwd_total = float(sum(fwd))
    rev_total = float(sum(rev))
    tpi = topological_protection_index(fwd_total, rev_total)
    base_score = interest_score(fwd)
    topo_bonus = min(tpi - 1.0, 3.0) * 0.5 if tpi > 1.5 else 0.0

    return {
        "improvements":                  fwd,
        "interest_score":                base_score + topo_bonus,
        "interest_score_base":           base_score,
        "topo_bonus":                    topo_bonus,
        "phase_entropy":                 phase_entropy(fwd),
        "curvature_var":                 curvature_variance(fwd),
        "stress_gradient":               stress_gradient(fwd),
        "surprise":                      surprise_score(fwd),
        # Topological protection — the new key metrics
        "reverse_improvements":          rev,
        "forward_total":                 fwd_total,
        "reverse_total":                 rev_total,
        "topological_protection_index":  tpi,
        "is_topologically_protected":    tpi > 2.0,
        "activation_energy_asymmetry":   fwd_total - rev_total,
    }


# ── Fourier warmup architecture ───────────────────────────────────────────────

def fourier_warmup(fn_start: np.ndarray, fn_end: np.ndarray,
                    n_warmup_terms: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fourier warmup: project both functions onto the low-frequency basis,
    compute the warmup signal, then use it to initialize the full morph.

    Low-frequency terms (k=0,1,2): the carrier wave, algebraic type
    High-frequency terms (k=3..7): fine structure, riding on carrier

    Step 1: Extract low-frequency skeleton of fn_start and fn_end
    Step 2: Compute what the carrier wave morph looks like (cheap)
    Step 3: Use this as initialization signal for the full morph

    This is P-code lifting: lift to IR (low-freq), compute direction,
    pull back to full representation with initialized carrier.
    """
    # Low-frequency projection (keep only first n_warmup_terms)
    fn_start_lf = fn_start.copy(); fn_start_lf[n_warmup_terms:] = 0.0
    fn_end_lf   = fn_end.copy();   fn_end_lf[n_warmup_terms:] = 0.0

    # The warmup signal: encode the low-frequency direction
    direction_lf = fn_end_lf - fn_start_lf
    SIG_LEN = 2 ** N
    warmup_sig = np.zeros(SIG_LEN)
    warmup_sig[:N] = direction_lf
    warmup_sig = wht(warmup_sig)

    # The warm start for the full morph: fn_start with low-freq corrected
    warm_start = fn_start.copy()
    warm_start[:n_warmup_terms] = fn_end[:n_warmup_terms]  # pre-tune carrier

    return warmup_sig, warm_start


# ── Automated explorer ────────────────────────────────────────────────────────

@dataclass
class ExplorationRecord:
    """One entry in the automated taxonomy."""
    start_name:    str
    end_name:      str
    signal_name:   str
    interest_score: float
    improvements:  List[float]
    metrics:       dict
    timestamp:     float = field(default_factory=time.time)
    is_interesting: bool = False


def automated_explorer(run_time_seconds: int = 300,
                        interest_threshold: float = 0.5,
                        output_file: str = "algebraic_origami_taxonomy.json",
                        verbose: bool = True) -> List[ExplorationRecord]:
    """
    Run the automated algebraic origami explorer.

    Continuously generates morph experiments from the function library,
    scores them for interest, and stores interesting results.

    The "periodic table filling" strategy:
    - Start from null signals (minimal structure)
    - Systematically vary: start function, end function, signal type
    - Record ONLY outputs with high interest score
    - The accumulated records ARE the taxonomy

    Run for run_time_seconds (default 5 minutes for demo, run for days).
    """
    functions = build_function_library()
    fn_names  = list(functions.keys())
    signal_designs = {
        "null":          np.zeros(2**N),
        "osc_even":      make_signal("oscillation_even"),
        "osc_odd":       make_signal("oscillation_odd"),
        "derivative":    make_signal("derivative"),
        "linearity":     make_signal("linearity"),
        "scale_growth":  make_signal("scale_growth"),
        "uniform":       make_signal("uniform"),
    }

    records   = []
    n_total   = 0
    n_stored  = 0
    start_t   = time.time()
    rng       = np.random.default_rng(int(start_t))

    if verbose:
        print(f"\n{'='*60}")
        print(f"ALGEBRAIC ORIGAMI EXPLORER")
        print(f"Running for {run_time_seconds}s, storing if interest > {interest_threshold}")
        print(f"Function library: {len(fn_names)} functions")
        print(f"Signal designs:   {len(signal_designs)} types")
        print(f"{'='*60}")

    while time.time() - start_t < run_time_seconds:
        # Random pair from library
        s_name = rng.choice(fn_names)
        e_name = rng.choice(fn_names)
        if s_name == e_name:
            continue
        sig_name = rng.choice(list(signal_designs.keys()))

        fn_s = functions[s_name]
        fn_e = functions[e_name]
        sig  = signal_designs[sig_name]

        result = morph_and_score(fn_s, fn_e, sig,
                                  n_steps=8, n_gens=25,
                                  rng_seed=int(time.time() * 1000) % 10000)
        n_total += 1

        is_interesting = result["interest_score"] > interest_threshold
        if is_interesting:
            n_stored += 1
            record = ExplorationRecord(
                start_name=s_name, end_name=e_name,
                signal_name=sig_name,
                interest_score=result["interest_score"],
                improvements=result["improvements"],
                metrics={k: v for k, v in result.items()
                          if k != "improvements"},
                is_interesting=True,
            )
            records.append(record)

            if verbose:
                tpi = result.get("topological_protection_index", 1.0)
                tpi_flag = " [PROTECTED]" if tpi > 2.0 else ""
                print(f"  [{n_total:4d}] STORED: {s_name[:15]}->{e_name[:15]} "
                      f"via {sig_name:<12} "
                      f"score={result['interest_score']:.3f}  "
                      f"TPI={tpi:.2f}{tpi_flag}")

        # Progress update every 20 experiments
        if n_total % 20 == 0 and verbose:
            elapsed = time.time() - start_t
            rate = n_total / elapsed
            print(f"\n  Progress: {n_total} experiments  {n_stored} stored  "
                  f"{elapsed:.0f}s elapsed  {rate:.1f}/s  "
                  f"[{n_stored/max(1,n_total)*100:.1f}% interesting]")

    elapsed = time.time() - start_t

    # Save taxonomy to JSON — now includes topological protection data
    taxonomy = [
        {"start": r.start_name, "end": r.end_name, "signal": r.signal_name,
         "score": r.interest_score,
         "entropy": r.metrics.get("phase_entropy", 0),
         "curvature": r.metrics.get("curvature_var", 0),
         "stress": r.metrics.get("stress_gradient", 0),
         "improvements": r.improvements,
         # NEW: topological protection fields (key for neural net training)
         "topological_protection_index": r.metrics.get("topological_protection_index", 1.0),
         "is_topologically_protected":   r.metrics.get("is_topologically_protected", False),
         "forward_total":                r.metrics.get("forward_total", 0),
         "reverse_total":                r.metrics.get("reverse_total", 0),
         "activation_energy_asymmetry":  r.metrics.get("activation_energy_asymmetry", 0),
         }
        for r in records
    ]
    Path(output_file).write_text(json.dumps(taxonomy, indent=2))

    if verbose:
        print(f"\n{'='*60}")
        print(f"EXPLORATION COMPLETE")
        print(f"  Total experiments: {n_total}")
        print(f"  Interesting (stored): {n_stored} ({n_stored/max(1,n_total)*100:.1f}%)")
        print(f"  Rate: {n_total/elapsed:.1f} experiments/second")
        print(f"  Taxonomy saved to: {output_file}")
        print(f"\nTOP 5 MOST INTERESTING:")
        top = sorted(records, key=lambda r: -r.interest_score)[:5]
        for r in top:
            print(f"  {r.start_name[:12]}->{r.end_name[:12]} "
                  f"via {r.signal_name}: score={r.interest_score:.3f}")

    # Plot if we have results
    if records:
        _plot_taxonomy(records)

    return records


def _plot_taxonomy(records: List[ExplorationRecord]):
    """Plot the discovered taxonomy as a scatter of interesting morphs."""
    entropies  = [r.metrics.get("phase_entropy", 0) for r in records]
    curvatures = [r.metrics.get("curvature_var", 0) for r in records]
    stresses   = [r.metrics.get("stress_gradient", 0) for r in records]
    scores     = [r.interest_score for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Algebraic Origami Explorer — Interesting Morphs",
                 fontsize=11, fontweight="bold")

    ax = axes[0]
    sc = ax.scatter(entropies, curvatures, c=scores, cmap="plasma",
                     s=60, alpha=0.8)
    ax.set_xlabel("Phase entropy (variety of states)")
    ax.set_ylabel("Curvature variance (non-uniform geometry)")
    ax.set_title("Phase entropy vs curvature\n(top-right = most interesting)")
    plt.colorbar(sc, ax=ax, label="Interest score")

    ax = axes[1]
    # Show improvement trajectories of top 5
    top5 = sorted(records, key=lambda r: -r.interest_score)[:5]
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    for rec, color in zip(top5, colors):
        ax.plot(rec.improvements, color=color, lw=1.5,
                label=f"{rec.start_name[:8]}->{rec.end_name[:8]}")
    ax.set_xlabel("Morph step")
    ax.set_ylabel("Improvement")
    ax.set_title("Top 5 improvement trajectories")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("algebraic_origami_taxonomy.png", dpi=140, bbox_inches="tight")
    plt.show()
    print("Plot saved -> algebraic_origami_taxonomy.png")


if __name__ == "__main__":
    import sys
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    print(f"Running algebraic origami explorer for {seconds} seconds...")
    print("For a full taxonomy run: python -m experiments.algebraic_origami_explorer 86400")
    print("(24 hours = 86400 seconds)")
    print()

    records = automated_explorer(
        run_time_seconds=seconds,
        interest_threshold=0.4,
        output_file="algebraic_origami_taxonomy.json",
    )
