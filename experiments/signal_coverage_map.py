"""
Coverage-guided signal molecule exploration.

Starts from a null signal and discovers the phase transitions in signal
molecule space that correspond to different algebraic relationship classes.

Analogous to AFL discovering JPEG structure from "HELLO":
  - Start: null signal (no Walsh coefficients)
  - Coverage: which algebraic behavior class does this signal trigger?
  - Mutation: flip/increment Walsh coefficient positions
  - Phase jump: when a mutation causes a new behavior class

The coverage bitmap maps the TOPOLOGY of signal-molecule space:
  - Which signals are "near" each other (one mutation apart)?
  - What are the phase transitions (class boundaries)?
  - What is the minimum path from null to each algebraic class?

This is NOT hypothesis testing — it is STRUCTURE DISCOVERY.
We are finding the natural classes in signal-molecule space without
knowing in advance what they are.
"""

import math
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from collections import defaultdict

from toolkit.euler_relay import _COS, _SIN, N
from toolkit.walsh import main_effects
from toolkit.signal_fuzzer import (FuzzerState, CoverageResult, CoverageBitmap,
                                    mutate_signal, generate_covering_array,
                                    covering_array_to_signals, afl_power_schedule)


# ── Real coevolution evaluator ────────────────────────────────────────────────

def make_real_evaluator(fn_a_target: np.ndarray,
                         fn_b_start: np.ndarray,
                         n_gens: int = 40,
                         pop_size: int = 30):
    """
    Real coevolution evaluator: run n_gens of evolution with injected signal.
    Returns behavior class based on actual convergence behavior.
    """
    rng = np.random.default_rng(99)

    def evaluate(signal: np.ndarray) -> CoverageResult:
        pop_a = rng.standard_normal((pop_size, N)) * 0.2 + fn_b_start
        me = main_effects(signal, N)
        me = np.abs(me) / (np.abs(me).max() + 1e-10)
        mut_std = 0.04 * (1.0 + 0.35 * me)

        def fit(c):
            return -float(np.linalg.norm(c - fn_a_target)**2)

        init_res = float(np.linalg.norm(pop_a[0] - fn_a_target))
        traj = []
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
            traj.append(float(-fits.max()))

        final_res = traj[-1]
        improvement = init_res - final_res

        # Phase classification based on trajectory shape
        if improvement < 0.5:
            phase = "null_no_effect"
        elif improvement > 4.0:
            phase = "strong_agonist"
        elif improvement > 2.5:
            phase = "moderate_agonist"
        elif improvement < 0.0:
            phase = "antagonist"
        else:
            phase = "weak_effect"

        failure = "DEGENERATE" if improvement < 0.1 else None

        return CoverageResult(
            signal_name="",
            signal=signal,
            algebraic_class=phase,
            final_residual=final_res,
            convergence_gen=None if failure else (
                next((i for i, r in enumerate(traj) if r < 0.5 * init_res), None)
            ),
            is_new_coverage=False,
            failure_mode=failure,
        )

    return evaluate


def run_coverage_map(n_rounds: int = 8, n_genes: int = N, verbose: bool = True):
    """
    Run coverage-guided exploration starting from null signal.
    Discover phase transitions in signal molecule space.
    """
    cos = _COS.copy()
    sin = _SIN.copy()

    evaluator = make_real_evaluator(sin, cos, n_gens=40, pop_size=30)
    rng = np.random.default_rng(42)
    bitmap = CoverageBitmap()
    corpus = [np.zeros(2**n_genes)]   # start from null
    corpus_names = ["null"]
    corpus_scores = [0.0]
    all_results = []
    phase_paths = defaultdict(list)   # phase -> mutation path that found it

    if verbose:
        print("="*60)
        print("COVERAGE-GUIDED SIGNAL EXPLORATION")
        print("Starting from null signal, discovering phase transitions")
        print("="*60)

    # Phase 1: CIT for systematic baseline
    ca = generate_covering_array(n_genes, t=2)
    cit_signals = covering_array_to_signals(ca, n_genes)

    for i, sig in enumerate(cit_signals[:20]):
        result = evaluator(sig)
        result.signal_name = f"CIT_{i}"
        is_new = bitmap.record(result)
        all_results.append(result)
        if is_new:
            corpus.append(sig)
            corpus_names.append(f"CIT_{i}")
            corpus_scores.append(abs(result.final_residual))
            phase_paths[result.algebraic_class].append(f"CIT_row_{i}")
            if verbose:
                print(f"  NEW PHASE [CIT_{i}]: {result.algebraic_class}  "
                      f"residual={result.final_residual:.4f}")

    # Phase 2: AFL mutation from promising seeds
    strategies = ["bit_flip", "arith", "interesting", "random", "zero_out"]

    for rnd in range(n_rounds):
        energies = afl_power_schedule(corpus_scores, rnd)
        for idx, energy in enumerate(energies):
            seed = corpus[idx]
            for _ in range(max(1, energy)):
                strat = strategies[rng.integers(0, len(strategies))]
                mutant = mutate_signal(seed, n_genes, strat, rng)
                result = evaluator(mutant)
                result.signal_name = f"r{rnd}_{strat}"
                is_new = bitmap.record(result)
                all_results.append(result)

                if is_new:
                    corpus.append(mutant)
                    corpus_names.append(f"r{rnd}_{strat}")
                    corpus_scores.append(abs(result.final_residual))
                    # Record the mutation path that discovered this phase
                    phase_paths[result.algebraic_class].append(
                        f"mutated_from:{corpus_names[idx]} via:{strat}")
                    if verbose:
                        print(f"  NEW PHASE [{result.algebraic_class}] "
                              f"via {strat} from {corpus_names[idx]}  "
                              f"res={result.final_residual:.4f}")

    bm = bitmap.summary()

    if verbose:
        print(f"\nCoverage summary:")
        print(f"  Total evaluations: {bm['total_runs']}")
        print(f"  Phase classes found: {bm['classes_covered']}")
        print(f"  Failure classes: {bm['crash_classes']}")
        print(f"\nPhase discovery paths:")
        for phase, paths in sorted(phase_paths.items()):
            print(f"  {phase}: {paths[0]}")

    # Plot phase distribution
    class_counts = defaultdict(int)
    for r in all_results:
        class_counts[r.algebraic_class] += 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Coverage-guided signal molecule exploration\n"
                 "Discovering phase transitions from null signal",
                 fontsize=11, fontweight="bold")

    ax = axes[0]
    classes = list(class_counts.keys())
    counts  = [class_counts[c] for c in classes]
    colors  = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    ax.bar(range(len(classes)), counts, color=colors)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Signals triggering this phase")
    ax.set_title("Phase class distribution\n(how many signals per class)")

    ax = axes[1]
    residuals = [r.final_residual for r in all_results]
    ax.hist(residuals, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0.1, color="green", lw=1.5, ls="--", label="agonist threshold")
    ax.axvline(1.0, color="orange", lw=1.5, ls="--", label="weak threshold")
    ax.axvline(5.0, color="red", lw=1.5, ls="--", label="null threshold")
    ax.set_xlabel("Final residual")
    ax.set_ylabel("Number of signals")
    ax.set_title("Residual distribution\n(phase transitions visible as gaps)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("signal_coverage_map.png", dpi=140, bbox_inches="tight")
    plt.show()
    print("Plot saved -> signal_coverage_map.png")

    return all_results, phase_paths


if __name__ == "__main__":
    results, paths = run_coverage_map(n_rounds=6)
