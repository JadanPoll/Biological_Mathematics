"""
Signal molecule fuzzer — coverage-guided exploration of the Walsh signal space.

Borrows from:
  AFL (American Fuzzy Lop):     coverage-guided mutation + power schedule
  t-way CIT (covering arrays):  systematic pairwise/3-way coverage
  Greybox fuzzing:              instrumented feedback, not blind random
  CRISPR screens:               systematic knockdown of each position

The fuzzing framework mapped to our system:
  Input:            Walsh coefficient pattern (signal molecule)
  Target:           Coevolution dynamics (black box)
  Coverage:         Which algebraic relationships / failure modes are triggered
  Interesting input: Signal that triggers a new algebraic behavior class
  Crash:            Type 7 failure / degenerate collapse
  Corpus:           Signal library (optimal signals for known relationships)
  Power schedule:   More experiments for under-explored signals

Key efficiency insight from CIT:
  To guarantee every PAIR of Walsh positions is exercised:
    Naive: 2^N = 256 tests (for N=8)
    CIT t=2 covering array: ~50-100 tests (5× efficiency gain)

  For N=16 (expression tree parameters):
    Naive: 65536 tests
    CIT t=2: ~200 tests (300× efficiency gain)

AFL-style coverage bitmap for our system:
  - Each algebraic relationship type = one "branch" in the coverage bitmap
  - Failure mode = one "crash class"
  - Signal that triggers a new algebraic class = added to corpus
  - Signal that crashes = analyzed for Type 7 boundary
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


# ── Coverage bitmap ───────────────────────────────────────────────────────────

@dataclass
class CoverageResult:
    """What happened when we injected this signal."""
    signal_name:       str
    signal:            np.ndarray
    algebraic_class:   str          # inferred relationship class
    final_residual:    float
    convergence_gen:   Optional[int]
    is_new_coverage:   bool         # did this signal trigger a new behavior class?
    failure_mode:      Optional[str]


class CoverageBitmap:
    """
    Tracks which algebraic behavior classes have been discovered.
    Equivalent to AFL's edge coverage bitmap.
    Each 'edge' = one distinct algebraic behavior class triggered by a signal.
    """

    def __init__(self, residual_buckets: List[float] = None):
        # Residual buckets define behavior classes
        # (0-0.01), (0.01-0.1), (0.1-1.0), (1.0-10.0), (10.0+)
        self.residual_buckets = residual_buckets or [0.01, 0.1, 1.0, 10.0, float('inf')]
        self.covered_classes:  Set[str] = set()
        self.crash_classes:    Set[str] = set()
        self.total_runs:       int = 0

    def bucket(self, residual: float, failure_mode: Optional[str]) -> str:
        """Map a (residual, failure_mode) to a behavior class string."""
        if failure_mode:
            return f"FAIL:{failure_mode}"
        for i, threshold in enumerate(self.residual_buckets):
            if residual <= threshold:
                return f"RES_BUCKET_{i}"
        return "RES_BUCKET_OVERFLOW"

    def record(self, result: CoverageResult) -> bool:
        """Record a result. Returns True if new coverage was found."""
        self.total_runs += 1
        cls = self.bucket(result.final_residual, result.failure_mode)
        if result.failure_mode:
            self.crash_classes.add(cls)
        is_new = cls not in self.covered_classes
        self.covered_classes.add(cls)
        return is_new

    def summary(self) -> dict:
        return {
            "total_runs":     self.total_runs,
            "classes_covered": len(self.covered_classes),
            "crash_classes":  list(self.crash_classes),
            "all_classes":    list(self.covered_classes),
        }


# ── t-way covering array generator ───────────────────────────────────────────

def generate_covering_array(n_factors: int, t: int = 2,
                             n_levels: int = 2) -> np.ndarray:
    """
    Generate a t-way covering array for n_factors binary factors.
    Guarantees every combination of t factors takes all t-tuples of values.

    For n_factors=8, t=2: guarantees every pair (i,j) of Walsh positions
    has been tested with both positions on and both off.

    This is the systematic alternative to exhaustive 2^N testing.

    Returns: array of shape (n_rows, n_factors) with values in {0, 1}.
    """
    if n_levels != 2:
        raise NotImplementedError("Only binary factors supported")

    # Simple greedy construction (Turán-type)
    # More sophisticated: use IPOG algorithm for tighter arrays
    rows = []
    covered = set()

    all_t_tuples = []
    from itertools import combinations, product
    for cols in combinations(range(n_factors), t):
        for vals in product(range(n_levels), repeat=t):
            all_t_tuples.append((cols, vals))

    uncovered = set(range(len(all_t_tuples)))

    while uncovered:
        # Greedy: find the row that covers the most uncovered tuples
        best_row = None
        best_count = -1

        for _ in range(100):  # sample random candidates
            row = np.random.randint(0, n_levels, n_factors)
            count = 0
            for idx in list(uncovered)[:50]:  # sample subset for speed
                cols, vals = all_t_tuples[idx]
                if tuple(row[list(cols)]) == vals:
                    count += 1
            if count > best_count:
                best_count = count
                best_row = row.copy()

        rows.append(best_row)
        # Remove covered tuples
        new_uncovered = set()
        for idx in uncovered:
            cols, vals = all_t_tuples[idx]
            if tuple(best_row[list(cols)]) != vals:
                new_uncovered.add(idx)
        uncovered = new_uncovered

    return np.array(rows)


def covering_array_to_signals(covering_array: np.ndarray,
                                n_genes: int,
                                base_value: float = 1.0) -> List[np.ndarray]:
    """
    Convert a covering array (rows of 0/1 for each Walsh position)
    to a list of signal molecules.

    Each row = one signal. Position k active (=1) means W[2^k] = base_value.
    """
    signals = []
    sig_len = 2 ** n_genes
    main_effect_positions = [1 << k for k in range(n_genes)]

    for row in covering_array:
        sig = np.zeros(sig_len)
        for k, active in enumerate(row):
            if active and main_effect_positions[k] < sig_len:
                sig[main_effect_positions[k]] = base_value
        signals.append(sig)

    return signals


# ── AFL-style fuzzer ──────────────────────────────────────────────────────────

@dataclass
class FuzzerState:
    """State of the signal molecule fuzzer."""
    corpus:         List[np.ndarray] = field(default_factory=list)
    corpus_scores:  List[float] = field(default_factory=list)   # activity score
    corpus_names:   List[str]   = field(default_factory=list)
    bitmap:         CoverageBitmap = field(default_factory=CoverageBitmap)
    round_num:      int = 0
    total_mutations: int = 0


def mutate_signal(signal: np.ndarray, n_genes: int,
                   strategy: str = "random",
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Mutate a signal molecule to generate a new test case.

    AFL mutation strategies mapped to Walsh coefficient space:
      bit_flip:      flip one Walsh main-effect position (0 ↔ value)
      byte_flip:     flip two adjacent positions
      arith:         add/subtract small delta to one position
      interesting:   set position to one of several "interesting" values
      random:        random perturbation of one position
      splice:        mix two corpus signals

    The AFL insight: use ALL strategies, weighted by what's been productive.
    """
    sig = signal.copy()
    sig_len = len(sig)
    main_eff_pos = [1 << k for k in range(n_genes) if (1 << k) < sig_len]

    if rng is None:
        rng = np.random.default_rng()

    if strategy == "bit_flip":
        pos = rng.choice(main_eff_pos)
        sig[pos] = 0.0 if abs(sig[pos]) > 0.1 else 1.0

    elif strategy == "arith":
        pos = rng.choice(main_eff_pos)
        sig[pos] += rng.choice([-0.5, -0.25, 0.25, 0.5])

    elif strategy == "interesting":
        pos = rng.choice(main_eff_pos)
        sig[pos] = rng.choice([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

    elif strategy == "random":
        pos = rng.choice(main_eff_pos)
        sig[pos] = rng.uniform(-2.0, 2.0)

    elif strategy == "zero_out":
        # AFL deterministic: zero out one position (pharmacophore knockdown)
        pos = rng.choice(main_eff_pos)
        sig[pos] = 0.0

    elif strategy == "amplify":
        # Amplify one position
        pos = rng.choice(main_eff_pos)
        sig[pos] *= 2.0

    return sig


def afl_power_schedule(corpus_scores: List[float], round_num: int) -> List[int]:
    """
    Power schedule: assign 'energy' (number of mutations) to each corpus entry.
    AFL's AFLFast insight: give more energy to less-explored corpus entries.

    Higher energy = more mutations generated from this seed.
    """
    n = len(corpus_scores)
    if n == 0:
        return []

    # AFLFast: energy inversely proportional to how many times it's been selected
    # We approximate by using scores: low-activity seeds get more energy
    max_score = max(corpus_scores) + 1e-10
    energies = [max(1, int(5 * (1.0 - s / max_score) + 1)) for s in corpus_scores]
    return energies


def signal_fuzzing_campaign(
        evaluate_fn,                        # fn(signal) -> CoverageResult
        seed_signals: Dict[str, np.ndarray], # initial corpus
        n_genes: int = 8,
        max_rounds: int = 10,
        use_cit: bool = True,
        cit_t: int = 2,
        verbose: bool = True) -> FuzzerState:
    """
    Run a coverage-guided signal fuzzing campaign.

    Phase 1 (CIT): systematically cover all t-way combinations
    Phase 2 (AFL): mutate interesting seeds for deeper exploration

    Returns the fuzzer state with corpus, bitmap, and discovered behaviors.
    """
    rng = np.random.default_rng(42)
    state = FuzzerState(bitmap=CoverageBitmap())

    # Initialize corpus with seeds
    for name, sig in seed_signals.items():
        state.corpus.append(sig.copy())
        state.corpus_names.append(name)
        state.corpus_scores.append(0.0)  # unknown score

    if verbose:
        print(f"\n{'='*60}")
        print(f"SIGNAL FUZZING CAMPAIGN")
        print(f"Seeds: {len(state.corpus)}  CIT: {use_cit} (t={cit_t})")
        print(f"{'='*60}")

    # Phase 1: CIT covering array
    if use_cit:
        if verbose:
            print(f"\nPhase 1: t-way CIT (t={cit_t}, guarantees systematic coverage)")
        ca = generate_covering_array(n_genes, t=cit_t)
        cit_signals = covering_array_to_signals(ca, n_genes)

        for i, sig in enumerate(cit_signals[:30]):  # limit for speed
            result = evaluate_fn(sig)
            is_new = state.bitmap.record(result)
            if is_new:
                state.corpus.append(sig)
                state.corpus_names.append(f"CIT_row_{i}")
                state.corpus_scores.append(abs(result.final_residual))
                if verbose:
                    print(f"  CIT row {i}: NEW coverage class '{result.algebraic_class}'")

    # Phase 2: AFL-style mutation campaign
    if verbose:
        print(f"\nPhase 2: AFL-style mutation ({max_rounds} rounds)")

    strategies = ["bit_flip", "arith", "interesting", "random",
                  "zero_out", "amplify"]

    for round_num in range(max_rounds):
        state.round_num = round_num
        energies = afl_power_schedule(state.corpus_scores, round_num)

        for idx, energy in enumerate(energies):
            seed = state.corpus[idx]
            for _ in range(energy):
                strategy = strategies[rng.integers(0, len(strategies))]
                mutant = mutate_signal(seed, n_genes, strategy, rng)
                result = evaluate_fn(mutant)
                state.total_mutations += 1

                is_new = state.bitmap.record(result)
                if is_new:
                    state.corpus.append(mutant)
                    state.corpus_names.append(
                        f"r{round_num}_m{state.total_mutations}_{strategy}")
                    state.corpus_scores.append(abs(result.final_residual))

    if verbose:
        bm = state.bitmap.summary()
        print(f"\nCampaign complete:")
        print(f"  Total runs:          {bm['total_runs']}")
        print(f"  Behavior classes:    {bm['classes_covered']}")
        print(f"  Crash classes:       {len(bm['crash_classes'])}")
        print(f"  Corpus size:         {len(state.corpus)}")
        print(f"  Unique signals:      {state.total_mutations}")

    return state


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from toolkit.euler_relay import _COS, _SIN, N
    from toolkit.walsh import main_effects
    from toolkit.signal_injection import optimal_signal_derivative, optimal_signal_identical

    # Simplified evaluate function for demo
    def quick_evaluate(signal: np.ndarray) -> CoverageResult:
        """Evaluate signal by measuring how it biases mutation variance."""
        me = main_effects(signal, N)
        me_norm = np.abs(me) / (np.abs(me).max() + 1e-10)

        # Classify by dominant Walsh structure
        even_energy = float(np.sum(me_norm[0::2]**2))
        odd_energy  = float(np.sum(me_norm[1::2]**2))
        total       = even_energy + odd_energy + 1e-8

        even_frac = even_energy / total
        mag = float(np.abs(me).max())

        if mag < 0.05:
            algebraic_class = "null"
            residual = 5.0
            failure = "DEGENERATE"
        elif even_frac > 0.8:
            algebraic_class = "even_dominated"
            residual = 0.05
            failure = None
        elif even_frac < 0.2:
            algebraic_class = "odd_dominated"
            residual = 0.08
            failure = None
        else:
            algebraic_class = "mixed"
            residual = 0.5
            failure = None

        return CoverageResult(
            signal_name=f"sig_{hash(signal.tobytes()) % 10000}",
            signal=signal,
            algebraic_class=algebraic_class,
            final_residual=residual,
            convergence_gen=None if failure else 50,
            is_new_coverage=False,
            failure_mode=failure,
        )

    seed_signals = {
        "null":        np.zeros(2**N),
        "derivative":  optimal_signal_derivative(_COS, "walsh_solution"),
        "identical":   optimal_signal_identical(_COS),
    }

    state = signal_fuzzing_campaign(
        evaluate_fn=quick_evaluate,
        seed_signals=seed_signals,
        n_genes=N,
        max_rounds=5,
        use_cit=True,
        cit_t=2,
        verbose=True,
    )

    print(f"\nCovering array size (t=2, N=8): ~{len(generate_covering_array(N, t=2))} rows")
    print(f"Exhaustive (2^8): 256 rows")
    print(f"Efficiency gain: ~{256 / len(generate_covering_array(N, t=2)):.1f}x")
