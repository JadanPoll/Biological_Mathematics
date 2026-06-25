# Biological Mathematics — Codebase Onboarding for LLM Sessions

## The Research Question (read this first)

Does there exist, for any two mathematically related objects, a smooth geometric
path through some space that makes each analytically tractable in terms of the
other? Can we detect whether such a path exists by measuring the "stress" of
attempting to morph one object toward the other?

This is NOT about solving hard conjectures. It is about building an empirical
instrument sensitive enough to detect that a mapping exists, even when no human
can follow the path analytically. The Mendel notebook analogy: record everything,
including all failures with precise failure modes. The explanation comes later.

## State of the Project

The instrument is built. The calibration is in progress. The hypothesis matrix
has 2520 testable experiments. The automated explorer runs at 22 experiments/sec.

**The single most important near-term action:** implement FABRIK relay chain.
FABRIK is O(n) per iteration, 5 iterations to convergence, no hyperparameters.
See toolkit/neb_relay.py for the NEB version (already implemented).
FABRIK reference: Aristidou & Lasenby 2011 (andreasaristidou.com/FABRIK.pdf).

---

## What Actually Works (do not rebuild these)

### The Gold Standards (max_error = 0 or analytical proof)
- `toolkit/complex_relay.py`: exact Euler rotation path in 1D complex space
- `toolkit/walsh.py`: WHT with involutory property verified to machine precision
- `toolkit/stress_metrics.py` (jsd, jsd_metric): Jensen-Shannon divergence,
  proper metric, all 4 analytical properties confirmed
- `toolkit/improved_metrics.py` (exact_wasserstein): Hungarian algorithm,
  0% error, O(n^3), replaces sliced Wasserstein for small populations

### Validated Relay Chain
- `toolkit/neb_relay.py`: NEB with nudging + string reparameterization.
  Result: Wasserstein variance = 0.000000, equal step distances.
  Also contains ReservoirSignal class (ReSTIR-inspired signal molecule).

### Working Experiment Infrastructure
- `experiments/runner.py`: CLI runner. `python -m experiments.runner compare MM-T0-001`
- `experiments/base.py`: Experiment dataclass + run_experiment()
- `experiments/registry.py`: all experiments registered here
- `notebook/db.py`: SQLite Mendel notebook (auto-created on first run)

### Working Autonomous Explorer
- `experiments/algebraic_origami_explorer.py`: 22 experiments/sec autonomously.
  Run: `python -m experiments.algebraic_origami_explorer 86400` (24 hours)
  Stores interesting morphs to algebraic_origami_taxonomy.json

### Key Validated Tests
- `experiments/instrument_validation.py`: master validation report (run to check tool health)
- `experiments/geometric_richness_validation.py`: 4-case canonical validation
- `experiments/type7_test.py`: confirms Type 7 dimensional insufficiency
- `experiments/level2_analytical.py`: confirms joint fitness Q1+Q2

---

## What Doesn't Work / Known Failures

### BROKEN: Independent fitness + signal exchange
`ga.py` with independent fitness is NOT joint optimization. It is two separate
optimizations with a communication channel. The signal barely helps.
FIX: use joint_fitness.py functions + joint CMA-ES (experiments/joint_covariance.py).

### BROKEN: Alternating CMA-ES (miscoordination)
Optimizing A given B, then B given A alternately does NOT converge to joint optimum.
It converges to Nash equilibria that may be far from joint optimum.
FIX: use simultaneous CMA-ES on [A,B] jointly (experiments/joint_covariance.py).

### BROKEN: Closed-cycle relay chain for Euler experiment
The 4-step closed cycle (A->B->C->D->A) fails because the Euler path is OPEN
(cos to -cos), not closed. The cycle forces a geometrically inconsistent return step.
FIX: use open-chain D-brane model (toolkit/open_chain_relay.py).

### PARTIAL: Our Ollivier-Ricci approximation
Returns ~0.72 for ALL paths (true value for unit circle = 1.0). 28% underestimate.
Reliable for RELATIVE comparison only. Use natural scale + geodesic deviation instead.

### SUPERSEDED: rung0_coevo.py
The original prototype. Superseded by the full experiment infrastructure.
Kept for historical reference only. Do not use for new experiments.

### NOT YET IMPLEMENTED (high priority)
- FABRIK relay chain (O(n), 5 iterations, game animation IK algorithm)
- SLERP relay chain (exact geodesic for rotations, gold standard validator)
- PBD/XPBD constraint-based relay (no oscillation artifacts)
- Additive signal molecules (encode delta not state, animation industry insight)

---

## File Map

### toolkit/ — Core mathematical tools

toolkit/walsh.py
  WHT, main_effects(), kl_divergence_spectra(), spectral_class(), effective_dimensionality()
  STATUS: RIGOROUS. Do not reimplement.
  NOTE: never compose wht(wht(x)) without rescaling by n=2^N_GENES.

toolkit/signals.py
  WalshLandscape, WalshSolution, RawSparse, TrajectoryNgram signal molecule classes.
  get_signal(name) returns the right one.
  STATUS: all 4 implemented. TrajectoryNgram needs warmup (first WINDOW=4 calls are state signals).

toolkit/joint_fitness.py
  make_pair_fitness(relationship_type) -> f(ind_a, ind_b) for joint optimization.
  Relationships: "derivative", "scalar_multiple", "additive_shift", "identical", "discovery".
  CRITICAL: always use normalization (norm_weight > 0) or degenerate zero solution will dominate.
  STATUS: all implemented, Level 2 Q1 confirmed analytically.

toolkit/ga.py
  Coevolutionary GA with N-collaborations, pluggable signals, timing.
  WARNING: this uses INDEPENDENT FITNESS by default (populations chase fixed targets).
  For joint optimization, use experiments/joint_covariance.py instead.

toolkit/stress_metrics.py
  jsd(), jsd_metric() — RIGOROUS symmetric metric, replaces KL divergence everywhere.
  sliced_wasserstein() — PARTIAL (15-25% error). Use exact_wasserstein() for small pops.
  ollivier_ricci_edge(), relay_ricci_curvatures() — PARTIAL (28% underestimate).
  relay_stress_report() — unified stress summary.

toolkit/improved_metrics.py
  exact_wasserstein() — RIGOROUS. Hungarian algorithm. 0% error. O(n^3).
  benettin_lyapunov() — RIGOROUS for smooth systems. Sign reliable, abs values not.
  calibrated_homology() — PARTIAL. Materials science scale calibration.
  full_characterization_v2() — run all improved metrics on a path.

toolkit/complex_relay.py
  run_complex_relay(n_steps, ...) — exact Euler path in 1D complex space.
  max_error = 0.000000. Gold standard.
  from_complex(), to_complex() — conversion between complex and coefficient representations.

toolkit/euler_relay.py
  euler_intermediate(k, n_steps) — analytical true intermediate at step k.
  euler_step(c, delta_theta) — exact phase rotation in coefficient space.
  run_euler_relay() — full relay chain (use complex_relay for the exact version instead).

toolkit/neb_relay.py
  neb_force() — nudged force decomposition (path-perp potential + path-para spring).
  reparameterize_string() — equal arc-length redistribution (String Method).
  ReservoirSignal — ReSTIR-inspired K-best-positions signal molecule.
  run_neb() — complete NEB relay chain.
  RESULT: Wasserstein variance = 0.000000 confirmed.

toolkit/checkpoint.py
  ExperimentCheckpoint — monitors generations, fires on failure mode detection.
  Failure modes: DEGENERATE, OSCILLATORY_WARNING, GROKKING_CANDIDATE,
  MODE_COLLAPSE_WARNING, TYPE_7_DIMENSIONAL_INSUFFICIENCY, MISCOORDINATION.
  CONFIRMED WORKING: Type 7 detected at gen 80 (residual > 20), 1.26x efficiency gain.
  THRESHOLDS: calibrated on Tier 0. Recalibrate for new function families.

toolkit/signal_injection.py
  make_signal(pattern) — artificial signal with specific Walsh structure.
  Patterns: linearity, oscillation, oscillation_even, oscillation_odd,
  scale_growth, derivative, uniform, zero.
  optimal_signal_derivative(), optimal_signal_scalar(), etc. — analytically optimal.
  build_signal_library() — precompute library for known relationships.
  match_to_library() — compare natural signal to library (library matching).
  PHARMACOPHORE FINDING: oscillation_even is the pharmacophore for derivative relationship.
  The signal encodes B's identity (even structure); A infers the transformation.

toolkit/walsh.py
  make_signal() moved here from wsar_mapping. CANONICAL LOCATION for signal patterns.

toolkit/computational_taxonomy.py
  TAXONOMY dict — maps each tool to its Ramanujan type and validation status.
  print_taxonomy() — shows the full table.
  run as script to see smoke test results.

toolkit/info_geometry.py
  jsd(), jsd_metric() — also available in stress_metrics.py (same implementation).
  population_fisher_rao() — PARTIAL (10-20% error for non-Gaussian).
  natural_gradient() — CMA-ES equivalent using empirical Fisher matrix.

### experiments/ — Experiment runners

experiments/base.py
  Experiment dataclass — declare with target_a, target_b, relationship, cfg, etc.
  joint_fitness_type field — set to "derivative", "scalar_2", etc. for joint mode.
  run_experiment(exp, signal_design, verbose) -> (run_id, CoevoResult).
  CRITICAL: joint_fitness_type=None uses INDEPENDENT fitness (old broken mode).
  Use joint_fitness_type="derivative" etc. for correct joint optimization.

experiments/registry.py
  REGISTRY dict — all registered experiments by ID.
  list_experiments(tier=0) — all Tier 0 experiments.
  Import: from experiments.registry import get_experiment

experiments/runner.py
  CLI: python -m experiments.runner [list|run|compare|run-all|report]
  Examples:
    python -m experiments.runner compare MM-T0-009 --gens 600 --pop 80
    python -m experiments.runner run-all --tier 0 --all-signals
    python -m experiments.runner report --tier 0

experiments/tier0/experiments.py
  9 Tier 0 experiments (MM-T0-001 through MM-T0-009).
  4 Joint fitness variants (MM-T0-J01 through MM-T0-J04).
  JOINT FITNESS EXPERIMENTS ARE THE IMPORTANT ONES — they use correct architecture.
  MM-T0-J02 (joint derivative, no fixed targets): residual=0.11, conv_gen=285. KEY RESULT.

experiments/joint_covariance.py
  Joint CMA-ES on [A,B] simultaneously (not alternating).
  Extracts A-B cross-covariance block = algebraic relationship fingerprint.
  KEY FINDINGS:
    derivative (cos->sin): cross_cov_max=0.998, residual=0.002
    scalar (cos->2cos): cross_cov_max=0.989, residual=0.001
    TYPE 7 (cos->cos2x): cross_cov_max=0.348, residual=58.9

experiments/type7_test.py
  CONFIRMED: cos->sin depth=1 in linear grammar (antiderivative, residual=0.000)
  CONFIRMED: cos->cos(2x) TYPE 7 in linear grammar (no path exists)
  CONFIRMED: cos->cos(2x) depth=2 in quadratic grammar (cos_double(scale_0.5))

experiments/algebraic_origami_explorer.py
  automated_explorer(run_time_seconds, interest_threshold, output_file) -> records.
  Interest criteria: phase entropy + curvature variance + stress gradient + surprise.
  Run continuously for taxonomy filling. Stores to algebraic_origami_taxonomy.json.
  RESULT: 1993 interesting morphs in 90 seconds (22/sec, ~2M/day if continuous).
  TOP FINDING: oscillatory->linear transitions are the most geometrically rich.

experiments/geometric_richness_validation.py
  4-case canonical validation battery:
  CASE 1: flat linear (expected: zero curvature, zero variance) -> natural_scale=0.57
  CASE 2: Euler rotation (expected: constant curvature ~0.72) -> natural_scale=0.78
  CASE 3: Type 7 cos->cos2x (expected: high stress) -> natural_scale=8.10 !!!
  CASE 4: fractional derivative (expected: matches rotation) -> natural_scale=0.39
  KEY FINGERPRINT: natural scale 10x higher for Type 7 than valid paths.

experiments/instrument_validation.py
  Master validation report for all 8 critical instruments.
  Run: python -m experiments.instrument_validation
  Current status: 3 RIGOROUS, 4 PARTIAL, 1 EMPIRICAL (see report for details).

experiments/sanity_hierarchy.py
  Level 0-4 lower-lifted partition checks.
  Level 0 (scalar): PASS. Level 1 (2D circle): PASS. Level 2 (N=2 poly): PASS (Q1+Q2).
  Level 3 (complex relay): PASS. Level 4 (Ricci on circle): 0.72 (calibrated, not exact).

### notebook/ — The Mendel Notebook Database

notebook/db.py
  init() — create tables (auto-called by run_experiment).
  upsert_object() — register a mathematical object.
  log_run() — record one experiment run.
  log_timeseries() — record per-generation measurements.
  summary(tier=None) — Mendel table summary.
  Uses SQLite at notebook/mendel.db (auto-created, safe to delete to reset).

notebook/capacity.py
  capacity_matrix() — which signal designs detect which relationship types.
  betweenness_score() — how "smooth" was the relay chain path.
  print_capacity_matrix() — print the full capacity matrix.
  FINDING: all current signal designs show NO vs. all relationship types
  (independent fitness is the root cause, not signal design).

---

## Key Design Decisions (and why)

### Why joint fitness, not independent fitness
Independent fitness makes each population chase its own target. The signal exchange
can modulate mutation speed but cannot change what they're optimizing toward.
Joint fitness defines fitness over PAIRS: fit_a(ind_a, partner_b) depends on B's state.
VALIDATION: 10x improvement in residual (1.19 -> 0.11) confirmed in TB-03.

### Why normalization is required
Without normalization, A=B=0 satisfies any joint constraint trivially (zero vector).
The normalization constraint forces populations onto the amplitude manifold.
VALIDATION: Level 2 analytical test confirms joint_fit(cos2, sin2)=0 (global max).

### Why simultaneous CMA-ES, not alternating
Alternating optimization converges to Nash equilibria (each pop locally optimal given other).
Nash equilibria can be far from the joint optimum (the cooperative coevolution problem).
Simultaneous CMA-ES on [A,B] jointly learns the A-B correlation structure.
VALIDATION: Level 2 Q2 confirmed with 4D joint CMA-ES.

### Why the phi_k = x^k/k! basis
Using raw x^k basis: at x=pi, x^7=3035. Coefficients blow up (Type 6 failure).
phi_k = x^k/k! keeps coefficients O(1), well-conditioned for optimization.
The derivative relationship is then exactly: A[j] = B[j+1] (unit shift, no scaling).

### Why the open-chain D-brane model (not closed cycle)
The Euler path from cos to -cos is OPEN (half rotation). A closed cycle forces a
return step from -cos back to cos in one delta step, which is geometrically impossible.
Fixed endpoints (D-branes) break the degenerate all-same-angle collapse.
VALIDATION: Open-chain relay with CMA-ES achieves mean step residual 0.63 (vs >4 for closed).

### Why NEB + String Method over pure gradient descent
Force decomposition (nudging) prevents corner-cutting (degenerate collapse).
Arc-length reparameterization gives equal step sizes analytically.
VALIDATION: Wasserstein variance = 0.000000 confirmed after implementation.

---

## The Failure Mode Taxonomy (use this to diagnose problems)

When an experiment fails, match to the closest type:

TYPE 1 — Wrong metric: formally correct, metrically fails.
  Symptom: consistent systematic offset (8-10 degree angular offset in Euler relay).
  Fix: lift to richer representation (complex 1D instead of real 8D).

TYPE 2 — Missing shadow: systematic predictable failure, always same direction.
  Symptom: signal always encodes gradient, never character.
  Fix: add directional signal (trajectory n-gram). NEB also helps.

TYPE 3 — Near-miss / wrong basis: error too structured to be random.
  Symptom: residual converges to specific nonzero value, not zero.
  Fix: check basis normalization (phi_k vs x^k), check truncation artifacts.

TYPE 4 — Conjectured from structure: satisfies necessary conditions, no convergence.
  Symptom: betweenness score looks reasonable but never hits threshold.
  Fix: patient data collection, more seeds, longer runs.

TYPE 5 — Crossroads / multiple valid solutions.
  Symptom: different seeds converge to different attractors, all seem valid.
  Fix: diversity penalty, subspace constraints, Gromov-Wasserstein alignment.

TYPE 6 — Wrong basis / incomplete representation.
  Symptom: formal algebra works but magnitudes grow without bound.
  Fix: normalize the basis (phi_k = x^k/k!).

TYPE 7 — Dimensional insufficiency: no valid path in this grammar.
  Symptom: residual > 20 at gen 80, cross-covariance < 0.4.
  Confirmed: cos->cos(2x) in linear grammar is TYPE 7.
  Fix: enrich grammar (quadratic operations) or lift to higher-dimensional space.

ML ANALOGS (look like failure but are not):
  GROKKING: high residual but accelerating decrease. Do NOT kill. Let run.
  DOUBLE_DESCENT: residual increases then decreases. Do NOT kill. Let run.
  MODE_COLLAPSE: all seeds converge to same answer. Diversity injection needed.
  POSTERIOR_COLLAPSE: cross-covariance drops after being high. Switch to directional signal.

---

## Quick Start for Common Tasks

### Run the full Tier 0 sweep
```bash
python -m experiments.runner run-all --tier 0 --all-signals
python -m experiments.runner report --tier 0
```

### Run a specific experiment with all three signal designs
```bash
python -m experiments.runner compare MM-T0-J02 --gens 600 --pop 80
```

### Run the autonomous taxonomy explorer for 1 hour
```bash
python -m experiments.algebraic_origami_explorer 3600
```

### Check all instrument validation statuses
```bash
python -m experiments.instrument_validation
```

### Run the 4-case geometric validation battery
```bash
python -m experiments.geometric_richness_validation
```

### Run the NEB relay chain (best current implementation)
```bash
python -m toolkit.neb_relay
```

### Run the complex relay chain (gold standard for rotation)
```bash
python -m toolkit.complex_relay
```

### Check the Mendel notebook
```bash
python show_taxonomy.py
python -m experiments.runner report --tier 0
```

---

## Installed Libraries

cma          — CMA-ES with Fisher Information Metric (natural gradient)
gudhi        — Persistent homology, Betti numbers, TDA
ripser       — Fast Vietoris-Rips persistent homology
networkx     — Graph algorithms
scipy        — Wasserstein-1D, optimization (linear_sum_assignment for exact Wasserstein)
numpy        — All numerical computation
matplotlib   — Plotting

NOT INSTALLED (needs C++ build tools on Windows Python 3.14):
  POT (Python Optimal Transport) — needs C++ build. For exact OT use:
    scipy.optimize.linear_sum_assignment (toolkit/improved_metrics.py exact_wasserstein)

NOT COMPATIBLE (Python 3.14):
  geomstats — incompatible (uses numpy.trapz removed in numpy 2.x)
  GraphRicciCurvature — fails on Python 3.14

---

## The Research Roadmap

DONE:
  - Joint fitness architecture (10x improvement confirmed)
  - Type 7 detection and confirmation
  - NEB relay chain (Wasserstein variance = 0)
  - Complex relay chain (exact, gold standard)
  - Instrument validation (3 rigorous, 4 partial, 1 empirical)
  - Autonomous taxonomy explorer (22/sec)
  - 1993 interesting morphs in first 90-second run

NEW SINCE LAST SESSION:
  - TPI (Topological Protection Index) added to algebraic_origami_explorer.py
  - Every morph now measures BOTH forward and reverse direction
  - Taxonomy JSON includes: topological_protection_index, is_topologically_protected,
    forward_total, reverse_total, activation_energy_asymmetry
  - Parts XXVIII-XXX added to joint_tractability_exploration.md (1713 lines total)

HIGHEST PRIORITY NEXT:
  - Run explorer for 12+ hours to accumulate ~1M taxonomy entries with TPI data
  - Train simple MLP: 16 inputs (two coefficient vectors) -> predict TPI + algebraic_depth
  - Add prime-sequence functions to library (first 8 primes as coefficients)
  - FABRIK relay chain (O(n), 5 iterations, game animation IK) -- 100x faster than CMA-ES
  - SLERP relay chain (exact geodesic validator for rotations)

MEDIUM PRIORITY:
  - Algebraic depth auto-logging in all experiments
  - 50+ experiments to build taxonomy density
  - Motion matching lookup from Mendel notebook
  - Half-pound filter for frequency-domain blending

LONG-TERM:
  - Neural path guiding (learn importance distribution)
  - Mixture of experts architecture (dimensional lifting)
  - Gromov-Wasserstein for cross-space alignment (needs POT)
  - Galois theory formalization of algebraic depth

---

## Files to NOT Touch

rung0_coevo.py — historical prototype, superseded
_append_md*.py — temporary scripts for documentation, can be deleted
notebook/mendel.db — the Mendel notebook, never delete (or backup first)

---

## The .md Document

joint_tractability_exploration.md — the complete research journal.
1913 lines, 32 Parts. Read sequentially or jump to:
  Part XIII: Failure Mode Taxonomy
  Part XIV: Checkpoint Architecture
  Part XV: Joint Fitness Correction
  Part XVIII: Key Empirical Findings
  Part XXII: Computation Domain Taxonomy
  Part XXIII: Game Animation Algorithms
  Part XXVI: Future Hypothesis Matrix
  Part XXVIII: Topological Protection + Activation Energy (TPI metric, neural net as catalyst)
  Part XXIX: Shor's Algorithm as Dimensional Lift (every lift introduces instabilities)
  Part XXXI: Natural vs. Unnatural Lifts -- Formal Definition (category theory + p-adic Hodge)
  Part XXXII: The Galois Parallel -- THE DEEPEST FRAMING (read this first in a new session)

---

## If You're Picking This Up Fresh

1. Read the research question at the top of this file.
2. Run `python -m experiments.instrument_validation` to see tool health.
3. Run `python -m experiments.runner report --tier 0` to see Mendel table.
4. The most informative experiment to run: `python -m toolkit.neb_relay`
   (demonstrates the current best relay chain implementation).
5. The most important unimplemented thing: FABRIK relay chain.
6. The gold standard everything else is calibrated against: `python -m toolkit.complex_relay`

The instrument is built. The calibration is in progress.
The territory is vast. The map is just beginning.
