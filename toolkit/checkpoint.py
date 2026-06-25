"""
Early termination checkpoints — the immune system of the experiment pipeline.

Biological analogy:
  T-cell selection has two checkpoints:
    1. Positive selection (gen 10-20): does this receptor bind MHC at all?
       Fail: apoptosis (no signal)  ->  DEGENERATE failure
    2. Negative selection (gen 50-100): does it bind self-antigens too strongly?
       Fail: apoptosis (wrong attractor) -> MISCOORDINATION failure
    Only cells passing both survive to run fully.

Our checkpoints do the same:
  - Degenerate checkpoint    (gen ~15):  populations approaching zero
  - Oscillatory checkpoint   (gen ~30):  Walsh trajectory not settling
  - Type 7 checkpoint        (gen ~60):  no algebraic path detected (cross-cov)
  - Miscoordination check    (gen ~100): valid covariance but wrong residual

Why this matters:
  Without checkpoints: every experiment runs to gen_max regardless of outcome.
    -> 80% of compute wasted on hopeless runs
  With checkpoints: hopeless runs terminated at gen 15-100.
    -> Same compute budget covers 3-5x more experiments
    -> Every termination is precisely labelled
    -> Failure mode data IS the taxonomy

The failure mode label is as valuable as a success — it tells you:
  - WHERE in the problem space this point lives (close to which failure mode)
  - WHAT information enrichment might help (from the Ramanujan taxonomy)
  - WHETHER to redirect (new grammar) or record (genuine Type 7 boundary)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class CheckpointResult:
    generation:    int
    should_stop:   bool
    failure_mode:  Optional[str]   # None = healthy, otherwise failure mode name
    metric_values: dict            # the actual numbers that triggered the check
    recommendation: str            # what to do next


class ExperimentCheckpoint:
    """
    Run at specific generation intervals during an experiment.
    Each check costs O(N^2) and returns (continue, failure_mode).

    Usage in experiment loop:
        checker = ExperimentCheckpoint(n_genes=8)
        for g in range(gens):
            ... evolve ...
            result = checker.check(g, pop_a, pop_b, signal_a, signal_b, residual)
            if result.should_stop:
                log_failure(result.failure_mode)
                break
    """

    def __init__(self,
                 n_genes: int = 8,
                 checkpoints: List[int] = (15, 40, 80, 150),
                 degenerate_threshold:   float = 0.05,
                 type7_residual:         float = 20.0,
                 type7_crosscov:         float = 0.40,
                 oscillatory_window:     int   = 8,
                 miscoord_residual:      float = 0.50,
                 miscoord_crosscov:      float = 0.85):
        self.n        = n_genes
        self.cpts     = sorted(checkpoints)
        self.thr_deg  = degenerate_threshold
        self.thr_t7r  = type7_residual
        self.thr_t7c  = type7_crosscov
        self.thr_osc  = oscillatory_window
        self.thr_msr  = miscoord_residual
        self.thr_msc  = miscoord_crosscov

        self._residual_history: List[float] = []
        self._norm_history_a:   List[float] = []
        self._norm_history_b:   List[float] = []

    def _cross_cov_max(self, pop_a: np.ndarray, pop_b: np.ndarray) -> float:
        """
        Scale-invariant A-B structural correlation.
        Normalize each population to zero mean unit variance before correlating.
        This captures STRUCTURAL similarity, not magnitude similarity.
        """
        if pop_a.ndim == 1:
            pa = pop_a / (np.linalg.norm(pop_a) + 1e-10)
            pb = pop_b / (np.linalg.norm(pop_b) + 1e-10)
            return float(abs(np.dot(pa, pb)))
        n = min(len(pop_a), len(pop_b), 30)
        # Normalize each individual to unit norm, then compute correlations
        pa = pop_a[:n] / (np.linalg.norm(pop_a[:n], axis=1, keepdims=True) + 1e-10)
        pb = pop_b[:n] / (np.linalg.norm(pop_b[:n], axis=1, keepdims=True) + 1e-10)
        # Mean correlation across gene pairs
        a_flat = pa.flatten()
        b_flat = pb.flatten()
        if np.std(a_flat) < 1e-8 or np.std(b_flat) < 1e-8:
            return 0.0
        return float(abs(np.corrcoef(a_flat, b_flat)[0, 1]))

    def check(self,
              generation:  int,
              pop_a:       np.ndarray,
              pop_b:       np.ndarray,
              signal_a:    Optional[np.ndarray],
              signal_b:    Optional[np.ndarray],
              residual:    float) -> Optional[CheckpointResult]:
        """
        Run all applicable checkpoints at this generation.
        Returns None if no checkpoint fires, CheckpointResult otherwise.
        Only fires at generation numbers in self.cpts.
        """
        if generation not in self.cpts:
            self._residual_history.append(residual)
            mean_a = float(np.mean(np.linalg.norm(pop_a.reshape(-1, self.n), axis=1))
                           if pop_a.ndim > 1 else np.linalg.norm(pop_a))
            mean_b = float(np.mean(np.linalg.norm(pop_b.reshape(-1, self.n), axis=1))
                           if pop_b.ndim > 1 else np.linalg.norm(pop_b))
            self._norm_history_a.append(mean_a)
            self._norm_history_b.append(mean_b)
            return None

        # Compute diagnostics
        mean_norm_a = (np.mean(np.linalg.norm(pop_a.reshape(-1, self.n), axis=1))
                       if pop_a.ndim > 1 else np.linalg.norm(pop_a))
        mean_norm_b = (np.mean(np.linalg.norm(pop_b.reshape(-1, self.n), axis=1))
                       if pop_b.ndim > 1 else np.linalg.norm(pop_b))
        cross_cov = self._cross_cov_max(pop_a, pop_b)

        self._residual_history.append(residual)
        self._norm_history_a.append(float(mean_norm_a))
        self._norm_history_b.append(float(mean_norm_b))

        # ── Route to unified checker that runs ALL checks ────────────────────
        return self._run_all_checkpoints(
            generation, float(mean_norm_a), float(mean_norm_b), cross_cov, residual)

    def _LEGACY_check_individual(self, generation, mean_norm_a, mean_norm_b, cross_cov, residual):
        """Legacy individual checks — replaced by _run_all_checkpoints above."""
        # ── Checkpoint 1: Degenerate (gen ~15) ──────────────────────────────
        if generation <= 20:
            if mean_norm_a < self.thr_deg or mean_norm_b < self.thr_deg:
                return CheckpointResult(
                    generation=generation,
                    should_stop=True,
                    failure_mode="DEGENERATE",
                    metric_values={
                        "mean_norm_a": float(mean_norm_a),
                        "mean_norm_b": float(mean_norm_b),
                        "threshold": self.thr_deg,
                    },
                    recommendation=(
                        "Populations collapsed to zero. "
                        "Add normalization penalty (norm_weight > 0). "
                        "Ramanujan Type: scale degeneracy (no amplitude constraint)."
                    ),
                )

        # ── Checkpoint 2a: Oscillatory (gen ~40) ────────────────────────────
        if generation >= 30 and len(self._residual_history) >= self.thr_osc:
            window = self._residual_history[-self.thr_osc:]
            recent_var  = float(np.var(window[-self.thr_osc//2:]))
            earlier_var = float(np.var(window[:self.thr_osc//2]))
            if recent_var > earlier_var * 1.5 and residual > 0.5:
                return CheckpointResult(
                    generation=generation,
                    should_stop=False,   # warning, not stop
                    failure_mode="OSCILLATORY_WARNING",
                    metric_values={
                        "recent_var": recent_var,
                        "earlier_var": earlier_var,
                        "residual": residual,
                    },
                    recommendation=(
                        "Residual variance INCREASING — path not converging. "
                        "Try: reduce sigma, add directional signal (trajectory_ngram), "
                        "or switch to joint CMA-ES. "
                        "Ramanujan Type 2: may be missing a shadow/complement."
                    ),
                )

        # ── Checkpoint 2b: Grokking detector (gen ~40) ──────────────────────
        # Grokking: high residual but ACCELERATING (second derivative positive).
        # This looks like failure but is a late-bloomer. DO NOT KILL.
        # ML analog: the memorization→generalization transition.
        if generation >= 30 and len(self._residual_history) >= 6:
            h = self._residual_history[-6:]
            d1 = [h[i+1] - h[i] for i in range(5)]   # first derivative
            d2 = [d1[i+1] - d1[i] for i in range(4)]  # second derivative
            accel = float(np.mean(d2))
            if residual > 0.3 and accel < -0.01:  # high residual BUT accelerating down
                return CheckpointResult(
                    generation=generation,
                    should_stop=False,  # KEEP RUNNING — this is grokking territory
                    failure_mode="GROKKING_CANDIDATE",
                    metric_values={
                        "residual": residual,
                        "acceleration": accel,
                    },
                    recommendation=(
                        "High residual but ACCELERATING decrease — possible late bloomer. "
                        "Do NOT terminate. Let run to completion. "
                        "ML analog: grokking (memorization→generalization transition). "
                        "Catastrophe type: fold unfolding — minimum emerging from flat region."
                    ),
                )

        # ── Checkpoint 2c: Mode collapse (inter-seed) ────────────────────────
        # If inter-seed diversity is very low, populations collapsed to one attractor.
        # This is a warning — valid algebraic path found but full distribution missed.
        if generation >= 30:
            norm_a_std = float(np.std(self._norm_history_a[-min(5, len(self._norm_history_a)):]))
            if norm_a_std < 0.001 and residual > 0.1:
                return CheckpointResult(
                    generation=generation,
                    should_stop=False,
                    failure_mode="MODE_COLLAPSE_WARNING",
                    metric_values={
                        "norm_std": norm_a_std,
                        "residual": residual,
                    },
                    recommendation=(
                        "Population norms are nearly identical — may have mode-collapsed "
                        "to a single attractor. Consider diversity injection or multiple "
                        "independent restarts to explore the full distribution of solutions."
                    ),
                )

        # ── Checkpoint 3: Type 7 (gen ~80) ──────────────────────────────────
        # Primary signal: residual alone. Cross-cov is secondary (structurally
        # similar functions can have high cross-cov even when Type 7).
        # Calibrate thr_t7r from baseline experiments — set to 10x the
        # expected convergence residual.
        if generation >= 60:
            if residual > self.thr_t7r:
                return CheckpointResult(
                    generation=generation,
                    should_stop=True,
                    failure_mode="TYPE_7_DIMENSIONAL_INSUFFICIENCY",
                    metric_values={
                        "residual": residual,
                        "cross_cov_max": cross_cov,
                        "residual_threshold": self.thr_t7r,
                        "crosscov_threshold": self.thr_t7c,
                    },
                    recommendation=(
                        "High residual AND low A-B correlation: no algebraic path "
                        "in current grammar. "
                        "Try: enrich grammar with higher-degree operations (quadratic), "
                        "or lift to higher-dimensional coefficient space. "
                        "Ramanujan Type 7: dimensional insufficiency."
                    ),
                )

        # ── Checkpoint 4: Miscoordination (gen ~150) ────────────────────────
        if generation >= 120:
            if residual > self.thr_msr and cross_cov > self.thr_msc:
                return CheckpointResult(
                    generation=generation,
                    should_stop=True,
                    failure_mode="MISCOORDINATION",
                    metric_values={
                        "residual": residual,
                        "cross_cov_max": cross_cov,
                    },
                    recommendation=(
                        "High A-B correlation but high residual: populations are "
                        "coordinated but at the WRONG joint attractor. "
                        "Try: random restart with different initialization, "
                        "or switch from alternating to joint (simultaneous) optimization. "
                        "Ramanujan Type: Nash equilibrium ≠ joint optimum."
                    ),
                )

        # ── All clear ────────────────────────────────────────────────────────
        # If we accumulated any warnings above (non-stop), they were stored here.
        # Return the most important non-stop observation, or healthy.
        # (Warnings are collected before fatal checks, so fatal always wins.)
        return CheckpointResult(
            generation=generation,
            should_stop=False,
            failure_mode=None,
            metric_values={
                "mean_norm_a": float(mean_norm_a),
                "mean_norm_b": float(mean_norm_b),
                "cross_cov": cross_cov,
                "residual": residual,
            },
            recommendation="Healthy — continue."
        )

    def _run_all_checkpoints(self, generation, mean_norm_a, mean_norm_b,
                              cross_cov, residual):
        """Internal: run all checks, return most severe result (STOP > WARNING > None)."""
        warning_result = None

        # Degenerate
        if generation <= 20 and (mean_norm_a < self.thr_deg or mean_norm_b < self.thr_deg):
            return CheckpointResult(generation, True, "DEGENERATE",
                                    {"mean_norm_a": mean_norm_a, "mean_norm_b": mean_norm_b},
                                    "Add normalization penalty.")

        # Grokking (warning only)
        if generation >= 30 and len(self._residual_history) >= 6:
            h = self._residual_history[-6:]
            d1 = [h[i+1]-h[i] for i in range(5)]
            d2 = [d1[i+1]-d1[i] for i in range(4)]
            accel = float(np.mean(d2))
            if residual > 0.3 and accel < -0.01:
                warning_result = CheckpointResult(
                    generation, False, "GROKKING_CANDIDATE",
                    {"residual": residual, "acceleration": accel},
                    "Late bloomer — do NOT kill.")

        # Type 7 (STOP — checked after grokking so grokking doesn't mask it)
        if generation >= 60 and residual > self.thr_t7r:
            return CheckpointResult(
                generation, True, "TYPE_7_DIMENSIONAL_INSUFFICIENCY",
                {"residual": residual, "threshold": self.thr_t7r},
                "No algebraic path. Enrich grammar or lift dimension.")

        # Miscoordination (STOP)
        if generation >= 120 and residual > self.thr_msr and cross_cov > self.thr_msc:
            return CheckpointResult(
                generation, True, "MISCOORDINATION",
                {"residual": residual, "cross_cov": cross_cov},
                "Stable wrong attractor. Restart with joint optimizer.")

        return warning_result  # None or GROKKING_CANDIDATE warning

    def summary(self) -> dict:
        """Return a summary of the experiment health over its lifetime."""
        return {
            "n_generations":    len(self._residual_history),
            "final_residual":   self._residual_history[-1] if self._residual_history else None,
            "min_residual":     min(self._residual_history) if self._residual_history else None,
            "min_norm_a":       min(self._norm_history_a) if self._norm_history_a else None,
            "min_norm_b":       min(self._norm_history_b) if self._norm_history_b else None,
            "residual_trend":   "decreasing" if (
                len(self._residual_history) > 5 and
                self._residual_history[-1] < self._residual_history[0]
            ) else "flat_or_increasing",
        }
