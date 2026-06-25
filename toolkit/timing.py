"""
Operation timing — measures wall-clock cost of each component.

Cheapness is a first-class concern: expensive operations constrain what
experiments are feasible, which constrains what hypotheses can be tested.
The timing log tells you where to optimize when you want to scale up.

Tracked operations:
  fitness_eval     — one call to fitness_fn(ind)
  signal_compute   — one full Walsh signal computation (2^N evals)
  ga_step          — one generation of the coevolutionary GA
  full_run         — one complete experiment run

Stored per experiment_run in the DB as a JSON blob.
Reported in the Mendel table alongside residuals.
"""

import time
import json
from contextlib import contextmanager
from collections import defaultdict


class Timer:
    """Accumulates timing for named operations."""

    def __init__(self):
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int]   = defaultdict(int)
        self._stack:  list             = []

    @contextmanager
    def measure(self, name: str):
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self._totals[name] += elapsed
        self._counts[name] += 1

    def total(self, name: str) -> float:
        return self._totals.get(name, 0.0)

    def count(self, name: str) -> int:
        return self._counts.get(name, 0)

    def mean(self, name: str) -> float:
        c = self._counts.get(name, 0)
        return self._totals[name] / c if c > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            k: {"total_s": round(self._totals[k], 4),
                "count":   self._counts[k],
                "mean_ms": round(self._mean_ms(k), 3)}
            for k in self._totals
        }

    def _mean_ms(self, k):
        c = self._counts.get(k, 0)
        return 1000 * self._totals[k] / c if c > 0 else 0.0

    def report(self, title: str = ""):
        d = self.to_dict()
        if title:
            print(f"\n  Timing: {title}")
        for name, v in sorted(d.items(), key=lambda x: -x[1]["total_s"]):
            print(f"    {name:<22} total={v['total_s']:>7.3f}s  "
                  f"n={v['count']:>5}  mean={v['mean_ms']:>7.3f}ms")

    def budget_breakdown(self) -> dict:
        """
        What fraction of total run time does each operation consume?
        Returns {operation: fraction_of_total}.
        """
        total = sum(self._totals.values())
        if total == 0:
            return {}
        return {k: round(v / total, 3) for k, v in self._totals.items()}
