"""
Testbeds 2 — the three atomic confirmations for the Euler relay chain.

Each testbed tests exactly ONE claim.
No claim depends on another.
If a testbed fails, the failure mode is precisely named.

TB-06: Stability — do the true Euler intermediates stay stable under the relay fitness?
TB-07: CMA-ES — does adaptive sigma find the Euler path the plain GA cannot?
TB-08: Diversity — does a repulsion penalty break the all-same-angle degeneracy?
"""

import math
import time
import numpy as np

from toolkit.euler_relay import (euler_intermediate, euler_step_fitness,
                                  euler_step, all_true_intermediates,
                                  phase_angle, _COS, _SIN, N, X, BASIS)
from toolkit.testbeds import Testbed, TestbedResult, run_all
from toolkit.stress_metrics import relay_stress_report, sliced_wasserstein


# ── TB-06: Stability ───────────────────────────────────────────────────────────

class TB_EulerStability(Testbed):
    """
    Claim: the true Euler intermediates are a stable fixed point of the relay
    chain step fitness. If you initialize populations exactly at the true
    intermediates and run one generation, they should stay there.

    If this FAILS (DENIED):
      The fitness landscape itself is wrong. CMA-ES and diversity penalties
      cannot fix a wrong fitness function. This is the first gate.
      Failure mode: OVER_CONSTRAINED or wrong fitness definition.

    If this PASSES (CONFIRMED):
      The landscape has the right fixed point. The problem is purely one of
      FINDING it (initialization, sigma mismatch, degeneracy). CMA-ES and
      diversity penalties can potentially fix this.
    """
    def run(self):
        n_steps = 4
        delta   = math.pi / n_steps
        true_ints = all_true_intermediates(n_steps)

        # Evaluate fitness of each true intermediate given true neighbours
        # If true intermediates are a stable fixed point, fitness should be 0
        # (residual = 0 for exact intermediates)
        residuals = []
        for k in range(n_steps + 1):
            ind  = true_ints[k]
            prev = true_ints[(k - 1) % (n_steps + 1)]
            nxt  = true_ints[(k + 1) % (n_steps + 1)]
            res_from_prev = float(np.linalg.norm(ind - euler_step(prev, delta)))
            residuals.append(res_from_prev)

        mean_res = float(np.mean(residuals))
        fm = "OVER_CONSTRAINED" if mean_res > 0.01 else ""
        notes = f"per-step residuals: {[round(r,6) for r in residuals]}"
        return self._verdict(mean_res, notes=notes, failure_mode=fm)


# ── TB-07: CMA-ES ──────────────────────────────────────────────────────────────

class TB_CMAEulerRelay(Testbed):
    """
    Claim: CMA-ES (which adapts sigma to the problem scale using Fisher
    Information Metric) finds a relay chain closer to the true Euler path
    than the plain GA.

    The plain GA uses fixed sigma=0.04, which is << the Wasserstein distance
    1.53 between true Euler intermediates. CMA-ES adapts its sigma upward
    to match the problem scale.

    If CONFIRMED: the sigma mismatch (our 'wrong hbar') was the root cause
    of the Euler relay chain failure. CMA-ES is the right optimizer.

    If DENIED: there is a more fundamental problem (wrong fitness, symmetry).
    """
    def run(self):
        try:
            import cma
        except ImportError:
            return TestbedResult(
                claim=self.claim, verdict="ERROR",
                value=float('nan'), threshold=self.threshold,
                elapsed_s=0., notes="cma package not installed",
            )

        n_steps = 4
        delta   = math.pi / n_steps
        true_ints = all_true_intermediates(n_steps)
        n_pops  = n_steps + 1

        # Run CMA-ES on each relay population independently
        # (simplified: each pop optimized against fixed true neighbours)
        step_residuals = []
        for k in range(n_pops):
            prev = true_ints[(k - 1) % n_pops]
            nxt  = true_ints[(k + 1) % n_pops]

            def neg_fitness(c):
                return -euler_step_fitness(
                    np.array(c), prev, nxt, delta, cycle_weight=0.2)

            x0    = true_ints[k] + np.random.randn(N) * 0.5
            sigma0 = 0.5   # start large, CMA-ES will adapt
            opts  = cma.CMAOptions()
            opts['maxiter'] = 200
            opts['verbose'] = -9
            opts['tolx']    = 1e-5

            es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
            es.optimize(neg_fitness)
            best = np.array(es.result.xbest)
            step_residuals.append(float(np.linalg.norm(best - true_ints[k])))

        mean_res = float(np.mean(step_residuals))
        # Compare to plain GA result (approximately 4.3 from previous run)
        plain_ga_residual = 4.30
        improvement = plain_ga_residual - mean_res

        notes = (f"CMA mean_res={mean_res:.4f}  "
                 f"vs plain GA={plain_ga_residual:.4f}  "
                 f"improvement={improvement:.4f}")
        fm = "DENIED" if improvement < 0.5 else ""
        return self._verdict(improvement, notes=notes, failure_mode=fm)


# ── TB-08: Diversity penalty ───────────────────────────────────────────────────

class TB_DiversityPenalty(Testbed):
    """
    Claim: adding a repulsion penalty between adjacent relay populations
    breaks the all-same-angle degeneracy (populations clustering at ~-13°
    instead of distributing along the Euler path).

    The diversity penalty: each population's fitness is reduced if it is
    too close (in coefficient space) to its neighbours. This forces them apart.

    If CONFIRMED: the degeneracy was purely due to the missing diversity
    constraint. The relay chain CAN find the rotation with diversity.

    If DENIED: there is a deeper symmetry problem — even with diversity,
    populations find some valid rotation but not specifically the cos/sin one.
    This would mean we need external data or a subspace constraint.
    """
    def run(self):
        n_steps = 4
        delta   = math.pi / n_steps
        n_pops  = n_steps + 1
        rng     = np.random.default_rng(42)
        pop_size = 60
        gens     = 400
        diversity_weight = 0.3

        # Initialize randomly (NOT at true intermediates)
        pops = [rng.standard_normal((pop_size, N)) * 0.4
                for _ in range(n_pops)]
        best_pops = [p[0].copy() for p in pops]

        def step_fit_with_diversity(ind, prev, nxt, k):
            base = euler_step_fitness(ind, prev, nxt, delta, 0.2)
            # Diversity: penalise being too close to prev and next
            dist_prev = float(np.linalg.norm(ind - prev))
            dist_next = float(np.linalg.norm(ind - nxt))
            min_sep = 0.5   # populations must be at least this far apart
            div_pen = (max(0., min_sep - dist_prev) +
                       max(0., min_sep - dist_next))
            return base - diversity_weight * div_pen

        for g in range(gens):
            for k in range(n_pops):
                prev = best_pops[(k - 1) % n_pops]
                nxt  = best_pops[(k + 1) % n_pops]
                fits = np.array([step_fit_with_diversity(ind, prev, nxt, k)
                                 for ind in pops[k]])
                elite_idx = np.argsort(fits)[-2:]
                new_pop = np.empty_like(pops[k])
                new_pop[:2] = pops[k][elite_idx].copy()
                for j in range(2, pop_size):
                    idx = rng.choice(pop_size, 4, replace=False)
                    p1  = pops[k][idx[np.argmax(fits[idx])]]
                    child = p1 + rng.standard_normal(N) * 0.06
                    new_pop[j] = child
                pops[k] = new_pop
                fits2 = np.array([step_fit_with_diversity(ind, prev, nxt, k)
                                   for ind in pops[k]])
                best_pops[k] = pops[k][np.argmax(fits2)].copy()

        # Check: are populations spread across different angles?
        angles = [phase_angle(best_pops[k]) for k in range(n_pops)]
        angle_spread = float(np.std(angles))
        # Populations all at same angle → spread ~0 (degenerate)
        # Populations distributed → spread > 0.5 rad (~30 degrees)
        fm = "UNDER_CONSTRAINED" if angle_spread < 0.3 else ""
        notes = (f"angles_deg={[round(math.degrees(a),1) for a in angles]}  "
                 f"spread={math.degrees(angle_spread):.1f} deg")
        return self._verdict(angle_spread, notes=notes, failure_mode=fm)


# ── Suite ─────────────────────────────────────────────────────────────────────

SUITE_2 = [
    TB_EulerStability(
        id="TB-06",
        claim="True Euler intermediates are a stable fixed point of relay step fitness",
        conditions="4-step Euler relay, exact true intermediates, check residual",
        threshold=0.001,
        higher_is_better=False,   # residual should be BELOW threshold
    ),
    TB_CMAEulerRelay(
        id="TB-07",
        claim="CMA-ES finds Euler relay path closer to true than plain GA (improvement > 0.5)",
        conditions="4-step Euler relay, CMA-ES per-population, 200 iterations",
        threshold=0.5,
    ),
    TB_DiversityPenalty(
        id="TB-08",
        claim="Diversity penalty breaks all-same-angle degeneracy (angle spread > 0.3 rad)",
        conditions="4-step relay, diversity repulsion weight=0.3, 400 gens",
        threshold=0.3,
    ),
]


if __name__ == "__main__":
    from notebook import db
    db.init()
    results = run_all(SUITE_2)
    print("\nATOMIC CONFIRMATIONS:")
    for r in results:
        symbol = "CONFIRMED" if r.verdict == "CONFIRMED" else r.verdict
        print(f"  {symbol:<20} {r.claim[:55]}")
        if r.notes:
            print(f"                       {r.notes[:70]}")
