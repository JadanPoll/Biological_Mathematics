"""
Mini-testbeds — transparent fingerprinting of technique sub-claims.

The discipline (from the user):
  For every technique or sub-technique we try, state the minimal thing
  that MUST be true for it to work. Write a small test that checks exactly
  that claim. Record whether it holds, and if not, WHICH assumption broke.

This builds a taxonomy of failure modes — degeneracy, divergence, oscillation,
miscoordination — that lets us prune techniques early and understand exactly
where the breakdown is, not just that it broke.

Each Testbed:
  - claim:        the sub-claim being tested (one sentence)
  - conditions:   what must hold for the claim to apply
  - run():        the minimal experiment (< 50 lines of GA / 10s of runtime)
  - expected:     quantitative success threshold
  - failure_mode: taxonomy label if it fails
  - verdict:      CONFIRMED / PARTIAL / DENIED / DEGENERATE / ERROR

Taxonomy of failure modes:
  DEGENERATE    both populations collapse to a trivial shared solution (A=B=0)
  DIVERGENT     populations run away from each other (KL increases monotonically)
  OSCILLATORY   populations cycle without converging (betweenness = 'reversed')
  MISCOORDINATED populations find a wrong joint attractor (stable but incorrect)
  UNDER_CONSTRAINED too many valid solutions; no selection pressure toward target
  OVER_CONSTRAINED constraints are mutually incompatible; fitness landscape is empty
"""

import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

# ── Framework ─────────────────────────────────────────────────────────────────

VERDICTS = {"CONFIRMED", "PARTIAL", "DENIED", "DEGENERATE",
            "DIVERGENT", "OSCILLATORY", "MISCOORDINATED",
            "UNDER_CONSTRAINED", "OVER_CONSTRAINED", "ERROR"}

@dataclass
class TestbedResult:
    claim:        str
    verdict:      str
    value:        float    # the key measured quantity
    threshold:    float    # what it needed to be
    elapsed_s:    float
    notes:        str = ""
    failure_mode: str = ""


@dataclass
class Testbed:
    id:           str
    claim:        str
    conditions:   str
    threshold:    float
    higher_is_better: bool = True   # True: value > threshold = CONFIRMED

    def run(self) -> TestbedResult:
        raise NotImplementedError

    def _verdict(self, value: float, notes: str = "",
                 failure_mode: str = "") -> TestbedResult:
        if self.higher_is_better:
            passed = value >= self.threshold
        else:
            passed = value <= self.threshold
        verdict = "CONFIRMED" if passed else "DENIED"
        if failure_mode:
            verdict = failure_mode
        return TestbedResult(
            claim=self.claim, verdict=verdict,
            value=value, threshold=self.threshold,
            elapsed_s=0., notes=notes, failure_mode=failure_mode,
        )


def run_all(testbeds: list, verbose: bool = True) -> list[TestbedResult]:
    results = []
    if verbose:
        print(f"\n{'='*72}")
        print("MINI-TESTBED SUITE — technique fingerprinting")
        print(f"{'='*72}")
        print(f"{'ID':<8} {'Verdict':<16} {'Value':>8} {'Threshold':>10}  Claim")
        print("-"*72)

    for tb in testbeds:
        t0 = time.perf_counter()
        try:
            r = tb.run()
        except Exception as e:
            r = TestbedResult(
                claim=tb.claim, verdict="ERROR",
                value=float('nan'), threshold=tb.threshold,
                elapsed_s=0., notes=str(e),
            )
        r.elapsed_s = time.perf_counter() - t0
        results.append(r)

        if verbose:
            v_color = r.verdict
            claim_short = r.claim[:45]
            print(f"{tb.id:<8} {v_color:<16} {r.value:>8.4f} {r.threshold:>10.4f}  {claim_short}")

    if verbose:
        confirmed = sum(1 for r in results if r.verdict == "CONFIRMED")
        print(f"\n  {confirmed}/{len(results)} claims confirmed")
        print(f"{'='*72}")
    return results


# ── Testbed implementations ───────────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TB_RelayEscapesZero(Testbed):
    """
    Claim: relay chain does NOT collapse to A=B=C=D=0.
    The cycle constraint creates a non-trivial minimum.
    If this fails: DEGENERATE — need normalisation constraint.
    """
    def run(self):
        from toolkit.relay_chain import RelayChainConfig, run as relay_run
        from toolkit.ga import CoevoConfig
        cfg = CoevoConfig(n_genes=8, pop_size=40, gens=200, seed=42,
                          n_collaborations=1, k_signal_samples=4, mut_std=0.05,
                          sig_weight=0.3, signal_design="walsh_solution", sig_freq=10,
                          elite=2, tourney=4, delta=0.2, coupling_mode="adaptive_kl")
        chain = RelayChainConfig(n_pops=4, step_fitness="derivative", cfg=cfg, cycle_weight=0.2)
        r = relay_run(chain, log_every=200)
        mean_norm = float(np.mean([np.linalg.norm(p) for p in r.populations]))
        fm = "DEGENERATE" if mean_norm < 0.05 else ""
        return self._verdict(mean_norm, notes=f"mean_norm={mean_norm:.4f}", failure_mode=fm)


class TB_DirectionalBetterThanState(Testbed):
    """
    Claim: trajectory_ngram (directional signal) gives lower cycle residual
    than walsh_solution (state signal) on the relay chain.
    If this fails: directional signal adds no information — .md Section 3.4 wrong.
    """
    def run(self):
        from toolkit.relay_chain import RelayChainConfig, run as relay_run
        from toolkit.ga import CoevoConfig

        def _run_relay(signal_design):
            cfg = CoevoConfig(n_genes=8, pop_size=40, gens=200, seed=42,
                              n_collaborations=1, k_signal_samples=4, mut_std=0.05,
                              sig_weight=0.35, signal_design=signal_design, sig_freq=10,
                              elite=2, tourney=4, delta=0.25, coupling_mode="adaptive_kl")
            chain = RelayChainConfig(n_pops=4, step_fitness="derivative",
                                     cfg=cfg, cycle_weight=0.2)
            r = relay_run(chain, log_every=200)
            return r.cycle_residual

        res_directional = _run_relay("trajectory_ngram")
        res_state       = _run_relay("walsh_solution")
        # Directional is better if its cycle residual is lower
        improvement = res_state - res_directional
        fm = "DENIED" if improvement < 0 else ""
        return self._verdict(improvement,
                             notes=f"directional={res_directional:.4f} state={res_state:.4f}",
                             failure_mode=fm)


class TB_JointFitnessBetterThanIndependent(Testbed):
    """
    Claim: joint fitness (fitness over pairs) gives lower derivative residual
    than independent fitness on the cos/sin derivative experiment.
    Core claim of the .md Section 3.1 architecture.
    """
    def run(self):
        import math
        from experiments.base import Experiment, rel_derivative, run_experiment, poly_eval
        from toolkit.ga import CoevoConfig

        N = 8
        X = np.linspace(-1.5, 1.5, 120)
        BASIS = np.stack([X**k / math.factorial(k) for k in range(N)], axis=1)
        true_cos = np.array([(-1.)**(k//2) if k%2==0 else 0. for k in range(N)])
        true_sin = np.array([(-1.)**((k-1)//2) if k%2==1 else 0. for k in range(N)])
        cfg = CoevoConfig(n_genes=N, pop_size=60, gens=300, seed=42,
                          n_collaborations=3, k_signal_samples=5, mut_std=0.04,
                          sig_weight=0.35, signal_design="walsh_solution", sig_freq=10,
                          elite=2, tourney=4, delta=0.25, coupling_mode="adaptive_kl")

        # Joint fitness experiment
        exp_joint = Experiment(
            object_id="TB-joint", name="joint deriv test", domain="test",
            expression_a="cos", expression_b="sin",
            target_a=true_cos, target_b=true_sin,
            relationship=rel_derivative(), relationship_desc="A=d/dx(B)",
            tier=0, ground_truth="proven",
            joint_fitness_type="derivative",
            cfg=cfg,
        )
        _, r_joint = run_experiment(exp_joint, signal_design="walsh_solution",
                                    verbose=False)

        # Independent fitness experiment
        exp_indep = Experiment(
            object_id="TB-indep", name="indep deriv test", domain="test",
            expression_a="cos", expression_b="sin",
            target_a=true_cos, target_b=true_sin,
            relationship=rel_derivative(), relationship_desc="A=d/dx(B)",
            tier=0, ground_truth="proven",
            joint_fitness_type=None,
            cfg=cfg,
        )
        _, r_indep = run_experiment(exp_indep, signal_design="walsh_solution",
                                    verbose=False)

        improvement = r_indep.residual - r_joint.residual
        fm = "DENIED" if improvement < 0 else ""
        return self._verdict(improvement,
                             notes=f"joint={r_joint.residual:.4f} indep={r_indep.residual:.4f}",
                             failure_mode=fm)


class TB_SignalIsNoise(Testbed):
    """
    Claim: signal exchange HARMS convergence on the identical-function experiment.
    If signal is noise, sig_weight=0 should outperform sig_weight=0.35.
    (This was confirmed in our baselines.py run — verifying it here formally.)
    """
    def run(self):
        import math
        from experiments.base import Experiment, rel_identical, run_experiment
        from toolkit.ga import CoevoConfig

        N = 8
        true_cos = np.array([(-1.)**(k//2) if k%2==0 else 0. for k in range(N)])
        base_cfg = dict(n_genes=N, pop_size=60, gens=300, seed=42,
                        n_collaborations=3, k_signal_samples=5, mut_std=0.04,
                        signal_design="walsh_landscape", sig_freq=10,
                        elite=2, tourney=4, delta=0.25, coupling_mode="adaptive_kl")

        exp = Experiment(
            object_id="TB-noise", name="signal noise test", domain="test",
            expression_a="cos", expression_b="cos",
            target_a=true_cos, target_b=true_cos,
            relationship=rel_identical, relationship_desc="A=B",
            tier=0, ground_truth="proven",
            joint_fitness_type=None,
        )

        exp.cfg = CoevoConfig(**{**base_cfg, "sig_weight": 0.35})
        _, r_signal = run_experiment(exp, signal_design="walsh_landscape", verbose=False)

        exp.cfg = CoevoConfig(**{**base_cfg, "sig_weight": 0.0, "seed": 43})
        _, r_nosig = run_experiment(exp, signal_design="walsh_landscape", verbose=False)

        harm = r_signal.residual - r_nosig.residual  # positive = signal is harmful
        notes = f"signal={r_signal.residual:.4f} nosignal={r_nosig.residual:.4f}"
        fm = "UNDER_CONSTRAINED" if harm < -0.02 else ""
        return self._verdict(harm, notes=notes, failure_mode=fm)


class TB_NormalisationKillsZero(Testbed):
    """
    Claim: adding a normalisation penalty to the relay chain step fitness
    prevents populations from collapsing to zero.
    """
    def run(self):
        from toolkit.equal_kl_relay import EqualKLConfig, run_equal_kl
        from toolkit.ga import CoevoConfig

        cfg = CoevoConfig(n_genes=8, pop_size=40, gens=150, seed=42,
                          n_collaborations=1, k_signal_samples=4, mut_std=0.05,
                          sig_weight=0.3, signal_design="trajectory_ngram", sig_freq=10,
                          elite=2, tourney=4, delta=0.25, coupling_mode="adaptive_kl")
        eq = EqualKLConfig(n_pops=4, cfg=cfg, norm_weight=0.4, equal_kl_weight=0.0,
                           cycle_weight=0.2)
        r = run_equal_kl(eq, log_every=150)
        mean_norm = float(np.mean([np.linalg.norm(p) for p in r.populations]))
        fm = "DEGENERATE" if mean_norm < 0.05 else ""
        return self._verdict(mean_norm, notes=f"mean_norm={mean_norm:.4f}",
                             failure_mode=fm)


# ── Default suite ─────────────────────────────────────────────────────────────

DEFAULT_SUITE = [
    TB_RelayEscapesZero(
        id="TB-01",
        claim="Relay chain does not collapse to zero — cycle constraint is non-trivial",
        conditions="4-pop relay, derivative steps, no normalisation",
        threshold=0.10,   # mean norm > 0.10
    ),
    TB_DirectionalBetterThanState(
        id="TB-02",
        claim="Directional signal (trajectory_ngram) gives lower cycle residual than state signal",
        conditions="4-pop relay, 200 gens, same seed",
        threshold=0.0,    # improvement > 0
    ),
    TB_JointFitnessBetterThanIndependent(
        id="TB-03",
        claim="Joint fitness over pairs gives lower derivative residual than independent fitness",
        conditions="cos/sin derivative, walsh_solution, 300 gens",
        threshold=0.5,    # improvement > 0.5
    ),
    TB_SignalIsNoise(
        id="TB-04",
        claim="Walsh landscape signal HARMS identical-function convergence (signal is noise here)",
        conditions="cos/cos identical, walsh_landscape, 300 gens",
        threshold=0.0,    # harm > 0 means signal hurts
    ),
    TB_NormalisationKillsZero(
        id="TB-05",
        claim="Normalisation penalty prevents relay chain from collapsing to zero",
        conditions="4-pop relay, norm_weight=0.4, 150 gens",
        threshold=0.10,   # mean norm > 0.10
    ),
]


if __name__ == "__main__":
    from notebook import db
    db.init()
    results = run_all(DEFAULT_SUITE)

    print("\nFAILURE TAXONOMY:")
    for r in results:
        if r.verdict not in ("CONFIRMED",):
            print(f"  {r.verdict:<20} — {r.claim[:55]}")
            if r.notes:
                print(f"                       notes: {r.notes}")
