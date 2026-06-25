"""
Tier 0 experiment declarations — the true-breeding controls.

Adding a new Tier 0 experiment = one Experiment(...) appended to this list.
No new files. No new scripts. No new boilerplate.
"""

import math
import numpy as np

from experiments.base import (Experiment,
                               rel_identical, rel_scalar, rel_derivative,
                               rel_additive_const)
from toolkit.ga import CoevoConfig

N = 8

def _cos(n=N): return np.array([(-1.0)**(k//2) if k%2==0 else 0. for k in range(n)])
def _sin(n=N): return np.array([(-1.0)**((k-1)//2) if k%2==1 else 0. for k in range(n)])
def _lin(slope=1., intercept=0., n=N):
    c = np.zeros(n); c[0]=intercept; c[1]=slope; return c
def _quad(a=1., n=N):
    c = np.zeros(n); c[2]=a; return c

_cfg = CoevoConfig(
    n_genes=N, pop_size=80, gens=600,
    n_collaborations=5, k_signal_samples=8,
    mut_std=0.04, sig_weight=0.35, delta=0.25,
    coupling_mode="adaptive_kl",
)

TIER0_EXPERIMENTS = [

    Experiment(
        object_id="MM-T0-001",
        name="Identical function — cos truncation",
        domain="analysis",
        expression_a="1 - x^2/2 + x^4/24 - x^6/720",
        expression_b="1 - x^2/2 + x^4/24 - x^6/720",
        target_a=_cos(), target_b=_cos(),
        relationship=rel_identical,
        relationship_desc="A = B  (identical)",
        tier=0, ground_truth="proven",
        properties={"oscillatory": True, "even": True},
        notes="Absolute floor. Both fit the same function. Residual=0 is the ceiling.",
        cfg=_cfg,
    ),

    Experiment(
        object_id="MM-T0-002",
        name="Identical function — sin truncation",
        domain="analysis",
        expression_a="x - x^3/6 + x^5/120 - x^7/5040",
        expression_b="x - x^3/6 + x^5/120 - x^7/5040",
        target_a=_sin(), target_b=_sin(),
        relationship=rel_identical,
        relationship_desc="A = B  (identical)",
        tier=0, ground_truth="proven",
        properties={"oscillatory": True, "odd": True},
        notes="Odd-function floor. Walsh main-effects should cluster at odd k.",
        cfg=_cfg,
    ),

    Experiment(
        object_id="MM-T0-003",
        name="Scalar multiple — 2*cos vs cos",
        domain="algebra",
        expression_a="2*(1 - x^2/2 + x^4/24 - x^6/720)",
        expression_b="1 - x^2/2 + x^4/24 - x^6/720",
        target_a=2*_cos(), target_b=_cos(),
        relationship=rel_scalar(2.0),
        relationship_desc="A = 2*B",
        tier=0, ground_truth="proven",
        properties={"even": True, "scalar_relation": True},
        notes="First non-trivial Tier 0. v1 failed here (cycle penalty fought A=2B). "
              "Test whether signal designs handle scaling correctly.",
        cfg=_cfg,
    ),

    Experiment(
        object_id="MM-T0-004",
        name="Additive constant shift — cos+0.5 vs cos",
        domain="algebra",
        expression_a="(1 - x^2/2 + ...) + 0.5",
        expression_b="1 - x^2/2 + ...",
        target_a=_cos() + np.array([0.5,0,0,0,0,0,0,0]),
        target_b=_cos(),
        relationship=rel_additive_const(0.5),
        relationship_desc="A = B + [0.5, 0, ...]  (constant shift in c_0)",
        tier=0, ground_truth="proven",
        properties={"even": True, "shift_relation": True},
        notes="Only c_0 differs. Walsh main-effect for k=0 should dominate.",
        cfg=_cfg,
    ),

    Experiment(
        object_id="MM-T0-005",
        name="Even symmetry — cos(x) vs cos(-x)",
        domain="symmetry",
        expression_a="cos(x)",
        expression_b="cos(-x) = cos(x)  [even function]",
        target_a=_cos(), target_b=_cos(),
        relationship=rel_identical,
        relationship_desc="A = B  (even: f(x) = f(-x))",
        tier=0, ground_truth="proven",
        properties={"even": True, "reflection_invariant": True},
        notes="Reflection symmetry check. Should match MM-T0-001 exactly.",
        cfg=_cfg,
    ),

    Experiment(
        object_id="MM-T0-006",
        name="Pure linear — 2x vs 2x",
        domain="algebra",
        expression_a="2x",
        expression_b="2x",
        target_a=_lin(slope=2.), target_b=_lin(slope=2.),
        relationship=rel_identical,
        relationship_desc="A = B  (identical linear)",
        tier=0, ground_truth="proven",
        properties={"linear": True},
        notes="Simplest possible case. Walsh: only k=1 main-effect nonzero. "
              "This is the 'additive' spectral signature the document describes.",
        cfg=_cfg,
    ),

    Experiment(
        object_id="MM-T0-007",
        name="Linear sum — x+x vs 2x",
        domain="algebra",
        expression_a="x + x  (written as sum)",
        expression_b="2x  (written as scalar multiple)",
        target_a=_lin(slope=2.), target_b=_lin(slope=2.),
        relationship=rel_identical,
        relationship_desc="A = B  (x+x = 2x, same coefficients)",
        tier=0, ground_truth="proven",
        properties={"linear": True},
        notes="The canonical Tier 0 example from the document. "
              "Both representations identical in phi_k basis.",
        cfg=_cfg,
    ),

    Experiment(
        object_id="MM-T0-008",
        name="Pure quadratic — x^2/2 vs x^2/2",
        domain="algebra",
        expression_a="x^2 / 2",
        expression_b="x^2 / 2",
        target_a=_quad(), target_b=_quad(),
        relationship=rel_identical,
        relationship_desc="A = B  (identical quadratic)",
        tier=0, ground_truth="proven",
        properties={"quadratic": True, "even": True},
        notes="Second-order landscape. Walsh: k=2 dominates. "
              "Slower convergence than linear expected (higher epistasis).",
        cfg=_cfg,
    ),

    # ── Joint fitness versions (what the .md actually proposed) ──────────────
    # These use fitness defined over PAIRS — no fixed external targets.
    # Both populations jointly minimise the description length of their relationship.
    Experiment(
        object_id="MM-T0-J01",
        name="Joint identical (cos) — no fixed targets",
        domain="analysis",
        expression_a="cos [no fixed target]",
        expression_b="cos [no fixed target]",
        target_a=_cos(), target_b=_cos(),   # used only for initialisation hint
        relationship=rel_identical,
        relationship_desc="A = B  (joint fitness: minimise ||A-B||)",
        tier=0, ground_truth="proven",
        properties={"oscillatory": True, "joint_fitness": True},
        joint_fitness_type="identical",
        notes="Joint fitness baseline. Both populations minimise ||A-B|| directly. "
              "No external target. This is the .md's proposed architecture.",
        cfg=_cfg,
    ),

    Experiment(
        object_id="MM-T0-J02",
        name="Joint derivative (cos/sin) — no fixed targets",
        domain="calculus",
        expression_a="cos [no fixed target]",
        expression_b="sin [no fixed target]",
        target_a=_cos(), target_b=_sin(),
        relationship=rel_derivative(),
        relationship_desc="A = d/dx(B)  (joint fitness: minimise ||A - shift_left(B)||)",
        tier=0, ground_truth="proven",
        properties={"oscillatory": True, "joint_fitness": True, "derivative_relation": True},
        joint_fitness_type="derivative",
        notes="The key test of the .md's architecture. Joint fitness rewards the "
              "derivative relationship without being told what the targets are. "
              "Compare residual and betweenness to MM-T0-009 (independent fitness).",
        cfg=_cfg,
    ),

    Experiment(
        object_id="MM-T0-J03",
        name="Joint scalar x2 (cos/2*cos) — no fixed targets",
        domain="algebra",
        expression_a="2*cos [no fixed target]",
        expression_b="cos [no fixed target]",
        target_a=2*_cos(), target_b=_cos(),
        relationship=rel_scalar(2.0),
        relationship_desc="A = 2*B  (joint fitness: minimise ||A - 2*B||)",
        tier=0, ground_truth="proven",
        properties={"joint_fitness": True, "scalar_relation": True},
        joint_fitness_type="scalar_multiple",
        joint_fitness_kwargs={"k": 2.0},
        notes="Scalar multiple with joint fitness. Previously failed with independent "
              "fitness (residual 2.3). Joint fitness should collapse this to ~0.",
        cfg=_cfg,
    ),

    Experiment(
        object_id="MM-T0-J04",
        name="Joint discovery (cos/sin) — relationship unknown",
        domain="calculus",
        expression_a="cos [discovery mode]",
        expression_b="sin [discovery mode]",
        target_a=_cos(), target_b=_sin(),
        relationship=rel_derivative(),
        relationship_desc="A = d/dx(B)  (system discovers the transformation)",
        tier=0, ground_truth="proven",
        properties={"oscillatory": True, "joint_fitness": True, "discovery_mode": True},
        joint_fitness_type="discovery",
        notes="Discovery mode: the system searches over the transformation grammar "
              "to find the shortest path from B to A. It should find 'derivative' "
              "as the winning transformation. This is the path-finding objective.",
        cfg=_cfg,
    ),

    # ── Approaching Tier 1 ────────────────────────────────────────────────────
    Experiment(
        object_id="MM-T0-009",
        name="Derivative seed — cos vs sin  [Tier 0 boundary]",
        domain="calculus",
        expression_a="cos(x)  [8-term]",
        expression_b="sin(x)  [8-term]",
        target_a=_cos(), target_b=_sin(),
        relationship=rel_derivative(),
        relationship_desc="A = d/dx(B)  (shift left in phi_k basis)",
        tier=0, ground_truth="proven",
        properties={"oscillatory": True, "derivative_relation": True},
        notes="This is the Rung 0 experiment from the prototype. "
              "Listed as Tier 0 boundary because the relationship is PROVEN "
              "and algebraically trivial in phi_k basis. "
              "Key test: do signal designs that encode 'mathematical character' "
              "converge faster / to lower residual than gradient signals?",
        cfg=_cfg,
    ),
]
