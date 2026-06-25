"""
Information topology experiment matrix.

Each experiment varies EXACTLY ONE mathematical property from a baseline.
This is Mendel's "one trait at a time" discipline applied to signal molecules.

The question per experiment: does the signal design detect this property change?
If residual rises significantly vs. the baseline — the signal is blind to this property.
If residual stays low — the signal encodes this property.

Baseline: MM-T0-001 (identical cos, residual ~0.09 with walsh_population)

Properties tested in isolation:
  T0-P01: scalar ratio   (k × cos vs cos, varying k = 1.5, 2, 3, 5)
  T0-P02: additive shift (cos + c vs cos, varying c = 0.1, 0.5, 1.0, 2.0)
  T0-P03: phase shift    (cos(x+φ) vs cos(x), varying φ)  — Tier 0/1 boundary
  T0-P04: frequency      (cos(2x) vs cos(x))              — Tier 1 territory
  T0-P05: algebraic type (sin vs cos)                     — derivative relationship
  T0-P06: odd vs even    (sin vs cos directly without derivative framing)

The residual × property value curves are the data.
A flat curve means the signal is blind to that property dimension.
A rising curve means the signal is sensitive to that dimension.
Together they map the information topology of the signal molecule space.
"""

import math
import numpy as np
from experiments.base import Experiment, rel_identical, rel_scalar, rel_additive_const
from toolkit.ga import CoevoConfig

N = 8
X     = np.linspace(-1.5, 1.5, 120)
BASIS = np.stack([X**k / math.factorial(k) for k in range(N)], axis=1)

def _cos(n=N): return np.array([(-1.0)**(k//2) if k%2==0 else 0. for k in range(n)])
def _sin(n=N): return np.array([(-1.0)**((k-1)//2) if k%2==1 else 0. for k in range(n)])
def _cos_phase(phi, n=N):
    """cos(x+phi) Taylor coefficients in phi_k basis."""
    # cos(x+phi) = cos(x)cos(phi) - sin(x)sin(phi)
    return np.cos(phi) * _cos(n) - np.sin(phi) * _sin(n)

_cfg = CoevoConfig(
    n_genes=N, pop_size=80, gens=600,
    n_collaborations=5, k_signal_samples=8,
    mut_std=0.04, sig_weight=0.35, delta=0.25,
    coupling_mode="adaptive_kl",
)

INFO_TOPOLOGY_EXPERIMENTS = [

    # ── Scalar ratio sweep ────────────────────────────────────────────────────
    Experiment(
        object_id="MM-T0-P01a", name="Scalar ratio k=1.5 (cos vs 1.5*cos)",
        domain="algebra", tier=0, ground_truth="proven",
        expression_a="1.5*(truncated cos)", expression_b="truncated cos",
        target_a=1.5*_cos(), target_b=_cos(),
        relationship=rel_scalar(1.5),
        relationship_desc="A = 1.5*B",
        properties={"scalar_ratio": 1.5, "property_varied": "scalar_ratio"},
        notes="Scalar ratio sweep step 1. Compare residual to baseline (k=1, MM-T0-001).",
        cfg=_cfg,
    ),

    Experiment(
        object_id="MM-T0-P01b", name="Scalar ratio k=2 (cos vs 2*cos)",
        domain="algebra", tier=0, ground_truth="proven",
        expression_a="2*(truncated cos)", expression_b="truncated cos",
        target_a=2.0*_cos(), target_b=_cos(),
        relationship=rel_scalar(2.0),
        relationship_desc="A = 2*B",
        properties={"scalar_ratio": 2.0, "property_varied": "scalar_ratio"},
        notes="Same as MM-T0-003 but in the topology matrix context.",
        cfg=_cfg,
    ),

    Experiment(
        object_id="MM-T0-P01c", name="Scalar ratio k=5 (cos vs 5*cos)",
        domain="algebra", tier=0, ground_truth="proven",
        expression_a="5*(truncated cos)", expression_b="truncated cos",
        target_a=5.0*_cos(), target_b=_cos(),
        relationship=rel_scalar(5.0),
        relationship_desc="A = 5*B",
        properties={"scalar_ratio": 5.0, "property_varied": "scalar_ratio"},
        notes="Larger ratio — does signal capacity degrade monotonically with ratio?",
        cfg=_cfg,
    ),

    # ── Additive shift sweep ──────────────────────────────────────────────────
    Experiment(
        object_id="MM-T0-P02a", name="Additive shift c=0.1",
        domain="algebra", tier=0, ground_truth="proven",
        expression_a="cos + 0.1", expression_b="cos",
        target_a=_cos() + np.array([0.1,0,0,0,0,0,0,0]),
        target_b=_cos(),
        relationship=rel_additive_const(0.1),
        relationship_desc="A = B + [0.1, 0, ...]",
        properties={"additive_shift": 0.1, "property_varied": "additive_shift"},
        notes="Small shift. Should be easier than c=0.5 (MM-T0-004).",
        cfg=_cfg,
    ),

    Experiment(
        object_id="MM-T0-P02b", name="Additive shift c=1.0",
        domain="algebra", tier=0, ground_truth="proven",
        expression_a="cos + 1.0", expression_b="cos",
        target_a=_cos() + np.array([1.0,0,0,0,0,0,0,0]),
        target_b=_cos(),
        relationship=rel_additive_const(1.0),
        relationship_desc="A = B + [1.0, 0, ...]",
        properties={"additive_shift": 1.0, "property_varied": "additive_shift"},
        notes="Shift magnitude comparable to the function amplitude. More distinct?",
        cfg=_cfg,
    ),

    # ── Phase shift sweep — approaching Tier 1 ────────────────────────────────
    Experiment(
        object_id="MM-T0-P03a", name="Phase shift phi=pi/8",
        domain="analysis", tier=0, ground_truth="proven",
        expression_a="cos(x + pi/8)", expression_b="cos(x)",
        target_a=_cos_phase(math.pi/8), target_b=_cos(),
        relationship=lambda a, b: np.linalg.norm(a - _cos_phase(math.pi/8)),
        relationship_desc="A = cos(x + pi/8)",
        properties={"phase_shift": math.pi/8, "property_varied": "phase_shift"},
        notes="Small phase shift. cos(x+phi) = cos*cos(phi) - sin*sin(phi). "
              "Mixed even/odd character — does the signal detect the mixing?",
        cfg=_cfg,
    ),

    Experiment(
        object_id="MM-T0-P03b", name="Phase shift phi=pi/4",
        domain="analysis", tier=0, ground_truth="proven",
        expression_a="cos(x + pi/4)", expression_b="cos(x)",
        target_a=_cos_phase(math.pi/4), target_b=_cos(),
        relationship=lambda a, b: np.linalg.norm(a - _cos_phase(math.pi/4)),
        relationship_desc="A = cos(x + pi/4)",
        properties={"phase_shift": math.pi/4, "property_varied": "phase_shift"},
        notes="45-degree shift. Equal mix of cos and sin character.",
        cfg=_cfg,
    ),

    Experiment(
        object_id="MM-T0-P03c", name="Phase shift phi=pi/2 (cos -> sin)",
        domain="analysis", tier=0, ground_truth="proven",
        expression_a="cos(x + pi/2) = -sin(x)", expression_b="cos(x)",
        target_a=-_sin(), target_b=_cos(),
        relationship=lambda a, b: np.linalg.norm(a - (-_sin())),
        relationship_desc="A = -sin(x)  (cos shifted by pi/2)",
        properties={"phase_shift": math.pi/2, "property_varied": "phase_shift"},
        notes="90-degree shift = the derivative relationship. "
              "Same as Tier 1 cos/sin but framed as phase shift. "
              "Connects to MM-T0-009.",
        cfg=_cfg,
    ),
]


def register_info_topology():
    """Register all information topology experiments."""
    from notebook import db
    db.init()
    for exp in INFO_TOPOLOGY_EXPERIMENTS:
        db.upsert_object(
            id=exp.object_id, name=exp.name, domain=exp.domain,
            expression_a=exp.expression_a, expression_b=exp.expression_b,
            relationship=exp.relationship_desc,
            tier=exp.tier, ground_truth=exp.ground_truth,
            properties=exp.properties, notes=exp.notes,
        )
    print(f"Registered {len(INFO_TOPOLOGY_EXPERIMENTS)} information topology experiments.")
