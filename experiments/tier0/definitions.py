"""
Tier 0 Mathematical Object Registry.

These are definitionally equivalent pairs — no algebraic lifting required.
They calibrate the FLOOR of the instrument:
  - What does "trivially liftable" look like spectrally?
  - How fast does the joint attractor form when A and B are identical?
  - What cycle-consistency residual corresponds to a correct Tier 0 attractor?

All pairs use the phi_k(x) = x^k/k! basis on X = [-1.5, 1.5].

Mendel equivalent: these are the true-breeding pea strains.
Every harder experiment is interpreted relative to these baselines.
"""

import math
import numpy as np
from notebook import db

X     = np.linspace(-1.5, 1.5, 120)
BASIS = np.stack([X**k / math.factorial(k) for k in range(8)], axis=1)


def poly_eval(c):
    return BASIS[:, :len(c)] @ c


# ── True coefficient vectors for known functions ──────────────────────────────

def _cos_coeffs(n=8):
    return np.array([(-1.0)**(k//2) if k%2==0 else 0.0 for k in range(n)])

def _sin_coeffs(n=8):
    return np.array([(-1.0)**((k-1)//2) if k%2==1 else 0.0 for k in range(n)])

def _linear_coeffs(slope=1.0, intercept=0.0, n=8):
    c = np.zeros(n); c[0] = intercept; c[1] = slope; return c

def _quad_coeffs(a=1.0, n=8):
    c = np.zeros(n); c[2] = a; return c   # a * x^2/2! in phi_k basis


# ── Tier 0 pair definitions ───────────────────────────────────────────────────

TIER0_OBJECTS = [

    dict(
        id="MM-T0-001",
        name="Same function (cos truncation)",
        domain="analysis",
        expression_a="1 - x^2/2",
        expression_b="1 - x^2/2",
        relationship="A = B  (identical)",
        tier=0,
        ground_truth="proven",
        properties={"oscillatory": True, "even": True, "additive_structure": True},
        target_a=_cos_coeffs(),
        target_b=_cos_coeffs(),
        expected_residual=lambda a, b: np.linalg.norm(a - b),   # should be ~0
        notes="Floor calibration. Both populations fit the same function. "
              "Joint attractor: A = B. Cycle residual = 0.",
    ),

    dict(
        id="MM-T0-002",
        name="Same function (sin truncation)",
        domain="analysis",
        expression_a="x - x^3/6",
        expression_b="x - x^3/6",
        relationship="A = B  (identical)",
        tier=0,
        ground_truth="proven",
        properties={"oscillatory": True, "odd": True, "additive_structure": True},
        target_a=_sin_coeffs(),
        target_b=_sin_coeffs(),
        expected_residual=lambda a, b: np.linalg.norm(a - b),
        notes="Odd-function floor calibration.",
    ),

    dict(
        id="MM-T0-003",
        name="Scalar multiple  (2*cos = cos + cos)",
        domain="algebra",
        expression_a="2*(1 - x^2/2)",
        expression_b="1 - x^2/2",
        relationship="A = 2*B",
        tier=0,
        ground_truth="proven",
        properties={"even": True, "scalar_relation": True},
        target_a=2 * _cos_coeffs(),
        target_b=_cos_coeffs(),
        expected_residual=lambda a, b: np.linalg.norm(a - 2*b),
        notes="Tests whether the joint attractor encodes a scalar multiple. "
              "Walsh spectra of A and B should differ only in magnitude.",
    ),

    dict(
        id="MM-T0-004",
        name="Additive shift  (cos + 0.5  vs  cos)",
        domain="algebra",
        expression_a="(1 - x^2/2) + 0.5",
        expression_b="1 - x^2/2",
        relationship="A = B + [0.5, 0, 0, ...]",
        tier=0,
        ground_truth="proven",
        properties={"even": True, "shift_relation": True},
        target_a=_cos_coeffs() + np.array([0.5, 0, 0, 0, 0, 0, 0, 0]),
        target_b=_cos_coeffs(),
        expected_residual=lambda a, b: np.linalg.norm(
            a - (b + np.array([0.5] + [0]*7))
        ),
        notes="Tests constant-shift detection. Only c_0 differs.",
    ),

    dict(
        id="MM-T0-005",
        name="Even-function reflection  (cos(x) vs cos(-x))",
        domain="symmetry",
        expression_a="cos(x)  [truncated 8-term]",
        expression_b="cos(-x) = cos(x)  [truncated 8-term]",
        relationship="A = B  (even symmetry)",
        tier=0,
        ground_truth="proven",
        properties={"even": True, "reflection_invariant": True},
        target_a=_cos_coeffs(),
        target_b=_cos_coeffs(),   # cos(-x) = cos(x) — same coefficients
        expected_residual=lambda a, b: np.linalg.norm(a - b),
        notes="Even functions are reflection-invariant. "
              "Spectral signature should be identical to MM-T0-001.",
    ),

    dict(
        id="MM-T0-006",
        name="Linear function  (2x vs 2x)",
        domain="algebra",
        expression_a="2x",
        expression_b="2x",
        relationship="A = B  (identical linear)",
        tier=0,
        ground_truth="proven",
        properties={"linear": True, "additive_structure": True},
        target_a=_linear_coeffs(slope=2.0),
        target_b=_linear_coeffs(slope=2.0),
        expected_residual=lambda a, b: np.linalg.norm(a - b),
        notes="Simplest possible case. Establishes absolute floor spectral signature "
              "for purely linear (first-order Walsh only) landscape.",
    ),

    dict(
        id="MM-T0-007",
        name="Linear sum decomposition  (x+x vs 2x)",
        domain="algebra",
        expression_a="x + x  (sum)",
        expression_b="2x  (scalar)",
        relationship="A = B  (x+x = 2x)",
        tier=0,
        ground_truth="proven",
        properties={"linear": True, "additive_structure": True},
        target_a=_linear_coeffs(slope=2.0),
        target_b=_linear_coeffs(slope=2.0),
        expected_residual=lambda a, b: np.linalg.norm(a - b),
        notes="The canonical Mendel Tier 0 example from the document. "
              "Both representations evaluate identically in coefficient vector space.",
    ),

    dict(
        id="MM-T0-008",
        name="Quadratic  (x^2 vs x^2)",
        domain="algebra",
        expression_a="x^2/2",
        expression_b="x^2/2",
        relationship="A = B  (identical quadratic)",
        tier=0,
        ground_truth="proven",
        properties={"quadratic": True, "even": True},
        target_a=_quad_coeffs(),
        target_b=_quad_coeffs(),
        expected_residual=lambda a, b: np.linalg.norm(a - b),
        notes="Pure second-order landscape. Walsh signal should show "
              "dominant main-effect only at k=2. No first-order (no linear term).",
    ),
]


def register_all():
    """Write all Tier 0 objects to the notebook database."""
    db.init()
    for obj in TIER0_OBJECTS:
        db.upsert_object(
            id=obj["id"],
            name=obj["name"],
            domain=obj["domain"],
            expression_a=obj["expression_a"],
            expression_b=obj["expression_b"],
            relationship=obj["relationship"],
            tier=obj["tier"],
            ground_truth=obj["ground_truth"],
            properties=obj["properties"],
            notes=obj["notes"],
        )
    print(f"Registered {len(TIER0_OBJECTS)} Tier 0 objects.")
