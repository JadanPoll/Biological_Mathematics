"""
Joint fitness functions — fitness defined over PAIRS (A, B), not individuals.

This is what the .md proposed from the start (Section 3.1):
  "fitness of any individual in A is evaluated by pairing with current
   best from B... computing joint MDL"

The key difference from independent fitness:
  - No fixed external targets
  - A's fitness depends on B's current best (and vice versa)
  - What gets minimized is the RELATIONSHIP between A and B,
    not how well each fits its own fixed data

For KNOWN controls (Tier 0): we specify the expected relationship explicitly.
  joint_fitness(A, B) = -||relationship(A, B)||_2

For DISCOVERY (Tier 1+): we minimize over a grammar of simple transformations.
  joint_fitness(A, B) = -min_{T in grammar} [||T(B) - A||_2 + lambda * |T|]
  where |T| is the description length of the transformation T.

The discovered T is the relationship. Its description length is the MDL cost.
Short T = simple geometric path = the relationship is algebraically shallow.
Long T  = complex path = the relationship requires deep algebraic lifting.

This is the path-finding objective, not the endpoint-checking objective.
"""

import math
import numpy as np
from typing import Callable

# ── Shared polynomial basis ───────────────────────────────────────────────────
X     = np.linspace(-1.5, 1.5, 120)
N     = 8
BASIS = np.stack([X**k / math.factorial(k) for k in range(N)], axis=1)

def poly_eval(c: np.ndarray) -> np.ndarray:
    return BASIS[:, :len(c)] @ c


# ── Known-relationship joint fitness (for controls) ───────────────────────────

def joint_identical(lambda_mdl: float = 0.0):
    """Both populations should converge to the same function. A = B."""
    def _fit(ind_a, ind_b):
        return -np.linalg.norm(ind_a - ind_b)
    return _fit


def joint_scalar(k: float, lambda_mdl: float = 1e-3):
    """A should be k times B. A = k*B."""
    def _fit(ind_a, ind_b):
        relationship_cost = np.linalg.norm(ind_a - k * ind_b)
        mdl_cost = lambda_mdl * abs(math.log2(abs(k) + 1))  # description length of k
        return -(relationship_cost + mdl_cost)
    return _fit


def joint_additive_shift(shift: float, lambda_mdl: float = 1e-3):
    """A = B + shift*e_0  (constant added to c_0 only)."""
    def _fit(ind_a, ind_b):
        expected = ind_b.copy()
        expected[0] += shift
        return -np.linalg.norm(ind_a - expected)
    return _fit


def joint_derivative(lambda_mdl: float = 1e-3):
    """A = d/dx(B) in phi_k basis: a_j = b_{j+1}."""
    def _fit(ind_a, ind_b):
        deriv_b = np.append(ind_b[1:], 0.)
        return -np.linalg.norm(ind_a - deriv_b)
    return _fit


# ── Discovery joint fitness (for unknown relationships) ───────────────────────

# Grammar of simple transformations — the search space for T
# Each entry: (name, transformation, description_length_bits)
_GRAMMAR = [
    ("identity",        lambda b: b.copy(),                   0),
    ("scale_2",         lambda b: 2.0 * b,                    2),
    ("scale_0.5",       lambda b: 0.5 * b,                    2),
    ("scale_-1",        lambda b: -1.0 * b,                   2),
    ("shift_+0.5",      lambda b: b + np.array([.5]+[0]*(len(b)-1)), 4),
    ("shift_-0.5",      lambda b: b + np.array([-.5]+[0]*(len(b)-1)), 4),
    ("derivative",      lambda b: np.append(b[1:], 0.),       3),
    ("antiderivative",  lambda b: np.append([0.], b[:-1]),     3),
    ("reflect_even",    lambda b: np.array([c if i%2==0 else 0. for i,c in enumerate(b)]), 4),
    ("reflect_odd",     lambda b: np.array([0. if i%2==0 else c for i,c in enumerate(b)]), 4),
    ("scale_pi",        lambda b: math.pi * b,                4),
    ("scale_e",         lambda b: math.e * b,                 4),
]


def joint_discovery(lambda_mdl: float = 0.05, grammar=None):
    """
    Discovery mode: find the simplest T in the grammar such that T(B) ≈ A.
    Fitness = -min_T [||T(B) - A|| + lambda * description_length(T)]

    This is the MDL joint objective from the document:
    shorter descriptions = simpler geometric paths = shallower algebraic relationship.

    The discovered transformation IS the relationship hypothesis.
    Record the best T for each run — that's the Mendel observation.
    """
    g = grammar or _GRAMMAR

    def _fit(ind_a, ind_b):
        best_score = float('inf')
        for name, transform, dl in g:
            try:
                t_b = transform(ind_b)
                residual = np.linalg.norm(ind_a - t_b)
                score = residual + lambda_mdl * dl
                if score < best_score:
                    best_score = score
            except Exception:
                pass
        return -best_score

    return _fit


def best_transformation(ind_a: np.ndarray, ind_b: np.ndarray,
                        grammar=None) -> tuple[str, float, float]:
    """
    Find which grammar element best maps B to A.
    Returns (transformation_name, residual, description_length).
    This is the relationship hypothesis — what the system thinks A and B are doing.
    """
    g = grammar or _GRAMMAR
    best = ("none", float('inf'), 0)
    for name, transform, dl in g:
        try:
            residual = float(np.linalg.norm(ind_a - transform(ind_b)))
            if residual < best[1]:
                best = (name, residual, dl)
        except Exception:
            pass
    return best


# ── Fitness function factory ──────────────────────────────────────────────────

JOINT_FITNESS_REGISTRY = {
    "identical":          joint_identical,
    "scalar_multiple":    joint_scalar,
    "additive_shift":     joint_additive_shift,
    "derivative":         joint_derivative,
    "discovery":          joint_discovery,
}


def make_pair_fitness(relationship_type: str, **kwargs) -> Callable:
    """
    Returns a fitness function f(ind_a, ind_b) -> float.
    Both populations use the same function — fitness is symmetric over the pair.

    Usage:
        fit = make_pair_fitness("derivative")
        fitness_a = lambda ind, partner: fit(ind, partner)
        fitness_b = lambda ind, partner: fit(partner, ind)
    """
    if relationship_type not in JOINT_FITNESS_REGISTRY:
        raise ValueError(f"Unknown relationship type '{relationship_type}'. "
                         f"Choose from: {list(JOINT_FITNESS_REGISTRY)}")
    return JOINT_FITNESS_REGISTRY[relationship_type](**kwargs)
