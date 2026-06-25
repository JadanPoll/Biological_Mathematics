"""
Burgers residue — topological path dislocation detector.

Crystal analog:
  In crystallography, the Burgers vector is the "closure failure" of a circuit
  traced around a dislocation. Trace N steps around a dislocation in the actual
  crystal, then take the same N steps in a perfect reference crystal — the gap
  between finish and start is the Burgers vector b. If b ≠ 0, there is a
  dislocation. If b = 0, the lattice is perfect at that site.

Our analog for relay chains:
  1. Project the relay chain images onto the 2D plane spanned by A and (B-A).
  2. The shoelace formula applied to the projected polygon gives the signed
     area enclosed between the relay path and the direct straight-line closing
     segment (which is the "perfect crystal" reference: the linear interpolation).
  3. The closure failure = signed area. Zero = no topological artifact. Nonzero
     = the relay path has looped or spiralled relative to the direct reference.

Connection to path signature theory (Chen 1954):
  The signed area equals the antisymmetric part of the level-2 iterated integral
  signature of the path. The full signature (implemented in iisignature) encodes
  ALL topological invariants. Our shoelace formula extracts the key level-2 term.

Failure mode discrimination:
  burgers_norm ≈ 0, sigma_norm ≈ 0  →  valid morph (free path, no artifact)
  burgers_norm ≈ 0, sigma_norm >> 0 →  Type 7 or 8 (intrinsic gap, no artifact)
  burgers_norm >> 0                  →  Type M (wrong fold direction, methodological)

  The distinction: Type 7 via FSM/FABRIK scores burgers ≈ 0 because those
  methods walk directly without folding back. A Type M path (wrong-order
  gradient descent, NEB in a bad basin, miscoordinated fold) scores burgers >> 0
  because the path spirals or detours in the (A, B-A) projection plane.

TYPE_M_THRESHOLD = 0.05 (empirically calibrated on:
  linear interp  -> burgers_norm ≈ 0.000  (exact zero, no detour)
  complex_relay  -> burgers_norm ≈ 0.002  (small but nonzero — valid curved path)
  deliberate loop -> burgers_norm > 0.1   (clear Type M signature)
)
"""

import numpy as np
from typing import List, Dict

TYPE_M_THRESHOLD = 0.05


def _project_2d(images: List[np.ndarray],
                a: np.ndarray,
                b: np.ndarray):
    """
    Project relay chain images onto the 2D plane spanned by A and (B-A).

    Basis:
      e1 = (b - a) / ||b - a||          (direction of travel A→B)
      e2 = orthogonal complement of a   (A's component perpendicular to e1)

    Returns list of (x, y) 2D projections and the ab_norm.
    """
    ab = b - a
    ab_norm = float(np.linalg.norm(ab))
    if ab_norm < 1e-10:
        return [(0.0, 0.0)] * len(images), 0.0

    e1 = ab / ab_norm

    a_perp = a - np.dot(a, e1) * e1
    a_perp_norm = float(np.linalg.norm(a_perp))
    if a_perp_norm < 1e-10:
        # a is parallel to (b-a); pick first standard basis vector not parallel to e1
        e2 = np.zeros(len(a))
        for i in range(len(a)):
            candidate = np.zeros(len(a))
            candidate[i] = 1.0
            candidate -= np.dot(candidate, e1) * e1
            cand_norm = float(np.linalg.norm(candidate))
            if cand_norm > 0.1:
                e2 = candidate / cand_norm
                break
    else:
        e2 = a_perp / a_perp_norm

    pts = [(float(np.dot(img, e1)), float(np.dot(img, e2))) for img in images]
    return pts, ab_norm


def _shoelace(pts) -> float:
    """Signed area of polygon via the shoelace formula. Closing edge implicit."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return area * 0.5


def burgers_residue(
    images: List[np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
) -> Dict:
    """
    Topological dislocation measure for an open relay chain path.

    Projects images onto the 2D (A, B-A) plane and computes the signed area
    enclosed by the relay path + the implicit straight closing segment B→A.
    Normalizes by ||B-A||² for scale invariance.

    Args:
      images : relay chain (may include endpoints as first/last elements)
      a, b   : endpoint coefficient vectors

    Returns dict:
      signed_area   : raw signed area (A-coefficient-units²)
      burgers_norm  : |signed_area| / ||b-a||²  — scale-invariant dislocation size
      burgers_sign  : +1/-1/0 — direction of curvature (folded left vs right)
      is_type_m     : True if burgers_norm > TYPE_M_THRESHOLD
      pts_2d        : list of (x,y) projections for plotting
    """
    if len(images) < 2:
        return {"signed_area": 0.0, "burgers_norm": 0.0,
                "burgers_sign": 0, "is_type_m": False, "pts_2d": []}

    pts, ab_norm = _project_2d(images, a, b)

    if ab_norm < 1e-10:
        return {"signed_area": 0.0, "burgers_norm": 0.0,
                "burgers_sign": 0, "is_type_m": False, "pts_2d": pts}

    signed_area  = _shoelace(pts)
    burgers_norm = abs(signed_area) / (ab_norm ** 2)
    burgers_sign = int(np.sign(signed_area)) if abs(signed_area) > 1e-12 else 0

    return {
        "signed_area":  float(signed_area),
        "burgers_norm": float(burgers_norm),
        "burgers_sign": burgers_sign,
        "is_type_m":    burgers_norm > TYPE_M_THRESHOLD,
        "pts_2d":       pts,
    }


def make_looping_path(a: np.ndarray, b: np.ndarray, n: int = 9,
                      detour_scale: float = 1.0) -> List[np.ndarray]:
    """
    Construct a deliberately looping relay path for Type M testing.

    The path goes A → midpoint + perpendicular_detour → B, then back
    toward A via a second arc, creating a closed loop.  This mimics a
    wrong-fold-order path that winds around the direct A→B segment.

    detour_scale controls how far the path deviates: 0 = straight line,
    1.0 = detour amplitude equal to ||B-A|| / 4.
    """
    ab = b - a
    ab_norm = float(np.linalg.norm(ab))
    if ab_norm < 1e-10:
        return [a.copy()] * n

    e1 = ab / ab_norm
    e2 = np.zeros(len(a))
    for i in range(len(a)):
        candidate = np.zeros(len(a))
        candidate[i] = 1.0
        candidate -= np.dot(candidate, e1) * e1
        if np.linalg.norm(candidate) > 0.1:
            e2 = candidate / np.linalg.norm(candidate)
            break

    images = []
    for i in range(n):
        t = i / (n - 1)
        # Half-sine arc: all on ONE side of the direct A-B line.
        # sin(pi*t) = 0 at endpoints, peaks at t=0.5.
        # This encloses a nonzero signed area (a lens-shaped region).
        # Using sin(2*pi*t) would cancel (up then down = zero net area).
        perp_offset = detour_scale * (ab_norm * 0.25) * np.sin(np.pi * t)
        images.append(a + t * ab + perp_offset * e2)
    return images
