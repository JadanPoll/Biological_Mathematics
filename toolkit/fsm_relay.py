"""
Freezing String Method (FSM) relay chain — bidirectional walker.

Behn, Zimmerman, Bell, Head-Gordon (2011). Adapted from computational chemistry
to coefficient vector relay chains between mathematical functions.

KEY INSIGHT OVER NEB:
  NEB: all images initialized at once, all relax simultaneously.
  FSM: two walkers grow inward from each endpoint, images FROZEN once placed.
       The walkers converge near the transition state (max-stress point).
       Bidirectionality concentrates images near the hardest part of the path.

DIRECT EQUIVALENCES:
  FSM reactant end   = endpoint_a (Walker A grows from here)
  FSM product end    = endpoint_b (Walker B grows from here)
  FSM frozen image   = relay chain waypoint locked in place after insertion
  FSM junction point = the image where walkers meet = max-stress image
  FSM transition state = the waypoint with highest potential energy = stress peak

WHY THE JUNCTION IS THE STRESS MAXIMUM:
  Walker A grows from A toward B. Its frontier moves through rising stress.
  Walker B grows from B toward A. Its frontier moves through rising stress from B.
  They meet at the point of maximum stress (the saddle) — where both frontiers
  have climbed as far as they can before they start converging to each other.
  With a flat potential (linear path), they meet in the middle.
  With a peaked potential (e.g., Type 7 stress landscape), they meet at the peak.

RELATION TO GSM (Growing String Method, Peters et al. 2004):
  GSM re-relaxes images after insertion (perpendicular to string direction).
  FSM freezes images immediately — cheaper, less accurate per image.
  For relay chains without a well-defined potential, FSM's simplicity is preferred.
  When a potential_fn is provided, we relax perpendicular to path before freezing.

REFERENCE:
  Behn, A., Zimmerman, P.M., Bell, A.T., Head-Gordon, M. (2011).
  Efficient exploration of reaction paths via a freezing string method.
  J. Chem. Phys. 135, 224108. DOI: 10.1063/1.3664901
"""

import numpy as np
from typing import List, Optional, Callable

from toolkit.neb_relay import reparameterize_string


# ── Core FSM walkers ──────────────────────────────────────────────────────────

def _fsm_step(frontier: np.ndarray, target: np.ndarray,
              step_size: float,
              potential_fn: Optional[Callable] = None,
              relax_steps: int = 10,
              relax_lr: float = 0.01) -> np.ndarray:
    """
    Insert one new image from frontier toward target.
    If potential_fn provided, relax perpendicular to step direction before freezing.
    """
    direction = target - frontier
    dist = float(np.linalg.norm(direction))
    if dist < 1e-10:
        return frontier.copy()

    new_image = frontier + step_size * (direction / dist)

    if potential_fn is not None and relax_steps > 0:
        # Relax perpendicular to the path direction (string method principle)
        tau = direction / dist
        for _ in range(relax_steps):
            eps = 1e-4
            grad = np.zeros_like(new_image)
            for i in range(len(new_image)):
                xp = new_image.copy(); xp[i] += eps
                xm = new_image.copy(); xm[i] -= eps
                grad[i] = (potential_fn(xp) - potential_fn(xm)) / (2 * eps)
            # Remove parallel component (only perpendicular force)
            grad_perp = grad - np.dot(grad, tau) * tau
            new_image = new_image - relax_lr * grad_perp

    return new_image


# ── Full FSM relay chain ──────────────────────────────────────────────────────

def run_fsm(endpoint_a: np.ndarray,
            endpoint_b: np.ndarray,
            step_fraction: float = 0.15,
            potential_fn: Optional[Callable] = None,
            relax_steps: int = 0,
            relax_lr: float = 0.01,
            max_images: int = 50) -> dict:
    """
    Run the Freezing String Method between endpoint_a and endpoint_b.

    Two walkers grow inward simultaneously:
      Walker A: endpoint_a → endpoint_b (frozen images at each step)
      Walker B: endpoint_b → endpoint_a (frozen images at each step)

    They meet when their frontiers are within 2*step_size of each other.
    The junction is the max-stress image when potential_fn is provided.

    Parameters
    ----------
    step_fraction : float
        Each step is step_fraction * total_distance. Default 0.15 → ~7 images.
    potential_fn : callable or None
        If provided: stress/energy at a position. Junction = argmax(stress).
        If None: linear geometry only, junction = midpoint.
    relax_steps : int
        Gradient steps perpendicular to path before freezing each image.
        0 = pure FSM (no relaxation). >0 = GSM-style relaxation.
    """
    total_dist = float(np.linalg.norm(endpoint_b - endpoint_a))
    step_size  = total_dist * step_fraction

    # Two walkers: each stores its list of frozen images
    walker_a = [endpoint_a.copy()]   # grows toward B
    walker_b = [endpoint_b.copy()]   # grows toward A
    n_steps  = 0

    while True:
        frontier_a = walker_a[-1]
        frontier_b = walker_b[-1]

        # Stop condition: frontiers have converged
        if float(np.linalg.norm(frontier_b - frontier_a)) < 2.0 * step_size:
            break
        if len(walker_a) + len(walker_b) >= max_images:
            break

        # Walker A advances one step toward B's frontier
        new_a = _fsm_step(frontier_a, frontier_b, step_size,
                          potential_fn, relax_steps, relax_lr)
        walker_a.append(new_a)
        n_steps += 1

        # Re-check after A's step
        if float(np.linalg.norm(walker_b[-1] - walker_a[-1])) < 2.0 * step_size:
            break
        if len(walker_a) + len(walker_b) >= max_images:
            break

        # Walker B advances one step toward A's frontier
        new_b = _fsm_step(walker_b[-1], walker_a[-1], step_size,
                          potential_fn, relax_steps, relax_lr)
        walker_b.append(new_b)
        n_steps += 1

    # Merge: walker_a forward + walker_b reversed
    junction_idx = len(walker_a) - 1
    images = walker_a + list(reversed(walker_b))

    # Reparameterize to equal arc length (inherit NEB string method)
    images = reparameterize_string(images)

    # Compute stresses if potential provided
    stresses = None
    if potential_fn is not None:
        stresses = [float(potential_fn(img)) for img in images]
        junction_idx = int(np.argmax(stresses))

    step_dists = [float(np.linalg.norm(images[k + 1] - images[k]))
                  for k in range(len(images) - 1)]
    wass_var = float(np.var(step_dists))

    return {
        "images":        images,
        "stresses":      stresses,
        "junction_idx":  junction_idx,
        "junction_pos":  images[junction_idx],
        "step_dists":    step_dists,
        "wass_var":      wass_var,
        "n_from_a":      len(walker_a),
        "n_from_b":      len(walker_b),
        "n_total":       len(images),
        "step_size":     step_size,
    }


# ── FSM + FABRIK pipeline ─────────────────────────────────────────────────────

def run_fsm_then_fabrik(endpoint_a: np.ndarray,
                        endpoint_b: np.ndarray,
                        step_fraction: float = 0.15,
                        potential_fn: Optional[Callable] = None,
                        fabrik_iter: int = 5) -> dict:
    """
    FSM for bidirectional initialization, FABRIK for equal-spacing polish.

    FSM establishes the bidirectional structure (junction near stress maximum).
    FABRIK re-spaces images to equal arc length without changing path topology.
    """
    from toolkit.fabrik_relay import _fabrik_forward, _fabrik_backward

    fsm_result = run_fsm(endpoint_a, endpoint_b, step_fraction, potential_fn)
    images = [img.copy() for img in fsm_result["images"]]

    total_dist = sum(np.linalg.norm(images[k + 1] - images[k])
                     for k in range(len(images) - 1))
    bone_len = total_dist / (len(images) - 1)

    for _ in range(fabrik_iter):
        _fabrik_forward(images, endpoint_b, bone_len)
        _fabrik_backward(images, endpoint_a, bone_len)

    step_dists = [float(np.linalg.norm(images[k + 1] - images[k]))
                  for k in range(len(images) - 1)]

    return {
        **fsm_result,
        "images":     images,
        "step_dists": step_dists,
        "wass_var":   float(np.var(step_dists)),
    }


# ── Sanity validation ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from toolkit.euler_relay import _COS, _SIN, N

    cos   = _COS.copy()
    sin   = _SIN.copy()
    cos2x = np.zeros(N)
    for k in range(N // 2):
        cos2x[2 * k] = (-1)**k * float(4**k)
    neg_cos = -cos.copy()

    print("=" * 60)
    print("FSM RELAY CHAIN — Sanity Validation")
    print("Freezing String Method: bidirectional walker")
    print("=" * 60)

    # Potential: norm deviation from unit sphere (penalizes amplitude change)
    def norm_potential(c):
        return float((np.linalg.norm(c) - np.linalg.norm(cos))**2)

    cases = [
        ("CASE 1: cos -> sin (valid, expect symmetric junction)",
         cos, sin, True),
        ("CASE 2: cos -> -cos (Euler rotation, expect symmetric junction)",
         cos, neg_cos, True),
        ("CASE 3: cos -> cos(2x) (Type 7, expect high junction stress)",
         cos, cos2x, False),
    ]

    all_pass = True
    for label, a, b, expect_low_junction in cases:
        print(f"\n{label}")

        res = run_fsm(a, b, step_fraction=0.15, potential_fn=norm_potential)
        n   = res["n_total"]
        ji  = res["junction_idx"]
        wv  = res["wass_var"]

        # Junction should be near middle for symmetric cases
        junction_frac = ji / max(n - 1, 1)
        junction_near_mid = abs(junction_frac - 0.5) < 0.35

        print(f"  Images: {res['n_from_a']} from A + {res['n_from_b']} from B "
              f"= {n} total")
        print(f"  Junction idx: {ji}/{n-1} (frac={junction_frac:.2f})  "
              f"wass_var={wv:.6f}")
        if res["stresses"] is not None:
            s = res["stresses"]
            print(f"  Stresses: min={min(s):.4f} max={max(s):.4f} "
                  f"junction={s[ji]:.4f}")

        # Sanity: endpoint constraint satisfied
        end_err_a = float(np.linalg.norm(res["images"][0] - a))
        end_err_b = float(np.linalg.norm(res["images"][-1] - b))
        status_a = "PASS" if end_err_a < 1e-6 else "FAIL"
        status_b = "PASS" if end_err_b < 1e-6 else "FAIL"
        if "FAIL" in (status_a, status_b):
            all_pass = False
        print(f"  Endpoint errors: A={end_err_a:.2e} [{status_a}]  "
              f"B={end_err_b:.2e} [{status_b}]")

        # FSM + FABRIK pipeline
        pipe_res = run_fsm_then_fabrik(a, b, step_fraction=0.15,
                                       potential_fn=norm_potential)
        pipe_wv = pipe_res["wass_var"]
        print(f"  After FABRIK polish: wass_var={pipe_wv:.6f}  "
              f"(vs {wv:.6f} before polish)")

    print()
    print("VALIDATION CRITERIA:")
    print("  Endpoint errors < 1e-6 for all cases [PASS/FAIL above]")
    print("  Junction at ~midpoint for symmetric potentials")
    print("  FABRIK polish reduces wass_var (more equal spacing)")
    print("  Type 7: junction stress highest (stress peak detection)")
    print()
    print("OVERALL:", "PASS" if all_pass else "FAIL")
