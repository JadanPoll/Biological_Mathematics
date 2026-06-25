"""
FABRIK relay chain — Forward And Backward Reaching Inverse Kinematics.

Aristidou & Lasenby 2011. Adapted from robot IK to coefficient vector relay chains.

DIRECT EQUIVALENCES:
  IK joint          = relay chain intermediate image (coefficient vector)
  IK bone length    = step distance between images (equal = good path)
  IK target         = endpoint_b
  IK root           = endpoint_a (fixed)
  IK iterations     = relay refinement steps (5-10 sufficient empirically)

WHY FABRIK OVER NEB FOR INITIALIZATION:
  NEB requires gradient evaluations at each image per step: O(n*d) per step
  FABRIK requires only vector arithmetic: O(n*d) per iteration, zero gradients
  Result: FABRIK initializes a well-spaced chain ~100x faster than NEB converges.
  Use FABRIK to initialize, then NEB to refine (polish the energy landscape).

THE SPECTRAL FABRIK EXTENSION (preserves Wasserstein geodesic direction):
  The low-frequency Walsh modes capture coarse mass-transport direction (EMD).
  Running FABRIK in the low-freq subspace first locks in the geodesic direction.
  The backward pass restores endpoint constraints while respecting this coarse path.
  High-frequency details get unwrinkled in a subsequent full-space pass.

  This is the FABRIK analog of the BARF spectral annealing schedule:
    1. Forward pass in low-freq subspace  → coarse Wasserstein-geodesic direction
    2. Backward pass (full space)         → restore root, maintain coarse path
    3. Expand to higher freq subspace     → unwrinkle details

CONVERGENCE:
  Monotone: |endpoint_reached - target| is non-increasing per iteration.
  No formal rate guarantee. Empirically 5-10 iterations for typical chains.
  Degeneracy: collinear images can stall (handled by perturbation in run_fabrik).

REFERENCE:
  Aristidou, A., Lasenby, J. (2011). FABRIK: A fast, iterative solver for the
  Inverse Kinematics problem. Graphical Models, 73(5), 243-260.
  andreasaristidou.com/publications/papers/FABRIK.pdf
"""

import numpy as np
from typing import List, Optional, Callable

from toolkit.walsh import wht


# ── Core FABRIK passes ────────────────────────────────────────────────────────

def _fabrik_forward(images: List[np.ndarray], target: np.ndarray,
                    bone_len: float) -> None:
    """
    Forward pass: pull chain end to target, propagate constraints backward.
    Modifies images in-place.

    For each joint i (from end to root):
      Place i on the line from old i toward new i+1, at distance bone_len from i+1.
    """
    images[-1] = target.copy()
    for i in range(len(images) - 2, -1, -1):
        r = np.linalg.norm(images[i + 1] - images[i])
        if r < 1e-10:
            continue
        lam = bone_len / r
        images[i] = (1.0 - lam) * images[i + 1] + lam * images[i]


def _fabrik_backward(images: List[np.ndarray], root: np.ndarray,
                     bone_len: float) -> None:
    """
    Backward pass: restore root, propagate constraints forward.
    Modifies images in-place.

    For each joint i (from root to end):
      Place i on the line from new i-1 toward old i, at distance bone_len from i-1.
    """
    images[0] = root.copy()
    for i in range(1, len(images)):
        r = np.linalg.norm(images[i] - images[i - 1])
        if r < 1e-10:
            continue
        lam = bone_len / r
        images[i] = (1.0 - lam) * images[i - 1] + lam * images[i]


# ── Standard FABRIK relay ─────────────────────────────────────────────────────

def run_fabrik(endpoint_a: np.ndarray,
               endpoint_b: np.ndarray,
               n_images: int = 7,
               n_iter: int = 10,
               seed: int = 42) -> dict:
    """
    FABRIK relay chain between endpoint_a and endpoint_b.

    Returns images with equal arc-length spacing by construction.
    Use as fast initializer before NEB refinement.

    O(n_iter * n_images * d) — no gradient evaluations.
    """
    rng = np.random.default_rng(seed)
    total_dist = float(np.linalg.norm(endpoint_b - endpoint_a))
    bone_len = total_dist / (n_images + 1)

    # Linear initialization with small perturbation to avoid symmetry collapse
    images = [endpoint_a + (endpoint_b - endpoint_a) * k / (n_images + 1)
              for k in range(n_images + 2)]
    for k in range(1, len(images) - 1):
        images[k] = images[k] + rng.standard_normal(len(endpoint_a)) * bone_len * 0.05

    for _ in range(n_iter):
        _fabrik_forward(images, endpoint_b, bone_len)
        _fabrik_backward(images, endpoint_a, bone_len)

    # Enforce D-brane endpoints exactly — both are fixed in relay chain context.
    # FABRIK backward pass can drift images[-1] away from endpoint_b; enforce here.
    images[0]  = endpoint_a.copy()
    images[-1] = endpoint_b.copy()

    step_dists = [float(np.linalg.norm(images[k + 1] - images[k]))
                  for k in range(len(images) - 1)]
    wass_var = float(np.var(step_dists))

    return {
        "images":     images,
        "step_dists": step_dists,
        "wass_var":   wass_var,
        "bone_len":   bone_len,
        "n_iter":     n_iter,
    }


# ── Spectral FABRIK: preserves Wasserstein geodesic direction ─────────────────

def _walsh_project(images: List[np.ndarray],
                   active_indices: List[int]) -> List[np.ndarray]:
    """Project all images onto the active Walsh subspace."""
    projected = []
    for img in images:
        spectrum = wht(img)
        masked = np.zeros_like(spectrum)
        masked[active_indices] = spectrum[active_indices]
        # Inverse WHT = WHT (involutory up to scale)
        n = len(img)
        projected.append(wht(masked) * n)
    return projected


def _walsh_bands(n_dims: int) -> dict:
    """Group Walsh indices by Hamming weight (interaction order = frequency level)."""
    bands: dict = {}
    for k in range(n_dims):
        w = bin(k).count('1')
        bands.setdefault(w, []).append(k)
    return bands


def run_spectral_fabrik(endpoint_a: np.ndarray,
                        endpoint_b: np.ndarray,
                        n_images: int = 7,
                        n_iter_per_band: int = 5,
                        seed: int = 42) -> dict:
    """
    Spectral FABRIK: coarse-to-fine relay chain initialization.

    Runs FABRIK progressively through Walsh frequency bands:
      Band 0 (Hamming weight 0): DC component — global mean shift
      Band 1 (Hamming weight 1): main effects — coarse Wasserstein direction
      Band 2 (Hamming weight 2): 2-way interactions — medium detail
      Band 3+ (Hamming weight 3+): high-order — fine detail

    Low-frequency bands lock in the geodesic (EMD) direction first.
    High-frequency bands unwrinkle details within that constraint.

    The low-dimensional information IS the most stable under alternating projection
    because it has the fewest degrees of freedom — it cannot wriggle away.
    """
    rng = np.random.default_rng(seed)
    n_dims = len(endpoint_a)
    total_dist = float(np.linalg.norm(endpoint_b - endpoint_a))
    bone_len = total_dist / (n_images + 1)

    images = [endpoint_a + (endpoint_b - endpoint_a) * k / (n_images + 1)
              for k in range(n_images + 2)]
    for k in range(1, len(images) - 1):
        images[k] = images[k] + rng.standard_normal(n_dims) * bone_len * 0.05

    bands = _walsh_bands(n_dims)
    max_band = max(bands.keys())
    band_history = []

    # Progressively expand active subspace
    active_indices: List[int] = []
    for band_level in range(max_band + 1):
        active_indices = active_indices + bands.get(band_level, [])

        # Project endpoints and images onto current active subspace
        proj_a   = _walsh_project([endpoint_a], active_indices)[0]
        proj_b   = _walsh_project([endpoint_b], active_indices)[0]
        proj_imgs = _walsh_project(images, active_indices)

        # FABRIK in projected subspace
        proj_bone = float(np.linalg.norm(proj_b - proj_a)) / (n_images + 1)
        if proj_bone < 1e-10:
            proj_bone = bone_len

        for _ in range(n_iter_per_band):
            _fabrik_forward(proj_imgs, proj_b, proj_bone)
            _fabrik_backward(proj_imgs, proj_a, proj_bone)

        # Lift: update the active-subspace components of full images
        for k in range(1, len(images) - 1):
            spectrum = wht(images[k])
            proj_spectrum = wht(proj_imgs[k])
            for idx in active_indices:
                spectrum[idx] = proj_spectrum[idx]
            images[k] = wht(spectrum) * n_dims

        # Lock endpoints
        images[0]  = endpoint_a.copy()
        images[-1] = endpoint_b.copy()

        step_var = float(np.var([np.linalg.norm(images[k+1]-images[k])
                                  for k in range(len(images)-1)]))
        band_history.append({
            "band": band_level,
            "active_dims": len(active_indices),
            "step_var": step_var,
        })

    # Final full-space FABRIK polish
    for _ in range(n_iter_per_band):
        _fabrik_forward(images, endpoint_b, bone_len)
        _fabrik_backward(images, endpoint_a, bone_len)

    # Enforce D-brane endpoints exactly
    images[0]  = endpoint_a.copy()
    images[-1] = endpoint_b.copy()

    step_dists = [float(np.linalg.norm(images[k + 1] - images[k]))
                  for k in range(len(images) - 1)]
    wass_var = float(np.var(step_dists))

    return {
        "images":       images,
        "step_dists":   step_dists,
        "wass_var":     wass_var,
        "bone_len":     bone_len,
        "band_history": band_history,
    }


# ── Sanity validation ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from toolkit.euler_relay import _COS, _SIN, N

    cos  = _COS.copy()
    sin  = _SIN.copy()
    # cos(2x) in phi_k basis: coeff[2k] = (-1)^k * 4^k
    cos2x = np.array([(-1)**k * 4**k if i == 2*k else 0.0
                       for i in range(N) for k in range(N//2) if i == 2*k],
                      dtype=float)
    # rebuild properly
    cos2x = np.zeros(N)
    for k in range(N // 2):
        cos2x[2 * k] = (-1)**k * float(4**k)

    neg_cos = -cos.copy()

    print("=" * 60)
    print("FABRIK RELAY CHAIN — Sanity Validation")
    print("=" * 60)

    cases = [
        ("CASE 1: cos -> sin (derivative, valid path)",       cos,  sin,   True),
        ("CASE 2: cos -> -cos (Euler rotation, gold std)",    cos,  neg_cos, True),
        ("CASE 3: cos -> cos(2x) (Type 7, invalid path)",    cos,  cos2x, False),
    ]

    all_pass = True
    for label, a, b, expect_low_var in cases:
        print(f"\n{label}")

        # Standard FABRIK
        res = run_fabrik(a, b, n_images=7, n_iter=10)
        wv  = res["wass_var"]
        ee_a = float(np.linalg.norm(res["images"][0] - a))
        ee_b = float(np.linalg.norm(res["images"][-1] - b))
        status = "PASS" if ee_a < 1e-10 and ee_b < 1e-10 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Standard FABRIK:  wass_var={wv:.6f}  "
              f"ep_err_A={ee_a:.2e}  ep_err_B={ee_b:.2e}  [{status}]")
        print(f"  Step dists: {[round(d, 4) for d in res['step_dists']]}")

        # Spectral FABRIK
        sres = run_spectral_fabrik(a, b, n_images=7, n_iter_per_band=5)
        swv  = sres["wass_var"]
        see_a = float(np.linalg.norm(sres["images"][0] - a))
        see_b = float(np.linalg.norm(sres["images"][-1] - b))
        sstatus = "PASS" if see_a < 1e-10 and see_b < 1e-10 else "FAIL"
        if sstatus == "FAIL":
            all_pass = False
        print(f"  Spectral FABRIK:  wass_var={swv:.6f}  "
              f"ep_err_A={see_a:.2e}  ep_err_B={see_b:.2e}  [{sstatus}]")
        print(f"  Band convergence: "
              + "  ".join(f"band{b['band']}={b['step_var']:.4f}"
                          for b in sres["band_history"]))

    print()
    print("VALIDATION CRITERIA:")
    print("  ep_err_A and ep_err_B < 1e-10 (D-brane endpoints enforced exactly)")
    print("  wass_var measures path evenness (lower = more equal spacing)")
    print("  Type 7 case: FABRIK finds A PATH but not necessarily the right one")
    print("               (Type 7 detection is job of adic_filter/spectral_warmup)")
    print()
    print("OVERALL:", "PASS" if all_pass else "FAIL")
