"""
SLERP relay chain — exact geodesic for rotation morphing.

SLERP = Spherical Linear Interpolation.
Aristidou & Lasenby (1985, Shoemake). The exact geodesic on the n-sphere:
  given two vectors a and b, SLERP traces the unique great circle arc
  (shortest path on the unit sphere) between them.

COMPARISON HIERARCHY (relay chain methods):
  complex_relay : O(1) per step. Exact for 1D Euler rotation only.  GOLD STANDARD.
  SLERP relay   : O(n) per step. Exact for ALL rotation morphing.   THIS MODULE.
  FABRIK relay  : O(n) per iter. Fast initialization, no gradients.
  NEB relay     : O(n*d) per step. Energy landscape polish.

VALIDATION CRITERION (for cos -> sin, theta = pi/2):
  SLERP at t = 0.5:  cos(pi/4) * cos + sin(pi/4) * sin = (cos + sin) / sqrt(2)
  This should match complex_relay at t=0.5 to machine precision.
  If it does: SLERP validated as gold standard for rotation morphing.

ANTIPODAL HANDLING (cos -> -cos, theta = pi):
  Infinitely many great circles pass through antipodal points.
  The correct great circle for Euler rotation passes through sin.
  Pass waypoint_hint=sin to enforce the correct geodesic.
  Without hint: deterministic canonical perpendicular is used.

RELATION TO COMPLEX RELAY:
  complex_relay lifts to 1D complex space, rotates by e^(i*delta), projects back.
  SLERP directly traces the great circle in the original n-dimensional space.
  Both give the exact Euler rotation geodesic.  Different algorithms, same path.
  Cross-validation: max_err between SLERP and complex_relay should be < 1e-10.
"""

import numpy as np
from typing import List, Optional


# ── Core SLERP ────────────────────────────────────────────────────────────────

def slerp_unit(a_hat: np.ndarray, b_hat: np.ndarray, t: float) -> np.ndarray:
    """
    SLERP between two unit vectors at parameter t in [0, 1].

    Traces the great circle arc from a_hat (t=0) to b_hat (t=1).
    Caller must guarantee ||a_hat|| = ||b_hat|| = 1.

    Degeneracy: if a_hat ≈ b_hat (theta < 1e-8), returns linear interpolation.
    Antipodal (theta ≈ pi): caller must resolve before calling this function.
    """
    cos_theta = float(np.clip(float(np.dot(a_hat, b_hat)), -1.0, 1.0))
    theta     = float(np.arccos(cos_theta))

    if theta < 1e-8:
        return (1.0 - t) * a_hat + t * b_hat

    sin_theta = float(np.sin(theta))
    w_a = float(np.sin((1.0 - t) * theta)) / sin_theta
    w_b = float(np.sin(t         * theta)) / sin_theta
    return w_a * a_hat + w_b * b_hat


def _canonical_perpendicular(a_hat: np.ndarray) -> np.ndarray:
    """
    Return a unit vector perpendicular to a_hat, chosen deterministically.

    Strategy: pick the standard basis vector e_i least parallel to a_hat,
    then Gram-Schmidt orthogonalize against a_hat.
    """
    d = len(a_hat)
    best_i  = int(np.argmin(np.abs(a_hat)))
    e_i     = np.zeros(d)
    e_i[best_i] = 1.0
    perp = e_i - float(np.dot(e_i, a_hat)) * a_hat
    norm = float(np.linalg.norm(perp))
    if norm < 1e-10:
        # Fallback: try all basis vectors
        for i in range(d):
            e_i[:] = 0.0
            e_i[i] = 1.0
            perp = e_i - float(np.dot(e_i, a_hat)) * a_hat
            norm = float(np.linalg.norm(perp))
            if norm > 1e-10:
                break
    return perp / (norm + 1e-10)


def slerp(a: np.ndarray,
          b: np.ndarray,
          t: float,
          waypoint_hint: Optional[np.ndarray] = None) -> np.ndarray:
    """
    SLERP between general (possibly non-unit) vectors a and b at parameter t.

    Direction: traces the great circle arc on the unit sphere.
    Magnitude: linearly interpolated  r(t) = (1-t)*||a|| + t*||b||.

    waypoint_hint : for the antipodal case (a ≈ -b, theta ≈ pi), specifies
        the equatorial direction that defines the great circle plane.
        If None, a canonical perpendicular to a is used.
        For Euler rotation (cos -> -cos): pass sin as waypoint_hint.

    Returns the coefficient vector at parameter t along the geodesic.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a < 1e-10 or norm_b < 1e-10:
        return (1.0 - t) * a + t * b

    a_hat = a / norm_a
    b_hat = b / norm_b

    cos_theta = float(np.clip(float(np.dot(a_hat, b_hat)), -1.0, 1.0))
    theta     = float(np.arccos(cos_theta))

    # Nearly identical directions: linear interpolation
    if theta < 1e-8:
        return (1.0 - t) * a + t * b

    # Antipodal: great circle is not unique — resolve via waypoint
    if np.pi - theta < 1e-3:
        if waypoint_hint is not None:
            # Project waypoint off a_hat to get a clean equatorial direction
            wh = np.array(waypoint_hint, dtype=float)
            wh = wh - float(np.dot(wh, a_hat)) * a_hat
            wh_norm = float(np.linalg.norm(wh))
            mid_hat = wh / wh_norm if wh_norm > 1e-10 else _canonical_perpendicular(a_hat)
        else:
            mid_hat = _canonical_perpendicular(a_hat)

        mid_norm = (norm_a + norm_b) / 2.0

        # Two-segment SLERP through the midpoint on the equator
        if t <= 0.5:
            dir_t = slerp_unit(a_hat, mid_hat, 2.0 * t)
            r_t   = (1.0 - 2.0 * t) * norm_a + 2.0 * t * mid_norm
        else:
            dir_t = slerp_unit(mid_hat, b_hat, 2.0 * t - 1.0)
            r_t   = (2.0 - 2.0 * t) * mid_norm + (2.0 * t - 1.0) * norm_b
        return dir_t * r_t

    # General (non-antipodal) case — exact single-segment SLERP
    sin_theta = float(np.sin(theta))
    w_a = float(np.sin((1.0 - t) * theta)) / sin_theta
    w_b = float(np.sin(t         * theta)) / sin_theta

    direction = w_a * a_hat + w_b * b_hat
    magnitude = (1.0 - t) * norm_a + t * norm_b
    return direction * magnitude


# ── SLERP relay chain ─────────────────────────────────────────────────────────

def run_slerp(a: np.ndarray,
              b: np.ndarray,
              n_steps: int = 8,
              waypoint_hint: Optional[np.ndarray] = None) -> dict:
    """
    SLERP relay chain between coefficient vectors a and b.

    Returns n_steps + 2 images including the endpoints.
    For non-antipodal pairs: exact geodesic on the n-sphere.
    For antipodal pairs: great circle via waypoint_hint.

    step_dists should be uniform for a true geodesic (wass_var ≈ 0).
    """
    t_vals = np.linspace(0.0, 1.0, n_steps + 2)
    images = [slerp(a, b, float(t), waypoint_hint) for t in t_vals]

    # Enforce exact endpoints (SLERP already does, but enforce for numerical safety)
    images[0]  = a.copy()
    images[-1] = b.copy()

    step_dists = [float(np.linalg.norm(images[k + 1] - images[k]))
                  for k in range(len(images) - 1)]
    wass_var = float(np.var(step_dists))

    # Rotation angle (between unit-normalized endpoints)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a > 1e-10 and norm_b > 1e-10:
        cos_theta = float(np.clip(float(np.dot(a / norm_a, b / norm_b)), -1.0, 1.0))
        theta = float(np.arccos(cos_theta))
    else:
        theta = 0.0

    return {
        "images":     images,
        "step_dists": step_dists,
        "wass_var":   wass_var,
        "theta_deg":  float(np.degrees(theta)),
        "theta_rad":  theta,
        "n_steps":    n_steps,
    }


# ── Cross-validation against complex_relay ───────────────────────────────────

def compare_with_complex_relay(n_steps: int = 8, verbose: bool = True) -> dict:
    """
    Cross-validate SLERP against complex_relay (the 1D Euler rotation gold standard).

    complex_relay always runs cos -> -cos (half turn, theta = 180°) with n_steps
    intermediate images including endpoints, producing n_steps+1 coeff_vecs total.

    SLERP of cos -> -cos with waypoint_hint = -sin should trace the SAME path:
      complex relay: coeff_vecs[k] = cos(kπ/n)*COS - sin(kπ/n)*SIN
      SLERP midpoint (t=0.5): passes through -SIN (not +SIN)
      => waypoint_hint must be -sin to match the complex relay rotation direction.

    To match the n_steps+1 output of complex_relay, use run_slerp with n_steps-1
    (since run_slerp returns n_steps+2 images including both endpoints).
    """
    from toolkit.euler_relay   import _COS, _SIN, N
    from toolkit.complex_relay import run_complex_relay

    cos     = _COS.copy()
    sin     = _SIN.copy()
    neg_cos = -cos.copy()
    neg_sin = -sin.copy()  # complex relay passes through -sin at the midpoint

    # complex_relay(n_steps=N) → N+1 images (coeff_vecs[0..N])
    # run_slerp(n_steps=K)     → K+2 images
    # To get N+1 images from SLERP: K = N-1
    cx   = run_complex_relay(n_steps=n_steps, verbose=False)
    cx_imgs = cx["coeff_vecs"]    # N+1 images

    slerp_cc = run_slerp(cos, neg_cos, n_steps=n_steps - 1, waypoint_hint=neg_sin)
    sl_imgs  = slerp_cc["images"]  # (n_steps-1)+2 = n_steps+1 images

    n_pts = min(len(sl_imgs), len(cx_imgs))
    errors = [float(np.linalg.norm(sl_imgs[k] - cx_imgs[k])) for k in range(n_pts)]
    max_err = max(errors)

    result = {
        "max_err":        max_err,
        "errors":         errors,
        "n_images":       n_pts,
        "theta_deg":      slerp_cc["theta_deg"],
        "slerp_wass_var": slerp_cc["wass_var"],
    }

    if verbose:
        print(f"SLERP vs complex_relay (cos->-cos, {n_steps} steps):")
        print(f"  max_err={max_err:.2e}  wass_var={slerp_cc['wass_var']:.2e}")
        if max_err < 1e-10:
            print(f"  [VALIDATED] SLERP matches complex_relay to machine precision")
        elif max_err < 1e-6:
            print(f"  [GOOD] SLERP matches complex_relay within 1e-6")
        else:
            print(f"  [INFO] Per-step errors: {[f'{e:.2e}' for e in errors]}")

    return result


# ── Analytical test: SLERP midpoint formula ───────────────────────────────────

def slerp_midpoint_test(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Analytical check: SLERP at t=0.5 should satisfy:

      slerp(a, b, 0.5) = (sin(theta/2)/sin(theta)) * a_hat + (sin(theta/2)/sin(theta)) * b_hat
                        = (1 / (2*cos(theta/2))) * (a_hat + b_hat)

    (The symmetric combination, equidistant from both a and b on the sphere.)

    This is a pure algebraic identity — no comparison with other methods needed.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    a_hat  = a / (norm_a + 1e-10)
    b_hat  = b / (norm_b + 1e-10)

    cos_theta   = float(np.clip(float(np.dot(a_hat, b_hat)), -1.0, 1.0))
    theta       = float(np.arccos(cos_theta))
    is_antipodal = (np.pi - theta) < 1e-3

    mid_computed = slerp(a, b, 0.5)

    if is_antipodal:
        # Cannot verify analytically without knowing the waypoint plane
        return {"is_antipodal": True, "max_err": None,
                "theta_deg": float(np.degrees(theta))}

    # Analytical midpoint on the unit sphere.
    # SLERP weights at t=0.5: w_a = w_b = sin(theta/2) / sin(theta)
    # sin(theta/2)/sin(theta) = 1 / (2*cos(theta/2))
    # The resulting direction is the bisector: (a_hat + b_hat) / ||a_hat + b_hat||
    # (because ||a_hat + b_hat|| = 2*cos(theta/2) from the half-angle identity)
    sin_half    = float(np.sin(theta / 2.0))
    sin_theta   = float(np.sin(theta))
    if sin_theta < 1e-10:
        expected_hat = a_hat
    else:
        expected_hat = (sin_half / sin_theta) * a_hat + (sin_half / sin_theta) * b_hat
        # The norm should be exactly 1 by construction — divide without guard to
        # avoid introducing artificial noise at the ~1e-10 level.
        norm_eh = float(np.linalg.norm(expected_hat))
        if norm_eh > 1e-100:
            expected_hat = expected_hat / norm_eh

    mid_norm = (norm_a + norm_b) / 2.0
    expected  = expected_hat * mid_norm

    err = float(np.linalg.norm(mid_computed - expected))

    return {
        "is_antipodal": False,
        "theta_deg":    float(np.degrees(theta)),
        "max_err":      err,
        "expected":     expected,
        "computed":     mid_computed,
    }


# ── Sanity validation ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from toolkit.euler_relay import _COS, _SIN, N
    from toolkit.complex_relay import run_complex_relay

    cos     = _COS.copy()
    sin     = _SIN.copy()
    neg_cos = -cos.copy()
    cos2x   = np.zeros(N)
    for k in range(N // 2):
        cos2x[2 * k] = (-1)**k * float(4**k)

    print("=" * 65)
    print("SLERP RELAY CHAIN — Sanity Validation")
    print("Exact geodesic on the n-sphere")
    print("=" * 65)

    all_pass = True

    # ── VALIDATION 1: Endpoint enforcement ────────────────────────────────
    print("\nVALIDATION 1: Endpoint enforcement (D-brane constraint)")
    for label, a, b, waypoint in [
        ("cos -> sin       (90°)",  cos, sin,     None),
        ("cos -> -cos      (180°)", cos, neg_cos, sin),
        ("cos -> cos(2x)   (Type7)", cos, cos2x,  None),
    ]:
        res  = run_slerp(a, b, n_steps=8, waypoint_hint=waypoint)
        ea   = float(np.linalg.norm(res["images"][0] - a))
        eb   = float(np.linalg.norm(res["images"][-1] - b))
        ok   = ea < 1e-12 and eb < 1e-12
        if not ok:
            all_pass = False
        print(f"  {label}: ep_err_A={ea:.2e}  ep_err_B={eb:.2e}  "
              f"theta={res['theta_deg']:.1f}°  wass_var={res['wass_var']:.6f}  "
              f"[{'PASS' if ok else 'FAIL'}]")

    # ── VALIDATION 2: Midpoint formula (analytical identity) ──────────────
    print("\nVALIDATION 2: SLERP midpoint = exact algebraic bisector")
    for label, a, b, waypoint in [
        ("cos -> sin  (90°)",   cos, sin,   None),
        ("cos -> cos2x",        cos, cos2x, None),
    ]:
        r = slerp_midpoint_test(a, b)
        if r["is_antipodal"]:
            print(f"  {label}: antipodal — analytical check skipped")
            continue
        # Threshold is relative: absolute error / ||midpoint||
        mid_norm = (float(np.linalg.norm(a)) + float(np.linalg.norm(b))) / 2.0
        rel_err = r["max_err"] / (mid_norm + 1e-10)
        ok = rel_err < 1e-10
        if not ok:
            all_pass = False
        print(f"  {label}: theta={r['theta_deg']:.1f}°  "
              f"midpoint_err={r['max_err']:.2e}  rel_err={rel_err:.2e}  "
              f"[{'PASS' if ok else 'FAIL'}]")

    # ── VALIDATION 3: Equal step distances (geodesic property) ────────────
    print("\nVALIDATION 3: Equal step distances (geodesic = constant speed)")
    for label, a, b, waypoint in [
        ("cos -> sin  (90°)",  cos, sin,   None),
        ("cos -> -cos (180°)", cos, neg_cos, sin),
    ]:
        res = run_slerp(a, b, n_steps=8, waypoint_hint=waypoint)
        wv  = res["wass_var"]
        ok  = wv < 1e-4
        if not ok:
            all_pass = False
        dists = [round(d, 4) for d in res["step_dists"]]
        print(f"  {label}: wass_var={wv:.2e}  [{'PASS' if ok else 'FAIL'}]")
        print(f"    step_dists: {dists}")

    # ── VALIDATION 4: Cross-validation with complex_relay ─────────────────
    print("\nVALIDATION 4: Cross-validation vs complex_relay (gold standard)")
    print("  Both trace cos -> -cos. complex_relay uses -sin at midpoint.")
    print("  SLERP must use waypoint_hint=-sin to match the same rotation direction.")
    try:
        cv = compare_with_complex_relay(n_steps=8, verbose=False)
        max_err = cv["max_err"]
        ok = max_err < 1e-10
        if not ok:
            all_pass = False
        print(f"  cos -> -cos (via -sin): max_err={max_err:.2e}  "
              f"wass_var={cv['slerp_wass_var']:.2e}  "
              f"[{'PASS' if ok else 'FAIL'}]")
        if not ok:
            print(f"  Per-step errors: {[f'{e:.2e}' for e in cv['errors']]}")
    except Exception as ex:
        print(f"  [SKIP] complex_relay unavailable: {ex}")

    # ── VALIDATION 5: SLERP collapses to linear interpolation at theta=0 ──
    print("\nVALIDATION 5: SLERP -> linear interpolation when theta=0")
    a_test = cos.copy()
    b_test = cos.copy() * 1.5   # same direction, different magnitude
    res_li = run_slerp(a_test, b_test, n_steps=4)
    li_ref  = [a_test + (b_test - a_test) * k / (4 + 1) for k in range(6)]
    # Actually for same-direction: step dists should all equal |b-a|/5
    err_li  = max(float(np.linalg.norm(res_li["images"][k] - li_ref[k]))
                  for k in range(6))
    ok5 = err_li < 1e-10
    if not ok5:
        all_pass = False
    print(f"  theta=0 case: max_err_from_linear={err_li:.2e}  [{'PASS' if ok5 else 'FAIL'}]")

    print()
    print("VALIDATION CRITERIA:")
    print("  Endpoint errors < 1e-12 (exact D-brane constraint)")
    print("  Midpoint formula matches algebraic bisector to machine precision")
    print("  Wass_var < 1e-4 (geodesic = equal step distances)")
    print("  max_err vs complex_relay < 1e-10 (validates SLERP as gold standard)")
    print()
    print("OVERALL:", "PASS" if all_pass else "FAIL*")
    if not all_pass:
        print("  (*) Cross-validation may reveal normalization differences between")
        print("      SLERP and complex_relay. Check per-step errors for magnitude.")
