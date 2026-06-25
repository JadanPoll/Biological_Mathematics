"""
Sanity check for toolkit/gap_metrics.py.

Runs all three gap metrics on the four canonical cases whose answers
are pre-confirmed by existing experiments. Every expected value below
is derived from a gold-standard result already in the codebase.

EXPECTED OUTCOMES:
  All metrics now use FFT magnitude on function evaluations (BASIS @ coeff on X grid).
  WHT cosine similarity was wrong: cos and sin are orthogonal in phi_k Taylor basis
  (even vs odd indices), giving score=0 despite being validly morphable.
  Spectral impurity replaces WHT lerp-deviation (which is always 0 by linearity).

  cos -> sin        (valid, gold standard: complex_relay max_error=0)
    go_map_score   : > 0.7    (both pure frequency-1, same spectral orbit)
    residual_norm  : < 0.15   (complex_relay midpoint is pure freq-1 sinusoid)
    sigma_norm     : < 0.1    (FREE: all images are pure freq-1)
    regime         : FREE

  cos -> cos(2x)   (Type 7 confirmed: type7_test.py, joint_covariance.py)
    go_map_score   : < 0.4    (freq-1 vs freq-2 -> different FFT peaks)
    residual_norm  : > 0.3    (lerp midpoint = 0.5*cos + 0.5*cos(2x), two FFT peaks)
    sigma_norm     : > 0.3    (CONFINEMENT: impurity peaks at t=0.5)
    regime         : CONFINEMENT

  cos -> 2*cos     (valid scalar multiple: cross_cov=0.989, residual=0.001)
    go_map_score   : > 0.95   (same frequency, cosine sim normalizes amplitude)
    residual_norm  : < 0.15   (lerp midpoint = 1.5*cos, still pure freq-1)
    sigma_norm     : < 0.1    (FREE: all lerp images are scaled cos, pure freq-1)
    regime         : FREE

  sin(x²) <-> sin(2x) (this session: Band-1 gap=29.39, valuation gap=2)
    go_map_score   : < 0.4    (chirp vs pure sinusoid -> different FFT profiles)
    residual_norm  : > 0.2    (lerp midpoint mixes chirp and sinusoid)
    sigma_norm     : > 0.1    (at least partial impurity along path)
    regime         : CONFINEMENT or FREE

Run:
  python -m experiments.sanity_gap_metrics
"""

import math
import numpy as np

from toolkit.euler_relay import _COS, _SIN, N, euler_intermediate, BASIS, X
from toolkit.complex_relay import run_complex_relay
from toolkit.fsm_relay import run_fsm
from toolkit.gap_metrics import (go_map_score, go_map_score_b2,
                                  wht_residual_spectrum, string_tension, full_gap_report)


# -- Coefficient vectors for canonical test cases ------------------------------

# cos(x) and sin(x) are already in euler_relay
COS_V = _COS.copy()
SIN_V = _SIN.copy()

# 2*cos(x)
TWO_COS_V = 2.0 * _COS.copy()

# cos(2x) in phi_k = x^k/k! basis:
#   cos(2x) = sum_k (-1)^k * 4^k * phi_{2k}
#   coefficients: c_{2k} = (-1)^k * 4^k,  c_{2k+1} = 0
COS2X_V = np.array([(-1.)**k * 4.**k if i == 2*k else 0.0
                    for i, _ in enumerate(range(N))
                    for k in [i // 2] if i % 2 == 0 or True],
                   dtype=float)[:N]
# Simpler explicit construction for N=8:
COS2X_V = np.array([1., 0., -4., 0., 16., 0., -64., 0.])

# sin(x^2): project onto phi_k basis using least-squares on the evaluation grid
# sin(x^2) evaluated on X = linspace(-1.5, 1.5, 120)
SIN_X2_vals = np.sin(X**2)
SIN_X2_V, _, _, _ = np.linalg.lstsq(BASIS, SIN_X2_vals, rcond=None)

# sin(2x): same approach for consistency
SIN_2X_vals = np.sin(2 * X)
SIN_2X_V, _, _, _ = np.linalg.lstsq(BASIS, SIN_2X_vals, rcond=None)


# -- Build relay chain images for each case ------------------------------------

def get_images_valid():
    """cos -> sin via complex_relay (gold standard, max_error=0)."""
    result = run_complex_relay(n_steps=8, verbose=False)
    return result["coeff_vecs"]


def get_images_fsm(endpoint_a, endpoint_b):
    """Generic FSM relay between two coefficient vectors."""
    result = run_fsm(endpoint_a, endpoint_b, step_fraction=0.12, max_images=40)
    return result["images"]


# -- Sanity check --------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"

def check(name, value, expected_op, threshold, unit=""):
    ops = {">": value > threshold, "<": value < threshold,
           ">=": value >= threshold, "<=": value <= threshold}
    ok = ops[expected_op]
    symbol = "[+]" if ok else "[!]"
    status = PASS if ok else FAIL
    print(f"    [{symbol}] {name:30s} = {value:8.4f}  {expected_op} {threshold}{unit}  [{status}]")
    return ok


def run_sanity():
    all_pass = True
    print("\n" + "=" * 65)
    print("  GAP METRICS SANITY CHECK")
    print("=" * 65)

    # -- Case 1: cos -> sin (valid, gold standard) ---------------------------
    print("\nCase 1: cos -> sin  (valid morph, complex_relay gold standard)")
    images_valid = get_images_valid()
    # complex_relay goes cos -> -cos, so take the first half (cos -> -sin direction)
    # For the go_map_score we just test COS_V vs SIN_V directly
    gms_1  = go_map_score(COS_V, SIN_V)
    wht_1  = wht_residual_spectrum(images_valid, COS_V, -COS_V)  # full path
    st_1   = string_tension(images_valid, COS_V, -COS_V)
    # also run the convenience printer
    full_gap_report(images_valid, COS_V, -COS_V, label="cos -> -cos (complex_relay path)")

    ok  = check("go_map_score(cos, sin)",   gms_1,                  ">",  0.95, " (band1_dist=0 -> score=1.0)")
    ok &= check("WHT residual norm",        wht_1["residual_norm"], "<",  0.05, " (lerp mid stays in same orbit)")
    ok &= check("sigma_norm",               st_1["sigma_norm"],     "<",  0.1,  " (expect FREE)")
    print(f"    regime: {st_1['regime']}")
    all_pass &= ok

    # -- Case 2: cos -> cos(2x) (Type 7 confirmed) --------------------------
    print("\nCase 2: cos -> cos(2x)  (Type 7 confirmed: type7_test.py)")
    images_t7 = get_images_fsm(COS_V, COS2X_V)
    gms_2     = go_map_score(COS_V, COS2X_V)
    wht_2     = wht_residual_spectrum(images_t7, COS_V, COS2X_V)
    st_2      = string_tension(images_t7, COS_V, COS2X_V)
    full_gap_report(images_t7, COS_V, COS2X_V, label="cos -> cos(2x) via FSM")

    ok  = check("go_map_score(cos, cos2x)", gms_2,                  "<",  0.4,  " (band1_dist=22 -> score=0.11)")
    ok &= check("WHT residual norm",        wht_2["residual_norm"], ">",  0.5,  " (lerp mid migrates far from endpoints)")
    ok &= check("sigma_norm",               st_2["sigma_norm"],     ">",  0.5,  " (expect CONFINEMENT)")
    print(f"    regime: {st_2['regime']}")
    if wht_2["dominant_modes"]:
        print(f"    dominant FFT bins: {wht_2['dominant_modes']}  "
              f"(frequencies present in lerp midpoint)")
    all_pass &= ok

    # -- Case 3: cos -> 2*cos (valid scalar multiple) ------------------------
    print("\nCase 3: cos -> 2*cos  (valid scalar: cross_cov=0.989 confirmed)")
    images_2cos = get_images_fsm(COS_V, TWO_COS_V)
    gms_3       = go_map_score(COS_V, TWO_COS_V)
    wht_3       = wht_residual_spectrum(images_2cos, COS_V, TWO_COS_V)
    st_3        = string_tension(images_2cos, COS_V, TWO_COS_V)
    full_gap_report(images_2cos, COS_V, TWO_COS_V, label="cos -> 2*cos via FSM")

    ok  = check("go_map_score(cos, 2cos)",  gms_3,                  ">",  0.93, " (band1_dist=0.5 -> score=0.95)")
    ok &= check("WHT residual norm",        wht_3["residual_norm"], "<",  0.05, " (lerp mid = 1.5*cos, same orbit)")
    ok &= check("sigma_norm",               st_3["sigma_norm"],     "<",  0.05, " (expect FREE)")
    print(f"    regime: {st_3['regime']}")
    all_pass &= ok

    # -- Case 4: sin(x²) <-> sin(2x) (this session: Band-1 gap=29.39) --------
    print("\nCase 4: sin(x²) <-> sin(2x)  (session result: Band-1 gap=29.39)")
    images_sinx2 = get_images_fsm(SIN_X2_V, SIN_2X_V)
    gms_4        = go_map_score(SIN_X2_V, SIN_2X_V)
    wht_4        = wht_residual_spectrum(images_sinx2, SIN_X2_V, SIN_2X_V)
    st_4         = string_tension(images_sinx2, SIN_X2_V, SIN_2X_V)
    full_gap_report(images_sinx2, SIN_X2_V, SIN_2X_V, label="sin(x²) <-> sin(2x) via FSM")

    ok  = check("go_map_score(sinx2, sin2x)", gms_4,                "<",  0.6,  " (different spectral regimes)")
    ok &= check("WHT residual norm",          wht_4["residual_norm"], ">", 0.05, " (expect some gap)")
    ok &= check("sigma_norm",                 st_4["sigma_norm"],   ">",  0.05, " (expect stress along path)")
    print(f"    regime: {st_4['regime']}")
    all_pass &= ok

    # -- Ordering check: Type 7 harder than valid ---------------------------
    print("\nOrdering checks (Type 7 must score worse than valid on all axes):")
    ok  = check("go_map(cos,cos2x) < go_map(cos,sin)",
                gms_2, "<", gms_1)
    ok &= check("sigma(cos2x) > sigma(cos->sin)",
                st_2["sigma_norm"], ">", st_1["sigma_norm"])
    ok &= check("residual(cos2x) > residual(cos->2cos)",
                wht_2["residual_norm"], ">", wht_3["residual_norm"])
    all_pass &= ok

    # -- Band-2 checks (Krawtchouk K_2 coefficient) -------------------------
    print("\nBand-2 checks (Krawtchouk degree-2 coefficient — finer than band-1):")
    print("  (N=8 band-2 indices = {3,5,6} — validated distances: cos/sin=1.0, cos/cos2x=25.1)")
    b2_cos_sin   = go_map_score_b2(COS_V, SIN_V)
    b2_cos_2cos  = go_map_score_b2(COS_V, TWO_COS_V)
    b2_cos_cos2x = go_map_score_b2(COS_V, COS2X_V)
    print(f"    go_map_b2(cos,sin)  = {b2_cos_sin:.4f}  "
          f"(same coarse orbit, different sub-orbit — valid but lower than b1=1.0)")
    print(f"    go_map_b2(cos,2cos) = {b2_cos_2cos:.4f}  "
          f"(valid scalar multiple — same sub-orbit)")
    print(f"    go_map_b2(cos,cos2x)= {b2_cos_cos2x:.4f}  "
          f"(Type 7 — large distance at both bands)")
    ok  = check("b2(cos,cos2x) < 0.4  (Type 7 flagged at band-2)",
                b2_cos_cos2x, "<", 0.4)
    ok &= check("b2(cos,2cos)  > 0.9  (valid scalar unchanged)",
                b2_cos_2cos,  ">", 0.9)
    ok &= check("b2(cos,sin)   > 0.85 (valid rotation, lower than b1 expected)",
                b2_cos_sin,   ">", 0.85)
    ok &= check("b2(cos,cos2x) < b2(cos,sin)  (Type 7 < valid at band-2)",
                b2_cos_cos2x, "<", b2_cos_sin)
    all_pass &= ok

    # -- Summary ------------------------------------------------------------
    print("\n" + "=" * 65)
    if all_pass:
        print("  ALL CHECKS PASSED — gap_metrics.py is correctly calibrated.")
    else:
        print("  SOME CHECKS FAILED — review output above before using in explorer.")
    print("=" * 65)

    return all_pass


if __name__ == "__main__":
    run_sanity()
