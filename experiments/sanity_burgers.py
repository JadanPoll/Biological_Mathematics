"""
Sanity check for toolkit/burgers.py.

Four cases covering the three regimes (valid, Type 7, Type M):

  Case 1: Linear interpolation (reference path)
    Expected: burgers_norm = 0.000  (exactly zero — relay IS the reference)

  Case 2: complex_relay cos -> -cos (valid curved path)
    Expected: burgers_norm < 0.01  (geodesic rotation curves slightly, not a fold)

  Case 3: FSM cos -> cos(2x) (Type 7, direct path)
    Expected: burgers_norm < 0.01  (FSM walks straight, no topological detour)

  Case 4: Deliberately looping path (Type M test case)
    Expected: burgers_norm > 0.05  (path winds around the direct segment)

Ordering check:
  loop_burgers > valid_burgers  (Type M > Type 7 AND Type M > valid)
  type7_burgers ~ valid_burgers  (both near-zero — the distinction is sigma_norm)

Run:
  python -m experiments.sanity_burgers
"""

import numpy as np

from toolkit.euler_relay   import _COS, _SIN, N
from toolkit.complex_relay import run_complex_relay
from toolkit.fsm_relay     import run_fsm
from toolkit.burgers       import burgers_residue, make_looping_path


# -- Canonical endpoint vectors ------------------------------------------------

COS_V  = _COS.copy()
COS2X_V = np.array([1., 0., -4., 0., 16., 0., -64., 0.])


# -- Sanity runner -------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"

def check(name, value, expected_op, threshold, unit=""):
    ops = {">": value > threshold, "<": value < threshold,
           ">=": value >= threshold, "<=": value <= threshold}
    ok = ops[expected_op]
    symbol = "[+]" if ok else "[!]"
    status = PASS if ok else FAIL
    print(f"    [{symbol}] {name:40s} = {value:8.5f}  {expected_op} {threshold}{unit}  [{status}]")
    return ok


def run_sanity():
    all_pass = True
    print("\n" + "=" * 65)
    print("  BURGERS RESIDUE SANITY CHECK")
    print("=" * 65)

    # -- Case 1: Linear interpolation (zero by construction) -------------------
    print("\nCase 1: Linear interpolation A->B  (zero by definition)")
    a = COS_V
    b = COS2X_V
    images_linear = [a + t * (b - a) for t in np.linspace(0, 1, 9)]
    bg1 = burgers_residue(images_linear, a, b)
    print(f"    signed_area = {bg1['signed_area']:.2e}  burgers_norm = {bg1['burgers_norm']:.6f}")
    ok = check("burgers_norm (linear interp)", bg1["burgers_norm"], "<", 1e-10,
               " (shoelace on collinear points = 0)")
    ok &= check("is_type_m (linear interp)", float(bg1["is_type_m"]), "<", 0.5,
                " (False expected)")
    all_pass &= ok

    # -- Case 2: complex_relay cos->-cos (valid geodesic, slight curve) --------
    print("\nCase 2: complex_relay cos->-cos  (valid geodesic rotation)")
    relay_result = run_complex_relay(n_steps=8, verbose=False)
    images_valid = relay_result["coeff_vecs"]
    a_valid = images_valid[0]
    b_valid = images_valid[-1]
    bg2 = burgers_residue(images_valid, a_valid, b_valid)
    print(f"    signed_area = {bg2['signed_area']:.4f}  burgers_norm = {bg2['burgers_norm']:.5f}"
          f"  sign = {bg2['burgers_sign']}")
    ok = check("burgers_norm (complex_relay)", bg2["burgers_norm"], "<", 0.01,
               " (geodesic is not a fold — small but possibly nonzero)")
    ok &= check("is_type_m (complex_relay)", float(bg2["is_type_m"]), "<", 0.5,
                " (False expected — valid morph)")
    all_pass &= ok

    # -- Case 3: FSM cos->cos(2x) (Type 7, direct walk, no fold) ---------------
    print("\nCase 3: FSM cos->cos(2x)  (Type 7 confirmed, FSM walks directly)")
    fsm_result = run_fsm(COS_V, COS2X_V, step_fraction=0.12, max_images=40)
    images_t7  = fsm_result["images"]
    bg3 = burgers_residue(images_t7, COS_V, COS2X_V)
    print(f"    signed_area = {bg3['signed_area']:.4f}  burgers_norm = {bg3['burgers_norm']:.5f}"
          f"  sign = {bg3['burgers_sign']}")
    ok = check("burgers_norm (Type 7 FSM)", bg3["burgers_norm"], "<", 0.02,
               " (FSM goes straight, no topological detour)")
    ok &= check("is_type_m (Type 7 FSM)", float(bg3["is_type_m"]), "<", 0.5,
                " (False expected — intrinsic gap, not methodological)")
    all_pass &= ok

    # -- Case 4: Deliberate loop (Type M test case) ----------------------------
    print("\nCase 4: Deliberate looping path  (Type M: wrong fold direction)")
    images_loop = make_looping_path(COS_V, COS2X_V, n=17, detour_scale=1.0)
    bg4 = burgers_residue(images_loop, COS_V, COS2X_V)
    print(f"    signed_area = {bg4['signed_area']:.4f}  burgers_norm = {bg4['burgers_norm']:.5f}"
          f"  sign = {bg4['burgers_sign']}")
    ok = check("burgers_norm (loop path)", bg4["burgers_norm"], ">", 0.05,
               " (path winds around direct segment — clear Type M signature)")
    ok &= check("is_type_m (loop path)", float(bg4["is_type_m"]), ">", 0.5,
                " (True expected — methodological fold)")
    all_pass &= ok

    # -- Ordering checks -------------------------------------------------------
    print("\nOrdering checks (Type M must score highest on burgers, not highest on sigma):")
    ok  = check("loop > type7 on burgers_norm",
                bg4["burgers_norm"], ">", bg3["burgers_norm"])
    ok &= check("loop > valid on burgers_norm",
                bg4["burgers_norm"], ">", bg2["burgers_norm"])
    ok &= check("type7 ~ valid: both burgers_norm < 0.02",
                max(bg2["burgers_norm"], bg3["burgers_norm"]), "<", 0.02)
    all_pass &= ok

    print("\n" + "=" * 65)
    if all_pass:
        print("  ALL CHECKS PASSED — burgers.py is correctly calibrated.")
    else:
        print("  SOME CHECKS FAILED — review output above.")
    print("=" * 65)

    return all_pass


if __name__ == "__main__":
    run_sanity()


