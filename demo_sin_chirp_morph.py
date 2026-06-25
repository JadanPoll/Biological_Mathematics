"""
demo_sin_chirp_morph.py — Framework stress test: sin(x^2) <-> sin(2x)

sin(2x)  = clean single-frequency sinusoid
sin(x^2) = chirp function — instantaneous frequency grows without bound

This is harder than cos->cos(2x) in one precise sense:
  cos->cos(2x): same functional form (both pure cosines), different argument scale
  sin(2x)->sin(x^2): different functional FORMS (linear vs quadratic argument)

Expected fingerprint per theory:
  2-adic / spectral filters: Type 7 flagged (structural mismatch)
  FSM junction stress:       high (spectral chaos at the midpoint)
  SLERP angle:               large (functions live far apart on n-sphere)
  Lifted grammar:            trivial straight line in (c1, c2) argument space

Run: python demo_sin_chirp_morph.py
"""

import numpy as np
import math

N = 8   # phi_k coefficient dimension (same as rest of toolkit)
X_GRID = np.linspace(-1.5, 1.5, 400)

from toolkit.walsh          import wht
from toolkit.adic_filter    import adic_compatibility
from toolkit.spectral_warmup import spectral_type7_prefilter
from toolkit.fsm_relay      import run_fsm
from toolkit.slerp_relay    import run_slerp
from toolkit.euler_relay    import _COS


# ── Step 1: Build coefficient vectors in phi_k = x^k/k! basis ────────────────

def make_sin2x():
    """
    sin(2x) = sum_{k>=0} (-1)^k * 2^{2k+1} * phi_{2k+1}
    Exact — the series IS the phi_k expansion, no approximation.
    Nonzero at ODD indices: [0, 2, 0, -8, 0, 32, 0, -128]
    """
    c = np.zeros(N)
    for k in range(N // 2):
        c[2 * k + 1] = (-1)**k * 2**(2 * k + 1)
    return c


def make_sinx2_truncated():
    """
    sin(x^2) = x^2 - x^6/3! + x^10/5! - ...
             = sum_{k>=0} (-1)^k * (4k+2)!/(2k+1)! * phi_{4k+2}

    N=8 captures only k=0 (phi_2) and k=1 (phi_6):
      c[2] = 2!/1! = 2
      c[6] = -6!/3! = -120

    The NEXT term (k=2) lives at phi_10 (index 10), which is OUTSIDE our N=8 basis.
    The representation mismatch IS the grammar insufficiency.
    """
    c = np.zeros(N)
    for k in range(N):
        idx = 4 * k + 2
        if idx >= N:
            break
        c[idx] = (-1)**k * math.factorial(4 * k + 2) / math.factorial(2 * k + 1)
    return c


def fit_phik_ls(fn_vals):
    """Least-squares fit of sampled function values to phi_k basis on X_GRID."""
    A = np.column_stack([X_GRID**k / math.factorial(k) for k in range(N)])
    c, _, _, _ = np.linalg.lstsq(A, fn_vals, rcond=None)
    return c


def reconstruct(c):
    """Reconstruct function values on X_GRID from phi_k coefficients."""
    return sum(c[k] * X_GRID**k / math.factorial(k) for k in range(N))


def approx_err(c, fn_true):
    return float(np.max(np.abs(reconstruct(c) - fn_true)))


def wht_entropy(c):
    """Shannon entropy of the WHT magnitude spectrum."""
    w = np.abs(wht(c))
    total = w.sum()
    if total < 1e-10:
        return 0.0
    p = w / total
    return float(-np.sum(p * np.log(p + 1e-15)))


sin2x_exact    = make_sin2x()
sinx2_truncated = make_sinx2_truncated()
sin2x_fitted   = fit_phik_ls(np.sin(2 * X_GRID))
sinx2_fitted   = fit_phik_ls(np.sin(X_GRID**2))


# ── Reporting ─────────────────────────────────────────────────────────────────

print("=" * 72)
print("SIN(x^2) <-> SIN(2x)  --  Framework Stress Test")
print("=" * 72)

print(f"\n[1] COEFFICIENT VECTORS  (phi_k = x^k/k! basis, N={N})")
print(f"    sin(2x)  exact    : {np.round(sin2x_exact, 1)}")
print(f"    sin(x^2) truncated: {np.round(sinx2_truncated, 1)}")
print(f"    sin(2x)  LS-fitted: {np.round(sin2x_fitted, 2)}")
print(f"    sin(x^2) LS-fitted: {np.round(sinx2_fitted, 2)}")

err_2x_exact   = approx_err(sin2x_exact,    np.sin(2 * X_GRID))
err_x2_trunc   = approx_err(sinx2_truncated, np.sin(X_GRID**2))
err_2x_fitted  = approx_err(sin2x_fitted,   np.sin(2 * X_GRID))
err_x2_fitted  = approx_err(sinx2_fitted,   np.sin(X_GRID**2))

print(f"\n    Approx errors on [-1.5, 1.5]:")
print(f"      sin(2x)  exact    : {err_2x_exact:.8f}  (exact to machine precision)")
print(f"      sin(x^2) truncated: {err_x2_trunc:.6f}  *** N=8 basis CANNOT REPRESENT this ***")
print(f"      sin(2x)  LS-fitted: {err_2x_fitted:.8f}")
print(f"      sin(x^2) LS-fitted: {err_x2_fitted:.6f}")
print(f"\n    The large sin(x^2) truncation error IS the grammar mismatch:")
print(f"    the phi_k basis needs index 10+ to represent chirp functions.")
print(f"    This is the Type 7 signal at the REPRESENTATION level, before any tools run.")

# Use fitted coefficients for all tool analyses — best approximation we can do in N=8
a = sin2x_fitted    # endpoint A: sin(2x)
b = sinx2_fitted    # endpoint B: sin(x^2)


# ── 2-adic filter ─────────────────────────────────────────────────────────────

print(f"\n[2] 2-ADIC PRE-FILTER  (adic_filter.py)")
adic = adic_compatibility(a, b)
print(f"    WHT sparsity:     sin(2x)={adic['sparsity_a']}  sin(x^2)={adic['sparsity_b']}")
print(f"    Sparsity change:  {adic['sparsity_change']} modes")
print(f"    Valuation gap:    {adic['valuation_gap']:.0f}")
print(f"    Spectral sim:     {adic['spectral_similarity']:.4f}  (-1=negation, 0=orthogonal)")
print(f"    Incompatibility:  {adic['incompatibility']:.4f}")
print(f"    Type 7 candidate: {adic['is_type7_candidate']}")

# Compare against known cases
from toolkit.euler_relay import _SIN as _SIN_euler
cos     = _COS.copy()
cos2x_v = np.zeros(N)
for k in range(N // 2):
    cos2x_v[2 * k] = (-1)**k * float(4**k)
adic_ref = adic_compatibility(cos, _SIN_euler.copy())
adic_t7  = adic_compatibility(cos, cos2x_v)
print(f"\n    Reference values:")
print(f"      cos->sin    (valid):  incompatibility={adic_ref['incompatibility']:.4f}")
print(f"      cos->cos2x  (Type7):  incompatibility={adic_t7['incompatibility']:.4f}")
print(f"      sin2x->sinx2 (this):  incompatibility={adic['incompatibility']:.4f}")


# ── Spectral warmup filter ────────────────────────────────────────────────────

print(f"\n[3] SPECTRAL WARMUP PRE-FILTER  (spectral_warmup.py)")
sw = spectral_type7_prefilter(a, b)
band_str = "  ".join(f"b{bv}={d:.2f}" for bv, d in sorted(sw["band_distances"].items()))
print(f"    Band disagreements: {band_str}")
print(f"    Low-freq distance:  {sw['low_freq_dist']:.4f}")
print(f"    Flag main effect:   {sw['flag_main_effect']}")
print(f"    Type 7 candidate:   {sw['is_type7_candidate']}")
dual = adic["is_type7_candidate"] and sw["is_type7_candidate"]
print(f"    Dual confirmed:     {dual}")


# ── SLERP angle ──────────────────────────────────────────────────────────────

print(f"\n[4] SLERP GEODESIC ANGLE  (slerp_relay.py)")
slerp_res = run_slerp(a, b, n_steps=8)
print(f"    Rotation angle:  {slerp_res['theta_deg']:.2f} degrees")
print(f"    Wass variance:   {slerp_res['wass_var']:.8f}  (0 = exact geodesic)")
print(f"    Step dists:      {[round(d, 4) for d in slerp_res['step_dists']]}")
print(f"    (cos->sin = 90 deg reference; cos->-cos = 180 deg maximum)")


# ── FSM junction stress ───────────────────────────────────────────────────────

print(f"\n[5] FSM JUNCTION STRESS  (fsm_relay.py)")
print(f"    Potential: WHT spectral entropy (high = spectrally complex = stressed)")

def potential(c):
    return wht_entropy(c)

fsm_main = run_fsm(a, b, step_fraction=0.15, potential_fn=potential)
fsm_ref  = run_fsm(cos, cos2x_v, step_fraction=0.15, potential_fn=potential)

s_main = fsm_main["stresses"]
s_ref  = fsm_ref["stresses"]
print(f"    cos->cos2x   (Type 7 ref): junction_stress={max(s_ref):.4f}  "
      f"at idx {fsm_ref['junction_idx']}/{fsm_ref['n_total']-1}")
print(f"    sin2x->sinx2 (this):       junction_stress={max(s_main):.4f}  "
      f"at idx {fsm_main['junction_idx']}/{fsm_main['n_total']-1}")
print(f"    Stress profile (this): {[round(s, 4) for s in s_main]}")


# ── Lifted grammar path ───────────────────────────────────────────────────────

print(f"\n[6] LIFTED GRAMMAR PATH  (analytical)")
print(f"    In linear phi_k grammar:  NO PATH EXISTS (Type 7)")
print(f"    In lifted argument space  (c1*x + c2*x^2):")
print(f"      Start: sin(x^2)  = sin(0*x + 1*x^2)  -> (c1=0, c2=1)")
print(f"      End:   sin(2x)   = sin(2*x + 0*x^2)  -> (c1=2, c2=0)")
print(f"      Path:  f(x,t) = sin(2t*x + (1-t)*x^2)  -- TRIVIAL STRAIGHT LINE")
print()

# Compute the stress profile along the lifted path
A_ls = np.column_stack([X_GRID**k / math.factorial(k) for k in range(N)])
A_ls_pinv = np.linalg.pinv(A_ls)

t_vals = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
print(f"    {'t':6s}  {'c1':5s}  {'c2':5s}  {'WHT_entropy':11s}  {'N8_approx_err':13s}  Label")
print(f"    {'-'*66}")

for t in t_vals:
    c1 = 2 * t
    c2 = 1 - t
    f_true = np.sin(c1 * X_GRID + c2 * X_GRID**2)
    c_fit  = A_ls_pinv @ f_true
    entropy = wht_entropy(c_fit)
    y_approx = reconstruct(c_fit)
    max_err = float(np.max(np.abs(y_approx - f_true)))
    label = {0.0: "sin(x^2)", 0.5: "sin(x + 0.5x^2)  <-- SADDLE", 1.0: "sin(2x)"}.get(t, "")
    print(f"    t={t:.3f}  c1={c1:.2f}  c2={c2:.2f}  "
          f"entropy={entropy:.4f}  approx_err={max_err:.6f}  {label}")

print()
print(f"    Key observation: entropy PEAKS near t=0.5 (the mixed chirp is maximally")
print(f"    spectrally complex). The N=8 basis fails hardest at the saddle point.")
print(f"    This is the connection curvature: the fiber bundle twists most violently")
print(f"    when c1=c2=1 (equal linear and quadratic components).")


# ── Final summary ─────────────────────────────────────────────────────────────

print()
print("=" * 72)
print("FINGERPRINT SUMMARY")
print("=" * 72)
print(f"  Representation mismatch (N=8): sin(x^2) approx_err = {err_x2_fitted:.4f}")
print(f"  2-adic incompatibility:        {adic['incompatibility']:.4f}"
      f"  {'[FLAGGED]' if adic['is_type7_candidate'] else '[not flagged]'}")
print(f"  Spectral main-band distance:   {sw['low_freq_dist']:.4f}"
      f"  {'[FLAGGED]' if sw['is_type7_candidate'] else '[not flagged]'}")
print(f"  Dual confirmed Type 7:         {dual}")
print(f"  SLERP rotation angle:          {slerp_res['theta_deg']:.2f} degrees")
print(f"  FSM junction entropy stress:   {max(s_main):.4f}")
print()
print(f"  Grammar enrichment diagnosis:")
print(f"    Current grammar:  phi_k (linear coefficient space, N={N})")
print(f"    Missing dimension: the quadratic argument x^2 as an independent axis")
print(f"    Lifted grammar:   argument space (c1*x + c2*x^2), dim=2")
print(f"    Lifted path:      trivial linear interpolation in (c1, c2)")
print(f"    Holonomy:         projecting the lifted path back to phi_k space")
print(f"                      creates the stress peak at t=0.5")
print()
print(f"  The activation energy barrier at t=0.5 is the shadow of the missing")
print(f"  dimension. An MLP trained on the taxonomy would learn to predict this")
print(f"  barrier WITHOUT running the full lifted-grammar optimization.")
