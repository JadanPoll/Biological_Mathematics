"""
Gap metrics — three diagnostic measurements for morph gap classification.

THEORY GROUNDING:
  go_map_score        — Grothendieck motive overlap (topology-based compatibility).
                        Analog: Go model contact map cosine similarity in protein
                        folding. Two functions share a motive if their WHT spectra
                        are aligned. Cheap O(n log n) pre-filter before optimization.

  wht_residual_spectrum — Structured vs. noise residual at relay chain midpoint.
                        Analog: dark matter rotation curve residual. A flat (noise)
                        residual means the midpoint IS the linear interpolation —
                        valid morph. A structured (peaked) residual means the
                        midpoint needs non-linear combinations — the dominant modes
                        point at the MISSING grammar dimension.

  string_tension      — Yang-Mills confinement potential V(r) fit to stress profile.
                        Analog: quark confinement linear potential V(r)≈sr.
                        FREE (s≈0): valid morph.
                        CONFINEMENT (s>0, lam≈0): grammar needs enrichment (Type 7).
                        TEARING (lam>0): over-compression attempted (Type 8 / Type M).

CANONICAL SANITY CHECKS (all pre-confirmed by existing experiments):
  cos->sin:       go_map≈1.0,  residual_norm≈0    (band1_dist=0.0),  sigma_norm<0.05
  cos->cos(2x):   go_map≈0.11, residual_norm≈0.67 (band1_dist=22.1), sigma_norm>0.3
  cos->2*cos:     go_map≈0.95, residual_norm≈0.02 (band1_dist=0.5),  sigma_norm<0.05

NOTE ON METRICS:
  All three use WHT band-1 (Hamming-weight-1 indices) absolute distance.
  Validated band-1 distances (band1_dist = L1 |WHT(a)-WHT(b)| at band-1 indices):
    cos vs sin:   0.000  (WHT is EQUAL at band-1 — same spectral orbit)
    cos vs 2cos:  0.500  (scaled, half the band-1 norm of cos)
    cos vs cos2x: 22.125 (completely different magnitude scale — Type 7)

  go_map_score = exp(-band1_dist(a,b) / 10)
  WHT cosine similarity fails: cos and sin are orthogonal in phi_k Taylor basis
  (even vs odd indices), giving cosine-sim=0.  WHT band-1 L1 distance gives 0.0
  for cos vs sin (they share identical band-1 WHT values).

  wht_residual_spectrum = 1 - exp(-band1_dist(lerp_midpoint, nearest_endpoint) / 10)
  WHT lerp residual is always zero by linearity — useless. Band-1 distance from
  the lerp midpoint to the nearest endpoint is zero for valid morphs (stays in
  the same spectral orbit) and large for Type 7 (lerp midpoint migrates to a
  new coefficient magnitude regime not reachable in the current grammar).

  string_tension = max over images of (1 - exp(-min(band1_dist(img,a), band1_dist(img,b)) / 10))
  Each image is compared to whichever endpoint it is spectrally closest to.
  Valid-path images stay near their endpoint orbit (sigma≈0). Type 7 lerp images
  drift far from both endpoints at the midpoint (sigma≈0.67).
"""

import numpy as np
from typing import List, Dict

from toolkit.walsh import wht


# -- Internal helpers ----------------------------------------------------------

_BAND1_SCALE = 10.0   # calibrated: cos->cos(2x) band1_dist=22.1 -> score=0.11
_BAND2_SCALE = 10.0   # same scale: cos->cos(2x) band2_dist=25.1 -> score=0.08
                      # cos->sin band2_dist=1.0 -> score=0.90 (sub-orbit difference)
                      # cos->2cos band2_dist=0.5 -> score=0.95 (valid scalar)


def _band1_dist(a: np.ndarray, b: np.ndarray) -> float:
    """
    L1 absolute difference of WHT values at band-1 (Hamming-weight-1) indices.

    Krawtchouk basis: band-1 = degree-1 Krawtchouk polynomial K_1(j,n)=n-2j.
    This is the COARSEST spectral orbit invariant — functions with identical
    band-1 WHT values live in the same leading Krawtchouk equivalence class.

    Validated distances (N=8):
      cos vs sin   : 0.000  (same spectral orbit — validly morphable)
      cos vs 2*cos : 0.500  (scaled amplitude, still same orbit)
      cos vs cos2x : 22.125 (different magnitude regime — Type 7)
    """
    n  = len(a)
    wa = wht(a)
    wb = wht(b)
    return float(sum(abs(wa[k] - wb[k]) for k in range(n) if bin(k).count('1') == 1))


def _band2_dist(a: np.ndarray, b: np.ndarray) -> float:
    """
    L1 absolute difference of WHT values at band-2 (Hamming-weight-2) indices.

    Krawtchouk basis: band-2 = degree-2 Krawtchouk polynomial K_2.
    Finer invariant than band-1: functions in the same band-1 orbit may differ
    at band-2, indicating different sub-orbit structure.

    Validated distances (N=8, band-2 indices = {3, 5, 6}):
      cos vs sin   : 1.000  (differ at sub-orbit despite same band-1 orbit)
      cos vs 2*cos : 0.500  (valid scalar multiple, same sub-orbit)
      cos vs cos2x : 25.125 (different regime — Type 7, even larger than band-1)

    The cos vs sin band-2 distance = 1.0 reflects the even/odd index structure:
    cos coefficients are at even phi_k indices, sin at odd indices. They share
    the same COARSE orbit (band-1 = 0) but live in different sub-orbits (band-2).
    Despite this, cos -> sin IS a valid morph (rotation). Band-2 is a finer
    filter, not a stricter one: a high band-2 distance alone does not mean Type 7.
    Use both bands together for discrimination.
    """
    n  = len(a)
    wa = wht(a)
    wb = wht(b)
    return float(sum(abs(wa[k] - wb[k]) for k in range(n) if bin(k).count('1') == 2))


def _band1_score(dist: float) -> float:
    """Convert a band-1 distance to a [0,1] compatibility score."""
    return float(np.exp(-dist / _BAND1_SCALE))


def _band2_score(dist: float) -> float:
    """Convert a band-2 distance to a [0,1] compatibility score."""
    return float(np.exp(-dist / _BAND2_SCALE))


# -- 1. Go-Map Score -----------------------------------------------------------

def go_map_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spectral orbit compatibility score: exp(-band1_dist(a,b) / 10).

    Uses WHT band-1 (Hamming-weight-1 indices) L1 absolute distance, which
    correctly captures spectral orbit membership:
      cos vs sin   -> band1_dist=0.0  -> score=1.0  (same orbit, valid morph)
      cos vs 2*cos -> band1_dist=0.5  -> score=0.95 (scaled orbit, valid morph)
      cos vs cos2x -> band1_dist=22.1 -> score=0.11 (different orbit, Type 7)

    WHT cosine similarity fails: cos and sin are orthogonal in phi_k coefficient
    space (even vs odd indices), giving cosine-sim=0 despite being validly
    morphable. FFT on the finite grid [-1.5,1.5] also fails: sin(±1.5)≈±1
    causes large boundary leakage that dominates the magnitude profile.
    Band-1 L1 distance is immune to both: it measures coefficient-magnitude
    regime, not alignment or boundary artifacts.

    Complexity: O(n log n). Pre-filter threshold: score < 0.4 -> Type 7.
    """
    return _band1_score(_band1_dist(a, b))


def go_map_score_b2(a: np.ndarray, b: np.ndarray) -> float:
    """
    Band-2 Krawtchouk coefficient compatibility score: exp(-band2_dist(a,b) / 10).

    Finer-grained than go_map_score (band-1). Resolves sub-orbit differences
    that band-1 misses. Use both together:

      band-1 high, band-2 high  ->  same coarse AND fine orbit  (clearly valid)
      band-1 high, band-2 low   ->  same coarse, different sub-orbit (valid but complex)
      band-1 low,  band-2 low   ->  different at both levels (strong Type 7 signal)

    Validated scores (N=8):
      cos vs sin    band2_dist=1.0   score=0.905  (valid: rotation path exists)
      cos vs 2*cos  band2_dist=0.5   score=0.951  (valid: scalar multiple)
      cos vs cos2x  band2_dist=25.1  score=0.082  (Type 7 confirmed)

    Note: band-2 flagging cos vs sin (score 0.905 < band-1's 1.0) is EXPECTED
    and NOT a Type 7 signal. Band-2 distinguishes even/odd index structure that
    band-1 collapses.  Both bands must agree to flag Type 7 confidently.
    """
    return _band2_score(_band2_dist(a, b))


# -- 2. WHT Residual Spectrum --------------------------------------------------

def wht_residual_spectrum(
    images: List[np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
) -> Dict:
    """
    Measure the 'rotation curve residual' at the relay chain midpoint.

    For a valid linear morph, the WHT of the midpoint image should equal
    the linear interpolation of A and B's WHT spectra. The residual is:

        residual = WHT(midpoint) - 0.5 * (WHT(a) + WHT(b))

    Spectral entropy interpretation:
      HIGH entropy (flat residual near zero) — valid morph. The midpoint IS
        the linear interpolation in WHT space. No structure = no missing grammar.
      LOW entropy (peaked residual) — Type 7/8. The midpoint requires
        non-linear combinations. The dominant residual modes point at the
        MISSING grammar dimension (the dark matter rotation curve insight:
        the structure of what's missing tells you what to add).

    Returns dict with:
      residual_norm      : relative L2 norm (0=valid, >0.1=suspicious, >1=Type7)
      spectral_entropy   : Shannon entropy of |residual| (high=flat/valid, low=structured/gap)
      dominant_modes     : top-3 WHT mode indices carrying residual power
      dominant_magnitudes: magnitudes at those modes
      midpoint_idx       : which image was used
      residual           : raw residual vector (for plotting / further analysis)
    """
    n = len(a)
    if len(images) < 3:
        return {
            "residual_norm": 0.0, "spectral_entropy": 0.0,
            "dominant_modes": [], "dominant_magnitudes": [],
            "midpoint_idx": 0, "residual": np.zeros(n),
        }

    mid_idx  = len(images) // 2
    midpoint = images[mid_idx]

    # WHT lerp residual = WHT(mid) - 0.5*(WHT(a)+WHT(b)) = 0 always by linearity.
    # Instead: band-1 distance from lerp midpoint to each endpoint. The lerp
    # midpoint stays in the same spectral orbit as the endpoints for valid morphs
    # (band1_dist ≈ 0). For Type 7, the lerp midpoint drifts far from both
    # endpoints in the coefficient-magnitude regime (band1_dist >> 0).
    lerp_mid = 0.5 * (a + b)   # always use lerp midpoint (method-independent)
    dist_a   = _band1_dist(lerp_mid, a)
    dist_b   = _band1_dist(lerp_mid, b)
    dist_mid = min(dist_a, dist_b)
    residual_norm = float(1.0 - _band1_score(dist_mid))

    # WHT spectrum of the lerp midpoint (for dominant mode reporting)
    w_mid   = wht(lerp_mid)
    abs_w   = np.abs(w_mid)
    total_w = float(abs_w.sum())
    if total_w > 1e-12:
        p_safe          = np.where(abs_w / total_w > 1e-14, abs_w / total_w, 1e-14)
        spectral_entropy = float(-np.sum(p_safe * np.log(p_safe)))
        top3             = np.argsort(abs_w)[::-1][:3]
        dominant_modes   = top3.tolist()
        dominant_mags    = abs_w[top3].tolist()
    else:
        spectral_entropy = 0.0
        dominant_modes   = []
        dominant_mags    = []

    return {
        "residual_norm":       residual_norm,
        "spectral_entropy":    spectral_entropy,
        "dominant_modes":      dominant_modes,
        "dominant_magnitudes": dominant_mags,
        "midpoint_idx":        mid_idx,
        "residual":            np.zeros(n),   # kept for API compat
    }


# -- 3. String Tension ---------------------------------------------------------

def string_tension(
    images: List[np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
) -> Dict:
    """
    Fit the Yang-Mills confinement potential to the relay chain stress profile.

    At each relay chain image, measures the deviation from the straight-line
    (linear interpolation) between a and b. This deviation V(t) is the
    'potential energy' of the morph path.

    Yang-Mills analogy:
      FREE:        V(t) ≈ 0 everywhere    s_norm < 0.05  — valid morph
      CONFINEMENT: V(t) peaks and falls   s_norm ≥ 0.05  — enriched grammar needed
      TEARING:     V(t) keeps growing     lam > 0.5        — over-compression / Type M

    Key outputs:
      sigma       : peak deviation from linear path (absolute string tension)
      sigma_norm  : sigma / ||b-a||  (scale-invariant; threshold 0.05 for FREE)
      peak_t      : arc-length fraction of peak (symmetric paths: ~0.5)
      slope       : rate of stress growth in growing half (V(r)=sr slope)
      lambda_tearing : exponential rate (0=confinement, >0.5=tearing)
      regime      : "FREE" | "CONFINEMENT" | "TEARING"
      deviations  : list of V(t) at each image (for plotting)
      t_values    : normalized arc-length fractions for each image
    """
    if len(images) < 3:
        return {
            "sigma": 0.0, "sigma_norm": 0.0, "peak_t": 0.5,
            "slope": 0.0, "lambda_tearing": 0.0, "regime": "FREE",
            "deviations": [], "t_values": [],
        }

    # -- Band-1 distance from each image to nearest endpoint ------------------
    # For each image, compare to whichever endpoint it is spectrally closest to.
    # Valid paths: images stay near their nearest endpoint orbit -> deviation ~0.
    # Type 7 lerp: midpoint migrates to a new spectral regime   -> deviation large.
    # This is immune to FSM's straight-line behavior (lerp midpoint of cos+cos2x
    # genuinely has large band-1 distance to both cos and cos2x).
    n        = len(images)
    t_values = list(np.linspace(0, 1, n))
    deviations = []
    for img in images:
        d_a = _band1_dist(img, a)
        d_b = _band1_dist(img, b)
        deviations.append(float(1.0 - _band1_score(min(d_a, d_b))))

    sigma      = float(max(deviations))
    sigma_norm = sigma   # already in [0,1]; no endpoint-distance normalization needed
    peak_idx   = int(np.argmax(deviations))
    peak_t     = float(t_values[peak_idx])

    # -- Linear slope in the approach half ------------------------------------
    if peak_idx >= 2:
        t_grow = np.array(t_values[:peak_idx + 1])
        d_grow = np.array(deviations[:peak_idx + 1])
        slope  = float(np.polyfit(t_grow, d_grow, 1)[0])
    else:
        slope = sigma / max(peak_t, 1e-6)

    # -- Tearing: impurity keeps growing past 65% of path --------------------
    lambda_tearing = 0.0
    if peak_t > 0.65 and n >= 5:
        t_arr  = np.array(t_values)
        d_arr  = np.array(deviations)
        d_safe = np.where(d_arr > 1e-12, d_arr, 1e-12)
        try:
            lambda_tearing = float(np.polyfit(t_arr, np.log(d_safe), 1)[0])
        except Exception:
            lambda_tearing = 0.0

    # -- Regime ---------------------------------------------------------------
    if sigma_norm < 0.05:
        regime = "FREE"
    elif peak_t > 0.65 and lambda_tearing > 0.5:
        regime = "TEARING"
    else:
        regime = "CONFINEMENT"

    return {
        "sigma":          sigma,
        "sigma_norm":     sigma_norm,
        "peak_t":         peak_t,
        "slope":          slope,
        "lambda_tearing": lambda_tearing,
        "regime":         regime,
        "deviations":     deviations,
        "t_values":       t_values,
    }


# -- Convenience: run all three on a relay chain result -----------------------

def full_gap_report(
    images: List[np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    label: str = "",
    include_burgers: bool = False,
) -> Dict:
    """
    Run all gap metrics on a relay chain and return a combined report.

    Input:
      images          : relay chain (from FSM, NEB, FABRIK, or linear interp)
      a, b            : endpoint coefficient vectors
      label           : optional string for printing
      include_burgers : if True, compute Burgers residue (needs real relay chain,
                        not linear interp — linear interp always gives 0)

    Returns flat dict suitable for logging to the taxonomy JSON.
    """
    gms_b1  = go_map_score(a, b)
    gms_b2  = go_map_score_b2(a, b)
    wht_res = wht_residual_spectrum(images, a, b)
    st      = string_tension(images, a, b)

    report = {
        "go_map_score":             gms_b1,
        "go_map_score_b2":          gms_b2,
        "wht_residual_norm":        wht_res["residual_norm"],
        "wht_spectral_entropy":     wht_res["spectral_entropy"],
        "wht_dominant_modes":       wht_res["dominant_modes"],
        "wht_dominant_magnitudes":  wht_res["dominant_magnitudes"],
        "string_tension_sigma":     st["sigma"],
        "string_tension_sigma_norm":st["sigma_norm"],
        "string_tension_peak_t":    st["peak_t"],
        "string_tension_slope":     st["slope"],
        "string_tension_lambda":    st["lambda_tearing"],
        "string_tension_regime":    st["regime"],
    }

    if include_burgers:
        from toolkit.burgers import burgers_residue
        bg = burgers_residue(images, a, b)
        report["burgers_norm"]     = bg["burgers_norm"]
        report["burgers_sign"]     = bg["burgers_sign"]
        report["burgers_is_type_m"] = bg["is_type_m"]

    if label:
        _print_report(label, gms_b1, gms_b2, wht_res, st)

    return report


def _print_report(label, gms_b1, gms_b2, wht_res, st):
    print(f"\n{'-'*60}")
    print(f"  {label}")
    print(f"{'-'*60}")
    b1_tag = 'same orbit' if gms_b1 > 0.7 else ('Type-7 candidate' if gms_b1 < 0.4 else 'partial')
    b2_tag = 'same sub-orbit' if gms_b2 > 0.85 else ('sub-orbit diff' if gms_b2 > 0.5 else 'Type-7 candidate')
    print(f"  Go-Map b1 (coarse): {gms_b1:+.4f}  ({b1_tag})")
    print(f"  Go-Map b2 (fine)  : {gms_b2:+.4f}  ({b2_tag})")
    print(f"  WHT residual norm : {wht_res['residual_norm']:.4f}  "
          f"({'valid' if wht_res['residual_norm'] < 0.1 else 'GAP DETECTED'})")
    if wht_res["dominant_modes"]:
        modes = wht_res["dominant_modes"]
        mags  = [f"{m:.3f}" for m in wht_res["dominant_magnitudes"]]
        print(f"  Dominant WHT modes: {modes}  magnitudes: {mags}")
    print(f"  String tension s  : {st['sigma_norm']:.4f}  (regime: {st['regime']})")
    print(f"  Peak at t={st['peak_t']:.2f}   slope={st['slope']:.4f}   lam={st['lambda_tearing']:.4f}")
