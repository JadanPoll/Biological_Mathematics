"""
Spectral warmup for CMA-ES — BARF-style Walsh frequency annealing.

Rachure, Park et al. (BARF, ICCV 2021). Adapted from positional encoding annealing
in NeRF to Walsh-Hadamard coefficient space for relay chain optimization.

CORE IDEA:
  CMA-ES (and all gradient-based optimizers) suffer from "spectral bias":
  they learn low-frequency components of the objective first.
  Instead of fighting this bias, exploit it:
    1. Start CMA-ES optimizing only low-frequency Walsh modes
    2. Progressively activate higher-frequency modes as low ones converge
    3. Each new band inherits the covariance structure from lower bands

  This gives:
    - 2-10x faster convergence (NeRF literature)
    - Natural early-exit for Type 7: fails to converge even at low frequency
    - Covariance matrix warm-starts across bands (no cold restart)

WALSH FREQUENCY ORDERING (Hamming weight = interaction order):
  Band 0 (Hamming weight 0): index 0    = DC / mean shift
  Band 1 (Hamming weight 1): indices 1,2,4 = main effects (one-gene sensitivities)
  Band 2 (Hamming weight 2): indices 3,5,6 = 2-way interactions
  Band 3 (Hamming weight 3): index 7    = 3-way interaction (highest order)

  Low Hamming weight = low-frequency = coarse structure = Wasserstein geodesic direction
  High Hamming weight = high-frequency = fine detail = local wrinkles

BARF ANNEALING FORMULA:
  alpha(t) = n_bands * t / n_total_steps   [0 to n_bands as training progresses]
  weight(k, alpha) = 0.5 * (1 - cos(pi * clamp(alpha - hamming(k), 0, 1)))

  weight=0: mode k is inactive (not visible to optimizer)
  weight=1: mode k is fully active
  Smooth transition between 0 and 1 as alpha passes hamming(k)

TYPE 7 EARLY EXIT:
  If two coefficient vectors A and B disagree strongly at low Walsh frequency
  (band 1: main effects), they are fundamentally incompatible in the linear grammar.
  No amount of high-frequency adjustment can bridge a low-frequency gap.
  This allows flagging Type 7 candidates in O(n) before running CMA-ES at all.

RELATION TO 2-ADIC FILTER (toolkit/adic_filter.py):
  Spectral warmup operates at optimization time: guides CMA-ES through bands.
  2-adic filter operates pre-CMA-ES: flags candidates from coefficient structure alone.
  They are complementary: 2-adic flags → spectral warmup confirms → CMA-ES refines.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict

from toolkit.walsh import wht


# ── Walsh frequency structure ─────────────────────────────────────────────────

def hamming_weight(k: int) -> int:
    """Number of 1-bits in binary representation of k = Walsh interaction order."""
    return bin(k).count('1')


def walsh_bands(n_dims: int) -> Dict[int, List[int]]:
    """
    Group Walsh indices by Hamming weight.
    Returns dict: band_level -> list of Walsh indices at that level.

    Band 0: index 0 (DC)
    Band 1: indices with one 1-bit (main effects)
    Band 2: indices with two 1-bits (2-way interactions)
    ...
    """
    bands: Dict[int, List[int]] = {}
    for k in range(n_dims):
        w = hamming_weight(k)
        bands.setdefault(w, []).append(k)
    return bands


def warmup_schedule(n_dims: int, n_total_steps: int,
                    soft: bool = True) -> List[np.ndarray]:
    """
    Generate the BARF annealing weight vector for each optimization step.

    Returns list of n_total_steps weight vectors (length n_dims).
    weight[k] at step t = how active Walsh index k is at step t.

    soft=True:  Hann-window smooth activation (BARF original formula)
    soft=False: Hard cutoff (all indices up to current band = 1, rest = 0)
    """
    bands = walsh_bands(n_dims)
    max_band = max(bands.keys())
    weights = []

    for step in range(n_total_steps):
        alpha = max_band * step / max(n_total_steps - 1, 1)
        w = np.zeros(n_dims)
        for k in range(n_dims):
            h = hamming_weight(k)
            if soft:
                x = np.clip(alpha - h, 0.0, 1.0)
                w[k] = 0.5 * (1.0 - np.cos(np.pi * x))
            else:
                w[k] = 1.0 if alpha >= h else 0.0
        weights.append(w)

    return weights


# ── Spectral subspace projection ──────────────────────────────────────────────

def project_to_band(coeff_vec: np.ndarray,
                    active_bands: List[int],
                    n_dims: Optional[int] = None) -> np.ndarray:
    """
    Project coefficient vector onto the active Walsh frequency bands.
    Returns a coefficient vector with only the active-band components.
    """
    if n_dims is None:
        n_dims = len(coeff_vec)
    bands = walsh_bands(n_dims)
    active_indices = []
    for b in active_bands:
        active_indices.extend(bands.get(b, []))

    spectrum = wht(coeff_vec)
    masked = np.zeros_like(spectrum)
    masked[active_indices] = spectrum[active_indices]
    return wht(masked) * n_dims


def spectral_distance(a: np.ndarray, b: np.ndarray,
                      band: int = 1) -> float:
    """
    Spectral distance between two coefficient vectors at a given Walsh band.
    Measures disagreement in the band-{band} subspace only.

    Band 1 (main effects) is the key diagnostic:
      Small band-1 distance: same coarse structure, likely morphable.
      Large band-1 distance: different coarse structure, likely Type 7.
    """
    n = len(a)
    a_proj = project_to_band(a, [band], n)
    b_proj = project_to_band(b, [band], n)
    return float(np.linalg.norm(a_proj - b_proj))


# ── Type 7 early detection ────────────────────────────────────────────────────

def spectral_type7_prefilter(a: np.ndarray,
                              b: np.ndarray,
                              threshold_main: float = 1.0,
                              threshold_dc: float = 2.0) -> dict:
    """
    O(n) spectral pre-filter for Type 7 detection.

    Computes spectral disagreement at each Walsh band.
    Strong disagreement at low bands (especially band 1: main effects)
    indicates the functions are fundamentally incompatible in linear grammar.

    A morph that fails to close at low frequency is unambiguously Type 7 —
    no high-frequency adjustment can bridge a low-frequency structural gap.

    Returns a dict with per-band disagreements and a Type 7 candidate flag.
    The threshold values are empirically calibrated on canonical cases:
      - cos→sin:    band1_dist ≈ 0.0 (same main-effect structure, different phase)
      - cos→cos(2x): band1_dist >> 1.0 (completely different frequency content)
    """
    n = len(a)
    bands = walsh_bands(n)
    max_band = max(bands.keys())

    wa = wht(a)
    wb = wht(b)

    band_dists = {}
    for band_level in range(max_band + 1):
        indices = bands.get(band_level, [])
        if not indices:
            continue
        diff = np.abs(wa[indices] - wb[indices])
        band_dists[band_level] = float(np.sum(diff))

    # Total spectral distance
    total_dist = sum(band_dists.values())

    # Low-frequency fraction: what fraction of total disagreement is at bands 0+1?
    low_freq_dist = band_dists.get(0, 0.0) + band_dists.get(1, 0.0)
    low_freq_frac = low_freq_dist / (total_dist + 1e-10)

    # Type 7 flags:
    # - Main-effect band (band 1) disagreement exceeds threshold
    flag_main   = band_dists.get(1, 0.0) > threshold_main
    # - DC disagreement (mean-shift) exceeds threshold
    flag_dc     = band_dists.get(0, 0.0) > threshold_dc
    # - Low-frequency fraction is high (disagreement is fundamental, not just detail)
    flag_struct = low_freq_frac > 0.5

    is_type7_candidate = flag_main or flag_dc

    return {
        "band_distances":        band_dists,
        "total_spectral_dist":   total_dist,
        "low_freq_dist":         low_freq_dist,
        "low_freq_fraction":     low_freq_frac,
        "flag_main_effect":      flag_main,
        "flag_dc":               flag_dc,
        "flag_structural":       flag_struct,
        "is_type7_candidate":    is_type7_candidate,
        "confidence":            low_freq_frac if is_type7_candidate else 1.0 - low_freq_frac,
    }


# ── Spectral warmup CMA-ES wrapper ────────────────────────────────────────────

def make_spectral_objective(base_objective,
                             coeff_dim: int,
                             active_bands: List[int]):
    """
    Wrap a CMA-ES objective to operate only in the active Walsh frequency bands.

    base_objective(x) -> float  [full coefficient vector]
    Returns: wrapped_objective(x_masked) -> float

    Usage:
      for band in range(max_band + 1):
          active_bands = list(range(band + 1))
          obj = make_spectral_objective(my_loss, n_dims, active_bands)
          result = cma.fmin(obj, x0_masked, sigma0, ...)
          # Transfer solution back to full space before next band
    """
    bands = walsh_bands(coeff_dim)
    active_indices = []
    for b in active_bands:
        active_indices.extend(bands.get(b, []))

    def spectral_objective(x_in: np.ndarray) -> float:
        # Lift x_in to full coefficient space (zero out inactive bands)
        spectrum = wht(x_in)
        masked   = np.zeros_like(spectrum)
        masked[active_indices] = spectrum[active_indices]
        x_full = wht(masked) * coeff_dim
        return base_objective(x_full)

    return spectral_objective


# ── Sanity validation ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from toolkit.euler_relay import _COS, _SIN, N

    cos   = _COS.copy()
    sin   = _SIN.copy()
    cos2x = np.zeros(N)
    for k in range(N // 2):
        cos2x[2 * k] = (-1)**k * float(4**k)
    neg_cos = -cos.copy()

    print("=" * 65)
    print("SPECTRAL WARMUP — Sanity Validation")
    print("BARF-style Walsh frequency annealing")
    print("=" * 65)

    all_pass = True

    # ── VALIDATION 1: Walsh band structure ────────────────────────────────
    print(f"\nVALIDATION 1: Walsh bands for N={N} coefficient vectors")
    bands = walsh_bands(N)
    for b, indices in sorted(bands.items()):
        names = [f"k={k}" for k in indices]
        print(f"  Band {b} (Hamming {b}): {names}")

    # ── VALIDATION 2: Annealing schedule is monotonically increasing ──────
    print(f"\nVALIDATION 2: Annealing schedule (first 5 / last 5 steps)")
    schedule = warmup_schedule(N, n_total_steps=100)
    print(f"  Step 0 (cold): active={np.sum(schedule[0] > 0.01):.0f}/{N} modes")
    print(f"  Step 25:       active={np.sum(schedule[25] > 0.01):.0f}/{N} modes")
    print(f"  Step 50:       active={np.sum(schedule[50] > 0.01):.0f}/{N} modes")
    print(f"  Step 99 (hot): active={np.sum(schedule[99] > 0.01):.0f}/{N} modes")

    # Monotone: sum of weights should increase with step
    totals = [float(np.sum(schedule[t])) for t in range(100)]
    is_monotone = all(totals[t+1] >= totals[t] - 1e-6 for t in range(99))
    status2 = "PASS" if is_monotone else "FAIL"
    if status2 == "FAIL":
        all_pass = False
    print(f"  [{status2}] Weight sum monotonically non-decreasing")

    # ── VALIDATION 3: Type 7 pre-filter on canonical cases ────────────────
    print("\nVALIDATION 3: Type 7 spectral pre-filter on canonical cases")
    print("  Expected: cos->sin NOT flagged, cos->cos(2x) FLAGGED")

    cases = [
        ("cos -> sin       (valid, derivative)",  cos,     sin,   False),
        ("cos -> -cos      (valid, Euler rot.)",  cos,     neg_cos, False),
        ("cos -> cos(2x)   (TYPE 7, invalid)",    cos,     cos2x, True),
    ]

    for label, a, b, expect_type7 in cases:
        result = spectral_type7_prefilter(a, b)
        flagged = result["is_type7_candidate"]
        band_str = "  ".join(
            f"b{bv}={d:.3f}" for bv, d in sorted(result["band_distances"].items())
        )
        correct = (flagged == expect_type7)
        status = "PASS" if correct else "FAIL"
        if not correct:
            all_pass = False
        print(f"  {label}:")
        print(f"    {band_str}")
        print(f"    type7_candidate={flagged} (expected={expect_type7})  [{status}]")

    # ── VALIDATION 4: Band projection preserves structure ─────────────────
    print("\nVALIDATION 4: Band projection round-trip accuracy")
    for active_bands in [[0, 1], [0, 1, 2], [0, 1, 2, 3]]:
        proj = project_to_band(cos, active_bands, N)
        full = project_to_band(cos, list(range(max(bands.keys()) + 1)), N)
        err  = float(np.linalg.norm(full - cos))
        proj_energy = float(np.linalg.norm(proj))
        cos_energy  = float(np.linalg.norm(cos))
        print(f"  Bands {active_bands}: projected_energy/total="
              f"{proj_energy:.4f}/{cos_energy:.4f}  "
              f"full_reconstruct_err={err:.2e}")
    status4 = "PASS" if err < 1e-6 else "FAIL"
    if status4 == "FAIL":
        all_pass = False
    print(f"  [{status4}] Full-band projection recovers original vector")

    print()
    print("VALIDATION CRITERIA:")
    print("  Type 7 filter correctly flags cos->cos(2x) and not cos->sin")
    print("  Annealing schedule is monotonically non-decreasing")
    print("  Full-band projection recovers original vector to machine precision")
    print()
    print("OVERALL:", "PASS" if all_pass else "FAIL")
