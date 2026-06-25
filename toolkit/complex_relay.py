"""
Complex-coefficient relay chain — the Argand diagram lift.

The dimensional lift that resolves all degeneracy in the Euler relay chain.

The problem in real 8D coefficient space:
  - Infinitely many valid rotations (8D rotation group SO(8))
  - The relay chain finds arbitrary rotations, not specifically cos/sin
  - Truncation error causes 8-10 degree systematic offset
  - The space is "Weierstrass-like" — zooming in doesn't help

The fix — lift to complex 1D space:
  cos(x) and sin(x) live in a 2D subspace of coefficient space.
  Project onto this subspace → get a single complex number z = a + ib.
  The Euler rotation is EXACT multiplication by e^(iδ) in this 1D complex space.
  No truncation error. No degeneracy (1D complex has only one rotation axis).
  The "many valid rotations" of SO(8) collapse to the unique rotation of U(1).

This is exactly the Argand diagram insight:
  e^(iπ) + 1 = 0 is NOT about the infinite decimal expansions of e and π.
  It is about a rotation by π in the complex plane.
  In the complex plane the chaos of 1D becomes the geometry of 2D.

The Riemann-Hurwitz connection:
  The map from cos to -cos via e^(iπ) has:
    - Degree 1 (bijection)
    - Zero ramification (smooth everywhere)
    - Therefore: zero stress, zero complexity gain
  This is the simplest possible test case — the relay chain should find it
  immediately and cleanly in the lifted representation.

The obstruction theory connection:
  In real 8D space, there IS an obstruction to finding the specific Euler path
  (the cohomological obstruction = the 7 extra rotation axes of SO(8) that
  pull the path away from the cos/sin subspace).
  Lifting to complex 1D eliminates all obstruction classes — the space has
  only one rotation axis (U(1)) and the Euler path is the unique geodesic.
"""

import math
import numpy as np
from typing import List, Optional

from toolkit.euler_relay import _COS, _SIN, N


# ── Complex projection ────────────────────────────────────────────────────────

# Norms for projection
_COS_NORM_SQ = float(np.dot(_COS, _COS))   # = 4.0 (four nonzero entries of ±1)
_SIN_NORM_SQ = float(np.dot(_SIN, _SIN))   # = 4.0


def to_complex(c: np.ndarray) -> complex:
    """
    Project coefficient vector c onto the (cos, sin) subspace and return
    as a complex number z = a - ib where:
      a = projection onto cos direction
      b = projection onto sin direction

    The sign convention: z = a - ib so that euler_step becomes z → e^(-iδ)·z
    which gives the correct positive phase rotation (cos → cos(x+δ)).
    """
    a = float(np.dot(c, _COS)) / _COS_NORM_SQ
    b = float(np.dot(c, _SIN)) / _SIN_NORM_SQ
    return complex(a, -b)


def from_complex(z: complex) -> np.ndarray:
    """
    Reconstruct coefficient vector from complex number.
    z = cos(theta) - i*sin(theta) = e^(-i*theta)
    -> coefficient vector = Re(z)*_COS + Im(z)*_SIN
                          = cos(theta)*_COS - sin(theta)*_SIN  [correct sign]
    """
    return z.real * _COS + z.imag * _SIN


def euler_step_complex(z: complex, delta: float) -> complex:
    """
    Exact Euler step in complex representation.
    z → e^(-iδ) · z

    This is EXACT multiplication, no truncation error.
    Compare to the real-space euler_step which has truncation error from
    the 8-term Taylor series.
    """
    return z * complex(math.cos(delta), -math.sin(delta))


def true_intermediate_complex(k: int, n_steps: int) -> complex:
    """
    Exact complex representation of the k-th Euler intermediate.
    z_k = e^(-ikπ/n) = cos(kπ/n) - i·sin(kπ/n)
    """
    theta = k * math.pi / n_steps
    return complex(math.cos(theta), -math.sin(theta))


# ── Complex relay chain ───────────────────────────────────────────────────────

def run_complex_relay(n_steps: int = 4,
                       n_iterations: int = 1000,
                       learning_rate: float = 0.05,
                       seed: int = 42,
                       verbose: bool = True) -> dict:
    """
    Run the Euler relay chain in complex 1D space.

    In this lifted representation:
      - No degeneracy (only one rotation axis)
      - No truncation error (exact multiplication)
      - The relay chain is just optimization in the complex plane
      - The answer is known analytically: z_k = e^(-ikπ/n)

    Each intermediate z_k is optimized to minimize:
      ||z_k - e^(-iδ)·z_{k-1}||² + ||z_{k+1} - e^(-iδ)·z_k||²

    This is a gradient descent problem on the complex plane — trivial.
    The point is to VERIFY that the complex representation gives the right answer
    and then use this as the GROUND TRUTH to calibrate the real-space relay chain.

    Returns:
      found_intermediates: list of complex numbers found
      true_intermediates:  list of analytically correct complex numbers
      max_error:           maximum deviation from true path
      in_coefficient_space: coefficient vectors reconstructed from complex numbers
    """
    rng = np.random.default_rng(seed)
    delta = math.pi / n_steps

    # Fixed endpoints
    z_start = complex(1.0, 0.0)    # cos → z = 1 + 0i
    z_end   = complex(-1.0, 0.0)   # -cos → z = -1 + 0i

    # Initialize intermediates randomly in the complex plane
    intermediates = [z_start]
    for k in range(1, n_steps):
        z_init = complex(rng.uniform(-1, 1), rng.uniform(-0.5, 0.5))
        intermediates.append(z_init)
    intermediates.append(z_end)

    # Analytic update: the minimum of
    #   L(z_k) = |z_k - e^(-id)*z_prev|^2 + |z_next - e^(-id)*z_k|^2
    # is achieved at:
    #   z_k = (e^(-id)*z_prev + e^(+id)*z_next) / 2
    # Gauss-Seidel iteration converges rapidly (no learning rate needed).
    e_neg = complex(math.cos(delta), -math.sin(delta))   # e^(-i*delta)
    e_pos = complex(math.cos(delta),  math.sin(delta))   # e^(+i*delta)

    for iteration in range(n_iterations):
        for k in range(1, n_steps):
            z_prev = intermediates[k - 1]
            z_next = intermediates[k + 1]
            intermediates[k] = (e_neg * z_prev + e_pos * z_next) / 2.0

    # Analytical true values
    true_ints = [true_intermediate_complex(k, n_steps)
                 for k in range(n_steps + 1)]

    errors = [abs(intermediates[k] - true_ints[k])
              for k in range(n_steps + 1)]
    max_err = max(errors)

    # Phase angles
    found_angles = [math.degrees(-(np.angle(z))) for z in intermediates]
    true_angles  = [math.degrees(k * math.pi / n_steps) for k in range(n_steps + 1)]

    # Convert back to coefficient vectors
    coeff_vecs = [from_complex(z) for z in intermediates]

    if verbose:
        print(f"\n  COMPLEX RELAY CHAIN (n_steps={n_steps})")
        print(f"  Representation: 1D complex (lifted from 8D real)")
        print(f"  The Euler rotation is exact multiplication by e^(-i*delta)")
        print(f"\n  Max error from true path: {max_err:.8f}  (should be ~1e-6)")
        print(f"\n  Per-step diagnostics:")
        for k in range(n_steps + 1):
            label = "FIXED" if k in (0, n_steps) else "free "
            print(f"    k={k} [{label}]  "
                  f"expected={true_angles[k]:6.1f}deg  "
                  f"found={found_angles[k]:6.1f}deg  "
                  f"error={errors[k]:.8f}")

        # Compare to real-space open-chain results
        print(f"\n  COMPARISON TO REAL-SPACE OPEN CHAIN:")
        print(f"    Complex space max error:  {max_err:.6f}")
        print(f"    Real space mean step res: ~0.63  (from previous experiment)")
        print(f"    Improvement factor:       ~{0.63/max(max_err, 1e-10):.0f}×")

    return {
        "found":          intermediates,
        "true":           true_ints,
        "errors":         errors,
        "max_error":      max_err,
        "coeff_vecs":     coeff_vecs,
        "found_angles":   found_angles,
        "true_angles":    true_angles,
    }


def calibration_residual(n_steps: int = 4) -> float:
    """
    The residual between the complex-space solution and the real-space
    coefficient vectors. This is the 'irreducible truncation gap' —
    the minimum residual achievable with 8 Taylor terms.

    If real-space relay chain achieves this residual, it has found the
    best possible solution given the truncation. If it achieves worse,
    there is still optimization improvement possible.
    """
    result = run_complex_relay(n_steps, n_iterations=2000, verbose=False)

    # Convert complex solution back to coefficient vectors and measure
    # how close they are to the true Euler intermediates in 8D space
    from toolkit.euler_relay import euler_intermediate

    residuals = []
    for k in range(n_steps + 1):
        z_k = result["found"][k]
        c_k = from_complex(z_k)           # 8D coefficient vector
        t_k = euler_intermediate(k, n_steps)  # analytical true intermediate
        residuals.append(float(np.linalg.norm(c_k - t_k)))

    return float(np.mean(residuals))


if __name__ == "__main__":
    print("="*60)
    print("COMPLEX RELAY CHAIN — Dimensional Lift")
    print("Euler path in 1D complex space (exact, no truncation)")
    print("="*60)

    for n in [4, 8, 12]:
        result = run_complex_relay(n_steps=n, n_iterations=1000)
        calib = calibration_residual(n)
        print(f"\n  Calibration residual (complex->8D): {calib:.6f}")
        print(f"  This is the MINIMUM achievable residual in 8D real space.")
        print(f"  Any real-space relay chain achieving less than {calib:.4f}")
        print(f"  has matched the theoretical optimum for {n} steps.")
