"""
Minimal 2D version of the morphing problem.

Instead of 8-dimensional Taylor coefficient vectors, use 2D unit circle points.
  - Function = point on unit circle: [cos(theta), sin(theta)]
  - Relationship = rotation angle between two points
  - Relay chain = sequence of points on the circle from A to B
  - Stress = how far the path deviates from the geodesic (great circle arc)

WHY THIS IS THE RIGHT MINIMAL VERSION:

1. The relationship is trivially computable: angle = arctan2(b[1], b[0]) - arctan2(a[1], a[0])
2. The relay chain has an EXACT solution: uniform angular steps along the arc
3. The grammar is complete: just {rotation_by_delta} for various delta
4. Phase space transitions are exact: each intermediate added constrains the path
5. Stress is geometrically interpretable: deviation from circular arc

PHASE SPACE TRANSITIONS (the key experiment):
  Start: A = [1, 0]  (0 degrees)
  End:   B = [cos(beta), sin(beta)]  (beta degrees)

  0 intermediates: ALL paths from A to B valid (infinite solutions)
  1 intermediate:  paths must pass near the midpoint on the arc
  2 intermediates: three-point constraint on the arc
  n intermediates: n+1 point constraint -> converges to unique geodesic

  The 'quantization' question: as n increases, how fast does the path
  space collapse to the single geodesic?

  This is the 2D version of the quantum well.  For the unit circle, the
  answer is analytic: n intermediates selects paths where the curvature
  is approximately uniform, and as n -> infinity only the constant-speed
  arc survives.

STRESS-VS-COMPLEXITY IN 2D:
  For pairs of points on the circle:
    Stress(0 degrees) = 0           [identical]
    Stress(30 degrees) = small      [close rotation]
    Stress(90 degrees) = medium     [quarter turn]
    Stress(135 degrees) = large     [three-quarter turn vs direct path]
    Stress(180 degrees) = maximum   [opposite point]
    Stress(off circle) = infinite   [no valid relay exists]

  If our stress metric works in 2D, it works.
  If it doesn't work in 2D, all higher-dimensional versions will also fail.
"""

import math
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from typing import List, Optional


# ── 2D unit circle representation ─────────────────────────────────────────────

def point_on_circle(theta_deg: float) -> np.ndarray:
    """[cos(theta), sin(theta)] - point on unit circle."""
    t = math.radians(theta_deg)
    return np.array([math.cos(t), math.sin(t)])


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle from a to b on the unit circle (in degrees)."""
    theta_a = math.degrees(math.atan2(a[1], a[0]))
    theta_b = math.degrees(math.atan2(b[1], b[0]))
    diff = (theta_b - theta_a) % 360
    if diff > 180:
        diff -= 360
    return diff


# ── Exact relay chain on the unit circle ─────────────────────────────────────

def exact_relay(a: np.ndarray, b: np.ndarray, n_steps: int) -> List[np.ndarray]:
    """
    Exact geodesic relay chain: uniform angular steps from a to b.
    This is the 'truth' that any approximation should converge to.
    """
    theta_a = math.atan2(a[1], a[0])
    theta_b = math.atan2(b[1], b[0])
    # Shortest arc
    diff = theta_b - theta_a
    if diff > math.pi:  diff -= 2*math.pi
    if diff < -math.pi: diff += 2*math.pi
    delta = diff / n_steps
    return [np.array([math.cos(theta_a + k*delta),
                      math.sin(theta_a + k*delta)])
            for k in range(n_steps + 1)]


# ── Simple relay runner for 2D ────────────────────────────────────────────────

def run_2d_relay(a: np.ndarray, b: np.ndarray,
                  n_steps: int = 4,
                  n_iters: int = 500,
                  constrain_to_circle: bool = True) -> List[np.ndarray]:
    """
    Find the minimum-stress relay chain from a to b with n_steps intermediates.
    Two modes:
      constrain_to_circle=True:  intermediates forced onto unit circle (correct)
      constrain_to_circle=False: intermediates free in 2D (L2 geodesic = straight line)

    The constrained version finds the arc (correct).
    The unconstrained version finds the chord (incorrect - straight line through origin).
    Comparing them demonstrates why the constraint matters.
    """
    # Initialize: linear interpolation (chord) then project to circle
    pops = [a.copy()]
    for k in range(1, n_steps):
        alpha = k / n_steps
        mid = (1-alpha)*a + alpha*b
        if constrain_to_circle:
            mid = mid / (np.linalg.norm(mid) + 1e-10)
        pops.append(mid.copy())
    pops.append(b.copy())

    # Optimize: each intermediate minimizes distance to neighbors
    for _ in range(n_iters):
        for k in range(1, n_steps):
            p, n_ = pops[k-1], pops[k+1]
            mid = 0.5 * (p + n_)   # analytic midpoint
            if constrain_to_circle:
                mid = mid / (np.linalg.norm(mid) + 1e-10)
            pops[k] = mid

    return pops


# ── Stress metrics for 2D ─────────────────────────────────────────────────────

def path_stress_2d(pops: List[np.ndarray]) -> dict:
    """
    Compute stress metrics for a 2D relay chain.

    For unit circle paths:
      Angular step variance: 0 = perfectly uniform rotation (geodesic)
      Deviation from arc:    0 = path follows the great circle
      Chord-vs-arc ratio:    1 = straight line, > 1 = detouring
    """
    angles = [math.degrees(math.atan2(p[1], p[0])) % 360 for p in pops]
    step_angles = [((angles[k+1] - angles[k] + 180) % 360) - 180
                   for k in range(len(angles)-1)]
    angular_var = float(np.var(step_angles))

    # Deviation from exact arc
    exact = exact_relay(pops[0], pops[-1], len(pops)-1)
    deviations = [float(np.linalg.norm(pops[k] - exact[k]))
                  for k in range(len(pops))]
    mean_dev = float(np.mean(deviations))

    # On-circle fraction (should be 1 for constrained relay)
    radii = [float(np.linalg.norm(p)) for p in pops]
    radius_var = float(np.var(radii))

    return {
        "angular_step_var":  angular_var,   # KEY: 0 = geodesic
        "mean_arc_deviation": mean_dev,
        "radius_variance":   radius_var,
        "step_angles":       step_angles,
        "is_geodesic":       angular_var < 0.01,
    }


# ── Phase space transition experiment ─────────────────────────────────────────

def phase_space_transition(beta_deg: float = 90.,
                            max_steps: int = 12,
                            verbose: bool = True) -> dict:
    """
    Measure how quickly the relay chain converges to the geodesic as
    more intermediates are added (the quantization experiment).

    For beta = 90 degrees (quarter turn):
      n=1: 1 intermediate -> moderate variance (not strongly constrained)
      n=4: 4 intermediates -> lower variance
      n=12: 12 intermediates -> near-zero variance (approaches geodesic)

    The rate of convergence is the 'quantization speed' - how fast does
    adding constraints collapse the path space to the geodesic?
    """
    a = point_on_circle(0)
    b = point_on_circle(beta_deg)

    n_values = list(range(1, max_steps + 1))
    variances = []

    for n in n_values:
        pops_constrained = run_2d_relay(a, b, n_steps=n, constrain_to_circle=True)
        stress = path_stress_2d(pops_constrained)
        variances.append(stress["angular_step_var"])

    if verbose:
        print(f"\n  Phase space transition (beta={beta_deg} deg):")
        print(f"  {'n_steps':>8} {'Ang Var':>10}  {'Is geodesic':>12}")
        print(f"  {'-'*34}")
        for n, v in zip(n_values, variances):
            print(f"  {n:>8} {v:>10.6f}  {'YES' if v < 0.01 else 'NO':>12}")

    return {"n_values": n_values, "variances": variances}


# ── Stress-vs-complexity in 2D ────────────────────────────────────────────────

def stress_vs_complexity_2d(n_steps: int = 4):
    """
    The minimal verifiable version of the core hypothesis.
    Five relationship classes in 2D.
    Expected: stress increases monotonically with rotation angle.
    """
    a = point_on_circle(0)   # fixed starting point

    cases = [
        {"name": "Identical (0 deg)",     "beta": 0,   "class": 0},
        {"name": "Near (30 deg)",          "beta": 30,  "class": 1},
        {"name": "Quarter turn (90 deg)", "beta": 90,  "class": 2},
        {"name": "Half turn (180 deg)",   "beta": 180, "class": 3},
        {"name": "Off circle (r=1.5)",    "beta": 90,  "class": 4, "r": 1.5},
    ]

    print(f"\n{'='*58}")
    print("2D STRESS-VS-COMPLEXITY  (minimal verifiable version)")
    print(f"{'='*58}")
    print(f"{'Class':>5} {'Ang Var':>10} {'Arc Dev':>10}  Name")
    print(f"{'-'*58}")

    results = []
    for case in cases:
        b = point_on_circle(case["beta"])
        if case.get("r"):
            b = b * case["r"]   # off-circle point

        pops = run_2d_relay(a, b, n_steps=n_steps,
                             constrain_to_circle=not case.get("r"))
        stress = path_stress_2d(pops)
        results.append({**case, **stress})
        print(f"  {case['class']:>3}  "
              f"{stress['angular_step_var']:>10.6f}  "
              f"{stress['mean_arc_deviation']:>10.6f}  "
              f"{case['name']}")

    # Check ordering
    ang_vars = [r['angular_step_var'] for r in results]
    ordered = all(ang_vars[i] <= ang_vars[i+1]*1.1
                  for i in range(len(ang_vars)-1))
    print(f"\nHypothesis (stress increases with rotation angle):")
    print(f"  Ordering holds: {ordered}")
    if ordered:
        print(f"  CONFIRMED: In 2D, angular variance IS a valid stress metric.")
        print(f"  The 8D case should behave the same way with the right representation.")
    else:
        print(f"  FAILED: check which class breaks ordering -> Ramanujan diagnosis")

    return results


if __name__ == "__main__":
    print("Running 2D minimal version...")

    # 1. Core hypothesis: stress vs complexity
    results = stress_vs_complexity_2d(n_steps=6)

    # 2. Phase space transitions: how fast does finer division select the geodesic?
    for beta in [30, 90, 180]:
        phase_space_transition(beta_deg=beta, max_steps=8)

    # 3. Constrained vs unconstrained: circle vs line
    print(f"\n{'='*55}")
    print("CONSTRAINED (arc) vs UNCONSTRAINED (chord)")
    print("For cos->sin (90 degree rotation), n=4 steps:")
    a = point_on_circle(0)
    b = point_on_circle(90)
    pops_arc   = run_2d_relay(a, b, n_steps=4, constrain_to_circle=True)
    pops_chord = run_2d_relay(a, b, n_steps=4, constrain_to_circle=False)
    s_arc   = path_stress_2d(pops_arc)
    s_chord = path_stress_2d(pops_chord)
    print(f"  Arc   (constrained): angular_var={s_arc['angular_step_var']:.6f}  "
          f"arc_dev={s_arc['mean_arc_deviation']:.6f}")
    print(f"  Chord (free):        angular_var={s_chord['angular_step_var']:.6f}  "
          f"arc_dev={s_chord['mean_arc_deviation']:.6f}")
    print(f"  Arc stress lower: {s_arc['angular_step_var'] < s_chord['angular_step_var']}")
