"""Quick verification that stress_metrics.py is correct."""

import math
import numpy as np
from toolkit.stress_metrics import (jsd, jsd_metric, sliced_wasserstein,
                                     hausdorff_dimension_path, relay_stress_report)
from toolkit.euler_relay import euler_intermediate, _COS, _SIN, N

print("=== Verifying stress metrics ===\n")

# 1. JSD: should be 0 for identical distributions, log(2) for orthogonal
p = np.array([1., 0., 0., 0.])
q = np.array([0., 0., 0., 1.])
r = np.array([1., 0., 0., 0.])
print(f"JSD(p, p) = {jsd(p, r):.6f}  (expected 0)")
print(f"JSD(p, q) = {jsd(p, q):.6f}  (expected {math.log(2):.6f} = log(2))")
print(f"sqrt_JSD(p,q) = {jsd_metric(p, q):.6f}  (metric, bounded)")

# 2. Sliced Wasserstein: two identical populations should give 0
rng = np.random.default_rng(42)
pop = rng.standard_normal((50, N)) * 0.3
print(f"\nSliced Wasserstein (same population) = {sliced_wasserstein(pop, pop, 20, rng):.6f}  (expected ~0)")

# Populations at cos vs sin should be nonzero
pop_cos = rng.standard_normal((50, N)) * 0.1 + _COS
pop_sin = rng.standard_normal((50, N)) * 0.1 + _SIN
w_cs = sliced_wasserstein(pop_cos, pop_sin, 20, rng)
print(f"Sliced Wasserstein (cos vs sin pop) = {w_cs:.4f}  (expected > 0)")

# 3. Hausdorff dimension: straight line should be ~1, random walk should be higher
straight_line = [np.array([k*0.1, k*0.1, k*0.1, k*0.1, 0, 0, 0, 0])
                  for k in range(20)]
random_walk   = [rng.standard_normal(N) * 2.0 for _ in range(20)]

hd_line = hausdorff_dimension_path(straight_line)
hd_rw   = hausdorff_dimension_path(random_walk)
print(f"\nHausdorff dim (straight line): {hd_line:.4f}  (expected ~1.0)")
print(f"Hausdorff dim (random walk):   {hd_rw:.4f}    (expected > 1.0)")

# 4. True Euler intermediates should form a smooth path (Hausdorff ~1)
n_steps = 8
true_path = [euler_intermediate(k, n_steps) for k in range(n_steps+1)]
hd_euler = hausdorff_dimension_path(true_path)
print(f"Hausdorff dim (true Euler path): {hd_euler:.4f}  (expected ~1.0 — smooth rotation)")

# 5. Relay stress report on true intermediates (should show geodesic)
print("\n=== Stress report on TRUE Euler intermediates (n=4) ===")
n = 4
true_ints = [euler_intermediate(k, n) for k in range(n+1)]
# Fake signals as WHT of true intermediate coefficient vectors
from toolkit.walsh import wht
sigs = [wht(np.pad(ti, (0, 2**N - N))) for ti in true_ints]
report = relay_stress_report(true_ints, sigs, true_ints, verbose=True)
print(f"\n  is_geodesic: {report['is_geodesic']}  (should be True for true path)")
print(f"  is_fractal:  {report['is_fractal']}   (should be False)")
