"""
Rung 0 -- Coevolutionary GA with Walsh spectral signal molecules
================================================================
Two coefficient-vector populations co-evolve:
  Population A  -> fits P_cos(x) = 1 - x^2/2   (2-term Taylor cosine)
  Population B  -> fits P_sin(x) = x - x^3/6   (2-term Taylor sine)

Basis: phi_k(x) = x^k / k!

True coefficients (unique optima):
  TRUE_COS = [1,  0, -1,  0]
  TRUE_SIN = [0,  1,  0, -1]

Derivative relationship:
  d/dx[sum_k b_k * phi_k(x)] = sum_j b_{j+1} * phi_j(x)
  => d/dx(TRUE_SIN) = [1, 0, -1, 0] = TRUE_COS  (unit left-shift of B)

Walsh signal molecules:
  Each population computes the WHT of its local fitness landscape
  (fitness values at all 2^4 = 16 corners of a +/-DELTA hypercube),
  and sends it to the other.  The main-effect at index 2^k encodes
  how sensitive the landscape is to flipping gene k alone.

Expected spectral signature:
  A (cos) should show large W[2^0] and W[2^2]  (even-index genes matter)
  B (sin) should show large W[2^1] and W[2^3]  (odd-index genes matter)
  The alternating even/odd pattern IS the spectral fingerprint of
  the complementary derivative relationship.
"""

import math
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# -- Hyperparameters ----------------------------------------------------------
N_COEFFS = 4      # Taylor terms: phi_0 ... phi_3
POP      = 60
GENS     = 1000
MUT_STD  = 0.06
L1_W     = 1e-5   # tiny MDL nudge
SIG_W    = 0.40   # signal-biasing strength
DELTA    = 0.30   # hypercube half-edge for WHT sampling
TOURNEY  = 4
SIG_FREQ = 8
ELITE    = 2

rng = np.random.default_rng(42)

# -- Basis and targets --------------------------------------------------------
X     = np.linspace(-1.5, 1.5, 120)
BASIS = np.stack([X**k / math.factorial(k) for k in range(N_COEFFS)], axis=1)

TRUE_COS = np.array([ 1.,  0., -1.,  0.])
TRUE_SIN = np.array([ 0.,  1.,  0., -1.])

TARGET_COS = BASIS @ TRUE_COS   # exact polynomial 1 - x^2/2
TARGET_SIN = BASIS @ TRUE_SIN   # exact polynomial x - x^3/6

# -- Core math ----------------------------------------------------------------
def poly_eval(c):
    return BASIS @ c

def wht(v):
    """Normalised Walsh-Hadamard transform.  len(v) must be power of 2."""
    u, h, n = np.array(v, dtype=float), 1, len(v)
    while h < n:
        for i in range(0, n, 2 * h):
            a = u[i:i+h].copy()
            b = u[i+h:i+2*h].copy()
            u[i:i+h]     = a + b
            u[i+h:i+2*h] = a - b
        h *= 2
    return u / n

def fitness(c, target):
    return -(np.mean((poly_eval(c) - target)**2) + L1_W * np.abs(c).sum())

def compute_signal(best, target):
    """
    WHT of 2^N fitness values at the corners of a +/-DELTA hypercube.
    For N=4: 16 evaluations.  Main-effects at indices 1,2,4,8.
    """
    n_corners = 2 ** N_COEFFS
    vals = np.empty(n_corners)
    for i in range(n_corners):
        perturb = np.array([(DELTA if (i >> k) & 1 else -DELTA)
                            for k in range(N_COEFFS)])
        vals[i] = fitness(best + perturb, target)
    return wht(vals)

# -- GA -----------------------------------------------------------------------
def tournament(pop, fits):
    idx = rng.choice(len(pop), TOURNEY, replace=False)
    return pop[idx[np.argmax(fits[idx])]].copy()

def step(pop, target, signal_from_other):
    fits = np.array([fitness(ind, target) for ind in pop])

    main_fx = np.abs(signal_from_other[1 << np.arange(N_COEFFS)])
    main_fx /= main_fx.max() + 1e-10
    mut_std = MUT_STD * (1.0 + SIG_W * main_fx)

    elite_idx = np.argsort(fits)[-ELITE:]
    new_pop   = np.empty_like(pop)
    new_pop[:ELITE] = pop[elite_idx].copy()

    for j in range(ELITE, POP):
        p1    = tournament(pop, fits)
        p2    = tournament(pop, fits)
        mask  = rng.random(N_COEFFS) < 0.5
        child = np.where(mask, p1, p2)
        child += rng.standard_normal(N_COEFFS) * mut_std
        new_pop[j] = child

    return new_pop, fits

# -- Run ----------------------------------------------------------------------
pop_a = rng.standard_normal((POP, N_COEFFS)) * 0.5
pop_b = rng.standard_normal((POP, N_COEFFS)) * 0.5

sig_from_a = np.zeros(2 ** N_COEFFS)
sig_from_b = np.zeros(2 ** N_COEFFS)

log_fit_a, log_fit_b = [], []
log_me_a,  log_me_b  = [], []

early_me_a = {}   # generation -> main-effects snapshot

print(f"Running {GENS} generations  (signal every {SIG_FREQ} gens, N={N_COEFFS})...")
for g in range(GENS):
    pop_a, fits_a = step(pop_a, TARGET_COS, sig_from_b)
    pop_b, fits_b = step(pop_b, TARGET_SIN, sig_from_a)

    log_fit_a.append(fits_a.max())
    log_fit_b.append(fits_b.max())

    if g % SIG_FREQ == 0:
        best_a     = pop_a[np.argmax(fits_a)]
        best_b     = pop_b[np.argmax(fits_b)]
        sig_from_a = compute_signal(best_a, TARGET_COS)
        sig_from_b = compute_signal(best_b, TARGET_SIN)
        if g % 200 == 0:
            print(f"  gen {g:4d}  fit_A={fits_a.max():.7f}  fit_B={fits_b.max():.7f}")

    log_me_a.append(sig_from_a[1 << np.arange(N_COEFFS)].copy())
    log_me_b.append(sig_from_b[1 << np.arange(N_COEFFS)].copy())

# -- Baseline: no signal exchange (independent evolution) ---------------------
print("\nRunning baseline (no signal exchange)...")
pop_a_base = rng.standard_normal((POP, N_COEFFS)) * 0.5
pop_b_base = rng.standard_normal((POP, N_COEFFS)) * 0.5
null_sig   = np.zeros(2 ** N_COEFFS)
log_fit_a_base, log_fit_b_base = [], []
for g in range(GENS):
    pop_a_base, fa = step(pop_a_base, TARGET_COS, null_sig)
    pop_b_base, fb = step(pop_b_base, TARGET_SIN, null_sig)
    log_fit_a_base.append(fa.max())
    log_fit_b_base.append(fb.max())

# -- Results ------------------------------------------------------------------
fits_a_f = np.array([fitness(ind, TARGET_COS) for ind in pop_a])
fits_b_f = np.array([fitness(ind, TARGET_SIN) for ind in pop_b])
best_a   = pop_a[np.argmax(fits_a_f)]
best_b   = pop_b[np.argmax(fits_b_f)]

deriv_b  = np.append(best_b[1:], 0.0)   # shift left: d/dx(B)[j] = B[j+1]
residual = np.linalg.norm(best_a - deriv_b)
err_a    = np.linalg.norm(best_a - TRUE_COS)
err_b    = np.linalg.norm(best_b - TRUE_SIN)

print("\n" + "="*60)
print("DERIVATIVE CHECK:  ||A_best - d/dx(B_best)||_2")
print(f"  Residual         : {residual:.7f}")
print(f"  ||A - true_cos|| : {err_a:.7f}")
print(f"  ||B - true_sin|| : {err_b:.7f}")
print(f"\n  Found  A         : {np.round(best_a,  5)}")
print(f"  d/dx(B_best)     : {np.round(deriv_b, 5)}")
print(f"  True cos         : {np.round(TRUE_COS, 5)}")
print(f"\n  Found  B         : {np.round(best_b,  5)}")
print(f"  True sin         : {np.round(TRUE_SIN, 5)}")
print("="*60)

# -- Walsh main-effect spectral fingerprint -----------------------------------
final_me_a = np.abs(sig_from_a[1 << np.arange(N_COEFFS)])
final_me_b = np.abs(sig_from_b[1 << np.arange(N_COEFFS)])
print(f"\nWalsh main-effects at convergence:")
print(f"  A (cos pop) W[2^k]: {np.round(final_me_a, 5)}  -- expect large at k=0,2")
print(f"  B (sin pop) W[2^k]: {np.round(final_me_b, 5)}  -- expect large at k=1,3")

# -- Plots --------------------------------------------------------------------
me_a_arr = np.array(log_me_a)
me_b_arr = np.array(log_me_b)
ks       = np.arange(N_COEFFS)

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle(
    "Rung 0 -- cos <-> sin  (N=4, Walsh signal molecules)",
    fontsize=13, fontweight="bold"
)

colors = ["steelblue", "darkorange", "seagreen", "crimson"]

# 1. Fitness convergence -- coevo vs baseline
ax = axes[0, 0]
ax.semilogy([-f for f in log_fit_a],      lw=1.8, label="A cos (coevo)",     color="steelblue")
ax.semilogy([-f for f in log_fit_b],      lw=1.8, label="B sin (coevo)",     color="darkorange")
ax.semilogy([-f for f in log_fit_a_base], lw=1.2, ls="--", label="A cos (baseline)", color="steelblue",  alpha=0.5)
ax.semilogy([-f for f in log_fit_b_base], lw=1.2, ls="--", label="B sin (baseline)", color="darkorange", alpha=0.5)
ax.set_xlabel("Generation")
ax.set_ylabel("Loss  (log scale)")
ax.set_title("Fitness: coevo vs baseline (no signal)")
ax.legend(fontsize=8)

# 2. Walsh main-effect trajectories -- show each gene as a separate line
ax = axes[0, 1]
gene_labels = [f"k={k} ({'even' if k%2==0 else 'odd'})" for k in range(N_COEFFS)]
for k in range(N_COEFFS):
    ax.plot(me_a_arr[:, k], color=colors[k], ls="-",  lw=1.5, alpha=0.85,
            label=gene_labels[k])
    ax.plot(me_b_arr[:, k], color=colors[k], ls="--", lw=1.5, alpha=0.85)
ax.set_xlabel("Generation")
ax.set_ylabel("|Walsh main-effect|  W[2^k]")
ax.set_title("Signal molecules  (solid=A/cos, dashed=B/sin)")
ax.legend(fontsize=8)

# 3. Learned vs true coefficients (bar chart)
ax = axes[1, 0]
w = 0.20
ax.bar(ks - 1.5*w, TRUE_COS, w, label="True cos",  color="steelblue",  alpha=0.38)
ax.bar(ks - 0.5*w, best_a,   w, label="Found A",   color="steelblue",  alpha=0.90)
ax.bar(ks + 0.5*w, TRUE_SIN, w, label="True sin",  color="darkorange", alpha=0.38)
ax.bar(ks + 1.5*w, best_b,   w, label="Found B",   color="darkorange", alpha=0.90)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Coefficient index k")
ax.set_ylabel("Value")
ax.set_title("Learned vs. true coefficients  (phi_k = x^k/k! basis)")
ax.set_xticks(ks); ax.set_xticklabels([f"k={k}" for k in ks])
ax.legend(fontsize=8)

# 4. Derivative check -- A_best vs d/dx(B_best)
ax = axes[1, 1]
ax.bar(ks - w/2, best_a,  w, label="A_best  (cos fit)", color="steelblue", alpha=0.9)
ax.bar(ks + w/2, deriv_b, w, label="d/dx(B_best)",      color="seagreen",  alpha=0.9)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Coefficient index k")
ax.set_ylabel("Value")
ax.set_title(f"Joint attractor: derivative check\n"
             f"||A - d/dx(B)||_2 = {residual:.5f}")
ax.set_xticks(ks); ax.set_xticklabels([f"k={k}" for k in ks])
ax.legend(fontsize=9)

plt.tight_layout()
out = "rung0_result.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPlot saved -> {out}")
