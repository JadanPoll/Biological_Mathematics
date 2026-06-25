"""
Compare unconstrained relay chain vs equal-KL constrained relay chain.
The equal-KL constraint selects paths with uniform curvature (geodesics).
"""
import numpy as np
from toolkit.equal_kl_relay import EqualKLConfig, run_equal_kl
from toolkit.ga import CoevoConfig

cfg = CoevoConfig(
    n_genes=8, pop_size=80, gens=700,
    n_collaborations=1, k_signal_samples=6,
    mut_std=0.05, sig_weight=0.40, delta=0.25,
    coupling_mode="adaptive_kl",
    signal_design="trajectory_ngram",
    sig_freq=10, elite=2, tourney=5,
)

print("="*60)
print("EQUAL-KL RELAY CHAIN (n=4 populations, 360 rotation)")
print("Constraints: derivative steps + normalisation + equal-KL")
print("="*60)

for n_pops, norm_w, kl_w in [(4, 0.3, 0.4), (6, 0.3, 0.4), (8, 0.3, 0.4)]:
    print(f"\n--- n_pops={n_pops} ---")
    eq_cfg = EqualKLConfig(
        n_pops=n_pops,
        cfg=CoevoConfig(**{**cfg.__dict__, "seed": 42}),
        step_weight=1.0,
        norm_weight=norm_w,
        equal_kl_weight=kl_w,
        cycle_weight=0.20,
    )
    result = run_equal_kl(eq_cfg, log_every=100)

    # True targets
    true_cos = np.array([(-1.)**(k//2) if k%2==0 else 0. for k in range(8)])
    true_sin = np.array([(-1.)**((k-1)//2) if k%2==1 else 0. for k in range(8)])

    print(f"  Pop 0 (A): {np.round(result.populations[0], 3)}")
    print(f"  True cos:  {np.round(true_cos, 3)}")
    dist_a = min(np.linalg.norm(result.populations[0] - true_cos),
                 np.linalg.norm(result.populations[0] + true_cos))
    print(f"  Dist to +/-cos: {dist_a:.4f}")
