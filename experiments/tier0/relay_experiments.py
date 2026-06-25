"""
Relay chain experiments — 360° rotation through phase space.

The fundamental test: can A↔B↔C↔D↔A with derivative steps at each node
find the {cos, sin, -cos, -sin} rotation without being told the targets?

This is the anchor-free version of the joint fitness experiments.
No fixed targets. No degenerate equality solution.
The cycle closure constraint IS the anchor.

Expected result at convergence:
  Pop A ≈ cos(x)   Walsh ME: even indices dominate
  Pop B ≈ sin(x)   Walsh ME: odd indices dominate
  Pop C ≈ -cos(x)  Walsh ME: even indices dominate (opposite sign)
  Pop D ≈ -sin(x)  Walsh ME: odd indices dominate (opposite sign)

The alternating_score should be high (even/odd alternation around the cycle).
The step_residuals should each be small.
The cycle_residual should be small (D→A closes the loop).

The spectral signature of the degenerate solution (all zeros) vs the rotation:
  Degenerate: all Walsh ME near zero, alternating_score → 0
  Rotation:   alternating ME, alternating_score → 1

This is the betweenness-constrained geometric path.
"""

import math
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from toolkit.relay_chain import RelayChainConfig, run as relay_run
from toolkit.ga import CoevoConfig


N   = 8
_cfg = CoevoConfig(
    n_genes=N, pop_size=80, gens=800,
    n_collaborations=1,   # relay chain has direct neighbour coupling; N-collab less important
    k_signal_samples=6,
    mut_std=0.05, sig_weight=0.40, delta=0.25,
    coupling_mode="adaptive_kl",
    signal_design="walsh_solution",   # solution-space signal best for structure detection
)


def run_rotation_chain(signal_design="walsh_solution", gens=800, seed=42):
    """
    Run the 360° rotation relay chain: cos → sin → -cos → -sin → cos.
    Returns the result and prints diagnostics.
    """
    cfg = CoevoConfig(**{**_cfg.__dict__,
                         "signal_design": signal_design,
                         "gens": gens,
                         "seed": seed})
    chain_cfg = RelayChainConfig(
        n_pops=4,
        step_fitness="derivative",
        cfg=cfg,
        cycle_weight=0.25,
    )

    print(f"\n{'='*60}")
    print(f"Relay chain: 360 degree rotation  (signal={signal_design})")
    print(f"4 populations: A->B->C->D->A, each step = d/dx")
    print(f"{'='*60}")

    result = relay_run(chain_cfg, log_every=80)

    # True targets for comparison
    true_targets = [
        np.array([(-1.)**( k//2) if k%2==0 else 0. for k in range(N)]),   # cos
        np.array([(-1.)**((k-1)//2) if k%2==1 else 0. for k in range(N)]), # sin
        np.array([(-1.)**(k//2+1) if k%2==0 else 0. for k in range(N)]),   # -cos
        np.array([(-1.)**((k+1)//2) if k%2==1 else 0. for k in range(N)]), # -sin
    ]
    labels = ["A (cos?)", "B (sin?)", "C (-cos?)", "D (-sin?)"]

    print(f"\nStep residuals (each should be < 0.10):")
    for k, (res, lbl) in enumerate(zip(result.step_residuals, labels)):
        best_k = result.populations[k]
        true_k = true_targets[k]
        dist_to_true = np.linalg.norm(best_k - true_k)
        print(f"  {lbl:<14} step_res={res:.4f}  dist_to_target={dist_to_true:.4f}"
              f"  spectral={result.spectral_classes[k]}")

    print(f"\nCycle closure residual: {result.cycle_residual:.4f}")
    print(f"Alternating score:      {result.alternating_score:.4f}  "
          f"(1=perfect even/odd alternation, 0=flat/degenerate)")
    print(f"Outcome:                {result.outcome}")

    _plot_relay(result, true_targets, labels,
                out=f"relay_chain_{signal_design}.png")
    return result


def _plot_relay(result, true_targets, labels, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"360 rotation relay chain — alternating score={result.alternating_score:.3f}",
                 fontsize=12, fontweight="bold")
    ks = np.arange(N)
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    w = 0.35

    for i, ax in enumerate(axes.flatten()):
        ax.bar(ks - w/2, result.populations[i],   w,
               label="Found",  color=colors[i], alpha=0.90)
        ax.bar(ks + w/2, true_targets[i], w,
               label="True",   color=colors[i], alpha=0.35)
        ax.axhline(0, color="k", lw=0.5)
        # Overlay Walsh main-effects
        ax2 = ax.twinx()
        ax2.plot(ks, result.walsh_signatures[i], "k--", lw=1, alpha=0.6,
                 label="Walsh ME")
        ax2.set_ylabel("Walsh ME", fontsize=7)
        ax.set_title(f"Pop {labels[i]}  step_res={result.step_residuals[i]:.3f}")
        ax.set_xticks(ks); ax.set_xticklabels([f"k={k}" for k in ks], fontsize=7)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"Plot saved -> {out}")


if __name__ == "__main__":
    import sys
    signal = sys.argv[1] if len(sys.argv) > 1 else "walsh_solution"
    run_rotation_chain(signal_design=signal, gens=800)
