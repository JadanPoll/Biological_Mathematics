"""
Tier 0 runner — runs all eight Tier 0 experiments, logs to the notebook DB,
and prints a Mendel-style summary table.

Usage:
    python -m experiments.tier0.run
    python -m experiments.tier0.run --id MM-T0-001   # run one pair
    python -m experiments.tier0.run --seeds 3         # repeat each N times
"""

import argparse
import math
import json
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from experiments.tier0.definitions import TIER0_OBJECTS, X, BASIS, register_all, poly_eval
from toolkit.ga import CoevoConfig, run as coevo_run
from notebook import db

N_COEFFS = 8
L1_W     = 1e-5


def _fitness_fn(target):
    """Returns a fitness function closure over a fixed target array."""
    def _f(ind, _partner=None):
        pred = poly_eval(ind)
        return -(np.mean((pred - target)**2) + L1_W * np.abs(ind).sum())
    return _f


def run_experiment(obj: dict, cfg: CoevoConfig, seed_offset: int = 0) -> str:
    """Run one Tier 0 coevolution experiment and log to DB. Returns run id."""
    cfg.seed += seed_offset

    target_a = obj["target_a"]
    target_b = obj["target_b"]
    expected = obj["expected_residual"]

    fit_a = _fitness_fn(poly_eval(target_a))
    fit_b = _fitness_fn(poly_eval(target_b))

    result = coevo_run(
        fitness_a=fit_a,
        fitness_b=fit_b,
        joint_residual=expected,
        cfg=cfg,
        log_every=max(1, cfg.gens // 40),
    )

    setup = dict(
        genome_type="coefficient_vector",
        signal_type="walsh_population",
        n_coeffs=N_COEFFS,
        pop_size=cfg.pop_size,
        gens=cfg.gens,
        n_collaborations=cfg.n_collaborations,
        cycle_consistency_weight=cfg.cycle_consistency_w,
        scaffolding=[],
        coupling_mode=cfg.coupling_mode,
    )
    terminal = dict(
        convergence_gen=result.convergence_gen,
        residual=result.residual,
        cycle_residual=result.cycle_residual,
        spectral_class=result.spectral_class_a,
        effective_dim_a_final=result.eff_dim_a,
        effective_dim_b_final=result.eff_dim_b,
        failure_mode=result.failure_mode,
        outcome=result.outcome,
        walsh_me_a_final=result.walsh_me_a.tolist(),
        walsh_me_b_final=result.walsh_me_b.tolist(),
        kl_final=result.kl_final,
    )

    run_id = db.log_run(obj["id"], setup, terminal,
                        notes=f"seed_offset={seed_offset}")

    # Log timeseries
    for key, series in result.timeseries.items():
        pass  # timeseries stored per-gen; write fit_a/fit_b
    for (gen, fa), (_, fb), (_, kl), (_, ed_a), (_, ed_b), (_, cy) in zip(
        result.timeseries["fit_a"], result.timeseries["fit_b"],
        result.timeseries["kl"],    result.timeseries["eff_dim_a"],
        result.timeseries["eff_dim_b"], result.timeseries["cycle"],
    ):
        db.log_timeseries(run_id, gen, fa, fb, kl, ed_a, ed_b, cy)

    return run_id, result


def _print_summary(results: list):
    """Print a Mendel-style table of Tier 0 results."""
    print("\n" + "="*80)
    print("TIER 0 MENDEL TABLE  —  spectral floor calibration")
    print("="*80)
    header = f"{'ID':<12} {'Name':<38} {'Conv':>6} {'Residual':>10} {'SpClass':<14} {'Outcome'}"
    print(header)
    print("-"*80)
    for obj_id, run_id, result in results:
        obj  = next(o for o in TIER0_OBJECTS if o["id"] == obj_id)
        conv = str(result.convergence_gen) if result.convergence_gen else "—"
        print(f"{obj_id:<12} {obj['name'][:38]:<38} {conv:>6} "
              f"{result.residual:>10.5f} {result.spectral_class_a:<14} {result.outcome}")
    print("="*80)

    # Aggregate: mean convergence gen and residual for Tier 0
    conv_gens = [r.convergence_gen for _, _, r in results if r.convergence_gen]
    residuals = [r.residual for _, _, r in results]
    print(f"\nTier 0 floor:  mean convergence gen = {np.mean(conv_gens):.0f}  "
          f"(std {np.std(conv_gens):.0f})")
    print(f"               mean residual        = {np.mean(residuals):.5f}  "
          f"(std {np.std(residuals):.5f})")
    print("\nThis is the baseline.  Any harder experiment is interpreted relative to it.")


def plot_tier0(results: list, out="tier0_summary.png"):
    """Walsh main-effect bar charts for all Tier 0 experiments."""
    n  = len(results)
    nc = math.ceil(n / 2)
    fig, axes = plt.subplots(2, nc, figsize=(4 * nc, 6))
    axes = axes.flatten()
    ks  = np.arange(N_COEFFS)

    for i, (obj_id, _, result) in enumerate(results):
        ax  = axes[i]
        obj = next(o for o in TIER0_OBJECTS if o["id"] == obj_id)
        w   = 0.35
        ax.bar(ks - w/2, result.walsh_me_a, w, label="Pop A", color="steelblue",  alpha=0.85)
        ax.bar(ks + w/2, result.walsh_me_b, w, label="Pop B", color="darkorange", alpha=0.85)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{obj_id}\n{obj['name'][:30]}", fontsize=8)
        ax.set_xticks(ks); ax.set_xticklabels([f"k={k}" for k in ks], fontsize=6)
        ax.legend(fontsize=6)
        res_str = f"res={result.residual:.4f}"
        ax.set_xlabel(f"{result.spectral_class_a}  {res_str}", fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Tier 0 Walsh Main-Effects at Convergence\n"
                 "(the spectral floor — all Tier 1+ interpreted relative to this)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"Plot saved -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",    default=None, help="Run only this MM-id")
    parser.add_argument("--seeds", type=int, default=1, help="Repeat N times per pair")
    parser.add_argument("--gens",  type=int, default=600)
    parser.add_argument("--pop",   type=int, default=80)
    parser.add_argument("--ncollab", type=int, default=5)
    args = parser.parse_args()

    register_all()

    objects = ([o for o in TIER0_OBJECTS if o["id"] == args.id]
               if args.id else TIER0_OBJECTS)

    cfg = CoevoConfig(
        n_genes=N_COEFFS,
        pop_size=args.pop,
        gens=args.gens,
        n_collaborations=args.ncollab,
        cycle_consistency_w=0.05,
        k_signal_samples=8,
        coupling_mode="adaptive_kl",
    )

    all_results = []
    for obj in objects:
        print(f"\nRunning {obj['id']}: {obj['name']} ...")
        for s in range(args.seeds):
            run_id, result = run_experiment(obj, cfg, seed_offset=s * 100)
            all_results.append((obj["id"], run_id, result))
            print(f"  seed {s}: residual={result.residual:.5f}  "
                  f"conv_gen={result.convergence_gen}  outcome={result.outcome}")

    _print_summary(all_results)
    plot_tier0(all_results, out="tier0_summary.png")


if __name__ == "__main__":
    main()
