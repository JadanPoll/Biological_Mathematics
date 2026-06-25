"""
CLI runner — one command to run any experiment.

Usage examples:
    # Run one experiment with all three signal designs
    python -m experiments.runner run MM-T0-001

    # Run with a specific signal design
    python -m experiments.runner run MM-T0-001 --signal walsh_solution

    # Run all Tier 0 experiments
    python -m experiments.runner run-all --tier 0

    # Compare signal designs on one experiment (runs all three, overlaid plot)
    python -m experiments.runner compare MM-T0-001

    # Print Mendel table from DB
    python -m experiments.runner report --tier 0

    # List registered experiments
    python -m experiments.runner list --tier 0
"""

import argparse
import sys
from experiments.registry import REGISTRY, get_experiment, list_experiments
from experiments.base import run_experiment, plot_experiment
from toolkit.ga import CoevoConfig
from notebook import db, report


def cmd_list(args):
    exps = list_experiments(tier=args.tier)
    print(f"\n{'ID':<14} {'Tier':>4} {'Domain':<16} {'Name'}")
    print("-" * 70)
    for e in exps:
        print(f"{e.object_id:<14} {e.tier:>4} {e.domain:<16} {e.name}")
    print(f"\n{len(exps)} experiments registered.")


def cmd_run(args):
    exp = get_experiment(args.id)
    signals = ([args.signal] if args.signal
               else ["walsh_landscape", "walsh_solution", "raw_sparse"])

    cfg_overrides = {}
    if args.gens:    cfg_overrides["gens"]    = args.gens
    if args.pop:     cfg_overrides["pop_size"] = args.pop
    if args.ncollab: cfg_overrides["n_collaborations"] = args.ncollab

    if cfg_overrides:
        exp.cfg = CoevoConfig(**{**exp.cfg.__dict__, **cfg_overrides})

    print(f"\n{'='*60}")
    print(f"Running: {exp.object_id} — {exp.name}")
    print(f"Signal designs: {signals}")
    print(f"Gens: {exp.cfg.gens}  Pop: {exp.cfg.pop_size}  "
          f"N-collab: {exp.cfg.n_collaborations}")
    print(f"{'='*60}")

    results = {}
    for sig in signals:
        for seed_offset in range(args.seeds):
            run_id, result = run_experiment(exp, signal_design=sig,
                                            seed_offset=seed_offset * 100)
            if sig not in results or result.residual < results[sig].residual:
                results[sig] = result   # keep best seed per signal

    print(f"\nSummary for {exp.object_id}:")
    print(f"  {'Signal':<20} {'Residual':>10} {'Conv':>6} {'Outcome'}")
    print(f"  {'-'*50}")
    for sig, res in results.items():
        conv = str(res.convergence_gen) if res.convergence_gen else "—"
        print(f"  {sig:<20} {res.residual:>10.5f} {conv:>6}  {res.outcome}")

    if len(signals) > 0:
        path = plot_experiment(exp, results)
        print(f"\nPlot saved -> {path}")


def cmd_compare(args):
    """Compare all three signal designs on one experiment."""
    args.signal = None
    args.seeds  = getattr(args, "seeds", 1)
    cmd_run(args)


def cmd_run_all(args):
    exps = list_experiments(tier=args.tier)
    print(f"\nRunning {len(exps)} experiments (tier={args.tier}) ...")
    for exp in exps:
        signals = (["walsh_landscape", "walsh_solution", "raw_sparse"]
                   if args.all_signals else [exp.cfg.signal_design])
        print(f"\n  {exp.object_id}: {exp.name}")
        for sig in signals:
            run_experiment(exp, signal_design=sig, verbose=True)
    print("\nAll done. Run 'python -m experiments.runner report' to see results.")


def cmd_report(args):
    db.init()
    report.mendel_table(tier=args.tier)


def main():
    parser = argparse.ArgumentParser(
        description="Mathematical Mendel Notebook — experiment runner")
    sub = parser.add_subparsers(dest="cmd")

    # list
    p = sub.add_parser("list", help="List registered experiments")
    p.add_argument("--tier", type=int, default=None)

    # run
    p = sub.add_parser("run", help="Run one experiment")
    p.add_argument("id")
    p.add_argument("--signal", default=None,
                   choices=["walsh_landscape", "walsh_solution", "raw_sparse"])
    p.add_argument("--seeds",   type=int, default=1)
    p.add_argument("--gens",    type=int, default=None)
    p.add_argument("--pop",     type=int, default=None)
    p.add_argument("--ncollab", type=int, default=None)

    # compare
    p = sub.add_parser("compare", help="Compare all signal designs on one experiment")
    p.add_argument("id")
    p.add_argument("--seeds",   type=int, default=1)
    p.add_argument("--gens",    type=int, default=None)
    p.add_argument("--pop",     type=int, default=None)
    p.add_argument("--ncollab", type=int, default=None)

    # run-all
    p = sub.add_parser("run-all", help="Run all experiments in a tier")
    p.add_argument("--tier", type=int, default=0)
    p.add_argument("--all-signals", action="store_true",
                   help="Run each experiment with all three signal designs")

    # report
    p = sub.add_parser("report", help="Print Mendel table from DB")
    p.add_argument("--tier", type=int, default=None)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help(); sys.exit(0)

    dispatch = {"list": cmd_list, "run": cmd_run,
                "compare": cmd_compare, "run-all": cmd_run_all,
                "report": cmd_report}
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
