"""
Experiment as data, not code.

Adding a new experiment = adding one dict to a tier's experiments list.
The infrastructure handles running, logging, diagnostics, and plotting.

Anatomy of an Experiment:
  - object_id        : references a math_objects record in the DB
  - name / domain    : human-readable metadata
  - target_a/b       : numpy arrays (the polynomial targets in phi_k basis)
  - relationship     : Callable(best_a, best_b) -> float  (lower = better)
                       This is what makes the experiment's success criterion explicit.
                       For identical functions: ||A - B||
                       For derivative:         ||A - shift_left(B)||
                       For scalar multiple:    ||A - 2*B||
  - signal_designs   : list of signal molecule names to sweep over
  - cfg              : CoevoConfig (can be shared across experiments)
  - tier / notes     : metadata
"""

import math
import json
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from toolkit.ga import CoevoConfig, run as _coevo_run, CoevoResult
from toolkit.walsh import spectral_class
from toolkit.joint_fitness import make_pair_fitness, best_transformation
from notebook import db

# ── Shared polynomial basis ───────────────────────────────────────────────────
X     = np.linspace(-1.5, 1.5, 120)
N     = 8
BASIS = np.stack([X**k / math.factorial(k) for k in range(N)], axis=1)
L1_W  = 1e-5


def poly_eval(c: np.ndarray) -> np.ndarray:
    return BASIS[:, :len(c)] @ c


# ── Standard relationship functions ──────────────────────────────────────────
def rel_identical(a, b):         return np.linalg.norm(a - b)
def rel_scalar(k):               return lambda a, b: np.linalg.norm(a - k * b)
def rel_shift(s):                return lambda a, b: np.linalg.norm(a - np.roll(b, -s)[:len(a)])
def rel_derivative():            return lambda a, b: np.linalg.norm(a - np.append(b[1:], 0.))
def rel_additive_const(c0_diff):
    def _r(a, b):
        expected = b.copy(); expected[0] += c0_diff
        return np.linalg.norm(a - expected)
    return _r


# ── Experiment dataclass ──────────────────────────────────────────────────────
@dataclass
class Experiment:
    object_id:          str
    name:               str
    domain:             str
    expression_a:       str
    expression_b:       str
    target_a:           np.ndarray
    target_b:           np.ndarray
    relationship:       Callable        # (best_a, best_b) -> float; 0 = correct
    relationship_desc:  str
    tier:               int
    ground_truth:       str
    properties:         dict = field(default_factory=dict)
    signal_designs:     List[str] = field(default_factory=lambda: [
                            "walsh_landscape", "walsh_solution", "raw_sparse"])
    cfg:                CoevoConfig = field(default_factory=CoevoConfig)
    notes:              str = ""
    # Joint fitness mode: if set, overrides independent target-based fitness.
    # The string is passed to make_pair_fitness() — e.g. "identical", "derivative".
    # None = use independent target-based fitness (old mode, kept for comparison).
    joint_fitness_type: Optional[str] = None
    joint_fitness_kwargs: dict = field(default_factory=dict)


# ── Run one experiment with one signal design ─────────────────────────────────
def run_experiment(exp: Experiment,
                   signal_design: Optional[str] = None,
                   seed_offset: int = 0,
                   verbose: bool = True) -> tuple[str, CoevoResult]:
    """
    Run `exp` with the given signal design (defaults to exp.cfg.signal_design).
    Logs everything to the notebook DB. Returns (run_id, result).
    """
    db.init()
    db.upsert_object(
        id=exp.object_id, name=exp.name, domain=exp.domain,
        expression_a=exp.expression_a, expression_b=exp.expression_b,
        relationship=exp.relationship_desc,
        tier=exp.tier, ground_truth=exp.ground_truth,
        properties=exp.properties, notes=exp.notes,
    )

    cfg = CoevoConfig(**{**exp.cfg.__dict__,
                         "signal_design": signal_design or exp.cfg.signal_design,
                         "seed": exp.cfg.seed + seed_offset})

    target_data_a = poly_eval(exp.target_a)
    target_data_b = poly_eval(exp.target_b)

    if exp.joint_fitness_type is not None:
        # Joint mode: fitness defined over PAIRS. No fixed targets.
        # This is what the .md proposed: both populations jointly minimise
        # the description length of their relationship.
        pair_fit = make_pair_fitness(exp.joint_fitness_type,
                                     **exp.joint_fitness_kwargs)
        def fit_a(ind, partner):
            return pair_fit(ind, partner)
        def fit_b(ind, partner):
            return pair_fit(partner, ind)
    else:
        # Independent mode (baseline): each population chases its own fixed target.
        # Kept for comparison — shows that independent fitness misses the path.
        def fit_a(ind, partner=None):
            return -(np.mean((poly_eval(ind) - target_data_a)**2)
                     + L1_W * np.abs(ind).sum())
        def fit_b(ind, partner=None):
            return -(np.mean((poly_eval(ind) - target_data_b)**2)
                     + L1_W * np.abs(ind).sum())

    result = _coevo_run(
        fitness_a=fit_a,
        fitness_b=fit_b,
        joint_residual=exp.relationship,
        cfg=cfg,
        log_every=max(1, cfg.gens // 40),
    )

    setup = dict(
        genome_type="coefficient_vector",
        signal_type=cfg.signal_design,
        n_coeffs=cfg.n_genes,
        pop_size=cfg.pop_size,
        gens=cfg.gens,
        n_collaborations=cfg.n_collaborations,
        cycle_consistency_weight=0.0,   # removed from optimization
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
        timing=result.timing,
    )
    run_id = db.log_run(exp.object_id, setup, terminal,
                        notes=f"signal={cfg.signal_design} seed_offset={seed_offset}")

    # Timeseries — full Walsh spectra included for betweenness model training
    ts = result.timeseries
    for i, (gen, fa) in enumerate(ts["fit_a"]):
        fb   = ts["fit_b"][i][1]
        kl   = ts["kl"][i][1]
        eda  = int(ts["eff_dim_a"][i][1])
        edb  = int(ts["eff_dim_b"][i][1])
        res  = ts["residual"][i][1]
        wa   = ts["walsh_a"][i][1] if i < len(ts["walsh_a"]) else None
        wb   = ts["walsh_b"][i][1] if i < len(ts["walsh_b"]) else None
        db.log_timeseries(run_id, gen, fa, fb, kl, eda, edb, res, wa, wb)

    if verbose:
        conv = result.convergence_gen if result.convergence_gen else "—"
        print(f"  [{cfg.signal_design:<18}]  residual={result.residual:.5f}  "
              f"conv={conv}  outcome={result.outcome}")

    return run_id, result


# ── Standard diagnostic plot for one experiment ───────────────────────────────
def plot_experiment(exp: Experiment,
                    results: dict,          # {signal_name: CoevoResult}
                    out: Optional[str] = None):
    """
    4-panel plot: fitness curves + Walsh signals + coefficients + residual trajectory.
    One curve per signal design, so you can compare them visually.
    """
    colors = {"walsh_landscape": "steelblue",
              "walsh_solution":  "darkorange",
              "raw_sparse":      "seagreen"}
    styles = {"walsh_landscape": "-",
              "walsh_solution":  "--",
              "raw_sparse":      ":"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"{exp.object_id} — {exp.name}", fontsize=12, fontweight="bold")

    # 1. Fitness convergence (Pop A)
    ax = axes[0, 0]
    for sig, res in results.items():
        gens = [g for g, _ in res.timeseries["fit_a"]]
        vals = [v for _, v in res.timeseries["fit_a"]]
        ax.plot(gens, vals, label=sig, color=colors[sig], ls=styles[sig], lw=1.5)
    ax.set_title("Pop A fitness convergence"); ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (-loss)"); ax.legend(fontsize=8)

    # 2. Joint residual trajectory
    ax = axes[0, 1]
    for sig, res in results.items():
        gens = [g for g, _ in res.timeseries["residual"]]
        vals = [v for _, v in res.timeseries["residual"]]
        ax.semilogy(gens, [max(v, 1e-6) for v in vals],
                    label=sig, color=colors[sig], ls=styles[sig], lw=1.5)
    ax.axhline(0.10, color="red", lw=0.8, ls="--", label="threshold=0.10")
    ax.set_title("Joint residual  (log scale)"); ax.set_xlabel("Generation")
    ax.set_ylabel("||A - expected(B)||"); ax.legend(fontsize=8)

    # 3. Walsh main-effects at convergence (best signal design only)
    ax   = axes[1, 0]
    ks   = np.arange(exp.cfg.n_genes)
    w    = 0.25
    best_sig = min(results, key=lambda s: results[s].residual)
    res  = results[best_sig]
    ax.bar(ks - w/2, res.walsh_me_a, w, label="Pop A", color="steelblue",  alpha=0.85)
    ax.bar(ks + w/2, res.walsh_me_b, w, label="Pop B", color="darkorange", alpha=0.85)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"Walsh main-effects at convergence\n(best design: {best_sig})")
    ax.set_xlabel("Gene k"); ax.set_xticks(ks)
    ax.legend(fontsize=8)

    # 4. Learned coefficients vs true
    ax = axes[1, 1]
    for i, (sig, res) in enumerate(results.items()):
        offset = (i - 1) * w
        ax.bar(ks + offset, res.best_a, w * 0.9,
               label=f"A [{sig[:8]}]", color=colors[sig], alpha=0.6 + 0.1*i)
    ax.bar(ks + len(results)*w, exp.target_a, w,
           label="True A", color="black", alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Found A coefficients across signal designs")
    ax.set_xlabel("Gene k"); ax.set_xticks(ks)
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = out or f"{exp.object_id}_result.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.show()
    return path
