"""
Signal information capacity analyzer.

Reads the Mendel notebook DB and computes what each signal molecule design
can reliably detect vs. what it misses.

The core idea: compare pairs of experiments that differ in exactly ONE
mathematical property (e.g., MM-T0-001 vs MM-T0-003 differ only in the
scalar factor). If a signal design gets MM-T0-001 right but MM-T0-003 wrong,
it can detect IDENTITY but not SCALAR RATIO.

Over time this builds a capacity matrix:

  Signal design    | identity | scalar_ratio | additive_shift | derivative | ...
  walsh_landscape  |    YES   |      NO      |      ?         |    NO      | ...
  walsh_solution   |    YES   |      NO      |      ?         |    NO      | ...
  raw_sparse       |    YES   |      NO      |      ?         |    NO      | ...

The NOes are as informative as the YESes. They tell you what information
structure is missing from the current signal vocabulary and what enrichments
(Dirichlet basis, trajectory n-grams, contrastive signal) you'd need to add.

Also computes:
  - Betweenness score per run: was the spectral trajectory a smooth path
    between A's initial spectrum and B's final spectrum?
  - Spectral trajectory manifold: the convex hull of valid trajectories
    that converged, used as the early-pruning boundary for future experiments.
"""

import json
import sqlite3
import numpy as np
from collections import defaultdict
from notebook.db import DB_PATH, _connect


# ── Relationship type definitions ─────────────────────────────────────────────
# Each entry: (experiment_ids_that_test_this, property_name, expected_outcome)
RELATIONSHIP_TESTS = {
    "identity":        (["MM-T0-001", "MM-T0-002", "MM-T0-005",
                         "MM-T0-006", "MM-T0-007", "MM-T0-008"], "correct"),
    "scalar_multiple": (["MM-T0-003"], "correct"),
    "additive_shift":  (["MM-T0-004"], "correct"),
    "derivative":      (["MM-T0-009"], "correct"),
}


def capacity_matrix() -> dict:
    """
    Compute the information capacity matrix from all runs in the DB.
    Returns: {signal_type: {relationship_type: {detectable, avg_residual, n_runs}}}
    """
    conn = _connect()
    rows = conn.execute("""
        SELECT r.signal_type, r.object_id, r.outcome, r.residual
        FROM experiment_runs r
        JOIN math_objects o ON r.object_id = o.id
        WHERE r.residual IS NOT NULL
    """).fetchall()
    conn.close()

    # {signal_type: {object_id: [residuals]}}
    by_signal_obj = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_signal_obj[r["signal_type"]][r["object_id"]].append(r["residual"])

    matrix = {}
    for signal_type, obj_data in by_signal_obj.items():
        matrix[signal_type] = {}
        for rel_type, (obj_ids, expected_outcome) in RELATIONSHIP_TESTS.items():
            residuals = []
            for oid in obj_ids:
                if oid in obj_data:
                    residuals.extend(obj_data[oid])
            if not residuals:
                matrix[signal_type][rel_type] = {"detectable": None, "n": 0}
            else:
                avg = np.mean(residuals)
                matrix[signal_type][rel_type] = {
                    "detectable": avg < 0.15,
                    "avg_residual": float(avg),
                    "n": len(residuals),
                }
    return matrix


def print_capacity_matrix():
    """Print the information capacity matrix as a readable table."""
    m = capacity_matrix()
    rel_types = list(RELATIONSHIP_TESTS)

    print("\n" + "="*80)
    print("SIGNAL INFORMATION CAPACITY MATRIX")
    print("What can each signal molecule design reliably transmit?")
    print("="*80)

    header = f"{'Signal design':<22}" + "".join(f"{r[:14]:>16}" for r in rel_types)
    print(header)
    print("-"*80)

    for sig, rel_data in sorted(m.items()):
        row = f"{sig:<22}"
        for rel in rel_types:
            d = rel_data.get(rel, {})
            if d.get("detectable") is None:
                cell = "      ?"
            elif d["detectable"]:
                cell = f"  YES({d['avg_residual']:.3f})"
            else:
                cell = f"   NO({d['avg_residual']:.3f})"
            row += f"{cell:>16}"
        print(row)

    print("="*80)
    print("\nYES = avg residual < 0.15 across all experiments testing that relationship")
    print("NO  = avg residual >= 0.15  (signal cannot encode this relationship type)")
    print("?   = no data yet")
    print("\nThe NOs are the research findings: what information enrichment is needed?")
    _print_enrichment_hypotheses(m)


def _print_enrichment_hypotheses(matrix: dict):
    """From the NOs, generate hypotheses about what signal enrichments are needed."""
    print("\nHYPOTHESES FROM CAPACITY GAPS:")
    print("-"*60)

    all_no = defaultdict(list)
    for sig, rel_data in matrix.items():
        for rel, d in rel_data.items():
            if d.get("detectable") is False:
                all_no[rel].append(sig)

    enrichment_map = {
        "scalar_multiple": (
            "Signal lacks RATIO information. Current designs transmit gene importance "
            "(magnitude) but not relative scale between populations.\n"
            "  Hypothesis: add ratio normalization — signal = spectrum_A / spectrum_B "
            "(elementwise) — so receiver knows 'I am N× larger than you in each dimension'."
        ),
        "additive_shift": (
            "Signal lacks OFFSET information. The constant term (c_0) shift is invisible.\n"
            "  Hypothesis: include raw c_0 value explicitly in signal alongside Walsh spectrum."
        ),
        "derivative": (
            "Signal lacks SHIFT-STRUCTURE information — which coefficient index maps to "
            "which in the other population.\n"
            "  Hypothesis: trajectory n-grams (send sequence of spectra, not snapshot) "
            "encode directional drift that would reveal the index-shift structure."
        ),
    }

    for rel, signals in sorted(all_no.items()):
        if rel in enrichment_map:
            print(f"\n  [{rel.upper()}] — not detectable by: {signals}")
            print(f"  {enrichment_map[rel]}")


# ── Betweenness score ─────────────────────────────────────────────────────────

def betweenness_score(run_id: str) -> dict:
    """
    Compute how 'between' the spectral trajectory is.

    A valid morphing from A to B should produce a Walsh spectrum trajectory
    that monotonically transitions from A's initial spectrum toward B's final
    spectrum. Violent jumps, reversals, or lateral moves in spectral space
    indicate the morphing is not on a valid path.

    Returns:
      path_length:    total L2 distance traveled in Walsh spectral space
      direct_dist:    L2 distance from start to end (lower bound)
      efficiency:     direct_dist / path_length  (1.0 = perfectly straight path)
      reversal_count: how many times the trajectory reversed direction
      betweenness:    "tight" | "wandering" | "reversed"
    """
    conn = _connect()
    rows = conn.execute("""
        SELECT generation, walsh_spectrum_a, walsh_spectrum_b
        FROM run_timeseries
        WHERE run_id = ? AND walsh_spectrum_a != '[]'
        ORDER BY generation
    """, (run_id,)).fetchall()
    conn.close()

    if len(rows) < 2:
        return {"betweenness": "insufficient_data"}

    spectra_a = [np.array(json.loads(r["walsh_spectrum_a"])) for r in rows]
    spectra_b = [np.array(json.loads(r["walsh_spectrum_b"])) for r in rows]

    # Measure trajectory properties for population A's signal
    dists = [np.linalg.norm(spectra_a[i+1] - spectra_a[i])
             for i in range(len(spectra_a)-1)]
    path_length  = float(sum(dists))
    direct_dist  = float(np.linalg.norm(spectra_a[-1] - spectra_a[0]))
    efficiency   = direct_dist / (path_length + 1e-10)

    # Count reversals: direction flip in the dominant Walsh coefficient
    dom_idx = np.argmax(np.abs(spectra_a[0]))
    proj    = [s[dom_idx] for s in spectra_a]
    signs   = np.sign(np.diff(proj))
    reversals = int(np.sum(np.diff(signs) != 0))

    # Cross-population divergence: did A and B move toward or away from each other?
    kl_start = _kl(spectra_a[0], spectra_b[0])
    kl_end   = _kl(spectra_a[-1], spectra_b[-1])
    converging = kl_end < kl_start

    if efficiency > 0.7 and reversals < 2:
        label = "tight"
    elif reversals > 4:
        label = "reversed"
    else:
        label = "wandering"

    return {
        "path_length": path_length,
        "direct_dist": direct_dist,
        "efficiency": efficiency,
        "reversal_count": reversals,
        "converging": converging,
        "kl_start": float(kl_start),
        "kl_end": float(kl_end),
        "betweenness": label,
    }


def _kl(a, b, eps=1e-10):
    a = np.abs(a) + eps;  b = np.abs(b) + eps
    a /= a.sum();         b /= b.sum()
    return float(np.sum(a * np.log(a / b)))


def betweenness_report(tier: int = None):
    """Print betweenness scores for all runs, grouped by experiment."""
    conn = _connect()
    q = """
        SELECT r.id, r.object_id, r.signal_type, r.outcome, r.residual
        FROM experiment_runs r
        JOIN math_objects o ON r.object_id = o.id
        {}
        ORDER BY r.object_id, r.signal_type
    """.format(f"WHERE o.tier={tier}" if tier is not None else "")
    runs = [dict(r) for r in conn.execute(q).fetchall()]
    conn.close()

    print(f"\n{'='*80}")
    print("BETWEENNESS REPORT — spectral trajectory quality")
    print(f"{'='*80}")
    print(f"{'Run ID':<14} {'Object':<14} {'Signal':<20} "
          f"{'Effic':>6} {'Rev':>4} {'Conv':>5} {'Label'}")
    print("-"*80)

    for run in runs:
        score = betweenness_score(run["id"])
        if score.get("betweenness") == "insufficient_data":
            continue
        print(f"{run['id']:<14} {run['object_id']:<14} "
              f"{(run['signal_type'] or '?'):<20} "
              f"{score['efficiency']:>6.3f} {score['reversal_count']:>4} "
              f"{'Y' if score['converging'] else 'N':>5} "
              f"{score['betweenness']}")
