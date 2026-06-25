"""
Notebook database — thin wrapper around SQLite.
All experiment results flow through here.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "mendel.db"
SCHEMA   = Path(__file__).parent / "schema.sql"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init():
    """Create tables if they don't exist."""
    conn = _connect()
    conn.executescript(SCHEMA.read_text())
    conn.commit()
    conn.close()


# ── Math Objects ──────────────────────────────────────────────────────────────

def upsert_object(id, name, domain, expression_a, expression_b,
                  relationship, tier, ground_truth, properties=None, notes=""):
    conn = _connect()
    conn.execute("""
        INSERT OR REPLACE INTO math_objects
          (id, name, domain, expression_a, expression_b, relationship,
           tier, ground_truth, properties, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (id, name, domain, expression_a, expression_b, relationship,
          tier, ground_truth, json.dumps(properties or {}), notes))
    conn.commit()
    conn.close()


def get_object(id):
    conn = _connect()
    row = conn.execute("SELECT * FROM math_objects WHERE id=?", (id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_objects(tier=None):
    conn = _connect()
    q = "SELECT * FROM math_objects" + (f" WHERE tier={tier}" if tier is not None else "")
    rows = conn.execute(q).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Experiment Runs ───────────────────────────────────────────────────────────

def new_run_id():
    return "EXP-" + uuid.uuid4().hex[:8].upper()


def log_run(object_id, setup: dict, terminal: dict, notes="") -> str:
    """
    Insert one experiment run.  Returns the run id.

    setup keys:    genome_type, signal_type, n_coeffs, pop_size, gens,
                   n_collaborations, cycle_consistency_weight,
                   scaffolding, coupling_mode
    terminal keys: convergence_gen, residual, cycle_residual, spectral_class,
                   effective_dim_a_final, effective_dim_b_final,
                   failure_mode, outcome,
                   walsh_me_a_final, walsh_me_b_final, kl_final
    """
    rid = new_run_id()
    conn = _connect()
    conn.execute("""
        INSERT INTO experiment_runs (
          id, object_id, run_date,
          genome_type, signal_type, n_coeffs, pop_size, gens,
          n_collaborations, cycle_consistency_weight, scaffolding, coupling_mode,
          convergence_gen, residual, cycle_residual, spectral_class,
          effective_dim_a_final, effective_dim_b_final,
          failure_mode, outcome,
          walsh_me_a_final, walsh_me_b_final, kl_final, notes, timing_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        rid, object_id, datetime.utcnow().isoformat(),
        setup.get("genome_type"), setup.get("signal_type"),
        setup.get("n_coeffs"), setup.get("pop_size"), setup.get("gens"),
        setup.get("n_collaborations"), setup.get("cycle_consistency_weight"),
        json.dumps(setup.get("scaffolding", [])), setup.get("coupling_mode"),
        terminal.get("convergence_gen"), terminal.get("residual"),
        terminal.get("cycle_residual"), terminal.get("spectral_class"),
        terminal.get("effective_dim_a_final"), terminal.get("effective_dim_b_final"),
        terminal.get("failure_mode"), terminal.get("outcome"),
        json.dumps(terminal.get("walsh_me_a_final", [])),
        json.dumps(terminal.get("walsh_me_b_final", [])),
        terminal.get("kl_final"), notes,
        json.dumps(terminal.get("timing", {}))
    ))
    conn.commit()
    conn.close()
    return rid


def log_timeseries(run_id, generation, fit_a, fit_b, kl,
                   eff_dim_a, eff_dim_b, joint_residual,
                   walsh_spectrum_a=None, walsh_spectrum_b=None):
    """
    Log one generation snapshot.
    walsh_spectrum_a/b are full 2^N arrays — the training signal for the
    betweenness model.  Store as JSON; query later for trajectory analysis.
    """
    conn = _connect()
    conn.execute("""
        INSERT OR IGNORE INTO run_timeseries
          (run_id, generation, fit_a, fit_b, kl_divergence,
           eff_dim_a, eff_dim_b, joint_residual,
           walsh_spectrum_a, walsh_spectrum_b)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (run_id, generation, fit_a, fit_b, kl,
          eff_dim_a, eff_dim_b, joint_residual,
          json.dumps(walsh_spectrum_a.tolist() if walsh_spectrum_a is not None else []),
          json.dumps(walsh_spectrum_b.tolist() if walsh_spectrum_b is not None else [])))
    conn.commit()
    conn.close()


def log_signal_capacity(signal_type, relationship_type, detectable,
                        evidence_run_ids, residual_with, residual_without, notes=""):
    conn = _connect()
    conn.execute("""
        INSERT INTO signal_capacity
          (signal_type, relationship_type, detectable, evidence_run_ids,
           residual_with, residual_without, notes)
        VALUES (?,?,?,?,?,?,?)
    """, (signal_type, relationship_type, int(detectable),
          json.dumps(evidence_run_ids), residual_with, residual_without, notes))
    conn.commit()
    conn.close()


# ── Query helpers ─────────────────────────────────────────────────────────────

def runs_for_object(object_id):
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM experiment_runs WHERE object_id=? ORDER BY run_date",
        (object_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def summary(tier=None):
    conn = _connect()
    q = """
        SELECT o.tier, r.outcome, r.spectral_class,
               AVG(r.residual) as avg_residual,
               AVG(r.convergence_gen) as avg_conv_gen,
               COUNT(*) as n_runs
        FROM experiment_runs r
        JOIN math_objects o ON r.object_id = o.id
        {}
        GROUP BY o.tier, r.outcome, r.spectral_class
        ORDER BY o.tier, r.outcome
    """.format(f"WHERE o.tier={tier}" if tier is not None else "")
    rows = conn.execute(q).fetchall()
    conn.close()
    return [dict(r) for r in rows]
