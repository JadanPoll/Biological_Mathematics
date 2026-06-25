-- Mathematical Mendel Notebook — SQLite schema
-- Three-level structure mirroring Darwin Core:
--   math_objects  (Taxon)   — what the mathematical object IS
--   experiment_runs (Occurrence) — one coevolution run
--   run_timeseries  (Event)      — per-generation measurements

CREATE TABLE IF NOT EXISTS math_objects (
    id          TEXT PRIMARY KEY,          -- e.g. MM-0001
    name        TEXT NOT NULL,
    domain      TEXT,                      -- e.g. "trigonometry", "algebra"
    expression_a TEXT,                     -- how Population A is parameterised
    expression_b TEXT,                     -- how Population B is parameterised
    relationship TEXT,                     -- expected joint attractor (e.g. "A = d/dx(B)")
    tier        INTEGER,                   -- 0=trivial .. 4=negative control
    ground_truth TEXT,                     -- "proven" | "suspected" | "open" | "negative"
    properties  TEXT,                      -- JSON: {oscillatory, additive, multiplicative, ...}
    notes       TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    id                          TEXT PRIMARY KEY,   -- e.g. EXP-0001
    object_id                   TEXT REFERENCES math_objects(id),
    run_date                    TEXT,

    -- Setup
    genome_type                 TEXT,   -- "coefficient_vector" | "expression_tree"
    signal_type                 TEXT,   -- "walsh_best" | "walsh_population" | "dirichlet"
    n_coeffs                    INTEGER,
    pop_size                    INTEGER,
    gens                        INTEGER,
    n_collaborations            INTEGER,  -- N-collab partners per fitness eval
    cycle_consistency_weight    REAL,
    scaffolding                 TEXT,     -- JSON list of intermediate MM-ids (empty = direct)
    coupling_mode               TEXT,     -- "fixed" | "adaptive_kl"

    -- Terminal measurements
    convergence_gen             INTEGER,  -- NULL if failed
    residual                    REAL,     -- ||A_best - expected(B_best)||_2
    cycle_residual              REAL,     -- cycle-consistency loss at convergence
    spectral_class              TEXT,     -- "smooth_decay"|"oscillatory"|"flat"|"nonmonotone"
    effective_dim_a_final       INTEGER,  -- how many Walsh main-effects were significant
    effective_dim_b_final       INTEGER,
    failure_mode                TEXT,     -- NULL | "miscoordination" | "collapse" | "stagnation"
    outcome                     TEXT,     -- "correct" | "wrong_attractor" | "failed"

    -- Final Walsh main-effects (JSON arrays, length = n_coeffs)
    walsh_me_a_final            TEXT,
    walsh_me_b_final            TEXT,
    kl_final                    REAL,

    notes                       TEXT,
    -- Wall-clock timing breakdown (JSON): {operation: {total_s, count, mean_ms}}
    -- Tells us where compute budget goes and what to optimize when scaling up.
    timing_json                 TEXT
);

CREATE TABLE IF NOT EXISTS run_timeseries (
    run_id              TEXT REFERENCES experiment_runs(id),
    generation          INTEGER,
    fit_a               REAL,
    fit_b               REAL,
    kl_divergence       REAL,
    eff_dim_a           INTEGER,
    eff_dim_b           INTEGER,
    joint_residual      REAL,
    -- Full Walsh spectra stored as JSON arrays (length 2^n_coeffs = 256 for n=8)
    -- These are the training signal for the betweenness model.
    -- Main-effects are at indices 2^k; full spectrum encodes all epistatic orders.
    walsh_spectrum_a    TEXT,
    walsh_spectrum_b    TEXT,
    PRIMARY KEY (run_id, generation)
);

-- Information capacity log: which signal designs can detect which relationship types.
-- Built up from comparing runs that differ in exactly one mathematical property.
CREATE TABLE IF NOT EXISTS signal_capacity (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_type     TEXT,
    relationship_type TEXT,   -- "identity"|"scalar_multiple"|"additive_shift"|"derivative"|...
    detectable      INTEGER,  -- 1=yes, 0=no, NULL=unknown
    evidence_run_ids TEXT,    -- JSON list of run_ids that support this
    residual_with    REAL,    -- avg residual when signal is used
    residual_without REAL,    -- avg residual with null signal (baseline)
    notes           TEXT,
    computed_at     TEXT DEFAULT (datetime('now'))
);

-- Mendel Tables: aggregate patterns across runs
CREATE TABLE IF NOT EXISTS aggregate_patterns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type    TEXT,    -- "spectral_floor" | "convergence_curve" | "tool_mapping"
    tier            INTEGER,
    signal_type     TEXT,
    n_runs          INTEGER,
    value           TEXT,    -- JSON summary
    computed_at     TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_runs_object ON experiment_runs(object_id);
CREATE INDEX IF NOT EXISTS idx_runs_outcome ON experiment_runs(outcome);
CREATE INDEX IF NOT EXISTS idx_ts_run ON run_timeseries(run_id);
