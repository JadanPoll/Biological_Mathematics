"""
Auto-generate Mendel tables and summary reports from the notebook DB.
"""

import json
from notebook import db


def mendel_table(tier=None):
    db.init()
    rows = db.summary(tier=tier)

    tier_str = f"Tier {tier}" if tier is not None else "All Tiers"
    print(f"\n{'='*76}")
    print(f"MENDEL TABLE — {tier_str}")
    print(f"{'='*76}")
    print(f"{'Tier':>4} {'Signal':<20} {'Outcome':<16} {'Avg Residual':>13} "
          f"{'Avg Conv':>9} {'N':>5}")
    print(f"{'-'*76}")

    for r in rows:
        print(f"{r['tier']:>4} {(r['spectral_class'] or '?'):<20} "
              f"{(r['outcome'] or '?'):<16} {(r['avg_residual'] or 0):>13.5f} "
              f"{(r['avg_conv_gen'] or 0):>9.0f} {r['n_runs']:>5}")

    print(f"{'='*76}")
    _signal_comparison(tier)


def _signal_comparison(tier=None):
    """Cross-tab: which signal design works best for which experiment?"""
    import sqlite3
    from notebook.db import DB_PATH

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    q = """
        SELECT o.id, o.name, r.signal_type,
               AVG(r.residual) as avg_res,
               MIN(r.residual) as min_res,
               COUNT(*) as n
        FROM experiment_runs r
        JOIN math_objects o ON r.object_id = o.id
        {}
        GROUP BY o.id, r.signal_type
        ORDER BY o.id, avg_res
    """.format(f"WHERE o.tier={tier}" if tier is not None else "")
    rows = [dict(r) for r in conn.execute(q).fetchall()]
    conn.close()

    if not rows:
        print("  (no runs yet)")
        return

    # Group by experiment
    from collections import defaultdict
    by_exp = defaultdict(list)
    for r in rows:
        by_exp[r["id"]].append(r)

    print(f"\nSIGNAL DESIGN COMPARISON")
    print(f"{'ID':<14} {'Signal':<20} {'Avg Residual':>13} {'Min Residual':>13} N")
    print(f"{'-'*66}")
    for exp_id, exp_rows in sorted(by_exp.items()):
        for i, r in enumerate(exp_rows):
            prefix = exp_id if i == 0 else " " * 14
            print(f"{prefix:<14} {(r['signal_type'] or '?'):<20} "
                  f"{r['avg_res']:>13.5f} {r['min_res']:>13.5f} {r['n']}")
        print()
