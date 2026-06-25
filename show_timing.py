import json, sqlite3
from notebook.db import DB_PATH
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
row = conn.execute('SELECT timing_json FROM experiment_runs LIMIT 1').fetchone()
timing = json.loads(row['timing_json'])
total = sum(v['total_s'] for v in timing.values())
print('TIMING BREAKDOWN (300 gens, pop=60, n_collab=5, walsh_landscape):')
for op, v in sorted(timing.items(), key=lambda x: -x[1]['total_s']):
    pct = 100 * v['total_s'] / total
    print(f"  {op:<22} {v['total_s']:>7.3f}s  {pct:>5.1f}%  mean={v['mean_ms']:>7.2f}ms  n={v['count']}")
print(f"  {'TOTAL':<22} {total:>7.3f}s")
