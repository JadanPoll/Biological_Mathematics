import json, numpy as np
data = json.load(open('algebraic_origami_taxonomy.json'))
print(f'Total interesting morphs discovered: {len(data)}')
top = sorted(data, key=lambda x: -x['score'])[:10]
print('\nTop 10 most interesting morphs:')
for r in top:
    print(f"  score={r['score']:.3f}  {r['start'][:16]}->{r['end'][:16]} via {r['signal']:<14}  H={r['entropy']:.2f} K={r['curvature']:.4f}")
scores = [r['score'] for r in data]
print(f'\nScore stats: min={min(scores):.3f} mean={np.mean(scores):.3f} max={max(scores):.3f}')

# Group by signal type
from collections import Counter
by_sig = Counter(r['signal'] for r in data)
print('\nBy signal type:')
for sig, cnt in by_sig.most_common():
    print(f'  {sig:<16}: {cnt} interesting morphs')
