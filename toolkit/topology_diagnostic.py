"""
Persistent homology diagnostic for relay chain paths.

The Type 7 prediction from obstruction theory:
  A relay chain path with no valid algebraic path in this dimension
  will create TOPOLOGICAL LOOPS (H1 components) in the path's point cloud.
  A valid path (one that follows the algebraic structure) will have
  only H0 components (connected components) and no H1 loops.

Why:
  - A valid path is a 1D manifold from A to B with no self-intersections
    -> only H0 (connected) in persistent homology
  - A Type 7 path (no algebraic connection) wanders chaotically in coefficient
    space, returning near points it visited before
    -> creates H1 loops (cycles that don't fill in)

This is the topological version of the Minsky-Papert impossibility proof:
  the path cannot reach its target without looping back, and those loops
  are measurable by persistent homology even without knowing
  what the target algebraic structure is.

Gudhi API:
  rips_complex = gudhi.RipsComplex(points=path_points, max_edge_length=r)
  simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
  diag = simplex_tree.persistence()
  H0 = [(b,d) for dim,b,d in ... if dim==0]   # connected components
  H1 = [(b,d) for dim,b,d in ... if dim==1]   # loops
  H1 persistence = H1 intervals that survive long (persistent loops)
"""

import math
import numpy as np
import gudhi
from typing import List, Optional
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


def path_persistence(path_points: List[np.ndarray],
                      max_edge_length: float = 2.0,
                      min_persistence: float = 0.05) -> dict:
    """
    Compute persistent homology of a relay chain path.

    path_points: list of coefficient vectors (each is a point in R^N)
    max_edge_length: Rips complex parameter (controls which pairs of points
                     are considered "close enough" to form an edge)

    Returns:
      h0: list of (birth, death) for H0 (connected components)
      h1: list of (birth, death) for H1 (loops)
      persistent_h1: H1 intervals with persistence > min_persistence
                     (these are REAL loops, not noise)
      has_persistent_loop: True if any H1 interval survives long enough
                           This is the Type 7 topological signature
      bottleneck: max H1 persistence (measure of loop strength)
    """
    points = np.array([p.flatten() for p in path_points])

    rips = gudhi.RipsComplex(points=points, max_edge_length=max_edge_length)
    st   = rips.create_simplex_tree(max_dimension=2)
    diag = st.persistence()

    h0 = [(b, d) for dim, (b, d) in diag if dim == 0 and d != float('inf')]
    h1 = [(b, d) for dim, (b, d) in diag if dim == 1]

    persistent_h1 = [(b, d) for b, d in h1
                     if d - b > min_persistence and d != float('inf')]
    bottleneck = max([d - b for b, d in persistent_h1], default=0.0)

    return {
        "h0":                  h0,
        "h1":                  h1,
        "persistent_h1":       persistent_h1,
        "has_persistent_loop": len(persistent_h1) > 0,
        "bottleneck":          float(bottleneck),
        "n_path_points":       len(points),
        "betti_0":             sum(1 for dim, _ in diag if dim == 0
                                   and _[1] == float('inf')),
        "betti_1":             sum(1 for dim, _ in diag if dim == 1
                                   and _[1] == float('inf')),
    }


def relay_topology(populations: List[np.ndarray],
                    n_samples: int = 20,
                    max_edge_length: float = 2.0,
                    min_persistence: float = 0.10) -> dict:
    """
    Run persistent homology on the relay chain path.

    Samples n_samples points from each population (or uses the
    best individual if single-point), then runs Rips complex.

    The key metric: persistent_h1
      - Valid path (degree-1 algebraic relationship): no persistent H1
      - Type 7 path (no algebraic path): persistent H1 loops appear
    """
    path_points = []
    for pop in populations:
        if pop.ndim == 1:
            path_points.append(pop.copy())
        else:
            # Sample a few representatives from each population
            n = min(n_samples, len(pop))
            path_points.extend(pop[:n])

    return path_persistence(path_points, max_edge_length, min_persistence)


# ── Type 7 topology test ──────────────────────────────────────────────────────

def test_topology_vs_type7(verbose: bool = True):
    """
    Test the prediction:
      cos->sin (degree-1, valid path): no persistent H1 loops
      cos->cos2x (degree-2, Type 7 in linear): persistent H1 loops appear

    Uses open-chain relay paths from type7_test.py.
    """
    import cma
    from toolkit.euler_relay import _COS, _SIN, N

    def _cos2x():
        c = np.zeros(N)
        for k in range(0, N, 2):
            c[k] = ((-1)**(k//2)) * (2.0**k)
        return c

    def build_relay_path(fn_a, fn_b, n_steps=5, seed=42):
        """Build relay chain path (intermediates from CMA-ES optimization)."""
        np.random.seed(seed)
        pops = [fn_b.copy()]
        for k in range(1, n_steps):
            alpha = k / n_steps
            pops.append((1-alpha)*fn_b + alpha*fn_a + np.random.randn(N)*0.3)
        pops.append(fn_a.copy())

        for k in range(1, n_steps):
            prev, nxt = pops[k-1], pops[k+1]
            def nf(c, p=prev, n_=nxt):
                return float(np.linalg.norm(np.array(c)-p)**2 +
                             np.linalg.norm(n_-np.array(c))**2)
            opts = cma.CMAOptions(); opts['maxiter']=100; opts['verbose']=-9
            es = cma.CMAEvolutionStrategy(pops[k].tolist(), 0.4, opts)
            es.optimize(nf)
            pops[k] = np.array(es.result.xbest)

        return pops

    cos = _COS.copy()
    sin = _SIN.copy()
    cos2 = _cos2x()

    results = {}
    for name, fn_a, fn_b in [
        ("cos->sin   (degree-1, valid)",     sin,  cos),
        ("cos->cos2x (degree-2, Type 7)",    cos2, cos),
    ]:
        # KEY DIAGNOSTIC: intermediate scatter across seeds.
        # Valid path:  all seeds find same algebraic intermediates -> LOW spread
        # Type 7 path: no algebraic attractor -> each seed finds different position -> HIGH spread
        # Use JOINT FITNESS (not L2) so the optimization is algebraically informed
        seed_intermediates = []  # shape: (n_seeds, n_steps-1, N)
        for seed in range(12):
            path = build_relay_path(fn_a, fn_b, n_steps=5, seed=seed*7)
            seed_intermediates.append([path[k] for k in range(1, 4)])  # middle 3

        seed_arr = np.array(seed_intermediates)  # (12, 3, N)
        # Spread = std of intermediate positions across seeds
        spread_per_step = np.mean(np.std(seed_arr, axis=0), axis=-1)  # (3,)
        mean_spread = float(np.mean(spread_per_step))

        # Also run topology on the union
        all_intermediates = [seed_arr[s, k] for s in range(12) for k in range(3)]
        topo = path_persistence(all_intermediates, max_edge_length=3.0, min_persistence=0.05)
        topo["mean_spread"] = mean_spread
        topo["spread_per_step"] = spread_per_step.tolist()
        results[name] = topo

        if verbose:
            print(f"\n  {name}")
            print(f"    Path points:  {topo['n_path_points']}")
            print(f"    H1 loops:     {len(topo['h1'])} total, "
                  f"{len(topo['persistent_h1'])} persistent")
            print(f"    Bottleneck:   {topo['bottleneck']:.4f}")
            print(f"    Seed spread:  {topo['mean_spread']:.5f}  "
                  f"(LOW=converged valid path, HIGH=scattered Type 7)")
            print(f"    Per-step:     {[round(x,4) for x in topo['spread_per_step']]}")

    # Check prediction
    valid_no_loop = not results["cos->sin   (degree-1, valid)"]["has_persistent_loop"]
    type7_has_loop = results["cos->cos2x (degree-2, Type 7)"]["has_persistent_loop"]

    if verbose:
        print(f"\n  Prediction check:")
        print(f"    cos->sin has no persistent H1: {valid_no_loop}")
        print(f"    cos->cos2x has persistent H1:  {type7_has_loop}")
        if valid_no_loop and type7_has_loop:
            print(f"    CONFIRMED: Topology distinguishes valid from Type 7 paths")
        elif valid_no_loop and not type7_has_loop:
            print(f"    PARTIAL: valid path clean, but Type 7 topology not yet visible")
            print(f"    -> may need more path samples or wider max_edge_length")
        else:
            print(f"    NOT CONFIRMED: adjust max_edge_length or min_persistence")

    return results, valid_no_loop and type7_has_loop


def plot_persistence_diagram(topo: dict, title: str, ax=None):
    """Plot persistence diagram (birth vs death) for H0 and H1."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 4))

    # H0 (blue)
    for b, d in topo["h0"]:
        ax.scatter(b, d, c="steelblue", s=20, alpha=0.7, label="H0" if b==topo["h0"][0][0] else "")

    # H1 (red)
    for b, d in topo["h1"]:
        if d == float('inf'):
            d = 3.0  # cap at finite value for plotting
        alpha = 0.4 if (d - b) < 0.1 else 0.9
        ax.scatter(b, d, c="firebrick", s=30, alpha=alpha,
                   label="H1" if b==topo["h1"][0][0] else "" if topo["h1"] else "")

    # Diagonal
    lim = ax.get_xlim()
    ax.plot([lim[0], lim[1]], [lim[0], lim[1]], "k--", lw=0.5, alpha=0.4)

    ax.set_xlabel("Birth"); ax.set_ylabel("Death")
    ax.set_title(f"{title}\nbottleneck={topo['bottleneck']:.3f}  "
                 f"persistent_H1={len(topo['persistent_h1'])}")
    if topo["h1"]:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="steelblue", label="H0 (components)"),
                            Patch(color="firebrick", label="H1 (loops)")],
                  fontsize=7)
    return ax


if __name__ == "__main__":
    print("="*60)
    print("PERSISTENT HOMOLOGY DIAGNOSTIC")
    print("Testing Type 7 topological signature:")
    print("  Valid paths:   no persistent H1 loops")
    print("  Type 7 paths:  H1 loops appear (path can't close)")
    print("="*60)

    results, confirmed = test_topology_vs_type7(verbose=True)

    # Plot persistence diagrams
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Persistence diagrams: valid path (cos->sin) vs Type 7 (cos->cos2x)",
                 fontsize=10, fontweight="bold")
    for ax, (name, topo) in zip(axes, results.items()):
        plot_persistence_diagram(topo, name[:25], ax)

    plt.tight_layout()
    plt.savefig("topology_diagnostic.png", dpi=140, bbox_inches="tight")
    plt.show()
    print("\nPlot saved -> topology_diagnostic.png")
