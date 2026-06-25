"""
Computational taxonomy — what each mathematical tool computes, when to use it,
and what failure looks like.

Organised by the Ramanujan misalignment types:

  Type 1 (formally correct, metrically wrong):
    Tool: Analytic continuation → for us: complex relay chain lift
    Indicator: formal algebra self-consistent, classical metric diverges
    Fix: change metric (complex 1D instead of real 8D)

  Type 2 (systematically incomplete — missing shadow):
    Tool: Harmonic weak Maass forms → for us: directional signal (trajectory n-gram)
    Indicator: failure systematic, not random; same direction every time
    Fix: add the missing complementary structure (the "shadow")

  Type 3 (almost integer — wrong basis):
    Tool: Complex multiplication / j-function → for us: right coordinate system
    Indicator: near-miss is too structured to be coincidence
    Fix: find the algebraic basis in which the near-miss becomes exact

  Type 4 (conjectured from structure, unproved):
    Tool: Etale cohomology / L-functions → for us: betweenness score
    Indicator: satisfies necessary conditions for a deep theorem
    Fix: patient data collection until the right instrument appears

  Type 5 (crossroads — too many coincidences):
    Tool: Algebraic geometry of intersection → for us: Gromov-Wasserstein
    Indicator: multiple independent algebraic structures simultaneously satisfied
    Fix: study the intersection variety, not the components separately

  Type 6 (incomplete basis):
    Tool: Change of basis (Dedekind, p-adic) → for us: phi_k = x^k/k! vs x^k
    Indicator: formal algebra works but magnitudes grow without bound
    Fix: normalise the basis
"""

import numpy as np


TAXONOMY = {
    "walsh_landscape_signal": {
        "computes": "WHT of fitness landscape at 2^N corners of hypercube",
        "encodes": "Epistatic interaction structure of search problem",
        "NOT": "Mathematical character of solution",
        "ramanujan_type": 6,  # incomplete basis — encodes gradient not identity
        "failure_mode": "Signal is noise for simple landscapes (confirmed TB-04)",
        "fix": "Use trajectory n-gram (Type 2 shadow) or complex lifting (Type 1)",
        "python": "toolkit/walsh.py",
        "installed": True,
    },
    "trajectory_ngram_signal": {
        "computes": "WHT of coefficient vectors over recent history window",
        "encodes": "DIRECTION of movement in solution space (velocity not position)",
        "ramanujan_type": 2,  # the 'shadow' that completes the state signal
        "failure_mode": "Needs warmup time (no history in first WINDOW iterations)",
        "fix": "Run longer; combine with state signal for both position+velocity",
        "python": "toolkit/signals.py TrajectoryNgram",
        "installed": True,
    },
    "jensen_shannon_divergence": {
        "computes": "Symmetric information distance between two distributions",
        "encodes": "How distinguishable two population states are",
        "NOT": "KL divergence — KL is asymmetric and not a metric",
        "ramanujan_type": 6,  # KL is the wrong basis; JSD is the right one
        "failure_mode": "None known — JSD is a proper metric",
        "python": "toolkit/stress_metrics.jsd / toolkit/info_geometry.js_distance",
        "installed": True,
    },
    "sliced_wasserstein": {
        "computes": "Earth Mover Distance between two point clouds",
        "encodes": "Physical transport effort to morph one population into another",
        "ramanujan_type": 3,  # measures 'almost integer' gap in physical space
        "failure_mode": "Approximate (sliced); exact OT needs C++ build tools",
        "python": "toolkit/stress_metrics.sliced_wasserstein",
        "installed": True,
    },
    "ollivier_ricci_curvature": {
        "computes": "Curvature of the relay chain path at each edge",
        "encodes": "Whether the path is a geodesic (equal curvature) or curved",
        "ramanujan_type": 4,  # conjectured structure without proof — curvature shows it
        "failure_mode": "Computationally expensive for large populations",
        "python": "toolkit/stress_metrics.relay_ricci_curvatures",
        "installed": True,
    },
    "fisher_rao_distance": {
        "computes": "Geodesic distance on statistical manifold of Gaussian distributions",
        "encodes": "Information-theoretic effort to morph one population state to another",
        "ramanujan_type": 1,  # formally correct; needs right metric (Fisher not Euclidean)
        "failure_mode": "Gaussian approximation may fail for non-Gaussian populations",
        "python": "toolkit/info_geometry.population_fisher_rao",
        "installed": True,
    },
    "cmaes": {
        "computes": "Natural gradient optimization via adaptive covariance matrix",
        "encodes": "Steepest descent in Fisher information geometry (not Euclidean)",
        "ramanujan_type": 1,  # right metric (Fisher) vs wrong metric (Euclidean GA)
        "failure_mode": "Requires function evaluations quadratic in dimension",
        "python": "import cma",
        "installed": True,
    },
    "complex_relay": {
        "computes": "Euler rotation path in 1D complex space (dimensional lift)",
        "encodes": "EXACT phase rotation without truncation error or degeneracy",
        "ramanujan_type": 1,  # wrong metric (real 8D) → right metric (complex 1D)
        "failure_mode": "Only applies to problems with U(1) rotation structure",
        "python": "toolkit/complex_relay",
        "installed": True,
    },
    "persistent_homology": {
        "computes": "Betti numbers and persistence diagrams of point clouds",
        "encodes": "Topological features (connected components, loops, voids) at all scales",
        "ramanujan_type": 5,  # crossroads — multiple topological structures intersecting
        "failure_mode": "Computationally expensive for high-dimensional data",
        "python": "import gudhi; import ripser",
        "installed": True,
    },
    "heat_kernel_SO2": {
        "computes": "Probability distribution over rotation angles after diffusion time t",
        "encodes": "The path integral measure over relay chain configurations",
        "ramanujan_type": 1,  # Ramanujan summation — series that works formally
        "failure_mode": "Only exact for SO(2); higher groups need more terms",
        "python": "toolkit/euler_relay.heat_kernel_SO2",
        "installed": True,
    },
    "gromov_wasserstein": {
        "computes": "Optimal transport between populations in DIFFERENT metric spaces",
        "encodes": "How far two algebraic structures are from being isometric",
        "ramanujan_type": 5,  # crossroads between different spaces
        "failure_mode": "NP-hard exactly; needs C++ build tools for POT library",
        "python": "ot.gromov_wasserstein2 (not installed — needs C++ build tools)",
        "installed": False,
    },
    "ricci_flow": {
        "computes": "Evolution of Riemannian metric to smooth curvature",
        "encodes": "The 'heat equation for geometry' — relaxes path stress",
        "ramanujan_type": 4,  # Perelman proved what the structure implied
        "failure_mode": "No general Python implementation for arbitrary manifolds",
        "python": "GraphRicciCurvature for graphs (NetworkX-based)",
        "installed": "partial",
    },
}


def print_taxonomy():
    print("\n" + "="*72)
    print("COMPUTATIONAL TAXONOMY — tools, what they measure, Ramanujan type")
    print("="*72)
    print(f"{'Tool':<28} {'Type':>4}  {'Installed':>9}  What it encodes")
    print("-"*72)
    for name, info in sorted(TAXONOMY.items()):
        installed = str(info.get("installed", "?"))
        r_type = info.get("ramanujan_type", "?")
        encodes = info.get("encodes", "")[:35]
        print(f"  {name:<26} {r_type:>4}  {installed:>9}  {encodes}")
    print("="*72)
    print("\nRamanujan Type Key:")
    print("  1 = Formally correct, metrically wrong (needs better metric)")
    print("  2 = Systematically incomplete (needs shadow / complement)")
    print("  3 = Almost integer / near-miss (needs right algebraic basis)")
    print("  4 = Conjectured from structure (patient data collection)")
    print("  5 = Crossroads / too many coincidences (study intersection)")
    print("  6 = Incomplete/wrong basis (change representation)")


if __name__ == "__main__":
    print_taxonomy()

    # Quick smoke test of installed tools
    print("\n\nSMOKE TEST:")
    import numpy as np
    from toolkit.info_geometry import js_distance, population_fisher_rao
    from toolkit.stress_metrics import sliced_wasserstein, ollivier_ricci_edge

    rng = np.random.default_rng(42)
    pop_a = rng.standard_normal((30, 8)) * 0.1 + np.array([1,0,-1,0,1,0,-1,0])
    pop_b = rng.standard_normal((30, 8)) * 0.1 + np.array([0,1,0,-1,0,1,0,-1])
    p = np.abs(rng.standard_normal(16)) + 0.1
    q = np.abs(rng.standard_normal(16)) + 0.1

    print(f"  JS distance (cos vs sin signals):      {js_distance(p,q):.4f}")
    print(f"  Fisher-Rao (cos pop vs sin pop):       {population_fisher_rao(pop_a, pop_b):.4f}")
    print(f"  Sliced Wasserstein (cos vs sin pop):   {sliced_wasserstein(pop_a, pop_b, 20, rng):.4f}")
    print(f"  Ollivier-Ricci curvature (cos/sin):   {ollivier_ricci_edge(pop_a, pop_b):.4f}")

    import gudhi
    print(f"\n  gudhi version: {gudhi.__version__}")

    import ripser
    from ripser import ripser as rips
    data = rng.standard_normal((20, 4))
    result = rips(data, maxdim=1)
    print(f"  ripser: computed H0/H1 for 20 points OK")
    print(f"  H0 components: {len(result['dgms'][0])}")
    print(f"  H1 loops:      {len(result['dgms'][1])}")
