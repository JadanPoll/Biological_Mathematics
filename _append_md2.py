extra = r"""

---

## Part XXI: Atoms Not Elsewhere Captured

*Individual insights too important to lose, not fitting cleanly into Parts XIII-XX.*

### The Schleyer Strain Energy Threshold (Anti-Bredt Molecules)

Bredt's rule states double bonds cannot be at bridgehead positions in small bicyclic molecules -- too much strain. Schleyer's refinement: strain energy (OS) provides a QUANTITATIVE threshold:
- OS < 17 kcal/mol: isolable molecule
- 17 < OS < 21 kcal/mol: observable molecule
- OS > 21 kcal/mol: unstable

This is a mathematical proof-of-concept for our stress-vs-existence claim: the stress (strain energy) of a morphing directly bounds whether the structure can exist. For our relay chain, the equivalent of OS < 17 is the Wasserstein variance threshold below which the path is an explorable geodesic.

The analogy extends: Bredt-violating molecules become stable in LARGER rings (more dimensional freedom) -- exactly our extradimensional bypass principle. More atoms = more degrees of freedom = strain distributes = formerly impossible becomes possible.

### The Reaction-Diffusion Cheaper Alternative

Turing's 1952 morphogenesis paper showed two chemicals (activator A and inhibitor B) produce stable spatial patterns through simultaneous coupled dynamics -- no alternation. The stable patterns ARE the algebraic relationships. Unstable combinations = Type 7.

The gradient flow equivalent of joint CMA-ES:
  da/dt = -(a - d/dx(b)) - 2*(||a|| - target)*a/||a||
  db/dt = +J*(a - d/dx(b)) - 2*(||b|| - target)*b/||b||

This is O(N) per timestep, simultaneous (not alternating), and has normalization built in. 200 steps costs 200 * 8 * 8 = 12,800 FLOPs versus CMA-ES which costs orders of magnitude more. If the stable attractor IS the correct algebraic relationship, this is the cheapest possible discovery mechanism.

### The Lotka-Volterra / Derivative Connection

Predator and prey populations in the classic Lotka-Volterra model oscillate 90 degrees out of phase -- prey peaks before predators. This is mathematically identical to the derivative relationship: d/dt(prey) leads predator by one quarter-period. In coefficient space, this IS d/dx(sin) = cos. Biology discovered the derivative relationship as a DYNAMIC STABILITY PROPERTY, not an algebraic fact.

This validates the biological lifting intuition: the algebraic relationship (derivative) IS a thermodynamic equilibrium structure in the biological context. The relay chain is trying to find the SAME attractor that Lotka-Volterra dynamics find naturally given the right resource constraints and coupling.

### The Quantum Supremacy Ratio (User Insight)

The user proposed reformulating quantum supremacy as a geometric ratio:

  ratio = volume_of_quantum_search_space / action_of_minimum_classical_path

When ratio < 1: classical computation finds the path efficiently.
When ratio >> 1: the path space is too diffuse for classical search -- quantum interference is needed to concentrate on the right path.

This was found to correspond to the Fluctuation Determinant in the stationary phase approximation of the Feynman path integral, and to the Sign Problem in quantum Monte Carlo. For our relay chain: the ratio is the number of valid rotation paths (8D rotation group SO(8) has many) divided by the length of the minimum-stress path (the complex rotation = pi/2). When this ratio is large, the relay chain wanders rather than converges.

### The KL -> JSD Correction

Throughout Parts I-XII and early implementation, we used KL divergence as the coupling strength signal. KL is asymmetric and is not a metric. The correct replacement is Jensen-Shannon divergence:
- Symmetric: JSD(p,q) = JSD(q,p)
- Bounded: JSD in [0, log(2)]
- sqrt(JSD) satisfies the triangle inequality -- a true metric

This was confirmed and implemented in toolkit/stress_metrics.py. All coupling strength measurements should use JSD rather than KL going forward.

### The Gromov-Hausdorff Connection

The Gromov-Hausdorff distance between two metric spaces measures how far they are from being isometric. For two algebraic structures, GH distance = minimum distortion of any map between them. Larger GH distance = any morphing must introduce more distortion = higher stress.

This gives a LOWER BOUND on the stress of any relay chain between two structures: stress >= f(GH_distance(structure_A, structure_B)).

This is directly computable (approximately, via POT library Gromov-Wasserstein distance) and gives a theoretical prediction for the Type 7 threshold: if the GH distance between A and B exceeds a grammar-dependent threshold, Type 7 is guaranteed.

Not yet implemented due to POT requiring C++ build tools on Python 3.14.
"""

with open("joint_tractability_exploration.md", "a", encoding="utf-8") as f:
    f.write(extra)
print("Appended Part XXI successfully.")
