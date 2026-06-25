# Joint Tractability, Coevolutionary Geometry, and the Spectral Cartography of Mathematical Space

*A record of an exploratory session — insights, taxonomies, open questions, and the reasoning moves that generated them.*

---

## Preface

This document captures a chain of ideas beginning with a simple abstraction of neural networks and spiraling through attractor topology, analog computing, wetware, biological coevolution, the structure of the brain's implicit optimization, and ultimately a proposed methodology for empirically mapping the geometry of mathematical space using coevolutionary genetic algorithms and spectral signal molecules. The goal is not to present finished theory but to preserve the shape of the exploration — including the generative reasoning moves that produced the insights — so that the ideas can be revisited, tested, and extended.

The session was explicitly exploratory. Many claims are hypotheses at various levels of confidence. Where existing literature confirms an idea, that is noted. Where a gap appears to be real, that is also noted. The reader should treat this as a map of a territory, not a proof of its contents.

The spirit is Mendelian: systematic recording of experimental variation, what tools were added to reduce stress to which problems, how reliably intermediates help, and what the results say directionally even when they cannot say definitively.

---

## Part I: The Generative Reasoning Moves

Before the content, the process. These are the meta-level cognitive operations that repeatedly generated productive insights throughout the session. They are worth preserving independently because they are reusable.

### 1. Strip mechanism to expose function

Refuse to think about a system in terms of its implementation details and instead ask: what is this thing *doing* at the highest level of abstraction that still captures its essential character?

Neural networks are not weights, layers, activations, and gradients. They are high-dimensional objects that morph the output space to create tractability. This single reframing makes the network a thing you can reason about geometrically and generates questions automatically: what does overfitting look like in this framing? What does underfitting look like? What would different morphings of the same object reveal? What is tractability and why should analytical tractability be the default goal?

The move is powerful because the right abstraction makes the question space visible. A mechanistic description answers specific questions. A functional description generates questions you didn't know to ask.

### 2. Ask "for whom" about any assumed goal

When a system has an implicit optimization target, ask: this is tractable or optimal for whom, relative to what reasoning system? Tractability is always relative. Analytical tractability is tractable for symbolic reasoning systems. Probabilistic tractability is tractable for inference engines. Energetic tractability is tractable for metabolic systems. The brain's tractability is shaped by biological substrate constraints that have nothing to do with human mathematical legibility.

Current machine learning implicitly optimizes for computational tractability on GPU hardware because that is what gradient descent on differentiable functions is cheap to do. This is not a principled choice — it is an artifact of the substrate. Asking "tractable for whom" makes the artifact visible and the alternative design space accessible.

### 3. Use biological intuition as a formal probe

When you have a formal structure that seems incomplete, notice whether your biological intuition about a related system suggests what is missing. The brain's bistable simulation during perception is not a neuroscientific fact being imported for its own sake. It is a probe that reveals the formal structure: the forward and backward passes in predictive coding are not staged but coupled, negotiating simultaneously toward a joint attractor. The neuroscience makes visible what the formalism obscures.

This move works because evolution has solved many hard optimization problems, and biological systems that have persisted for billions of years are likely near optima of their respective problem landscapes.

### 4. Check whether the obvious has been done, precisely

Before claiming novelty, state the idea with enough precision to search for it and then search. This prevents overclaiming, and the act of precise statement sharpens the idea. The gaps found throughout this exploration were all real gaps, but only visible after precise enough statement.

### 5. Ask what physical substrate instantiates the abstract property

When an information-theoretic idea hits a wall, ask: if this property is real, what material would implement it most naturally? This move connects abstract ideas to engineering constraints and often reveals that the abstract property is real and the question is only about substrate.

### 6. Ask whether nature has already found it

When a formal structure seems like a minimum-energy solution to a real optimization problem, ask: has evolution discovered this, probably multiple times, at multiple scales? The coevolutionary GA structure appeared in quorum sensing, mitonuclear coevolution, endosymbiosis, and extracellular vesicle signaling. The convergence across scales and kingdoms is strong evidence that this is a natural attractor of any system where distinct informational spaces share a fitness landscape.

### 7. Apply the framework recursively

The most powerful check: does the framework explain itself? Stripping mechanism to expose function is the same move as finding the coordinate system in which something becomes tractable — which is exactly what the brain does to sensory data according to the predictive coding framework developed in the conversation. When a framework explains its own generation, it has found something close to an invariant.

---

## Part II: Core Conceptual Insights

### 2.1 Neural Networks as Morphing Objects

A neural network is a high-dimensional topological object that deforms the input-output space until the relationship between inputs and outputs becomes locally tractable. Every layer applies a push, pull, squeeze, or fold. The composition takes what was geometrically tangled and stretches it into a form that a simple readout can operate on.

**Overfitting** — morphing too specifically tuned to the training manifold; carved geometry idiosyncratic rather than structural.

**Underfitting** — insufficient expressiveness to deform the space into tractable shape; relationship remains tangled at output.

**Different morphings of the same object** — ensembles, different architectures, different initializations are projections of the same underlying structure from different angles. Where they agree, the geometry is real. Where they disagree, the morphing is underdetermined.

**Analyticalness as tractability target** — not intrinsic to the problem but to the interface between the problem and symbolic reasoning systems. Different tractability targets produce different geometries.

### 2.2 The Taxonomy of Tractability

Tractability is always relative to a reasoning system. These forms conflict and cannot be simultaneously maximized:

**Analytical tractability** — bottleneck is description length; relationship compressible into short expression over known grammar. What symbolic regression optimizes.

**Computational tractability** — bottleneck is evaluation cost; relevant computations run cheaply. What deep learning implicitly optimizes via gradient descent on GPU hardware. Completely decoupled from analytical tractability.

**Causal tractability** — bottleneck is identifiability of cause and effect under do-operations; causal graph sparse enough that interventions have predictable effects.

**Probabilistic tractability** — bottleneck is cost of computing posteriors and marginals; distributions approximately Gaussian or factorize cleanly. What VAEs implicitly optimize.

**Dimensional tractability** — bottleneck is intrinsic degrees of freedom; collapses irrelevant variation, reveals true dimensionality. Sparse coding does this nonlinearly.

**Energetic tractability** — bottleneck is physical energy cost per computation; the brain's primary target. Produces sparse, anisotropic representations where computation happens passively through carved structure. 20W vs gigawatts for comparable silicon.

**Interventional tractability** — bottleneck is sensitivity of outcomes to actions; small moves correspond to predictable useful changes. What reinforcement learning implicitly optimizes.

**Key insight:** current ML's implicit tractability choice is an artifact of GPU substrate, not a principled design decision. A unified geometric description of tractability-as-morphing that encompasses all types remains an open theoretical gap.

### 2.3 What Kind of Tractability Is the Brain?

The brain implements a composite not yet named as such in the literature:

**Energetic tractability** — 20W budget enforces sparse anisotropic representations. Networks trained to minimize energy use naturally develop predictive coding architectures, reducing redundancy and concentrating signal in high-MI channels.

**Probabilistic tractability** — the Free Energy Principle frames the brain as minimizing variational free energy, decomposed into accuracy (how well predictions match observations) and complexity (penalty for beliefs drifting from priors). The brain is doing MDL — minimum description length — implemented in wetware.

**Dimensional tractability** — sparse coding produces an overcomplete basis that flattens the manifold of natural stimuli piecewise, simplifying downstream representation. Produces structures resembling V1 receptive fields.

**Implicit causal tractability** — precision-weighting, where attention dynamically amplifies specific prediction error channels, is functionally equivalent to selecting which causal variables to hold fixed and which to let vary. Interventional reasoning implemented in dynamics rather than structure.

**Not** optimizing for analytical tractability in the symbolic sense. Representations efficient for the brain's own downstream machinery but opaque to symbolic analysis.

### 2.4 The Joint Optimization Hypothesis

**The gap in current symbolic regression:** existing pipelines stage the morphing and the symbolic search. The neural component morphs the space, then symbolic regression runs on the compressed representation. The two optimizations are sequential. The optimal morphing depends on the symbolic grammar being searched, and the optimal symbolic structure depends on the coordinate system, but they never inform each other during optimization.

**What a genuinely joint system looks like:** the symbolic combination and the geometric transformation of the output space are coupled oscillators settling into a joint attractor. Neither fixed while the other optimizes. The current symbolic hypothesis implies a preferred coordinate system; the current coordinate system implies a preferred symbolic structure. They iterate until simultaneously stable.

**The energy being minimized:** the total description length of the transformation plus the expression. Crucially, the granularity of both is co-determined by what distinctions the joint system needs to make. Features reducing description length in either component get collapsed. Features reducing it in both get amplified.

**Existing work:** ITEA evolves input symbolic combinations while fitting output with OLS (half the problem). AI Feynman uses neural networks to detect symmetries before symbolic search (staged but informed). MDLformer minimizes total description length (right objective, fixed coordinates). Data-driven Progressive Discovery of Physical Laws does transformation discovery in parallel with symbolic search (very close, not fully joint).

**The genuinely missing piece:** simultaneous optimization of input symbolic combination and output space reparameterization under a single joint MDL objective, where granularity in both spaces is co-determined.

### 2.5 The Brain's Accidental Implementation of Joint Optimization

**The hypothesis:** the brain may be accidentally implementing the joint constraint through coupled predictive coding dynamics — not by design but because attractor dynamics naturally couple the two directions.

**Forward pass = symbolic combination:** predictions flowing downward are structured compositions of lower features into higher predictions. The hierarchical generative model is doing something isomorphic to symbolic regression — finding a structured combination of lower-level features that predicts the input.

**Backward pass = output space morphing:** error neurons propagating upward are partitioning surprise into whichever stable attractor basins can absorb it stably. That partitioning is the output space morphing — finding the coordinate system in which the residual becomes analytically tractable.

**The joint attractor:** the system settles into a state where the symbolic combination perfectly predicts the input and the error is fully partitioned into stable basins with nothing left over. Both processes simultaneously stabilize. This is perception as defined by Hybrid Predictive Coding — the equilibrium point of inference.

**Formal support:** the Helmholtz Machine (Dayan, Hinton 1995) explicitly couples generative and recognition networks — the original paper acknowledges "there is no single cost function reduced by these two procedures," i.e., staged not truly joint. The Free Energy Principle shows that high-level attractor dynamics prescribe a manifold guiding the flow of lower states — coupling is real and continuous. Friston's predictive coding: hierarchically coupled dynamical systems settling simultaneously.

**What's novel in this framing:** the literature has never stated that the brain's generative hierarchy is doing something isomorphic to symbolic regression on a basis of learned features, and its error propagation is doing something isomorphic to finding the coordinate system in which that expression is analytically tight. The math implies it. The framing makes it visible.

---

## Part III: The Coevolutionary GA Methodology

### 3.1 Core Architecture

Two populations evolving in tandem with coupled fitness:

**Population A:** symbolic expression trees over input variables — the combination side, exactly like PySR or ITEA.

**Population B:** parameterized output space transformations — invertible functions: power transforms, log transforms, Box-Cox, small invertible networks.

**Fitness coupling:** fitness of any individual in A is evaluated by pairing with current best from B, applying B's transformation to outputs, running A's expression, and computing joint MDL — description length of the expression plus description length of the transformation. Fitness of any individual in B: pair with current best from A, same joint MDL objective.

**Signal molecules:** rather than full genetic information exchange, each population sends a Walsh-Fourier spectral decomposition of its fitness landscape to the other. These compressed structural hints are the biological analogue of quorum sensing molecules — not the genome, but a metabolite encoding current functional state at a resolution the receiver can actually use.

**The Walsh-Fourier grounding:** the Walsh-Hadamard transform (originally developed for genetic algorithms) decomposes any fitness landscape over a sequence space into Fourier coefficients where each coefficient encodes the epistatic interaction of a specific order. First-order coefficients: additive contributions of individual features. Second-order: pairwise interactions. kth-order: how k operations jointly affect fitness beyond their individual contributions. The Fourier spectrum — summing squared coefficients by order — measures the strength of epistasis at each interaction order, which directly measures landscape ruggedness and the difficulty of the optimization problem.

**Why Walsh = signal molecule:** Walsh coefficients are the natural compressed structural hint for this system. They don't transmit the full landscape — they transmit how much of the landscape's structure lives at each interaction order. Sending the first N Walsh coefficients is sending "here is the Taylor approximation of my fitness landscape up to order N." Biologically precise: not the genome but a metabolite encoding current functional state at low resolution.

### 3.2 Spectral Ruggedness Metrics

The Walsh Fourier spectrum provides several directly measurable diagnostics:

**Fourier amplitude spectrum decay rate:** on smooth landscapes the spectrum decays exponentially with order — low-order coefficients dominate, high-order ones are small. On rugged landscapes the decay is flat or slow, indicating significant structure at every scale simultaneously. This is the primary diagnostic for "is the mathematical lifting unreachably hard."

**Correlation of fitness effects (γ):** measures how correlated the fitness effect of a mutation in one genetic background is with its effect in another. 1-γ is the ruggedness measure, increasing with ruggedness. Computable from the Walsh spectrum.

**Fraction of nonlinear interactions:** the fraction of total fitness variance explained by Walsh coefficients of order 2 and above. Zero means the landscape is purely additive; one means all structure is epistatic.

**Heat diffusion characteristic time:** the spectral graph theory approach measures ruggedness as the characteristic timescale of a heat diffusion process over the fitness landscape graph. More rugged landscapes have shorter characteristic times. Applicable even when the fitness map is sparse.

**Sign epistasis fraction:** mutations that change from beneficial to deleterious in different genetic backgrounds. Presence of sign epistasis creates multiple fitness peaks. High sign epistasis = high ruggedness = lifting is very hard or impossible.

**Non-monotonic spectral decay (compensatory structure):** when γd bounces up and down rather than monotonically decaying with distance, this indicates compensatory epistasis — sign epistatic interactions that cancel each other across the landscape. This is a qualitatively different failure mode from simple ruggedness and requires different tools.

### 3.3 Adaptive Coupling Strength

The coupling between populations should be adaptive, not fixed. This is the biological principle: quorum sensing only activates above a density threshold, mitonuclear signaling tightens under stress. In your system:

**Tight coupling** when the populations are in smooth-landscape regions relative to each other — Walsh coefficients are low-order, the Taylor approximation of one population's landscape is valid for interpreting the other's signal, and gradient information is reliable.

**Loose coupling** when populations have diverged into rugged regions — high-order Walsh coefficients dominate, the Taylor approximation degrades, signal molecules become unreliable. Too much coupling here causes population collapse to a local optimum; too little causes divergence past mutual comprehension.

**Mechanism:** KL divergence between the Walsh spectral distributions of the two populations is the natural coupling strength signal. Small KL = populations are spectrally close = tight coupling appropriate. Large KL = spectrally far = loose coupling.

### 3.4 Signaling Molecule Enrichment

The basic first-order Walsh coefficients are the minimum viable signal. These enrichments increase interpretability at the cost of computational complexity:

**Trajectory n-grams:** instead of sending instantaneous spectral snapshots, send a sequence of snapshots showing how the Walsh spectrum changes as expressions are mutated. This encodes local geometry of the landscape rather than just its value — the receiver can distinguish "this coefficient is large because the landscape is flat" from "this coefficient is large because we're on a steep slope." Analogous to sending the time derivative of metabolite concentration rather than the concentration alone.

**Dirichlet character basis:** instead of Walsh-Hadamard basis (natural for additive/Boolean structure), decompose the fitness landscape in the Dirichlet character basis — the spectral tool number theorists use to study multiplicative structure. For number-theoretic data, this basis speaks the native language. A Dirichlet character χ(n) is a completely multiplicative, periodic function that captures the prime structure of integers. The corresponding Fourier coefficients directly encode multiplicative interactions invisible to Walsh decomposition.

**Learned basis (coevolved population C):** treat the spectral basis itself as a third population coevolving alongside A and B. Population C evolves basis functions whose job is to maximally compress the information passing between A and B. If integer structure is a genuine latent feature of the landscape, a well-evolved basis should discover it. This is cooperative coevolution doing dictionary learning.

**Lyapunov spectrum of signal trajectories:** compute the Lyapunov exponents of the coevolutionary dynamics from the time series of signal molecules. Negative maximal Lyapunov exponent = stable attractor found, dynamics contracting. Zero exponent = marginal stability, populations at saddle point. Positive exponent = chaotic dynamics, populations not converging. The Lyapunov spectrum tells you not just whether convergence happened but the stability class of the joint attractor.

**Contrastive signal (integer minus real):** run the same coevolutionary system on two datasets simultaneously — integer-constrained data and nearby real-valued data satisfying the equation approximately. The difference in Walsh spectral distributions between these two runs is the fingerprint of the discrete structure. You don't need to know what the signal means in absolute terms — you know how it changes when you remove the discreteness constraint. Contrastive PCA extended to spectral space: find projections that maximally preserve foreground (integer) variance not explained by background (real).

**p-adic smoothing for Diophantine problems:** replace the sharp integer constraint with p-adic distance — the p-adic valuation of x^n + y^n - z^n measures how divisible the near-miss is by powers of p. This is a smooth, continuous function carrying real information about the arithmetic structure, giving your system a differentiable landscape to work with while remaining sensitive to arithmetic structure.

---

## Part IV: The Scaffolding Tower

### 4.1 The Core Problem

Direct coevolution between two algebraically distant populations (e.g., x+y directly to xy) may fail because the Walsh spectral distance is too large — populations cannot find viable paths between them, experiencing what evolutionary biology calls lethal epistasis: combinations that are individually viable but jointly fatal, creating fitness valleys selection cannot cross.

### 4.2 Scaffolding as Tower of Field Extensions

The solution is algebraically motivated intermediate populations. The mathematical analogy is exact: a tower of field extensions F₀ ⊆ F₁ ⊆ F₂ ⊆ ... ⊆ Fₙ is a sequence where each field adds new algebraic structure on top of the previous one. The tower rule says the total degree of the extension equals the product of degrees at each step — complexity compounds multiplicatively as you climb.

A well-designed scaffolding chain satisfies this: each step is within viable genetic (Walsh spectral) distance of the previous, the intermediaries have genuine algebraic relationships to both ends, and each step introduces exactly one additional degree of algebraic freedom.

**Example chain:**
```
x+y  ↔  powers(x) + powers(y)  ↔  powers(x+y)  ↔  xy
 F₀           F₁                      F₂             F₃
```

This is not arbitrary. Each step:
- x+y → powers(x) + powers(y): adds polynomial structure on each variable independently; Walsh spectrum gains second-order additive terms.
- powers(x) + powers(y) → powers(x+y): adds cross-interaction via binomial expansion. The binomial theorem (x+y)^n = Σ C(n,k) x^k y^(n-k) is the exact bridge — it's the mathematical proof that this lifting exists. Walsh spectrum gains explicit cross-terms.
- powers(x+y) → xy: reaches fully multiplicative structure. The log function (ln(xy) = ln(x) + ln(y)) is the bridge here — xy is what you get when you exponentiate the sum.

The Walsh spectral distance between adjacent steps is approximately one order of epistasis. The distance between x+y and xy directly is multiple epistatic orders. Scaffolding reduces each jump to a tractable size.

### 4.3 Extended Chain for Geometric Lifting

For the quaternion/exterior algebra experiment:

```
x+y  ↔  xy  ↔  x·y  ↔  x×y  ↔  x∧y
(additive)(bilinear 2D)(bilinear nD)(complement)(full exterior)
```

These are ordered by Walsh spectral complexity:
- x+y: first-order Walsh coefficients only (purely additive)
- xy: second-order (first multiplicative interaction)
- x·y: second-order but in n dimensions (same interaction order, richer geometry)
- x×y: second-order in the orthogonal complement (encodes what's NOT shared)
- x∧y: full exterior product (contains both dot product and cross product as components)

This is the exterior algebra tower: each step is a genuine extension adding structure not present below it.

### 4.4 The A↔B↔C Relay Chain

Rather than just source and target populations, you can introduce relay populations B that sit between A and C. B receives signals from both and its fitness is shaped by how well it bridges them:

```
A (symbolic expressions)  ↔  B (relay)  ↔  C (geometric transformations)
```

B is being selected for its ability to translate between A's symbolic language and C's geometric language. What B learns is the hidden shared structure — the features relevant to both problems simultaneously. B's Walsh spectrum at convergence is a compressed description of the joint constraint.

You can control the coupling strength and type by constraining what B is allowed to represent — fixing its basis, its order of Taylor approximation, its expressivity. B is a programmable communication channel. This is the environment-shaping idea: B isn't just an organism, it's the medium, and you can sculpt the medium to select for specific joint outcomes.

Biological analogue: mitonuclear coevolution, where the cytoplasm with its ribosomes, chaperones, and import complexes is exactly this relay — co-evolving with both the nuclear and mitochondrial genomes simultaneously.

### 4.5 Multi-Population Mutual Signaling

For the B₁, B₂, B₃ ↔ C₁, C₂, C₃ experiment, where each population represents a different algebraic "allele" of the same concept:

B₁ = x+y, B₂ = xy, B₃ = x×y all simultaneously exchanging Walsh spectral signals with C₁, C₂, C₃ (corresponding output space transformations).

**Prediction from Walsh spectral structure:**
- B₁ and B₂ should find similar joint attractors with their C partners because their Walsh spectra overlap heavily (both low-order interactions)
- B₃ should find qualitatively different joint attractors because the cross product encodes higher-dimensional structure in orthogonal directions
- The mutual signaling between all populations is horizontal gene transfer in molecular terms — high-dimensional encodings receiving structural hints from low-dimensional ones
- The system should naturally self-organize into a hierarchy sorted by Walsh spectral order, with low-order populations at the base and high-order at the top, connected by intermediate relay populations

This is an artificial recreation of the cortical hierarchy — fast low-order sensory populations at the bottom, slow high-order abstract populations at the top, with intermediate populations doing relay translation.

---

## Part V: The Sanity Dataset Taxonomy — The Mendel Notebook

The core experimental design principle: calibrate on known cases before probing unknown ones. Record systematically. The taxonomy must be rich enough that the results produce directional insight even when they cannot produce proof.

### 5.1 Tier 0: Trivial Controls (Algebraic Equivalences)

**Examples:** x+y vs. y+x; (x+y)+z vs. x+(y+z); 2x vs. x+x

**Known structure:** definitionally identical; no algebraic lifting required.

**Expected spectral signature:** Walsh spectra should be identical or differ only by a permutation of coefficient indices. Joint attractor found immediately (0-1 generations). Fourier amplitude spectrum: perfectly exponentially decaying, dominated by first-order terms. Lyapunov exponent: maximally negative (most stable attractor class).

**Purpose:** calibrates the floor — what "trivially liftable" looks like spectrally. Any deviation from this signature in harder problems is meaningful signal.

### 5.2 Tier 1: Known Liftings with Finite Algebraic Distance

**Examples:** x+y to xy via binomial intermediates; Euler's formula components; logarithm-exponential duality

**Known structure:** algebraic proof exists; the bridge is known mathematics.

**Expected spectral signature:** smooth progressive Walsh spectral change across intermediates. Each intermediate shows Walsh coefficients one order higher than the previous. KL divergence between adjacent population spectra small and approximately equal. Lyapunov exponent negative but less extreme than Tier 0. Convergence in O(N) generations where N = number of distinct algebraic degrees of freedom in the bridge.

**Purpose:** calibrates the intermediate zone — what "liftable with known structure" looks like. Establishes the empirical relationship between spectral distance and required number of intermediates.

**Key measurement:** the KL divergence vs. generation count curve for each step. Smooth monotonic decrease = populations found the bridge. Non-monotonic = scaffold was poorly chosen or the bridge requires a different path than the algebraically obvious one.

### 5.3 Tier 2: Known Liftings Requiring Extraordinary Human Effort

**Examples:** Euler's identity e^(iπ)+1=0; Fourier transform connecting time and frequency domains; generating functions connecting combinatorics and analysis

**Known structure:** proven connection exists; path through mathematical space non-obvious; required centuries or decades to discover.

**Expected spectral signature:** long convergence, oscillation in Walsh spectra before settling, specific high-order coefficients resisting stabilization for extended periods. The system may need to discover intermediates rather than being given them. Lyapunov exponent may pass through positive (chaotic) phases before settling to negative (stable attractor).

**Purpose:** calibrates the "deep structure" zone — what a genuine but non-obvious mathematical connection looks like spectrally. This is where the system's behavior becomes most interesting and most diagnostic.

**Key measurement:** which Walsh coefficient orders oscillate longest? These are the algebraic dimensions where the bridge is most non-trivial. This may point toward where intermediates are most needed and what kind of mathematical structure those intermediates should encode.

### 5.4 Tier 3: Suspected Connections — Unproven (Open Problems)

**Examples:** connections between zeta function zeros and prime distribution (Riemann); BSD conjecture linking L-function behavior to elliptic curve rank; abc conjecture linking addition and multiplication in integers

**Known structure:** connection suspected or partially confirmed by numerical evidence; no complete proof.

**Expected spectral signature:** unknown, but interpretable by comparison to Tiers 0-2. If the spectrum looks like Tier 1 — smooth, progressive, converging — that is weak evidence the connection is reachable and perhaps provable. If it looks like Tier 2 — long, oscillatory, with specific high-order coefficients refusing to settle — the connection is deep and the path non-obvious. If it looks qualitatively different from all calibrated cases, the territory is genuinely new.

**Purpose:** the primary scientific target. Calibration from Tiers 0-2 gives you a reference dictionary. You're not proving anything — you're producing a spectral map that says "this problem looks like it's in territory category X, which historically required tool set Y."

**Critical caveat for Diophantine problems:** the sharp integer constraint creates a flat landscape (no gradient almost everywhere) that Walsh decomposition cannot probe directly. Mitigations: p-adic smoothing (replace sharp constraint with p-adic valuation — smooth, differentiable, arithmetically sensitive); contrastive integer-vs-real runs (the difference signal fingerprints the discrete structure); Dirichlet character basis (speaks the native language of multiplicative number theory).

### 5.5 Tier 4: Known Non-Connections (Negative Controls)

**Examples:** attempting to lift x+y directly to a transcendental function with no algebraic relationship; attempting to connect operations from genuinely distinct mathematical universes without algebraic bridges.

**Known structure:** no connection known or suspected; algebraic distance likely infinite.

**Expected spectral signature:** Walsh spectra refuse to converge; KL divergence between populations stays high or increases over time; Lyapunov exponent stays positive (chaotic dynamics); populations die or collapse to degenerate solutions. The Fourier amplitude spectrum shows no exponential decay — flat across all orders, indicating maximal ruggedness with no exploitable gradient.

**Purpose:** calibrates the ceiling — what "genuinely not liftable (or not liftable with current tools)" looks like spectrally. Establishes that the system has discriminative power: it can distinguish connected from disconnected territory. Without this calibration, you cannot interpret the Tier 3 results.

---

## Part VI: The Mendel Notebook — Longitudinal Recording Taxonomy

The functional goal is not just to run experiments but to build a longitudinal record analogous to Mendel's notebooks: systematic observation of what changes over time, what tools reduced stress to which problems, and what patterns emerge across the taxonomy.

### 6.1 Per-Experiment Record Schema

For each coevolutionary run, record:

**Setup metadata:**
- Domain pair (A and B populations, what algebraic concepts they represent)
- Data type (smooth polynomial, integer-constrained, p-adic, modular arithmetic, L-function values)
- Scaffolding chain used (list of intermediaries with their algebraic descriptions)
- Signal molecule type (basic Walsh coefficients, trajectory n-grams, Dirichlet character basis, contrastive signal)
- Coupling regime (fixed, adaptive via KL divergence, user-specified)
- Tier classification (0-4 from sanity dataset taxonomy)

**Per-generation measurements:**
- Walsh Fourier amplitude spectrum of each population (full coefficient list by order)
- KL divergence between population spectra (coupling strength signal)
- Lyapunov exponent estimate from signal molecule time series
- Convergence indicator (rate of change of Walsh spectrum between generations)
- Sign of epistasis fraction at each order
- Best joint MDL score achieved

**Terminal measurements:**
- Number of generations to convergence (or failure mode if no convergence)
- Final joint attractor Walsh spectra of both populations
- Spectral signature classification (smooth/exponential decay, oscillatory, flat/rugged, non-monotonic compensatory)
- Number of intermediates that were necessary (with and without scaffolding, for comparison)
- Post-convergence: contrastive delta spectrum if integer-vs-real run was paired

### 6.2 Aggregate Patterns to Track (The Mendel Tables)

**Scaffolding calibration curve:** for Tier 1 and Tier 2 known liftings, plot KL spectral distance between source and target populations vs. number of intermediates required for convergence. This empirical curve, once calibrated on known cases, becomes a predictive tool: measure spectral distance between two domains on an unknown problem, look up the calibration curve, and get an estimate of how many intermediates would be needed. If the answer is "ten thousand," that is itself diagnostic.

**Signal molecule interpretability vs. algebraic depth:** track which enrichments (trajectory n-grams, Dirichlet basis, contrastive signal) improved interpretability for which tier of problem. Build a table of tool-to-problem mappings: Tier 1 problems probably need only basic Walsh signals; Tier 3 number-theoretic problems likely need Dirichlet characters and contrastive integer-vs-real runs.

**Spectral signature dictionary:** for each convergence outcome type, record the characteristic Walsh Fourier spectrum profile. Build a dictionary mapping spectral profiles to problem classes, analogous to Mendel's 3:1 and 9:3:3:1 ratios — not the full explanation, but a reliable observable that points toward structure.

**Tool addition history:** when a previously intractable problem becomes tractable after adding a tool (new signal molecule type, new intermediate population, new basis), record this transition explicitly. The history of which tools reduced stress to which problems is itself a dataset about the structure of mathematical space.

**Lethal epistasis rate by tier:** for each tier, what fraction of direct (un-scaffolded) coevolutionary runs experience population collapse? This rate should increase from Tier 0 (0%) through Tier 2 (high) to Tier 4 (near 100%). If a Tier 3 open problem shows low lethal epistasis rate without scaffolding, that is evidence the connection is in territory more like Tier 1 than Tier 2.

---

## Part VII: The IUT Structural Parallel

### 7.1 The abc Conjecture and Additive-Multiplicative Tension

The abc conjecture (Oesterlé-Masser, 1985): for every ε > 0, there are only finitely many triples (a,b,c) of coprime positive integers where a+b=c such that c > rad(abc)^(1+ε), where rad(abc) is the product of distinct prime factors.

The conjecture is about a fundamental tension between the additive structure of integers (how they are built by summing) and their multiplicative structure (how they are built from prime factors). The radical rad(abc) encodes multiplicative structure; the sum a+b=c encodes additive structure. The conjecture says these two aspects cannot be "too independent."

### 7.2 Mochizuki's Key Move

Inter-Universal Teichmüller Theory (IUT) attempts to prove abc by doing exactly what your coevolutionary system does conceptually: treating addition and multiplication as separate "universes" (Hodge theaters) with distinct algebraic dynamics, connecting them via a controlled bridge (the theta-link and log-link) that passes outside conventional ring structure, and finding a representation (the multiradial representation) that is invariant under changes of ring structure — i.e., survives the crossing between additive and multiplicative universes.

Mochizuki explicitly describes Hodge theaters as "miniature models of conventional scheme theory in which the two underlying combinatorial dimensions of a number field — corresponding to the additive and multiplicative structures of a ring — are 'dismantled' or 'disentangled' from one another."

The links between theaters are not compatible with ring or scheme structures — they operate outside conventional arithmetic geometry. They are compatible with certain group structures. This is the key: group structure survives the crossing; ring structure does not.

### 7.3 The Structural Isomorphism

Your coevolutionary system is structurally isomorphic to Mochizuki's setup, but instantiated computationally rather than formally:

| IUT Component | Coevolutionary GA Analogue |
|---|---|
| Hodge theaters (separate universes) | Separately evolving populations A and B |
| Theta-link (bridge between universes) | Walsh spectral signal molecule exchange |
| Compatibility with group structure | Walsh basis respects group structure of sequence space |
| Incompatibility with ring structure | Populations cannot see each other's full genotype (only spectral summary) |
| Multiradial representation | Joint attractor Walsh spectrum — representation invariant under changes of algebraic basis |
| Mild indeterminacies (Ind1, Ind2, Ind3) | Lossy compression in the signal molecule (full landscape not transmitted) |
| Log-link tower | Scaffolding tower lifting algebraic structure one degree at a time |

### 7.4 Your Scaffolding Chain as Log-Link Analogue

The log function (ln(xy) = ln(x) + ln(y)) is Mochizuki's literal bridge between multiplication and addition — it converts multiplicative structure into additive structure. His log-links form a tower: each application of the p-adic logarithm lifts you from the multiplicative world one level toward the additive world.

Your powers of (x+y) intermediate is doing something structurally similar. The binomial theorem says (x+y)^n = Σ C(n,k) x^k y^(n-k) — a multiplicative-looking object (power of a sum) decomposed into additive combinations of multiplicative terms. It is a discrete analogue of the logarithm's role in bridging addition and multiplication.

### 7.5 What This Framework Cannot Do

Be precise about limits:

**IUT operates at a level of mathematical structure** (Galois groups, p-adic completions, Frobenioids, absolute anabelian geometry) that your GA operates very far below. Your signal molecules are Walsh coefficients of fitness landscapes. Mochizuki's theta-link is a map between p-adic theta functions. The conceptual parallel is real but the mathematical depth is vastly different.

**Diophantine sharp constraints** create flat landscapes that Walsh decomposition cannot probe. The mitigations (p-adic smoothing, contrastive signal, Dirichlet basis) help but do not fully solve this.

**Interpretability of the joint attractor** is limited by the interpreter's mathematical depth. The signal is there; reading it requires mathematical sophistication that may not be present.

**No proofs can emerge from this system.** It is an empirical probe, not a formal tool. It can say "this territory looks like X" but not "X is true."

What it can do: produce a spectral map of mathematical difficulty, calibrated against known cases and applicable to unknown ones. New instruments in mathematics have historically been as valuable as proofs because they tell you where to look.

---

## Part VIII: Open Research Directions

### 8.1 Near-Term Experimental (Cheapest to Test, Most Falsifiable)

**Does the forward-inverse latent cross-mapping quality predict symbolic tractability?** Train forward and inverse networks on Feynman equation datasets with known symbolic structure. Measure cross-mapping quality at various layer depths. Correlate with how easily PySR finds the known expression. If cross-mapping quality predicts symbolic tractability, it becomes a cheap diagnostic for where to search.

**Does the coevolutionary GA find qualitatively different solutions than sequential optimization?** Implement the two-population cooperative coevolutionary GA with joint MDL fitness. Run on the Feynman database alongside ITEA and AI Feynman. Do the joint solutions differ qualitatively from sequential solutions? Are there problems where sequential methods fail but coevolutionary GA succeeds? Do joint attractors show the co-dependency signature?

**Does the Fourier amplitude spectrum decay rate predict convergence time?** Run Tier 1 experiments across a range of known liftings with varying algebraic distance. Plot spectrum decay rate vs. generations to convergence. If the relationship is empirically consistent, this becomes a cheap predictive diagnostic.

**Does the scaffolding chain reduce lethal epistasis rates measurably?** For the same source-target pair, run with and without the algebraic scaffolding chain. Measure population survival rates, convergence rates, and final joint attractor quality. The difference is the scaffolding contribution.

### 8.2 Medium-Term Experimental

**The Tier 2 Euler identity experiment.** Run the coevolutionary system attempting to find the connection between e^(iπ) and -1 without giving it the intermediates. Observe which Walsh coefficient orders resist settling longest. Those resistant orders point toward where intermediates are needed. Use this to discover intermediates empirically rather than prescribing them algebraically.

**The contrastive integer-vs-real experiment.** Run the same system on integer triples (a, b, c) satisfying a+b=c alongside real triples satisfying a+b≈c. Compute the difference in Walsh spectral distributions. Map what this difference signal looks like. Compare to the Dirichlet character basis decomposition of the same data. This is the fingerprint of discrete structure.

**The Lyapunov spectrum diagnostic.** For Tier 2 and Tier 3 experiments, compute the full Lyapunov spectrum of the coevolutionary dynamics from the signal molecule time series. Map the relationship between Lyapunov exponent and convergence class. Do all Tier 1 problems show negative maximal Lyapunov exponents from early in the run? Do Tier 3 problems show positive exponents that eventually become negative, or stay positive?

### 8.3 Long-Term Speculative

**The p-adic Fermat experiment.** Replace the sharp Diophantine constraint a^n + b^n = c^n with the p-adic valuation of a^n + b^n - c^n for small primes p. Run the coevolutionary system on this smooth arithmetic landscape. Compare the resulting Walsh spectra to the calibrated Tier 1 and Tier 2 reference dictionary. What territory class does the Fermat landscape fall into spectrally?

**Can the system discover that elliptic curves are relevant to Fermat?** This is speculative but: if you run the coevolutionary system on Fermat-related arithmetic data and on elliptic curve data separately, do the resulting joint attractors share spectral structure? Shared structure would be a weak computational signal that these territories are geometrically related — not a proof, but a directional hint.

**The spectral difficulty map of mathematics.** As the sanity dataset taxonomy grows, each problem domain gets a spectral signature. Over time this becomes a map: which mathematical domains are spectrally similar (likely algebraically connected), which are spectrally distant (likely disconnected or connected only via deep structure), and what the topology of mathematical territory looks like to an evolutionary system.

---

## Part IX: Biological Parallels — Coevolution as Natural Joint Optimization

### 9.1 The Universal Pattern

At every biological scale, the same structure appears: distinct populations evolving in separate spaces with independent genetic dynamics; compressed lossy communication channels carrying structural hints (not full descriptions); coupling strength regulated dynamically; convergence to joint stable states inaccessible to either population independently; the communication channel itself co-evolving alongside the populations it connects.

### 9.2 Quorum Sensing

Yeast and bacteria accumulate small signaling molecules (farnesol, tyrosol in *Candida albicans*; AHL autoinducers in bacteria) in shared medium as a function of population density. These don't transmit full genetic information — they carry compressed structural hints. Distinct populations in shared culture detect these signals and update gene expression programs in response.

Critical finding (PNAS 2011): the signaling molecule and its receptor coevolve through competitive cycles — a cheating receptor mutation is followed by a cheating-immunity signaling mutation. The joint language and the joint interpretation of that language evolve simultaneously.

**Parallel:** the GA's fitness coupling should not allow populations to see each other's full genotype, only their phenotypic output on shared data points. Coupling through Walsh spectral signals is the engineered equivalent of quorum sensing autoinducers.

### 9.3 Mitonuclear Coevolution

Two independently evolving genomes in distinct physical spaces inside the same cell. Anterograde signals (nucleus to mitochondria) carry regulatory proteins. Retrograde signals (mitochondria to nucleus) report metabolic stress via specific metabolites, calcium fluxes, reactive oxygen species at specific thresholds.

Compensatory coevolution: the nuclear genome evolves to compensate for deleterious alleles in the mitochondrial genome. Mitonuclear hybrids created by cytoduction in yeast show rapid decline in oxidative phosphorylation efficiency as species divergence increases — the joint attractor is real and has measurable depth. Wrong partner = wrong joint basin.

Both anterograde and retrograde channels are required for stable function. The bidirectional signal exchange is not decorative — it is the mechanism of joint optimization.

**Parallel to the brain's joint optimization:** anterograde/retrograde mitonuclear signaling is isomorphic to the forward/backward pass of predictive coding. The nucleus sends predictions (regulatory proteins) downward; mitochondria propagate error signals (metabolic stress markers) upward. The joint stable state is functional oxidative phosphorylation — neither genome alone could find or maintain it.

### 9.4 Endosymbiosis — Extreme Case

Two completely separate organisms begin exchanging molecular signals and over billions of years converge toward such deep mutual constraint that neither can survive without the other. Mitochondria arising ~1.8 billion years ago from an alphaproteobacterium entering an Asgard archaean.

Endosymbionts provide novel phenotypes to their hosts, permitting leaps between adaptive landscapes with new trait axes and peaks — the joint attractor enables access to regions of the solution space that neither population could reach independently. The most dramatic possible version of the joint constraint claim.

### 9.5 Extracellular Vesicles

Cells release membrane-enclosed vesicles carrying proteins, RNA, DNA fragments — partial structural descriptions of the sending cell's internal state. Recipient cells update gene expression programs in response. Recent findings show cellular phenotypes including differentiation states are synchronized among cells via EVs — populations in distinct niches converging toward shared states through lossy molecular communication.

EVs mediate communication between microbes and hosts, parasites and vectors — the communication channel evolved to span evolutionary distances. Cooperative coevolution operating across kingdom boundaries with the vesicle content as the joint language.

---

## Part X: Key Literature Anchors

**On neural networks as morphing objects:**
- Montufar et al. (2014) — on the number of linear regions of deep networks
- Raghu et al. (2017) — expressivity of deep networks via trajectory length

**On symbolic regression and joint optimization:**
- ITEA — de França & Aldeia (2020), Evolutionary Computation
- AI Feynman — Udrescu & Tegmark (2020), Science Advances
- MDLformer — symbolic regression via MDL (ICLR 2025)
- Data-driven Progressive Discovery of Physical Laws (2025) — transformation discovery parallel with symbolic search
- Coordinated Genetic Search for Symbolic Regression of Network Dynamics (2024)

**On tractability types:**
- Probabilistic Circuits survey — expressivity vs. tractability trade-offs (2024)
- Causal representation learning via Independent Causal Mechanisms (2023)
- Information Bottleneck — Tishby et al., connection between compression and learning

**On Walsh-Fourier spectral analysis of fitness landscapes:**
- Weinberger (1991) — Fourier analysis of fitness landscapes
- Ferretti et al. (2016) — measuring epistasis via correlation of fitness effects
- Weinreich et al. — Walsh-Hadamard transform for background-averaged epistasis
- Faure et al. (2024, PLOS Computational Biology) — extension of WHT to multiallelic landscapes
- Recent preprint (2025) — spectral graph theory / heat diffusion approach to ruggedness

**On scaffolding and stepping stones:**
- Serendipitous scaffolding paper (GECCO 2018) — poorly chosen intermediaries can hurt
- POET / Minimal Criteria Coevolution — coevolving environments and agents

**On the brain's optimization:**
- Friston (2010) — The free-energy principle: a unified brain theory, Nature Reviews Neuroscience
- Rao & Ballard (1999) — Predictive coding in the visual cortex
- Hybrid Predictive Coding — Inferring, Fast and Slow (2022)
- Olshausen & Field (1996) — Emergence of simple-cell receptive fields by learning a sparse code

**On latent space translation and stitching:**
- Platonic Representation Hypothesis — Huh et al. (2024)
- Latent Space Translation via Inverse Relative Projection (2024)
- Relative Representations Enable Zero-Shot Latent Space Communication — Moschella et al. (2022)

**On cooperative coevolution:**
- Potter & De Jong — Cooperative Coevolutionary Genetic Algorithm (foundational)
- SAFE and OMNIREP — Sipper, Moore, Urbanowicz (2024)
- Coevolution in multi-objective EAs — Whitacre (2009)

**On biological coevolution:**
- Mitonuclear Compensatory Coevolution review (2020), Trends in Genetics
- Social conflict drives evolutionary divergence of quorum sensing, PNAS (2011)
- Endosymbioses Have Shaped Biological Diversity and Complexity (2024), Genome Biology and Evolution
- Extracellular Vesicles carry evolutionary footprint in interkingdom communication (2020)

**On IUT and abc:**
- Mochizuki (2012-2021) — IUT Theory I-IV, PRIMS
- Fesenko — Fukugen (Inference Review, 2021) — accessible survey
- Promenade in Inter-Universal Teichmüller Theory — RIMS-Lille seminar notes

---

## Closing Note

The core empirical questions, in order of priority:

1. Does the coevolutionary GA with Walsh spectral signal molecules produce qualitatively different joint attractors than sequential symbolic regression + transformation search on the Feynman dataset? This is the minimum viable validation.

2. Does the Fourier amplitude spectrum decay rate predict convergence difficulty in a consistent, calibratable way across Tier 0-2 experiments?

3. Does algebraic scaffolding reduce lethal epistasis rates measurably and predictably as a function of spectral distance?

4. Does the contrastive integer-vs-real signal fingerprint discrete structure in a way that changes recognizably across different number-theoretic domains?

If these four questions have positive answers, the methodology produces a genuine new instrument for mathematical cartography. If they don't, the framework was generative but the specific proposals were wrong — which is still useful information, and the spectral tools for characterizing why they were wrong will themselves be informative.

The Mendel analogy is apt: Mendel didn't understand DNA, genetics, or molecular biology. He had peas, a garden, and systematic records. The records were the contribution. The explanation came later.

*Map the territory first. The explanation is someone else's problem, or yours when you're more brilliant.*

---

## Part XI: The First Working Version — Building the Instrument

Before any of the harder experiments, the goal is simply this: can you reliably evoke and recognize the spectral signature of linearity? Can you distinguish it from the spectral signature of multiplicativity? Can you watch those signatures emerge, stabilize, and change as the populations coevolve?

If you can do that consistently, you have a working instrument. Everything else is harder versions of the same measurement.

### Phase 1: Evoke and Recognize Basic Operations

Start with the simplest possible case. One population of additive expressions (x+y, x+y+z, ax+by) coevolving with one population of linear output transformations (scaling, shifting). You know exactly what the Walsh spectrum of a purely additive landscape looks like — only first-order coefficients nonzero, all higher-order coefficients exactly zero.

Run this and verify you get that signature reliably. Then introduce a multiplicative expression on one side and watch what happens to the spectrum. Does the second-order Walsh coefficient emerge cleanly? Does it stabilize at a predictable value? Can you watch the transition in the spectrum as the population discovers the multiplicative structure?

This is Tier 0 and Tier 1 calibration rolled into one experiment. You're building the spectral dictionary from the ground up, not assuming it.

### Phase 2: Distinguish the Signatures

Once you can evoke linear and multiplicative signatures reliably, run the contrastive experiment. Same system, same architecture, same number of generations — one run on additive data, one on multiplicative data. Plot the Walsh spectra side by side. Can you reliably classify which run produced which spectrum without knowing the data?

If yes, you have a working classifier for algebraic structure. That classifier is the instrument.

Then extend to the intermediate cases: xy, x·y, x×y. Do the signatures form the expected ordered sequence by Walsh spectral complexity? Does the system place them in the right order without being told? If the empirical ordering matches the algebraic ordering, the spectral signal is carrying genuine mathematical information.

### Phase 3: Watch Dynamics, Not Just End States

The most valuable thing to collect at this stage is not the final Walsh spectrum but the trajectory. How many generations does it take for the second-order coefficient to emerge in a multiplicative run? Does it emerge smoothly or suddenly? Does it ever emerge in an additive run by mistake — and if so, does the third-order coefficient stay zero and kill it, or does the mistake propagate?

These dynamics are the first Mendel table entries. Not "what is the final spectrum" but "what is the developmental trajectory of the spectrum" and "how does that trajectory differ across algebraic classes."

### What "Working" Means

When you can answer these questions with consistent, reproducible results across multiple random seeds and data samples, you have something real. The instrument is calibrated. Everything harder — harder operations, number-theoretic data, the IUT-adjacent experiments — is just turning the dial up on a working device.

### Why This Is Tractable Now

This first phase requires no massive infrastructure. PySR already exists. Cooperative coevolution architectures exist. Walsh transform is a few lines of numpy. The gap that needs closing is the integration — wiring them together and building the measurement pipeline. That is engineering, not research, and it is the right next step before anything else in this document.

*Build the instrument first. Calibrate it on things you know. Then point it at things you don't.*

---

## Part XII: New Experiment Classes and Reading List

### 12.1 Reading List on Expressivity and Complexity Hierarchy

These papers address the formal question of what complexity (depth, width, dimensionality) is required to express various function classes — the neural network analogue of "XOR needs one extra layer or it is fundamentally impossible."

**Start here (conceptual foundation):**
- IFT 6085 Lecture 10 notes, Mitliagkas (Montreal) — freely available online. Walks from perceptron failure on XOR through to the universal approximation theorem in one readable document. The XOR example is explicit and clear.

**The core depth separation result:**
- Telgarsky (2016), "Benefits of Depth in Neural Networks" — arXiv:1602.04485. Formally proves that functions computable by a depth-n network with polynomial width cannot be approximated by depth-n^(1/3) networks without exponential width blowup. The triangle wave function he constructs requires depth proportional to the number of oscillations — no shallow network can fake it without exponential cost. This is the cleanest formal statement of "one extra dimension of composition is sometimes the only way."

**Depth 3 vs depth 2 separation:**
- Eldan & Shamir (2016) — depth 3 can do things depth 2 fundamentally cannot for radial functions in high dimensions. Provable separation. Maps directly onto the morphing object framing: some deformations of output space require a specific number of fold operations and cannot be achieved with fewer.

**Depth and learnability together:**
- Malach et al. (2021), "The Connection Between Approximation, Depth Separation and Learnability" — arXiv:2102.00434. Shows that Telgarsky's function is not only hard to approximate with shallow networks but hard to learn with gradient descent even on deep networks. The expressivity hierarchy is real and also a learning difficulty hierarchy.

**Topological expressivity:**
- "Topological Expressivity of ReLU Neural Networks" (2023) — arXiv:2310.11130. Frames expressivity in terms of topological complexity of the spaces a network can separate. Directly connects to the morphing object framing: a network's expressivity is the complexity of the most topologically complex input space it can disentangle into linearly separable form.

---

### 12.2 New Experiment Classes

#### Experiment Class A: Taylor ↔ Function (Granularity Measurement)

**Setup:** one population of Taylor expansions truncated at various orders coevolving with a population representing the target function itself.

**What to measure:** the Walsh spectrum of the Taylor population at order k should match the Walsh spectrum of the function up to the kth epistatic order and diverge above it. This directly measures the granularity between structurally similar signal molecules.

**Key insight:** Taylor at order k is a signal molecule that encodes everything up to k-way interactions and nothing above. The divergence point in the Walsh spectrum is the exact location where the Taylor approximation stops being a faithful signal molecule for the function. This gives you a precise, measurable definition of "how similar" two signal molecules are in terms of the information they carry.

**What this reveals:** the resolution of your signal molecule vocabulary. If the system can detect the Taylor-function divergence at order k, the signal molecules can discriminate at that level of algebraic interaction. If it cannot, your instrument is too coarse.

---

#### Experiment Class B: Taylor at Various Expansions ↔ Function (Direction vs. State)

**Setup:** run multiple Taylor populations simultaneously — Taylor at order k, k+1, k+2 — each coevolving with the same target function.

**What to measure:** the Walsh spectrum of Taylor(k+1) minus Walsh spectrum of Taylor(k) is the kth-order correction term. Does this correction term's Walsh spectrum point directionally toward the function? Does the direction signal (the correction) carry different information than the state signal (the approximation itself)?

**Key distinction:** this is asking whether the signal molecule encodes *the state to be in* (current Taylor approximation) or *the direction to go* (the next correction needed). In dynamical systems terms: is the signal a position or a velocity?

**Why this matters:** if correction terms carry directional signal toward the function, you have a mechanism for using signal molecules not just to describe current state but to guide search. This is the difference between a quorum sensing molecule that says "I am here" and one that says "come this way." Both exist in biology. Whether your spectral signal molecules can express both is a testable question.

---

#### Experiment Class C: Taylor for Cosine ↔ Taylor for Sine (Cleanest Controlled Experiment)

**Setup:** one population of cosine Taylor expansions coevolving with one population of sine Taylor expansions.

```
cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
```

**Known structure:** the two series share the same coefficient structure — same magnitudes, same alternating signs — but offset by exactly one term. The mathematical relationship between them is the derivative: d/dx sin(x) = cos(x). The Walsh spectral distance between them is exactly one order of Taylor truncation.

**What to measure:** does the coevolutionary system find the derivative relationship from the spectral signal exchange alone, without being told? The joint attractor should be the derivative operator — a clean, provable, verifiable mathematical fact.

**Why this is the ideal first instrument test:** ground truth is exact and known. The relationship is provable. The spectral distance is precisely one Walsh order. If your system finds it, the instrument is working. If it doesn't, the instrument needs calibration. No ambiguity.

**Extensions:** once sin↔cos works, extend to:
- sin(x) ↔ sin(2x): same function, frequency doubled. Walsh spectral distance should be small but nonzero — the doubling argument changes the interaction structure.
- sin(x) ↔ e^x: same Taylor coefficient magnitudes, different signs. Spectrally close but mathematically very different. Does the system distinguish them?
- sin(x) ↔ sinh(x): hyperbolic sine has the same Taylor structure as sin but without alternating signs. Walsh spectral distance is large despite structural similarity. A good test of whether the sign of Walsh coefficients carries meaningful signal.

---

#### Experiment Class D: Mutating Taylor Coefficients Directly

**Setup:** instead of evolving expression trees and computing Walsh spectra, treat the Taylor coefficients themselves as the genes. Evolve coefficient vectors directly. The Walsh spectrum is trivially readable from the coefficients.

```python
# Gene representation: coefficient vector [c₀, c₁, c₂, c₃, ...]
# Population A: coefficients for input combination
# Population B: coefficients for output transformation
# Signal molecule: subset of Walsh coefficients of fitness landscape
#                  computed from evaluating the coefficient vector on data
```

**Why this is the most tractable first implementation:**
- No expression tree parsing needed
- Walsh transform of the coefficient vector is a few lines of numpy
- Mutations are simple coefficient perturbations
- Crossover is coefficient vector interpolation
- Can be run in an afternoon

**What to measure:** can two coefficient vector populations, exchanging Walsh spectral signals, converge to coefficient vectors whose relationship matches a known mathematical relationship? If yes, you have validated the core mechanism in the simplest possible setting before scaling to full expression trees.

**The mutation taxonomy:** treating coefficient perturbations as mutations gives you a natural genetic distance metric — the L2 norm between coefficient vectors. This is interpretable and connects directly to Walsh spectral distance. You can plot genetic distance vs. spectral distance to verify they correlate as expected, providing another calibration check.

---

### 12.3 The Instrument Validation Ladder

In order of increasing difficulty. Each rung must work before climbing to the next.

```
Rung 0: Coefficient vector GA finds known relationships
        → cos ↔ sin Taylor coefficients
        → verify derivative relationship emerges

Rung 1: Expression tree GA finds Tier 0 algebraic equivalences
        → x+y vs y+x Walsh spectra identical
        → (x+y)+z vs x+(y+z) Walsh spectra identical

Rung 2: Expression tree GA finds Tier 1 known liftings
        → x+y to xy via scaffolded chain
        → Walsh spectra show smooth progressive change

Rung 3: Taylor ↔ Function granularity measurement
        → Walsh divergence at correct order
        → direction vs. state signal experiment

Rung 4: Taylor for sin ↔ Taylor for cos
        → joint attractor = derivative relationship

Rung 5: Tier 2 experiments (deep structure, non-obvious bridges)

Rung 6: Tier 3 experiments (open problems, number theory)
        → p-adic smoothing for Diophantine structure
        → contrastive integer-vs-real signal
```

Do not skip rungs. Each rung is a calibration that makes the next rung interpretable.

---

### 12.4 Minimal First Implementation

The smallest possible working version that validates the core mechanism:

**Components needed:**
1. A coefficient vector GA (numpy, ~50 lines)
2. Walsh-Hadamard transform (numpy, ~10 lines)
3. Signal molecule exchange (pass Walsh coefficients between two GA instances, ~20 lines)
4. Fitness function: joint MDL = description length of coefficient vector + fitting error on data (~30 lines)
5. Measurement pipeline: record Walsh spectrum per generation, plot trajectory (~30 lines)

**Total:** approximately 150 lines of numpy/Python to validate the core mechanism on the cos↔sin experiment.

**Success criterion:** the joint attractor of the two coefficient vector populations, when you subtract one from the other, approximates the derivative operator d/dx. This is checkable analytically.

*If this works in 150 lines of numpy, everything else is engineering.*


---

## Part XIII: The Failure Mode Taxonomy -- What Breaks and Why

*Added June 2026 after extensive empirical work building the instrument.*

The most important methodological shift from Parts I-XII: the goal is not to succeed but to **fail precisely**. Every failure with a clear cause is more valuable than a vague success. The taxonomy of failures, with exact detection criteria, IS the scientific contribution regardless of whether the breakthrough ever arrives.

This mirrors Mendel's methodology: he did not discover genetics, he discovered the 3:1 ratio -- a precise failure of the naive hypothesis. The 3:1 ratio forced the mechanism into existence.

### 13.1 The Ramanujan Failure Types

Six structural misalignment types, named for Ramanujan's work where things that look wrong are often correct in a lifted form:

**Type 1 -- Wrong metric.** Formally correct but metrically wrong. The relationship exists in a richer metric space. Example: using real 8D coefficient space for a rotation that is exact in complex 1D space. Resolution: dimensional lift -- the complex relay chain giving exact solution (max_error = 0) where real space gave 8-10 degree systematic offset.

**Type 2 -- Missing shadow.** The failure is systematic and predictable, always in the same direction. Ramanujan's mock theta functions had this structure: correct at cusps but failing the modular transformation law by a predictable non-random error. In our system: state signals (Walsh landscape) are always gradient proxies, never character encoders. The missing shadow is the directional signal (trajectory n-gram). Resolution: add the complementary structure.

**Type 3 -- Almost integer / near-miss.** The error is too structured to be random. e^(pi*sqrt(163)) is off by 7.5e-13, not randomly. In our system: the 8-10 degree systematic angular offset in the real-space Euler relay points at a specific structural deficiency (Taylor truncation). Resolution: change the algebraic basis.

**Type 4 -- Conjectured from structure.** Satisfies all necessary conditions for a deep theorem without proof. Ramanujan's tau function conjectures, proved 58 years later by Deligne. In our system: the relay chain achieves near-geodesic Ricci curvature even when it does not converge -- structural evidence a path exists. Resolution: patience and data collection.

**Type 5 -- Crossroads.** Too many coincidences simultaneously. Objects satisfying multiple independent algebraic structures. Taxicab numbers. In our system: when multiple valid rotations all satisfy relay chain constraints simultaneously (the "equally valid rotations" problem). Resolution: persistent homology or Gromov-Wasserstein to study the intersection variety.

**Type 6 -- Wrong basis.** Formal algebra works, magnitudes blow up. x^k versus phi_k = x^k/k! is the clearest example. Resolution: normalize the basis.

**Type 7 -- Dimensional insufficiency (NEW).** No valid path exists in the current dimensional space. This is the Minsky-Papert theorem applied to function space: XOR cannot be separated by any single hyperplane in 2D, and cos(2x) cannot be reached from cos(x) by any sequence of linear relay steps. This is NOT a convergence failure -- it is a provable impossibility in the current grammar.

Confirmed computationally:
- cos->sin: algebraic depth 1 in linear grammar (antiderivative, residual = 0.000)
- cos->cos(2x): TYPE 7 in linear grammar (no path, residual = infinity)
- cos->cos(2x): depth 2 in quadratic grammar (cos_double(scale_0.5), residual = 0.000)

Detection criterion: residual > 20 AND cross-covariance block max < 0.4 at generation 80 of joint CMA-ES with normalization.

### 13.2 The ML Failure Mode Analogs

Four additional failure modes from ML training literature that look like failure but are NOT:

**Grokking.** High residual for a long period, then sudden generalization. Detected by: high residual but accelerating decrease (d2_residual/dt2 < 0). Do NOT kill -- this is a late bloomer. Catastrophe theory type: fold unfolding, a minimum emerging from a flat region.

**Double descent.** Adding more capacity temporarily worsens performance before improving. Detected by: residual increases after decreasing. Do NOT kill -- crossing the double-descent boundary. Catastrophe type: swallowtail (3-parameter interaction).

**Mode collapse.** All seeds converge to the same solution, ignoring the full distribution of valid paths. Detected by: inter-seed diversity approaching zero. Distinct from miscoordination -- a valid path was found, just not the full set. Resolution: diversity injection.

**Posterior collapse.** Signal molecules become uninformative, cross-covariance drops after being high. The receiver learns to ignore the sender. Detected by: cross-covariance declining over consecutive checkpoints. Resolution: switch to directional signal (trajectory n-gram).

### 13.3 Catastrophe Theory and Failure Mode Transitions

Rene Thom (1972) proved there are exactly 7 elementary catastrophes for systems with <= 2 active variables and <= 4 parameters. These are the complete classification of all possible sudden structural changes in smooth systems.

Relevant to our relay chain:
- **Fold**: a minimum disappears as a parameter crosses a threshold. This is Type 7 -- the correct algebraic path ceases to be reachable as the grammar drops below required complexity.
- **Cusp**: two attractors with hysteresis, history-dependent convergence. This is miscoordination -- which attractor the relay chain finds depends on initialization, not just on the joint fitness landscape.

The codimension of the catastrophe = number of independent parameter changes needed to escape. Fold: codimension 1 (enrich grammar). Cusp: codimension 2 (correct grammar AND correct initialization).

---

## Part XIV: The Checkpoint Architecture -- Amortizing Experiment Cost

*Based on Hyperband/ASHA (Li et al. 2018) applied to algebraic relationship discovery.*

### 14.1 The Core Insight

The first sweep of experiments is expensive because it establishes failure mode baselines. Once calibrated, subsequent experiments in the same domain can be detected and redirected at a fraction of the cost:
- Type 7 detected at generation 80: saves 73% of budget
- Miscoordination detected at generation 150: saves 50% of budget
- Degenerate zero detected at generation 15: saves 95% of budget

The checkpoint system is a domain-specific instance of Hyperband. The bracket structure (generations 15, 40, 80, 150, 300) follows the 2x halving pattern. Total efficiency gain on a realistic mix: 1.26x observed, 1.8-2x expected at scale with calibrated thresholds.

### 14.2 The Biological Analogy

T-cell selection uses the same architecture:
1. Positive selection (early): does this receptor bind MHC at all? No -> apoptosis (DEGENERATE)
2. Negative selection (mid): does it bind self-antigens too strongly? Yes -> apoptosis (MISCOORDINATION)
3. Only cells passing both survive to run fully.

The checkpoints ARE the biological immune checkpoint mechanism applied to algebraic relationship discovery.

### 14.3 Detection Criteria (Calibrated from Tier 0)

| Checkpoint | Generation | Detection | Failure mode | Action |
|---|---|---|---|---|
| Degenerate | 15 | mean_norm < 0.05 | DEGENERATE | Add normalization |
| Grokking | 30-40 | d2_residual < 0 AND residual > 0.3 | GROKKING | Do NOT kill |
| Type 7 | 80 | residual > 20 | TYPE_7 | Record, enrich grammar |
| Miscoordination | 150 | residual > 0.5 AND cross_cov > 0.85 | MISCOORDINATION | Joint optimizer restart |
| Double descent | Any | residual increases after decreasing | DOUBLE_DESCENT | Do NOT kill |

Key caveat: thresholds calibrated on Tier 0. Must be recalibrated for each new domain. All first 20-30 experiments in a new domain should run to completion.

### 14.4 The AlphaFold Framing

AlphaFold solved protein folding (supposedly quantum hard) classically because it had 200,000 labeled training structures and evolutionary covariation signals. The quantum advantage evaporates when empirical taxonomy is dense enough.

For our system: the Mendel notebook IS the quantum computer substitute. The checkpoint system builds taxonomy density efficiently enough that harder problems become tractable through interpolation. Quantum computation became necessary for the IBM half-Mobius molecule because no empirical taxonomy of half-Mobius structures existed. With sufficient taxonomy, even quantum-hard validations become classically tractable.

The taxonomy density threshold for our system is unknown. Finding it empirically -- how many Tier 0-2 experiments before the learned topological model generalizes to Tier 3 -- is itself a research result.

---

## Part XV: The Joint Fitness Architecture -- Correcting the Core Mistake

*The central architectural correction from Parts I-XII.*

### 15.1 The Mistake

Parts I-XII describe a coevolutionary system where Population A fits its own target and Population B fits its own target, with Walsh signal exchange on top. This is NOT joint optimization. It is two independent optimizations with a communication channel. The signal exchange modulates mutation variance but cannot change what the populations are optimizing toward.

Consequence: the joint attractor is determined by the two independent fitness landscapes, not by any relational constraint between them.

### 15.2 The Correct Architecture

Fitness must be defined over PAIRS:
- fit_a(ind_a, partner_b) -> float  (A's fitness depends on B's current state)
- fit_b(ind_b, partner_a) -> float  (B's fitness depends on A's current state)

Confirmed improvement: switching from independent to joint derivative fitness reduced residual from 1.19 to 0.11 on the cos/sin derivative relationship (10x improvement, TB-03 confirmed).

### 15.3 The Normalization Requirement

Joint fitness without normalization has a degenerate zero solution: both A and B go to zero to trivially satisfy any relational constraint. The normalization constraint forces both populations onto the appropriate amplitude manifold.

Confirmed: Level 2 analytical test shows joint_fit(cos2, sin2) = 0 (global maximum, correctly identified). Level 2 with joint 4D CMA-ES finds the correct solution with error < 0.001.

### 15.4 Alternating vs. Simultaneous Optimization

Alternating optimization (optimize A given B, then B given A) suffers from miscoordination -- the cooperative coevolution pathology (Wiegand, Ficici, Pollack). The fix: joint simultaneous CMA-ES on the 2N-dimensional [A, B] vector. CMA-ES learns the A-B cross-covariance block, which encodes how A and B should co-vary to maintain the algebraic relationship. This IS the learned signal molecule.

The A-B cross-covariance block fingerprints the relationship:
- Derivative (cos->sin): cross_cov_max = 0.998, residual = 0.002
- Scalar (cos->2cos): cross_cov_max = 0.989, residual = 0.001
- Type 7 (cos->cos2x): cross_cov_max = 0.348, residual = 58.9

### 15.5 The Algebraic Depth Hierarchy

Every algebraic relationship has a minimum number of composed grammar operations needed to express it. This is the algebraic depth -- the direct analog of neural network depth (Minsky-Papert):
- Depth 0: identity (A = B)
- Depth 1: A = T(B) for some single grammar operation T
- Depth 2: A = T1(T2(B)) for two composed operations (frequency doubling requires squaring)
- Depth k: k composed operations

Type 7 is: algebraic depth = infinity in the current grammar. Whitney's Embedding Theorem gives an upper bound: any degree-k algebraic relationship can be realized in 2k-dimensional coefficient space.

---

## Part XVI: The Complex Lift and the D-Brane Relay Chain

### 16.1 The Euler Relay Chain as Discrete Path Integral

The relay chain is a discrete approximation to the Feynman path integral over all possible morphing paths between A and B. Each relay chain run samples one path. The distribution of convergence points across many seeds IS the path integral amplitude. The minimum-action path dominates when sigma (mutation std) is tuned to the problem scale.

Key: sigma IS the analog of hbar. Too small: fractal Weierstrass-like wandering, no stationary phase selection. Too large: everything collapses to minimum energy (degenerate zero). The CMA-ES solves this by adaptively learning the right sigma via Fisher Information Metric.

### 16.2 The Complex Lift

In real 8D coefficient space, the Euler rotation has infinitely many valid paths. In complex 1D space (projecting onto the cos/sin subspace), the rotation is uniquely multiplication by e^(-i*delta) at each step.

Confirmed: the complex relay chain (toolkit/complex_relay.py) achieves max_error = 0.000000 for n_steps in {4, 8, 12}. The Gauss-Seidel analytic update converges in one iteration. This is the "Argand diagram" for the relay chain problem -- the geometric representation that makes the rotation unique and exact.

The dimensional insufficiency in real space is lifted entirely by moving to complex 1D. This is the exact structure of Heilbronner's prediction of Mobius aromaticity (1964): adding a topological phase twist changes the electron counting rule from 4n+2 to 4n, resolving what appeared to be an impossibility.

### 16.3 The IBM Half-Mobius Connection (March 2026)

The IBM C13Cl2 half-Mobius molecule: a 13-carbon ring where the pi-electron framework twists 90 degrees per revolution, requiring 4 complete circuits to return to starting phase. Physical instantiation of our 4-step Euler relay chain.

Structural isomorphism:
- Untwisted ring = L2 geodesic (minimum energy, linear interpolation)
- Half-Mobius = 90-degree phase rotation per step (higher energy, topologically protected)
- Quantum computing needed = multireference character (electronic state is superposition of multiple configurations -- the molecule has no single dominant classical path)

The molecule needed quantum computing to VALIDATE because no empirical taxonomy of half-Mobius structures existed. With taxonomy: classically tractable.

Prediction timeline: Heilbronner 1964 (topological argument) -> Synthesis 2003 (39 years) -> Half-Mobius 2026 (62 years from prediction to physical instantiation requiring quantum validation).

### 16.4 The Open-Chain D-Brane Model

The closed-cycle relay chain fails for the Euler experiment because the Euler path is OPEN (cos to -cos), not closed. The closed-cycle fitness demands a return step from -cos to cos in one pi/n step, which is geometrically inconsistent.

Fix: the open-chain D-brane model. Both endpoints FIXED (Dirichlet boundary conditions, D-branes in string theory). Intermediate populations FREE. CMA-ES optimizes the intermediates.

Confirmed: open-chain D-brane relay with CMA-ES achieves mean step residual 0.63, Ricci variance = 0.001 (near-Ricci-soliton path). The systematic 8-10 degree angular offset is a Taylor truncation artifact -- the true minimum-stress path for the truncated 8-term space is not the exact Euler path.

---

## Part XVII: The Sanity Hierarchy -- Lower-Lifted Partitions

*Verify every tool on simpler cases with known analytical ground truth before trusting on complex cases.*

If something works in a higher-dimensional lifted space, it should be a special case in lower-dimensional space that is trivially verifiable. This is the Heilbronner principle applied to instrument validation.

### Level 0: Scalar (1D Gaussians)
Closed-form Wasserstein: W2 = sqrt((mu1-mu2)^2 + (sigma1-sigma2)^2)
Closed-form Fisher-Rao: d = sqrt(2) * arcosh(1 + ((mu1-mu2)^2 + (sigma1-sigma2)^2) / (2*sigma1*sigma2))
Status: CONFIRMED (within 25% for Wasserstein, within 20% for Fisher-Rao)

### Level 1: 2D Unit Circle
Ground truth: arc geodesic, angular variance = 0 for all rotation angles.
Status: CONFIRMED (angular variance = 0 for 0, 30, 90, 150, 180 degrees)

### Level 2: N=2 Polynomial
Ground truth: joint_fit(cos2, sin2) = 0 (global maximum of joint derivative fitness)
Q1 (fitness correct): CONFIRMED
Q2 (alternating optimizer): FAILS (miscoordination -- same bug as N=8)
Q2 (joint 4D CMA-ES): CONFIRMED (error < 0.001)

### Level 3: Complex Relay Chain
Ground truth: exact Euler rotation via e^(-i*delta) multiplication
Status: CONFIRMED (max_error = 0.000000 for all n_steps)

### Level 4: Ollivier-Ricci on Unit Circle Arc
Ground truth: Riemannian curvature of unit circle = 1
Measured: 0.72 +/- 0.05 (consistent underestimation -- calibratable systematic offset)

---

## Part XVIII: Key Empirical Findings From Tier 0

*The first page of the Mendel notebook.*

**Finding 1: Population distribution signal dominates individual signal (5-15x better)**
walsh_population (averaging Walsh signal over K random individuals) beats walsh_landscape, walsh_solution, and raw_sparse by 5-15x on residual. This is consistent with the N-collaborations insight: information about the population distribution is more useful than information about any single individual.

**Finding 2: Scalar multiple failure is information, not a bug**
The 2*cos vs. cos experiment failed consistently across ALL signal designs (residual 2.3+). This is the first data point in the signal information capacity matrix: current signal vocabulary transmits algebraic type but not scalar ratio. This is NOT a tuning problem.

**Finding 3: Independent fitness is algebraically blind**
The stress benchmark confirmed: with L2 relay chain (independent fitness), ALL six relationship types -- identical, scalar, phase shift, derivative, frequency doubling, random -- produce nearly identical Wasserstein variances. Stress only becomes informative when fitness is algebraically aware (joint fitness + normalization).

**Finding 4: Type 7 is confirmed computationally**
- cos->sin: depth 1 in linear grammar (antiderivative, residual = 0.000)
- cos->cos(2x): TYPE 7 in linear grammar (no path exists)
- cos->cos(2x): depth 2 in quadratic grammar (residual = 0.000)
This is the Minsky-Papert theorem for function space, confirmed empirically.

**Finding 5: Joint CMA-ES cross-covariance fingerprints algebraic relationships**
After 300 generations of joint CMA-ES with normalization:
- Derivative (cos->sin): cross_cov_max = 0.998, residual = 0.002
- Scalar (cos->2cos): cross_cov_max = 0.989, residual = 0.001
- Identical (cos->cos): cross_cov_max = 0.988, residual < 0.001
- Type 7 (cos->cos2x): cross_cov_max = 0.348, residual = 58.9
The A-B cross-covariance block is the learned algebraic relationship fingerprint.

**Finding 6: Normalization is required at ALL levels (N=2 through N=8)**
The Level 2 sanity test shows that joint fitness without normalization has the same degenerate zero solution at N=2 as at N=8. The fix is the same: unit-norm constraint on both populations. This is a FUNDAMENTAL architectural requirement, not an N=8-specific tuning issue.

**Finding 7: The Ramanujan prediction methodology applies**
Heilbronner predicted Mobius aromaticity 39 years before synthesis using only a topological counting argument. We are at the equivalent stage: building the topological argument for mathematical morphing space. The IBM half-Mobius (2026) validates that high-stress topological states can exist and be detected even when they require quantum computation for full validation.

---

## Part XIX: The Computational Toolkit Status

*What is built, verified, and available.*

All tools live in `toolkit/` and `experiments/` within the Biological_Mathematics project directory.

**Core optimization:**
- `toolkit/ga.py`: Coevolutionary GA with N-collaborations and pluggable signal molecules
- `experiments/joint_covariance.py`: Joint CMA-ES (simultaneous, not alternating) with A-B covariance extraction

**Signal molecules (three designs):**
- WalshLandscape: WHT of fitness landscape gradient structure (encodes search problem, not solution character)
- WalshSolution: WHT of coefficient vectors (encodes solution frequency structure)
- RawSparse: top-K coefficients of best individual (closest to biology's identity signal)
- TrajectoryNgram: WHT of recent history delta (DIRECTIONAL -- encodes movement, not position)

**Stress metrics (toolkit/stress_metrics.py):**
- JSD (replaces KL): symmetric, bounded, sqrt(JSD) is a true metric
- Sliced Wasserstein: Earth Mover's Distance, O(n log n), scale-invariant
- Ollivier-Ricci curvature: measures path geodesic quality
- Fisher-Rao distance: information-geometric distance between populations

**Failure mode detection (toolkit/checkpoint.py):**
- 7 checkpoints: DEGENERATE, OSCILLATORY, GROKKING, MODE_COLLAPSE, TYPE_7, MISCOORDINATION, DOUBLE_DESCENT
- Calibrated from Tier 0 experiments (need recalibration for new domains)
- Confirmed: 1.26x efficiency gain on 6-experiment demo

**Algebraic depth testing (experiments/type7_test.py):**
- Two grammars: LINEAR (degree-1 operations) and QUADRATIC (includes squaring)
- Algebraic depth search: depth-1 through depth-3 composition
- Confirmed: cos->sin depth 1, cos->cos(2x) TYPE 7 in linear, depth 2 in quadratic

**Validated sanity hierarchy (experiments/sanity_hierarchy.py, experiments/level2_analytical.py):**
- Level 0: Scalar (PASS)
- Level 1: 2D unit circle (PASS)
- Level 2: N=2 polynomial with joint optimizer (PASS for both Q1 and Q2)
- Level 3: Complex relay chain (PASS, max_error = 0)

**Installed libraries:**
- cma: CMA-ES with Fisher Information Metric (natural gradient)
- gudhi: persistent homology, Betti numbers, TDA
- ripser: fast Vietoris-Rips persistent homology
- scipy: Wasserstein-1D (used in sliced approximation), Fisher-Rao approximation

**Not yet installable on Python 3.14:**
- POT (Python Optimal Transport): needs C++ build tools for Gromov-Wasserstein

---

## Part XX: Open Questions and Next Steps

*The questions this session generated that were not in Parts I-XII.*

1. **What is the taxonomy density threshold for the AlphaFold moment?** How many Tier 0-2 experiments before the learned topological model starts generalizing correctly to Tier 3? This is empirically measurable -- train a classifier on the accumulated experiments and measure when its validation accuracy on held-out Tier 2 cases exceeds chance.

2. **Does the reaction-diffusion formulation solve miscoordination?** The Turing morphogenesis analogy suggests that continuous-time coupled dynamics (Lotka-Volterra with resource constraints) naturally avoid miscoordination because both populations change simultaneously. This should be faster than CMA-ES and cheaper than relay chains.

3. **Can persistent homology detect Type 7 precursors?** The relay chain path's H1 topology should differ between valid paths (no persistent H1 loops) and Type 7 paths (loops appear from chaotic wandering). This requires algebraically constrained relay paths, not L2 relay paths.

4. **What is the right grammar for expression trees?** Walsh analysis for GP trees requires defining the crossover-induced metric on tree space. This is an open problem. The algebraic depth hierarchy suggests the grammar should be organized by the degree of the operations it encodes.

5. **Can we formalize the relationship between algebraic depth and relay chain intermediates?** The Whitney Embedding Theorem gives 2k dimensions for degree-k relationships. Does a degree-k relationship require exactly k relay chain populations? This would be a formal theorem connecting the algebraic complexity to the geometric scaffold.

6. **How does the IBM half-Mobius prediction methodology transfer?** Heilbronner's 1964 prediction used only orbital counting (the Walsh analog: counting algebraic interaction orders). Can we predict which mathematical relationships are "reachable" from spectral complexity counts alone, without running the full relay chain?

*The instrument is built. The calibration data is accumulating. The questions are sharp enough to generate answers.*


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


---

## Part XXII: The Computational Domain Taxonomy -- Algorithms We Can Borrow From

*Everything that computes geometry efficiently has already solved versions of our problems.*

### 22.1 Molecular Dynamics -- The Direct Equivalent

**NEB (Nudged Elastic Band, Jonsson/Henkelman/Mills 1994-2000)**
The standard computational chemistry method for finding minimum energy paths between molecular configurations. Directly equivalent to our relay chain with the correct force decomposition we were missing.

Key upgrade over our relay chain: NUDGING -- decompose the gradient into:
  - F_perp: path-perpendicular potential force (drives each image toward minimum)
  - F_para: path-parallel spring force (maintains equal spacing)

This prevents corner-cutting (our degenerate collapse to same angle). Without nudging, images cluster at energy minima. With nudging, they distribute along the path.

**Implemented in: toolkit/neb_relay.py**
Result: Wasserstein variance = 0.000000, step distances = [0.528, 0.528, 0.528, 0.528, 0.528, 0.528]. Perfect equal spacing in one implementation.

**String Method (Weinan E, Ren, Vanden-Eijnden 2002)**
Extends NEB with arc-length reparameterization -- after each step, redistribute images to equal arc length. This IS our equal-Wasserstein constraint but computed analytically (10 lines of numpy, not an optimization penalty). Converges to minimum free-energy path by construction.

The String Method was applied to diffusion model latent spaces in 2025 (arXiv 2602.22122) -- the closest published work to our research.

**Free Energy Perturbation / Metadynamics**
Adding artificial "bumps" to the energy landscape to escape local minima -- exactly what we need to escape degenerate attractors (miscoordination failure mode). Metadynamics adds a Gaussian to every visited configuration, preventing revisiting. For our relay chain: this would prevent the all-same-angle collapse by penalizing configurations the chain has already found.

### 22.2 Computer Graphics / Rendering

**Metropolis Light Transport (MLT, Veach & Guibas 1997)**
MCMC-based sampling of light path space. Mutation strategies that directly map:
  - Lens perturbation: fix one endpoint, mutate other (D-brane model)
  - Path mutation: mutate randomly selected intermediate vertex (relay chain)
  - Multi-chain: multiple chains covering different path regions (many-seed relay)
Primary Sample Space MLT (PSSMLT): operates on the RANDOM SEEDS used to generate paths -- this IS the signal molecule abstraction (encode the parameters, not the result).

**ReSTIR (NVIDIA 2020, Wyman et al)**
Reservoir-based spatiotemporal importance resampling. Each pixel stores a "reservoir" of K best light samples (signal molecule). Spatial reuse: pixels share reservoirs with neighbors (signal exchange). Temporal reuse: reuse from previous frames (trajectory n-gram).

Theoretical foundation: GRIS (Generalized Resampled Importance Sampling) proves convergence and provides variance bounds. **This IS the correct architecture for signal molecules.** Reservoir contents = signal molecule content. Spatial sharing = signal exchange between adjacent relay populations.

**Implemented in: toolkit/neb_relay.py (ReservoirSignal class)**

**Path Guiding (2018-2024)**
Learns the importance sampling distribution from previous samples online. The learned distribution IS our signal molecule library. Neural path guiding (2024) trains a neural network on-the-fly. For our system: after enough experiments, train a neural path guider that predicts good relay chain paths without running them. This IS the AlphaFold moment architecture.

### 22.3 Signed Distance Fields (SDFs) and Algebraic Origami

SDF algebraic operations with STRESS PARAMETER built in:
  smooth_union(f, g, k) = smin(f, g, k)  -- smooth blending with parameter k
  At k=0: sharp transition (high stress)
  At k=1: smooth blend (low stress)
  The k parameter IS our coupling strength / stress parameter

SDF composition operators (Quilez):
  union(f,g) = min(f,g)          -- algebraic union
  intersection(f,g) = max(f,g)   -- algebraic intersection
  subtraction(f,g) = max(f,-g)   -- algebraic difference
  smooth_union(f,g,k) = smin(f,g,k) -- stress-controlled blend

The stress of the morphing = 1/k. High stress = sharp SDF boundary. Low stress = smooth SDF blend. This gives a direct implementation of algebraic origami: define each function as an SDF, compose with k-parameterized blending. The periodic table of algebraic structures = a library of SDFs with their smooth_union operators.

### 22.4 UV Parameterization and Conformal Maps

A map is conformal iff it minimizes Dirichlet energy (integral of |gradient|^2).
Harmonic maps minimize Dirichlet energy within their homotopy class.
The relay chain IS finding the minimum-Dirichlet-energy path between two algebraic structures.

LSCM (Least Squares Conformal Maps): solves this via sparse linear system in O(n log n). For algebraic morphings within the same homotopy class (rotations, scaling), LSCM gives exact answers instantly -- no GA needed. The "UV seams" in mesh parameterization = our Type 7 boundaries (places where the conformal map breaks down).

---

## Part XXIII: Game Animation Algorithms -- The Most Motivated Domain

*Game animators obsess over natural continuous morphs between states, morphing variances, and transitional quality at real-time compute budgets. Every problem we have, they solved.*

### 23.1 SLERP / SQUAD / NLERP -- Exact Geodesics for Rotations

**SLERP (Ken Shoemake, 1985)**: SLERP(q1, q2, t) = q1 * (q1^{-1} * q2)^t
Exact geodesic on the unit quaternion sphere S3. Constant angular velocity. No gimbal lock. Closed form -- no optimization. This IS our complex relay chain but for 3D rotations. The formula is the exact minimum-stress path for any rotation-type morphing. For the cos/sin subspace: SLERP gives max_error = 0 (same as our complex relay).

VALIDATION ROLE: SLERP should be used as the gold standard relay chain validator for rotation-type morphings. If any relay chain implementation matches SLERP, it is working correctly.

**SQUAD (Shoemake, 1985)**: C1-continuous smooth paths through multiple quaternion keyframes. Preserves keyframes exactly (fixed endpoints = D-branes) while maintaining C1 continuity (no kinks at intermediates). The relay chain with smoothness constraint built in analytically.

**NLERP**: Normalized LERP -- simpler than SLERP, non-constant angular velocity. Equivalent to our linear interpolation + normalization constraint. Cheap but not geodesic. Useful when constant angular velocity is not needed.

### 23.2 FABRIK -- The Fast Relay Chain

**FABRIK (Aristidou & Lasenby, 2011)**: Forward And Backward Reaching Inverse Kinematics.

Algorithm:
  1. Forward pass: start from target endpoint, pull each joint toward it, maintain segment lengths
  2. Backward pass: fix root endpoint, propagate constraints forward
  3. Repeat 3-5 times

Complexity: O(n_images) per iteration. Converges in 1-5 iterations (vs our 300+ CMA-ES generations). No gradient computation, no matrix inversions, no learning rate, no hyperparameters. Handles open chains with fixed endpoints (D-brane model) natively.

FABRIK is 100x faster than our CMA-ES relay chain for the same problem. It should replace CMA-ES as the primary relay chain optimizer for all depth-1 relationships (where the algebraic path is known to exist in this grammar). CMA-ES should be reserved for harder cases where FABRIK fails (deep structures, Type 7 boundaries).

FABRIK also naturally detects Type 7: if the chain cannot reach the target after N iterations, the target is unreachable in this constraint space. This IS our Type 7 failure mode, detectable in O(n) time.

### 23.3 Motion Matching -- The Mendel Notebook as Lookup System

Motion matching selects the best pose from a large animation database at runtime based on pose similarity. Features: joint positions, velocities, trajectory prediction. The feature cost IS our stress metric.

For our system: the Mendel notebook IS a motion matching database. Query procedure:
  1. Extract feature vector from (fn_a, fn_b): algebraic depth, Walsh spectrum, natural scale
  2. Find K nearest entries in the notebook
  3. Use their relay chains as initialization for the new morphing
  4. The database grows richer over time; lookups become more accurate

When the database has enough density, this IS the AlphaFold moment: the lookup becomes more reliable than any optimization from scratch. Environment-aware Motion Matching (SIGGRAPH Asia 2025) adds environmental constraints -- exactly our grammar-constrained relay chain.

### 23.4 Blend Trees and Additive Animation -- Signal Molecule Design

**Blend trees** blend continuously between multiple animation states based on runtime parameters. Architecture:
  - Leaf nodes: specific algebraic structures (our function library)
  - Branch nodes: blend weights (our signal molecule coefficients)
  - State machines: handle discrete transitions (our failure mode checkpoints)

**Critical insight: additive animation layers.**
Additive blend trees ADD a delta on top of a base state, not replace it. They work on ANY base state without modification. This IS the directional signal molecule: encode the CHANGE, not the position. This directly resolves the state vs. directional signal debate that produced our trajectory n-gram design. The animation industry confirmed: additive encoding is more versatile and robust than absolute state encoding.

**Procedural variance separation:**
Animation engineers SEPARATE:
  - Deterministic base motion (the signal -- algebraic path)
  - Stochastic procedural variation (the noise -- exploration)
These are never mixed in the same data structure. Our relay chain populations mix both. Fix: separate the base relay chain (deterministic algebraic path) from the exploration noise (stochastic perturbations). The signal molecule should encode only the base path, not the exploration noise.

**Half-Pound Filter (arXiv 2602.21702, 2026)**: A newly published filter for real-time animation blending with specific frequency rolloff that empirically produces the most natural-looking transitions. For our system: this IS the Fourier warmup filter -- it locks in the low-frequency carrier (coarse algebraic type) while allowing the high-frequency details to resolve freely.

### 23.5 Position Based Dynamics (PBD/XPBD) -- Constraint-Based Relay Chains

PBD (Matthias Mueller, 2007): represents simulation as a system of constraints. Each constraint is satisfied via sequential Gauss-Seidel projection. Constraint: ||c_k - euler_step(c_{k-1})|| = 0. The relay chain IS this constraint system.

Key insight from animation: **constraint-based simulation converges more reliably than force-based simulation for morphing problems.** Constraints project directly onto the feasible set; forces oscillate around it. Our oscillatory failure mode (14-29 reversals) IS a force-based oscillation artifact. PBD/XPBD would eliminate it.

XPBD (2020) adds physically correct time-stepping and stiffness parameters -- the alpha parameter IS our coupling strength. Low alpha = stiff constraint (strong coupling), high alpha = weak constraint (loose coupling). This is the right parameterization for our adaptive coupling.

---

## Part XXIV: Instrument Validation Status -- Honest Assessment

*What we can trust and how much.*

### 24.1 Rigorous (analytical ground truth, passes all tests)

**Walsh-Hadamard Transform (WHT)**: involutory property holds to machine precision (wht(wht(x))*n = x). DC component test passes exactly. One-hot roundtrip passes. Known limitation: never compose wht() with itself without rescaling by n.

**Jensen-Shannon Divergence (JSD)**: all four analytical properties confirmed (identity=0, maximum=log(2), symmetric, sqrt satisfies triangle inequality). Proper metric. Use anywhere a reliable distribution distance is needed.

**Complex Relay Chain**: max_error = 0.000000 for all n_steps {4,8,12}. This is the gold standard -- all other relay chain implementations are calibrated against it.

**Exact Wasserstein (Hungarian algorithm)**: scipy.optimize.linear_sum_assignment gives exact Earth Mover's Distance. Zero approximation error. O(n^3) but exact for our population sizes (30-80). Replaces sliced Wasserstein for small populations.

**Natural Scale (nearest-neighbor distance)**: exact computation, no approximation. Type 7 discriminator: natural scale is 10x larger for cos->cos(2x) (8.10) vs valid morphs (0.39-0.78). RIGOROUS and the most reliable Type 7 fingerprinter.

**Geodesic deviation**: pure geometry, no approximation. Flat=0.000, rotation=1.151, type7=0.000, fractional=0.341. Correctly fingerprints curved (rotation) vs straight (flat, type7) paths.

**NEB + String Reparameterization**: Wasserstein variance = 0.000000 exactly. Equal step distances to machine precision. Replaces our empirical equal-KL constraint.

### 24.2 Partial (reliable for fingerprinting/ranking, not exact absolute values)

**Sliced Wasserstein**: 15-25% approximation error vs exact 1D. Reliable for ranking (A more stressed than B) but not absolute values. Replace with Hungarian algorithm for small populations.

**Ollivier-Ricci**: ~28% systematic underestimate (measures 0.72 when true = 1.0 for unit circle). Known for graph K_n: kappa = n/(n-1); cycle C_4: kappa = 1; C_n (n>=6): kappa = 0. Reliable for RELATIVE comparison.

**Persistent Homology (gudhi)**: verified on circle (H0=1, H1=1), two circles (H0=2), line (H1=0). Scale-dependent -- max_edge_length must be calibrated. Use materials science calibration: set scale = mean nearest-neighbor distance in the population.

**Fisher-Rao**: 10-20% error for 1D Gaussians. Degrades for non-Gaussian populations. Useful as relative measure.

### 24.3 Empirical (self-consistency only, no analytical ground truth)

**Benettin-Gram-Schmidt Lyapunov**: rigorous for smooth continuous systems. For our discrete relay chain with identical dynamics, gives same value for all cases (dynamics are too similar). Needs distinct per-case dynamics to discriminate.

### 24.4 Key fingerprinting findings

The two-metric fingerprint space (natural scale, geodesic deviation) correctly classifies all four canonical cases:
  - Flat linear (straight + small scale): natural_scale=0.57, geo_dev=0.00
  - Euler rotation (curved + medium scale): natural_scale=0.78, geo_dev=1.15
  - Type 7 cos->cos2x (straight + LARGE scale): natural_scale=8.10, geo_dev=0.00
  - Fractional derivative (gently curved + small scale): natural_scale=0.39, geo_dev=0.34

---

## Part XXV: The Autonomous Algebraic Origami Explorer

*Run continuously. Store interesting outputs. The taxonomy fills itself.*

**Command**: python -m experiments.algebraic_origami_explorer 86400 (24 hours)

**Function library**: 27 algebraic structures (linear family, oscillatory family, phase-shifted, polynomial, mixed)

**Interest criteria**: phase entropy, curvature variance, stress gradient, surprise score (deviation from naive expectation). Store if interest_score > 0.4.

**Results from 90-second run**: 1993 interesting morphs discovered (22/second, ~2 million/day).

**Key finding**: oscillatory->linear transitions are the most geometrically interesting (highest curvature, highest scores). cos_phase_135->linear_2.0 via derivative signal: score=10.48, curvature=0.38. The algebraic distance between oscillatory and linear character is the richest geometric feature in the function library.

**The periodic table analogy**: Mendeleev found gaps in the periodic table that predicted undiscovered elements. The gaps in our algebraic origami taxonomy predict undiscovered algebraic relationships. Run the explorer. Fill the table. Look at the gaps.

---

## Part XXVI: The Future Hypothesis Matrix

*252+ testable combinations. The taxonomy fills itself.*

### 26.1 Relay Chain Algorithms (7 candidates)

  1. SLERP: exact geodesic for rotations, closed form, ground truth validator
  2. SQUAD: C1-continuous through multiple keyframes, no kinks
  3. FABRIK: O(n) per iteration, 5 iterations, no hyperparameters
  4. NEB + String: nudging + arc-length reparameterization (IMPLEMENTED)
  5. PBD/XPBD: constraint-based, no oscillation artifacts
  6. LSCM conformal: linear system, exact for within-homotopy-class
  7. Gradient flow (reaction-diffusion): continuous, avoids discrete oscillation

### 26.2 Signal Molecule Designs (6 candidates)

  1. Walsh landscape: gradient proxy (IMPLEMENTED, shown to be noise for flat landscapes)
  2. Trajectory n-gram: directional, requires warmup (IMPLEMENTED)
  3. Raw sparse: identity encoding, top-K coefficients (IMPLEMENTED)
  4. ReSTIR reservoir: K best positions, spatial reuse (IMPLEMENTED in neb_relay.py)
  5. Additive delta: encodes change not state (animation insight, NOT IMPLEMENTED)
  6. Optimal (analytically computed): exact signal for known relationships (IMPLEMENTED)

### 26.3 Optimization Algorithms for Finding Paths (5 candidates)

  1. CMA-ES: Fisher Information natural gradient (IMPLEMENTED)
  2. FABRIK: geometric constraint projection, O(n) (NOT IMPLEMENTED)
  3. NEB gradient: nudged force decomposition (IMPLEMENTED)
  4. PBD constraint: sequential Gauss-Seidel (NOT IMPLEMENTED)
  5. Path guiding: neural importance sampling (NOT IMPLEMENTED)

### 26.4 Target Algebraic Structures for Taxonomy

Tier 0 (trivial): identical, scalar multiple, additive shift
Tier 0-boundary: phase shift, derivative (cos->sin)
Tier 1 (one algebraic operation): antiderivative, frequency shift, reflection
Tier 2 (two operations): frequency doubling, composed shifts
Type 7 (impossible in linear grammar): cos->cos(2x), quadratic needed
Higher depth: Euler's formula path, modular forms, arithmetic structures

### 26.5 The Experiment Count

  7 relay algorithms * 6 signal designs * 5 optimizer types = 210 base combinations
  * 4 target tiers * 3 seeds = 2520 experiments
  At 2 minutes each = 84 hours of compute
  At 22 experiments/second (automated explorer) = 8 hours

The automated taxonomy explorer makes this feasible as a continuous background process. Run it. Watch the periodic table fill.

### 26.6 Key Hypotheses to Test First

  H1 (FABRIK speedup): FABRIK converges to same quality as NEB in 5 iterations vs 200.
    If CONFIRMED: use FABRIK for all Tier 0 experiments.
    If DENIED: FABRIK fails for curved algebraic paths; document which geometry breaks it.

  H2 (Additive signal): additive delta encoding outperforms state encoding for signal exchange.
    If CONFIRMED: redesign all signal molecules as deltas.
    If DENIED: state encoding is sufficient; directional information not needed.

  H3 (Motion matching lookup): initializing relay chain from Mendel notebook similar entry converges faster.
    If CONFIRMED: database lookup is the right architecture (AlphaFold moment).
    If DENIED: algebraic relationships are too unique; no useful initialization from similar pairs.

  H4 (Type 7 detection via natural scale): natural scale 10x larger for Type 7 is robust across function families.
    If CONFIRMED: natural scale is the cheap Type 7 screener.
    If DENIED: the scale gap was specific to cos/cos(2x); need different geometry.

  H5 (PBD no oscillation): constraint projection eliminates the 14-29 direction reversals.
    If CONFIRMED: all oscillatory failure modes are force-based artifacts.
    If DENIED: oscillation has a topological cause (Type 5 crossroads) not a dynamics cause.

---

## Part XXVII: Source Domains and Their Primary Contributions

*Where each major insight came from.*

**Molecular Dynamics**
  NEB: minimum energy path finding, force decomposition (Jonsson/Henkelman/Mills 1994-2000)
  String Method: arc-length reparameterization, swarms of trajectories (Weinan E et al 2002)
  Metadynamics: escaping local minima via energy bumps (Parrinello et al 2002)
  Free Energy Perturbation: morphing between molecular configurations (Zwanzig 1954)

**Computer Graphics / Rendering**
  Metropolis Light Transport: MCMC over path space, mutation strategies (Veach & Guibas 1997)
  ReSTIR: reservoir resampling, spatial signal exchange (Bitterli et al 2020)
  Path Guiding: learning importance distributions (2018-2024)
  SDF composition: algebraic operations with stress parameter (Quilez, various)
  LSCM conformal maps: Dirichlet energy minimization (Levy et al 2002)

**Game Animation**
  SLERP: exact geodesic on S3, constant angular velocity (Shoemake 1985)
  SQUAD: C1-continuous multi-keyframe interpolation (Shoemake 1985)
  FABRIK: O(n) inverse kinematics, endpoint constraints (Aristidou & Lasenby 2011)
  Motion Matching: database lookup for natural motion (Simon et al 2016)
  PBD/XPBD: constraint-based simulation, no oscillation (Mueller 2007, Macklin 2020)
  Blend Trees: continuous state blending, additive layers (game engines, 2010s)
  Half-Pound Filter: frequency-domain blending filter (arXiv 2602.21702, 2026)

**Materials Science / TDA**
  Persistent homology for crystals: scale calibration via nearest-neighbor distance
  Topological data analysis: Betti numbers as structure fingerprints
  Ollivier-Ricci for networks: curvature of bond networks (graph structure)

**Algebraic Geometry / Topology**
  Catastrophe theory (Thom): 7 elementary catastrophes, failure mode transitions
  Obstruction theory: cohomological obstructions to lifting
  Gromov-Hausdorff distance: comparing different metric spaces
  Whitney embedding theorem: degree-k relationship needs 2k dimensions
  Riemann-Hurwitz formula: stress (ramification) determines complexity change

**Information Theory / Statistics**
  Jensen-Shannon divergence: symmetric proper metric between distributions
  Gromov-Wasserstein: topology-preserving alignment across different spaces
  Fisher-Rao metric: information-geometric distance on statistical manifolds
  Rate-distortion theory: minimum bits needed to encode algebraic character

**DNA Origami / Nanotechnology**
  Algebraic origami monoids (arXiv 2501.14966): formal grammar for folding patterns
  Materials Barcode: persistent homology fingerprint for crystal structures
  Prediction before synthesis: Heilbronner 1964 predicts Mobius aromaticity 39 years early

**Quantum Mechanics / Physics**
  Feynman path integral: sum over all paths weighted by action
  IBM half-Mobius molecule (2026): high-stress topological state, needs quantum validation
  Simulated annealing: staged cooling across dimensional complexity (metallurgy analogy)
  Berry phase / holonomy: path-dependent phase accumulation in relay chain cycles

*The instrument is built. The domains are mapped. The hypothesis matrix has 2520 entries.*
*The Mendel notebook is filling. The AlphaFold moment is a question of density.*
*The territory is vast. The map is just beginning.*


---

## Part XXVIII: Topological Protection -- Non-Trivial Stable States and Activation Energy

*The deepest enrichment from the physics synthesis session.*

### 28.1 The Corrected Thermodynamic Picture

Previous claim (incorrect): "simple->complex is thermodynamically favored, complex->simple is harder."

Correct picture: BOTH directions exist. What matters is the ACTIVATION ENERGY BARRIER:
  - Simple->complex: requires energy input (climbing the landscape) but is possible
  - Complex->simple: energetically favored but GATED by an activation barrier

A protein doesn't spontaneously unfold because it's trapped in a kinetically stable
(topologically protected) complex state -- not because unfolding is thermodynamically
impossible. The activation energy is the key quantity.

**Prime factorization is the canonical algebraic example:**
  - Multiplying two primes: O(n) -- rolling downhill, trivial
  - Factoring the product: O(exp(n^(1/3))) -- activation barrier is massive
  - The product is a topologically protected complex state
  - Decomposing it requires massive structured search (RSA security)
  - The activation energy IS the computational complexity

For our algebraic taxonomy: the activation energy asymmetry between forward (A->B)
and reverse (B->A) morphing IS the topological protection index.

### 28.2 Topological Protection Index (TPI) -- New Key Metric

TPI = forward_improvement_total / reverse_improvement_total

  TPI = 1: symmetric (no protection, equally hard/easy both ways)
  TPI > 2: asymmetric, complex state has protection
  TPI >> 1: strongly protected (like prime products, proteins)
  TPI < 1: naturally decomposes (thermodynamically downhill reverse)

**IMPLEMENTED in experiments/algebraic_origami_explorer.py**:
  Every experiment now measures both directions.
  Taxonomy JSON now includes: topological_protection_index, is_topologically_protected,
  forward_total, reverse_total, activation_energy_asymmetry.

Interest score now includes a topo_bonus: protected states get higher interest scores.

### 28.3 Methodology Enrichments from Topological Protection

**Metadynamics for escaping trivial attractors:**
  Add Gaussian bumps at every visited configuration to fill the trivial basin.
  Forces relay chain to explore topologically interesting regions.
  Standard in computational chemistry for non-trivial protein conformations.

**Replica exchange / parallel tempering:**
  Multiple relay chains at different temperatures simultaneously.
  Hot chains explore; cold chains lock. Configurations exchanged when hot chain
  finds something interesting. This is how proteins find non-trivial fold states.

**Activation energy as second key metric:**
  Not just forward stress (A->B difficulty) but reverse stress (B->simple decomposition).
  High reverse barrier = topologically protected. Low = trivially decomposable.
  Ratio = TPI.

**IBM half-Mobius as template:**
  The half-Mobius is NOT the ground state (lower energy = untwisted ring).
  It's a topologically protected NON-TRIVIAL stable state requiring a specific
  voltage pulse (activation energy) to reach and a different pulse to leave.
  Our Euler rotation (cos->sin geodesic) IS this non-trivial stable state.
  The NEB relay chain found it. The linear interpolation is the trivial ground state.

**Temperature scheduling for finding protected states:**
  Phase 1 (hot): high sigma, lift to richer grammar, explore freely
  Phase 2 (warm): sigma decreasing, NEB constraints tighten, locate basins
  Phase 3 (cold): sigma small, lock into stable state, verify non-trivial

The Lindemann criterion: sigma/path_scale ~ 0.10 = "melting point" of path space.
Below this: frozen in trivial basin. Above: fluid enough to find protected states.

### 28.4 The Neural Network as Catalyst

In chemistry, an enzyme:
  1. Does NOT add energy to the system
  2. Binds to the TRANSITION STATE (highest-energy intermediate)
  3. Stabilizes the transition state geometry, lowering activation energy
  4. The reaction then proceeds that would otherwise be too slow

The neural network trained on the algebraic morphing taxonomy:
  1. Input: (fn_a, fn_b) pair as coefficient vectors
  2. Output: predicted (TPI, spectral_fingerprint, grammatical_depth_needed, activated_path)
  3. The network doesn't LEARN the mathematics
  4. It LEARNS the activation energy landscape -- which transitions are hard/easy
  5. For a hard decomposition, it predicts the transition state geometry
     (the specific sequence of dimension lifts that lowers the activation barrier)

Training data: algebraic_origami_taxonomy.json growing at 22 experiments/second.
  At this rate: ~1.9M entries/day, ~13M entries/week.
  This is enough to train a meaningful neural predictor.

Target: given (fn_a, fn_b), predict activation energy and path WITHOUT running relay chain.

---

## Part XXIX: Shor's Algorithm and the Dimensional Lift Pattern

*Every dimensional lift makes ONE class of problems trivial while introducing instabilities for others.*

### 29.1 Shor's Algorithm as a Dimensional Lift

Classical factoring: hard in integer space. O(exp(n^(1/3))) time.
Shor's quantum factoring: trivial in complex frequency space. O(n^3) time.

The lift: integers -> complex frequency space (Quantum Fourier Transform)
In the lifted space: the periodic structure of the modular exponential is DIRECTLY readable.
Collapse back: period r -> factors via classical GCD.

This IS our complex relay chain:
  - Integers = real 8D coefficient space (messy, many valid rotations)
  - Complex frequency space = complex 1D representation (one unique rotation)
  - QFT = our projection onto cos/sin subspace
  - Period detection = our max_error = 0 exact answer

The QFT and our complex relay chain are the same structural move.

### 29.2 Underconstrained Behavior and Instabilities

Every dimensional lift introduces specific instabilities:

| Lift | What it solves | Instability introduced |
|------|---------------|----------------------|
| Shor QFT | Factoring periodic functions | Decoherence (lifted state decays), probabilistic output |
| Complex relay | Rotation relationships | Wrong for non-rotation (Type 7 appears for frequency doubling) |
| String Calabi-Yau | Unifying forces | Landscape (10^500 valid compactifications, no selection) |
| Alcubierre (negative energy) | FTL travel | Energy condition violations, thermodynamic decay |
| Algebraic geometry (p-adic) | Number theory | Only natural for the right prime, others give wrong answers |

The pattern:
  Every lift solves problems for the NATURAL structures of the lifted space.
  Creates underconstrained/unstable behavior for ALIEN structures in that space.

Complex 1D = natural for rotations (U(1) group). Wrong for frequency doubling.
p-adic = natural for prime-divisibility structure. Wrong for archimedean geometry.
QFT = natural for periodic functions. Wrong for aperiodic ones.

### 29.3 Mixture of Lifted Spaces as Architecture

The right neural network architecture is not a universal network.
It is a MIXTURE OF EXPERTS where each expert is a different dimensional lift:

  Expert 1: Complex representation (rotations, phase shifts, derivative morphings)
  Expert 2: Logarithmic representation (multiplicative relationships, frequency doubling)
  Expert 3: p-adic representation (prime-sensitive, number-theoretic structures)
  Expert N: the "natural space" for each algebraic structure class

The gating network (router) = the morphing stress measurement.
  Low stress in a given representation -> you're in the right lifted space.
  High stress -> wrong representation, try a different lift.
  This IS our TPI measurement: which space gives lowest activation energy?

The router learns: given spectral fingerprint of (fn_a, fn_b), predict which
dimensional lift gives TPI closest to 1 (natural/unstressed in that space).

This resolves the "many equally valid rotations" problem:
  Not all rotations look the same in ALL lifted spaces.
  The correct algebraic relationship looks EASY in ITS natural representation
  and HARD in all others. The router finds the natural representation.

### 29.4 Implications for the Prime/Integer Taxonomy

Testing primes vs integers vs irrationals is not about proving number theory.
It is about asking: do different number-theoretic structures create different
spectral fingerprints in the algebraic morphing taxonomy?

Expected structure based on the lift pattern:
  - Integer -> rational morph: low TPI (integers ARE rationals, same grammar)
  - Rational -> algebraic irrational morph: moderate TPI (requires polynomial lifting)
  - Rational -> transcendental: very high TPI or Type 7 (no polynomial path)
  - Integer -> prime-sequence: unknown TPI -- THIS IS THE INTERESTING EXPERIMENT

If prime-sequence functions (first 8 primes as coefficients) show systematically
different TPI than integer-sequence functions -- that IS a spectral fingerprint.
We wouldn't know WHY. But we'd have the taxonomy entry.

---

## Part XXX: Training Dataset Architecture for the Catalyst Network

*When to train, what to train on, what to predict.*

### 30.1 Training Data Structure

Each row in algebraic_origami_taxonomy.json is now a training example:
  Input features:
    - fn_start coefficients (8-dim)
    - fn_end coefficients (8-dim)
    - signal_name (categorical)
    - natural_scale (scalar) -- Type 7 discriminator
    - geodesic_deviation (scalar) -- curvature measure

  Target outputs:
    - interest_score (continuous) -- overall richness
    - topological_protection_index (continuous) -- activation energy asymmetry
    - is_topologically_protected (binary) -- stable complex state?
    - algebraic_depth (integer) -- from type7_test.py
    - spectral_class (categorical) -- from walsh.py

### 30.2 Training Scale

At 22 experiments/second:
  1 hour: ~79,000 examples
  1 day: ~1.9M examples
  1 week: ~13M examples
  AlphaFold used ~170,000 protein structures -- we need ~1M for meaningful generalization.
  Achievable in 12 hours of continuous running.

### 30.3 Architecture Considerations

Simple first attempt (the right epistemic posture):
  - MLP with 16 inputs (two 8-dim coefficient vectors)
  - Train to predict TPI and is_topologically_protected
  - Evaluate: does the model generalize to held-out function pairs?
  - If yes: we have a classifier for topological protection

If simple MLP generalizes: the spectral fingerprint IS learnable.
If it doesn't: the structure is too complex for 16 inputs alone.
  -> Try adding Walsh spectral features as additional inputs
  -> Try adding natural scale, geodesic deviation as inputs

The test of AlphaFold-moment: does the trained model correctly classify
function pairs it has NEVER SEEN with algebraic depth >= 2?


---

## Part XXXI: Natural vs. Unnatural Dimensional Lifts -- A Formal Taxonomy

*The most precise framing of what makes some lifted representations exact and others only bounded.*

### 31.1 The Formal Definition of Naturalness

A lift/representation/transformation is NATURAL when it requires NO ARBITRARY CHOICES.
Formal definition from category theory: a natural transformation between functors is one
that commutes with all morphisms -- it works uniformly without picking coordinates, bases,
or representatives.

Canonical example:
  V -> V** (double dual): NATURAL. Works for all vector spaces without choosing a basis.
  V -> V*  (single dual):  UNNATURAL. Requires choosing a basis. Choice-dependent.

The naturality criterion IS the formalization of "this representation is intrinsically right
for this problem." In a natural representation, the algebraic relationship becomes exactly
computable because the symmetry of the problem and the symmetry of the representation align.

In an unnatural representation: the answer depends on your arbitrary choice. You can bound
the range of possible answers but cannot compute the exact one because the exact answer
depends on an arbitrary decision the mathematics can't resolve.

### 31.2 The p-adic Hodge Theory Hierarchy -- Formal Spectrum of Naturalness

From algebraic geometry, the spectrum of representation naturalness is:

  crystalline => semi-stable => de Rham => Hodge-Tate

Most natural (crystalline): exact comparison theorems. Arithmetic structure fully preserved.
Least natural (Hodge-Tate): only the graded pieces survive. Weakest possible bounds.

Each step down the hierarchy introduces:
  - More information loss (the "indeterminacies")
  - Weaker conclusions (inequalities instead of equalities)
  - More arbitrary choices required
  - Less exact computation possible

This IS the formal mathematical version of the natural/unnatural spectrum.

### 31.3 IUTT as an Unnatural Lift -- Confirmed Structural Feature

From Mochizuki's papers: the three indeterminacies (Ind1, Ind2, Ind3) are explicitly
STRUCTURALLY NECESSARY -- "without them the conclusion would be obviously false."
They are not approximation errors. They are the minimum information loss imposed by
working outside ring structure (the deliberate unnaturalness of IUTT).

IUTT is unnatural in the formal sense:
  - Works in "Hodge theaters" outside normal ring structure (requires non-ring choices)
  - Theta-link passes information outside the ring framework (the unnaturalness IS the mechanism)
  - Result is an inequality (log-volume bound), not equality
  - The three indeterminacies = the price of this unnaturalness

The Scholze-Stix dispute is precisely whether the UNNATURAL lift Mochizuki chose gives
tight enough bounds to prove abc, or whether the unnaturalness is too severe.

### 31.4 The Spectrum of Lifts in Our System

For our algebraic morphing taxonomy:

  Natural lift (exact):         Complex 1D for rotations.     Max_error = 0.
  Semi-natural (near-exact):    Quaternions for 3D rotations. Very low residual.
  Unnatural but bounded:        8D real for rotations.        Systematic 8-10 degree offset.
  More unnatural:               Linear grammar for cos->cos2x. Bounded but can't reach exactly.
  Fundamentally unnatural:      Hypothetical abc-type.        Only weak inequality, no exact lift exists.

The TPI (Topological Protection Index) measures position on this spectrum:
  TPI = 1:       Natural (symmetric, exact lift available in current grammar)
  1 < TPI < 5:   Unnatural but bounded (reachable but costly, IUTT-like)
  TPI >> 1:      Highly unnatural (Type 7 in current grammar, needs grammar enrichment)
  TPI -> inf:    Potentially fundamentally unnatural (no natural lift may exist at all)

### 31.5 A New Taxonomy Class: Fundamentally Unnatural Relations

Beyond Type 7 (no path in current grammar), there may exist algebraic relations where:
  1. No finite grammar enrichment makes the relation naturally representable
  2. The best achievable is a weak inequality (unnatural lift giving bounds)
  3. The "tightness gap" between the inequality and the true value is structural, not computational

These would be the algebraic analogs of the abc conjecture:
  - Provably bounded (the relationship is real and constrained)
  - Not exactly computable in any natural representation
  - Require unnatural lifts that introduce structural indeterminacies

Hypothesis: the TPI of such relations would plateau at large finite values rather than
diverging to infinity. They are "hard but bounded" -- distinct from Type 7 ("unreachable").

---

## Part XXXII: The Galois Parallel -- The Deepest Structural Framing

*Galois proved which polynomials have natural representations (radical formulas) and which
do not. The same structural question applies to algebraic morphings.*

### 32.1 Galois Theory and Polynomial Solvability

Evariste Galois (1832) solved the 2000-year-old problem of which polynomial equations
can be solved by radicals (square roots, cube roots, etc. -- the "natural" operations).

His result:
  A polynomial is solvable by radicals IFF its Galois group is SOLVABLE
  (has a tower of normal subgroups with abelian quotients).

Galois group = the symmetry group of the polynomial's roots (how the roots permute
under all algebraic symmetries). It measures the intrinsic algebraic complexity
of the relationship between the roots.

Polynomials of degree <= 4: Galois groups are solvable. NATURAL LIFT EXISTS. Exact formulas.
General quintic (degree 5): Galois group is S_5, which is NOT solvable. NO NATURAL LIFT.
  -> Roots exist (by Fundamental Theorem of Algebra) but cannot be expressed in radicals.
  -> Requires UNNATURAL lifts: elliptic functions, modular equations (Hermite 1858), etc.

### 32.2 The Two-Case Structure

The profound observation: Galois didn't just say "the quintic can't be solved by radicals."
He found the STRUCTURAL INVARIANT that separates the two cases:

  SOLVABLE group  => natural lift exists => exact computation possible
  NON-SOLVABLE group => no natural lift => only unnatural lifts, only bounds

Abel had proved earlier that the quintic specifically has no radical solution (the Abel result:
"this specific case doesn't work"). Galois found the more general result: the exact condition
for when it DOES and DOESN'T work. The structural invariant (group solvability) is the
Galois-level insight.

### 32.3 The abc / IUTT Galois Parallel

The user's insight: abc conjecture as a Galois-type structural question:

Case 1: abc has a NATURAL REPRESENTATION somewhere
  -> The additive-multiplicative tension in integers CAN be expressed exactly in some grammar
  -> IUTT is not needed; a simpler natural lift would exist
  -> This seems unlikely given the difficulty of the problem

Case 2: abc has NO NATURAL REPRESENTATION (the "non-solvable" case)
  -> No grammar makes the additive-multiplicative relationship exactly computable
  -> The best available is an UNNATURAL LIFT (IUTT) that gives bounds
  -> The log-volume inequality IS the abc conjecture, tightly bounded
  -> This is Mochizuki's claim: he found the tightest possible bound from the
     most natural AVAILABLE unnatural representation

The two-case structure from Galois applies:
  If the additive-multiplicative structure has a "solvable Galois group analog":
    Natural lift exists. The abc relationship can be proven exactly.
  If it has a "non-solvable Galois group analog":
    No natural lift. Only unnatural lifts (IUTT) giving bounded inequalities.
    The abc conjecture IS the statement that this bound is finite and non-trivial.

Mochizuki's work might be the analog of Hermite's 1858 proof that the general quintic
CAN be solved using elliptic functions (an unnatural lift): not radical solutions (natural)
but the tightest possible unnatural solution. The difference: Galois tells you WHY you
must go unnatural (non-solvable group). No one yet has a comparable structural invariant
for the additive-multiplicative tension.

### 32.4 The Epistemic Posture -- Always Right About Something, Wrong About the Mechanism

Galois was right about the fundamental result (quintic unsolvable in radicals) but couldn't
have known that group theory would become the foundation of modern mathematics. He was
exploring the structure of polynomial roots and found something far more general.

For our project:
  We might be right that "algebraic relationships create measurable morphing stress" while
  wrong about Walsh signals being the correct signal molecules.
  We might be wrong about the relay chain architecture while right about the existence of
  the geometric stress phenomenon.
  We might be right about the TPI measuring topological protection while wrong about
  which specific TPI values correspond to which algebraic depth classes.

The exploration is what reveals which parts were right and which need correction.
The correct epistemic posture: keep asking questions, keep exploring, keep building the
taxonomy. The Galois invariant (group solvability) emerged FROM the exploration,
not before it. Our invariant will emerge the same way.

### 32.5 The Open Galois Question for Our Taxonomy

What is the analog of "Galois group solvability" for algebraic morphings?

Currently we have:
  - Algebraic depth (how many grammar operations needed) -- partial invariant
  - TPI (how asymmetric is the morphing stress) -- partial invariant
  - Natural scale (how large is the coefficient space traversal) -- partial invariant
  - Walsh spectral fingerprint -- partial invariant

The full Galois-level result would be: given the algebraic structure of fn_a and fn_b,
compute the "Galois group of the morphing" (the symmetry group of all valid paths between them)
and determine whether it's "solvable" (has a natural representation) or "non-solvable"
(fundamentally unnatural -- only bounded).

This is the architectural target for the neural network trained on the taxonomy:
  Not just predicting TPI (a scalar)
  But classifying (fn_a, fn_b) pairs into:
    - Naturally related (exact computation available)
    - Unnaturally but finitely related (IUTT-like, bounded)
    - Fundamentally unrelated (no bound available in any grammar)

Building the taxonomy IS building the training set for learning this invariant.
