"""
Signal molecule designs — three hypotheses about what "mathematical character" means.

Each design encodes the sender population's current state differently.
The taxonomy of which design works for which experiment class IS the research finding.

Design 1 — WalshLandscape
  WHT of fitness values at 2^N corners of a hypercube around sampled individuals.
  Encodes: gradient/epistatic structure of the SEARCH PROBLEM.
  Biology analogy: reporting your metabolic stress level, not your identity.

Design 2 — WalshSolution
  WHT applied directly to the coefficient vectors of sampled individuals.
  Encodes: frequency structure of the SOLUTIONS themselves.
  Biology analogy: reporting your gene expression pattern.
  For cos: even-indexed WHT coefficients dominate.
  For sin: odd-indexed WHT coefficients dominate.
  Hypothesis: this is what the receiver needs to know about the sender's nature.

Design 3 — RawSparse
  Top-K coefficient magnitudes of the best individual, zeroed elsewhere.
  Encodes: the IDENTITY of the solution, maximally direct.
  Biology analogy: secreting a metabolite that IS your functional output.
  Closest to quorum sensing: the molecule = the organism's current state.

The receiver uses whichever signal to modulate its per-gene mutation variance.
Which signal best preserves mathematical character across the exchange is empirical.
"""

import numpy as np
from abc import ABC, abstractmethod
from toolkit.walsh import wht, main_effects


class SignalMolecule(ABC):
    """Base class for signal molecule designs."""

    @abstractmethod
    def emit(self, pop: np.ndarray, fitness_fn, n_genes: int,
             rng: np.random.Generator, k_samples: int = 8,
             delta: float = 0.25) -> np.ndarray:
        """
        Compute the signal emitted by a population.
        Returns: array of length 2^n_genes (padded to power of 2 if needed).
        """

    @abstractmethod
    def receive(self, signal: np.ndarray, n_genes: int) -> np.ndarray:
        """
        Extract per-gene mutation modulation from a received signal.
        Returns: array of length n_genes in [0, 1] (0=no bias, 1=max bias).
        """

    @property
    @abstractmethod
    def name(self) -> str: ...


# ── Design 1: Walsh Landscape ─────────────────────────────────────────────────

class WalshLandscape(SignalMolecule):
    """
    WHT of fitness landscape around sampled individuals.
    The 'classic' design from the document — encodes epistatic gradient structure.
    """

    @property
    def name(self): return "walsh_landscape"

    def emit(self, pop, fitness_fn, n_genes, rng, k_samples=8, delta=0.25):
        n_corners = 2 ** n_genes
        k = min(k_samples, len(pop))
        idxs = rng.choice(len(pop), k, replace=False)
        spectra = []
        for i in idxs:
            best = pop[i]
            vals = np.empty(n_corners)
            for corner in range(n_corners):
                perturb = np.array([(delta if (corner >> b) & 1 else -delta)
                                    for b in range(n_genes)])
                vals[corner] = fitness_fn(best + perturb)
            spectra.append(wht(vals))
        return np.stack(spectra).mean(axis=0)

    def receive(self, signal, n_genes):
        me = np.abs(signal[1 << np.arange(n_genes)])
        me /= me.max() + 1e-10
        return me


# ── Design 2: Walsh Solution ──────────────────────────────────────────────────

class WalshSolution(SignalMolecule):
    """
    WHT of the coefficient vectors of sampled individuals.
    Encodes the frequency structure of the SOLUTIONS, not the fitness landscape.

    For cos (coefficients [1,0,-1,0,1,0,-1,0]):
      WHT concentrates energy at even Walsh indices.
    For sin (coefficients [0,1,0,-1,0,1,0,-1]):
      WHT concentrates energy at odd Walsh indices.
    Hypothesis: the receiver reads even/odd Walsh dominance and adjusts accordingly.
    """

    @property
    def name(self): return "walsh_solution"

    def emit(self, pop, fitness_fn, n_genes, rng, k_samples=8, delta=None):
        # fitness_fn unused — we encode the solutions directly
        k = min(k_samples, len(pop))
        idxs = rng.choice(len(pop), k, replace=False)
        # Pad each coefficient vector to length = power of 2
        p = 1
        while p < n_genes:
            p <<= 1
        spectra = []
        for i in idxs:
            padded = np.zeros(p)
            padded[:n_genes] = pop[i]
            spectra.append(wht(padded))
        return np.stack(spectra).mean(axis=0)

    def receive(self, signal, n_genes):
        # Use absolute magnitude of WHT coefficients at gene indices as bias
        p = len(signal)
        # Map WHT indices back to gene indices: coefficient k lives at index k
        me = np.abs(signal[:n_genes])
        me /= me.max() + 1e-10
        return me


# ── Design 3: Raw Sparse ──────────────────────────────────────────────────────

class RawSparse(SignalMolecule):
    """
    Top-K nonzero entries of the best individual's coefficient vector.
    Most direct identity signal — the molecule IS the solution, compressed.

    Closest to quorum sensing: the secreted metabolite is a direct product
    of the organism's current functional state, not a description of it.

    The receiver reads which genes are large and explores more there.
    For cos: large at k=0,2,4,6.  For sin: large at k=1,3,5,7.
    Hypothesis: this is the cleanest encoding of mathematical character.
    """
    TOP_K = 4   # number of nonzero entries to keep

    @property
    def name(self): return "raw_sparse"

    def emit(self, pop, fitness_fn, n_genes, rng, k_samples=8, delta=None):
        # Use best individual by fitness
        fits = np.array([fitness_fn(ind) for ind in pop])
        best = pop[np.argmax(fits)].copy()
        # Sparsify: keep only TOP_K largest-magnitude coefficients
        sparse = np.zeros_like(best)
        top_idx = np.argsort(np.abs(best))[-self.TOP_K:]
        sparse[top_idx] = best[top_idx]
        # Pad to power of 2 for consistent downstream handling
        p = 1
        while p < n_genes:
            p <<= 1
        out = np.zeros(p)
        out[:n_genes] = sparse
        return out

    def receive(self, signal, n_genes):
        # Direct read: which genes have nonzero signal?
        me = np.abs(signal[:n_genes])
        me /= me.max() + 1e-10
        return me


# ── Design 4: Trajectory N-gram ──────────────────────────────────────────────

class TrajectoryNgram(SignalMolecule):
    """
    DIRECTIONAL signal — encodes how the solution is MOVING, not where it is.

    From the .md Section 3.4:
    "Instead of sending instantaneous spectral snapshots, send a sequence of
     snapshots showing how the Walsh spectrum changes as expressions are mutated.
     This encodes local geometry of the landscape rather than just its value —
     the receiver can distinguish 'this coefficient is large because the landscape
     is flat' from 'this coefficient is large because we're on a steep slope.'
     Analogous to sending the time derivative of metabolite concentration rather
     than the concentration alone."

    Implementation:
    - Maintain a sliding window of K recent solution spectra
    - Signal = latest spectrum - oldest spectrum (direction of movement)
    - Receiver uses the direction to bias mutations ALONG the movement direction
      (positive component = gene moving this way, accelerate it)
      (negative component = gene moving that way, slow it down)

    For the rotation relay chain:
    - State signal: A looks like cos (even ME dominate)
    - Directional signal: A's even ME are SHRINKING, odd ME are GROWING → heading toward sin
    - B receives this direction and aligns its own movement: "if A going cos→sin,
      I should go sin→-cos" (next step in the rotation)

    This is the signal that makes relay chains work properly.
    """
    WINDOW = 4   # number of recent spectra to maintain

    def __init__(self):
        self._history: list = []   # recent WHT(coefficient_vector) snapshots

    @property
    def name(self): return "trajectory_ngram"

    def emit(self, pop, fitness_fn, n_genes, rng, k_samples=8, delta=None):
        # Compute current solution spectrum (WHT of best individual's coefficients)
        fits = np.array([fitness_fn(ind) for ind in pop])
        best = pop[np.argmax(fits)].copy()
        p = 1
        while p < n_genes: p <<= 1
        padded = np.zeros(p); padded[:n_genes] = best
        current = wht(padded)

        # Update history
        self._history.append(current.copy())
        if len(self._history) > self.WINDOW:
            self._history.pop(0)

        if len(self._history) < 2:
            return current   # not enough history yet — emit state signal

        # Directional signal: difference between newest and oldest in window
        # Positive = component growing; negative = component shrinking
        direction = self._history[-1] - self._history[0]
        return direction

    def receive(self, signal, n_genes):
        """
        Use the directional signal to bias mutation.
        Genes moving in the positive direction get LARGER mutations (accelerate).
        Genes moving in the negative direction also get larger mutations (they're active).
        The magnitude of movement = how much to amplify.
        """
        me = np.abs(signal[:n_genes])   # magnitude of directional change per gene
        me /= me.max() + 1e-10
        return me

    def reset(self):
        """Reset history between experiments."""
        self._history = []


# ── Registry ──────────────────────────────────────────────────────────────────

SIGNAL_DESIGNS: dict[str, SignalMolecule] = {
    "walsh_landscape":   WalshLandscape(),
    "walsh_solution":    WalshSolution(),
    "raw_sparse":        RawSparse(),
    "trajectory_ngram":  TrajectoryNgram(),
}


def get_signal(name: str) -> SignalMolecule:
    if name not in SIGNAL_DESIGNS:
        raise ValueError(f"Unknown signal design '{name}'. "
                         f"Choose from: {list(SIGNAL_DESIGNS)}")
    return SIGNAL_DESIGNS[name]
