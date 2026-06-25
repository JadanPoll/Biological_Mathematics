"""
Fit the fold-catastrophe 3/2 scaling law to the 509-morph taxonomy.

Theory (confirmed by PubMed, fold catastrophe paper):
  Barrier height ΔG ~ |a - a_c|^(3/2) near the critical control parameter a_c.

Our operationalization:
  Control parameter : gap_go_map_b1  (spectral orbit compatibility, 1=same, 0=different)
  Barrier height    : gap_sigma_norm (string tension peak height, 0=FREE, 1=CONFINEMENT)
  Incompatibility   : x = 1 - gap_go_map_b1  (so x=0 is valid, x=1 is most Type-7-like)

Predicted relationship:
  gap_sigma_norm = C * (1 - gap_go_map_b1)^alpha

If the theory is correct: alpha ~ 1.5 (3/2).
If alpha ~ 1.0: linear confinement (different catastrophe type).
If alpha ~ 2.0: quadratic onset (cusp catastrophe, codimension 2).
If alpha << 1: subcritical (TEARING regime, not fold).

The fit uses only records where both endpoints are in the known function library
(filter_coverage != "unknown"). Records with gap_sigma_regime = "TEARING" are
excluded from the main fold fit (tearing is a different catastrophe branch).

Run:
  python -m experiments.fit_fold_scaling
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


TAXONOMY = Path("algebraic_origami_taxonomy_enriched.json")


def load_data():
    records = json.loads(TAXONOMY.read_text())
    xs, ys, regimes, labels = [], [], [], []
    for r in records:
        b1   = r.get("gap_go_map_b1")
        sig  = r.get("gap_sigma_norm")
        reg  = r.get("gap_sigma_regime", "")
        cov  = r.get("filter_coverage", "unknown")
        if b1 is None or sig is None or cov == "unknown":
            continue
        x = 1.0 - b1   # incompatibility: 0=same orbit, 1=fully incompatible
        xs.append(x)
        ys.append(sig)
        regimes.append(reg)
        labels.append(f"{r.get('start','?')[:10]}->{r.get('end','?')[:10]}")
    return np.array(xs), np.array(ys), regimes, labels


def fold_model(x, C):
    """Pure fold: sigma = C * x^(3/2). Fixed exponent alpha=1.5."""
    return C * np.power(np.maximum(x, 0), 1.5)


def power_law(x, C, alpha):
    """Free power law: sigma = C * x^alpha. alpha is fitted."""
    return C * np.power(np.maximum(x, 0), alpha)


def load_canonical_anchors():
    """
    Add the four canonical test cases as anchors for the fold fit.

    These span the full range (0 to 0.9 incompatibility) needed to resolve
    the fold exponent. The 509-morph taxonomy only covers the valid-orbit range
    (go_map_b1 > 0.86), so the fit without anchors measures within-orbit
    scaling rather than the confinement transition itself.

    All values pre-computed by experiments/sanity_gap_metrics.py and the
    demo_sin_chirp_morph.py stress test.
    """
    import sys, os
    sys.path.insert(0, os.getcwd())
    import numpy as np
    from toolkit.euler_relay import _COS, _SIN, N
    from toolkit.gap_metrics import go_map_score, string_tension

    COS_V   = _COS.copy()
    SIN_V   = _SIN.copy()
    COS2X_V = np.array([1., 0., -4., 0., 16., 0., -64., 0.])
    import math
    X = np.linspace(-1.5, 1.5, 120)
    BASIS = np.column_stack([X**k / math.factorial(k) for k in range(N)])
    SIN_X2_V, _, _, _ = np.linalg.lstsq(BASIS, np.sin(X**2), rcond=None)
    SIN_2X_V, _, _, _ = np.linalg.lstsq(BASIS, np.sin(2 * X), rcond=None)

    pairs = [
        ("cos->sin",      COS_V, SIN_V),
        ("cos->2cos",     COS_V, 2*COS_V),
        ("cos->cos2x",    COS_V, COS2X_V),
        ("sinx2->sin2x",  SIN_X2_V, SIN_2X_V),
    ]

    xs, ys, regimes, labels = [], [], [], []
    for name, a, b in pairs:
        n_lerp = 9
        images = [a + t * (b - a) for t in np.linspace(0, 1, n_lerp)]
        b1  = go_map_score(a, b)
        st  = string_tension(images, a, b)
        xs.append(1.0 - b1)
        ys.append(st["sigma_norm"])
        regimes.append(st["regime"])
        labels.append(name)
        print(f"  Anchor {name:20s}: x={1-b1:.4f}  sigma={st['sigma_norm']:.4f}  {st['regime']}")
    return np.array(xs), np.array(ys), regimes, labels


def run_fit():
    print("=" * 65)
    print("  FOLD CATASTROPHE 3/2 SCALING FIT")
    print("=" * 65)

    xs_tax, ys_tax, reg_tax, lab_tax = load_data()
    print(f"\nLoaded {len(xs_tax)} records from taxonomy.")

    print(f"\nCanonical anchor cases (span the full incompatibility range):")
    xs_anc, ys_anc, reg_anc, lab_anc = load_canonical_anchors()

    # Combine: taxonomy (many points, small x range) + anchors (4 pts, full range)
    xs = np.concatenate([xs_tax, xs_anc])
    ys = np.concatenate([ys_tax, ys_anc])
    regimes = reg_tax + reg_anc
    labels  = lab_tax + lab_anc
    print(f"\nCombined dataset: {len(xs)} records "
          f"(x range: {xs.min():.4f} to {xs.max():.4f})")
    print(f"  Taxonomy records:  {len(xs_tax)} (x in [{xs_tax.min():.4f}, {xs_tax.max():.4f}])")
    print(f"  Anchor cases:      {len(xs_anc)} (x in [{xs_anc.min():.4f}, {xs_anc.max():.4f}])")

    # Exclude TEARING (different catastrophe branch)
    mask_fold = np.array([r in ("FREE", "CONFINEMENT") for r in regimes])
    xf, yf = xs[mask_fold], ys[mask_fold]
    n_tearing = (~mask_fold).sum()
    print(f"  Records with regime FREE or CONFINEMENT: {mask_fold.sum()}")
    print(f"  Records with regime TEARING (excluded):  {n_tearing}")

    # -- Descriptive statistics -----------------------------------------------
    print(f"\nDescriptive statistics (fold records only):")
    print(f"  x = 1 - go_map_b1 (incompatibility):")
    print(f"    min={xf.min():.4f}  max={xf.max():.4f}  "
          f"mean={xf.mean():.4f}  median={np.median(xf):.4f}")
    print(f"  y = sigma_norm (string tension):")
    print(f"    min={yf.min():.4f}  max={yf.max():.4f}  "
          f"mean={yf.mean():.4f}  median={np.median(yf):.4f}")
    r_result = pearsonr(xf, yf)
    r = float(r_result.statistic) if hasattr(r_result, 'statistic') else float(r_result[0])
    p = float(r_result.pvalue)    if hasattr(r_result, 'pvalue')    else float(r_result[1])
    print(f"  Pearson r(x,y) = {r:.4f}  p = {p:.2e}")

    # -- Fit 1: Fixed alpha = 3/2 (fold catastrophe prediction) ---------------
    print(f"\nFit 1: Fixed fold exponent alpha=1.5  [sigma = C * x^1.5]")
    try:
        popt1, pcov1 = curve_fit(fold_model, xf, yf, p0=[1.0],
                                  bounds=([0.0], [np.inf]), maxfev=5000)
        C_fold = popt1[0]
        y_pred_fold = fold_model(xf, C_fold)
        ss_res  = np.sum((yf - y_pred_fold) ** 2)
        ss_tot  = np.sum((yf - yf.mean()) ** 2)
        r2_fold = 1.0 - ss_res / ss_tot
        rmse_fold = float(np.sqrt(np.mean((yf - y_pred_fold) ** 2)))
        print(f"  C = {C_fold:.4f}")
        print(f"  R^2 = {r2_fold:.4f}  RMSE = {rmse_fold:.4f}")
        print(f"  Interpretation: sigma_norm = {C_fold:.3f} * (1 - go_map_b1)^1.5")
    except Exception as e:
        print(f"  Fit failed: {e}")
        C_fold, r2_fold, rmse_fold = None, None, None

    # -- Fit 2: Free exponent (let alpha vary) ---------------------------------
    print(f"\nFit 2: Free power law  [sigma = C * x^alpha]")
    try:
        popt2, pcov2 = curve_fit(power_law, xf, yf, p0=[1.0, 1.5],
                                  bounds=([0.0, 0.1], [np.inf, 10.0]), maxfev=5000)
        C_free, alpha_free = popt2
        perr = np.sqrt(np.diag(pcov2))
        y_pred_free = power_law(xf, C_free, alpha_free)
        ss_res   = np.sum((yf - y_pred_free) ** 2)
        r2_free  = 1.0 - ss_res / ss_tot
        rmse_free = float(np.sqrt(np.mean((yf - y_pred_free) ** 2)))
        print(f"  C     = {C_free:.4f}  +/- {perr[0]:.4f}")
        print(f"  alpha = {alpha_free:.4f}  +/- {perr[1]:.4f}")
        print(f"  R^2 = {r2_free:.4f}  RMSE = {rmse_free:.4f}")
        print(f"  Interpretation: sigma_norm = {C_free:.3f} * (1 - go_map_b1)^{alpha_free:.3f}")
    except Exception as e:
        print(f"  Fit failed: {e}")
        alpha_free, r2_free, rmse_free = None, None, None

    # -- Interpretation -------------------------------------------------------
    print(f"\nINTERPRETATION:")
    if alpha_free is not None:
        if abs(alpha_free - 1.5) < 0.3:
            print(f"  alpha={alpha_free:.3f} ~ 1.5 -> FOLD CATASTROPHE confirmed.")
            print(f"  The confinement transition IS a codimension-1 fold.")
            print(f"  Our gap metrics measure the fold catastrophe barrier height.")
        elif alpha_free < 1.0:
            print(f"  alpha={alpha_free:.3f} < 1.0 -> LINEAR or SUBCRITICAL onset.")
            print(f"  Not a pure fold: confinement grows faster than predicted.")
        elif alpha_free > 2.0:
            print(f"  alpha={alpha_free:.3f} > 2.0 -> QUADRATIC onset.")
            print(f"  Closer to CUSP catastrophe (codimension 2).")
        else:
            print(f"  alpha={alpha_free:.3f} between 1.0 and 2.0 -> partial fold.")
            print(f"  Mixture of catastrophe types or non-universal dataset.")

    # -- Regime breakdown -------------------------------------------------------
    free_mask = np.array([r == "FREE"        for r in regimes])
    conf_mask = np.array([r == "CONFINEMENT" for r in regimes])
    tear_mask = np.array([r == "TEARING"     for r in regimes])
    print(f"\nRegime breakdown (all {len(xs)} records):")
    print(f"  FREE:         {free_mask.sum():4d}  ({100*free_mask.mean():.1f}%)"
          f"  mean_sigma={ys[free_mask].mean():.4f}" if free_mask.any() else "  FREE: 0")
    print(f"  CONFINEMENT:  {conf_mask.sum():4d}  ({100*conf_mask.mean():.1f}%)"
          f"  mean_sigma={ys[conf_mask].mean():.4f}" if conf_mask.any() else "  CONFINEMENT: 0")
    print(f"  TEARING:      {tear_mask.sum():4d}  ({100*tear_mask.mean():.1f}%)"
          f"  mean_sigma={ys[tear_mask].mean():.4f}" if tear_mask.any() else "  TEARING: 0")

    # -- Top Type-7 scoring morphs ---------------------------------------------
    type7_scores = np.array([r.get("gap_type7_score", 0.0)
                              for r in json.loads(TAXONOMY.read_text())
                              if r.get("gap_type7_score") is not None])
    all_r = [r for r in json.loads(TAXONOMY.read_text())
             if r.get("gap_type7_score") is not None]
    top_t7 = sorted(all_r, key=lambda r: -r["gap_type7_score"])[:8]
    print(f"\nTop 8 Type-7 scoring morphs (gap_type7_score = 1 - b1*(1-sigma)):")
    print(f"  {'Start':20s}  {'End':20s}  b1      sigma   regime      t7_score")
    for r in top_t7:
        print(f"  {r.get('start','?')[:20]:20s}  {r.get('end','?')[:20]:20s}  "
              f"{r.get('gap_go_map_b1',0):.4f}  "
              f"{r.get('gap_sigma_norm',0):.4f}  "
              f"{r.get('gap_sigma_regime','?'):12s}  "
              f"{r.get('gap_type7_score',0):.4f}")

    # -- Compare dual-confirmed vs gap_type7_score ----------------------------
    print(f"\nAgreement between dual-confirmed Type 7 and gap_type7_score > 0.9:")
    all_gap = [r for r in json.loads(TAXONOMY.read_text())
               if r.get("gap_type7_score") is not None
               and r.get("filter_coverage") not in ("unknown", None)]
    dual_t7     = [r for r in all_gap if r.get("dual_confirmed_type7")]
    gap_flagged = [r for r in all_gap if r.get("gap_type7_score", 0) > 0.9]
    both        = [r for r in all_gap if r.get("dual_confirmed_type7")
                   and r.get("gap_type7_score", 0) > 0.9]
    print(f"  Dual-confirmed Type 7 (2-adic+spectral): {len(dual_t7)}")
    print(f"  Gap-flagged (gap_type7_score > 0.9):     {len(gap_flagged)}")
    print(f"  Agree on both:                           {len(both)}")
    if dual_t7:
        agree_rate = len(both) / len(dual_t7)
        print(f"  Agreement rate (how many dual-confirmed are also gap-flagged): {agree_rate:.1%}")

    print("\n" + "=" * 65)
    print("  FIT COMPLETE")
    print("=" * 65)

    return {
        "n_records":  len(xs),
        "r_pearson":  r,
        "C_fold":     C_fold,
        "r2_fold":    r2_fold,
        "C_free":     C_free,
        "alpha_free": alpha_free,
        "r2_free":    r2_free,
    }


if __name__ == "__main__":
    run_fit()
