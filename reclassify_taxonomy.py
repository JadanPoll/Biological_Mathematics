"""
Reclassify the algebraic origami taxonomy with the O(n) pre-filters.

Applies:
  1. 2-adic pre-filter   (toolkit/adic_filter.py)   — resolution-level compatibility
  2. Spectral warmup filter (toolkit/spectral_warmup.py) — low-frequency disagreement

Both filters are O(n) — no gradient evaluations needed.
Processing 509 morphs takes < 1 second.

New fields added to each record:
  adic_incompatibility  : float  — combined 2-adic incompatibility score
  adic_sparsity_change  : int    — WHT sparsity change (primary Type 7 signal)
  adic_valuation_gap    : float  — difference in 2-adic valuations of dominant modes
  adic_is_type7         : bool   — 2-adic filter Type 7 flag
  spectral_low_freq_dist: float  — spectral disagreement at low Walsh bands
  spectral_flag_main    : bool   — main-effect band disagreement flag
  spectral_is_type7     : bool   — spectral filter Type 7 flag
  dual_confirmed_type7  : bool   — BOTH filters agree → high-confidence Type 7
  filter_coverage       : str    — "both" | "adic_only" | "spectral_only" | "neither"

Also adds SLERP step-distance variance for valid morphs (Type 7 candidates skip):
  slerp_wass_var        : float  — geodesic step evenness (low = well-spaced path exists)
  slerp_theta_deg       : float  — rotation angle in n-sphere (degrees)

Output:
  algebraic_origami_taxonomy_enriched.json  — original + new fields
  Prints summary statistics to stdout.
"""

import json
import time
import numpy as np
from pathlib import Path

from experiments.algebraic_origami_explorer import build_function_library
from toolkit.adic_filter     import adic_compatibility
from toolkit.spectral_warmup import spectral_type7_prefilter
from toolkit.slerp_relay     import run_slerp
from toolkit.gap_metrics     import go_map_score, go_map_score_b2, wht_residual_spectrum, string_tension


TAXONOMY_IN  = Path("algebraic_origami_taxonomy.json")
TAXONOMY_OUT = Path("algebraic_origami_taxonomy_enriched.json")


def enrich_record(record: dict, lib: dict) -> dict:
    """
    Add pre-filter and SLERP fields to one taxonomy record.
    Returns the record unchanged (plus new fields) if function names are known.
    If either name is not in the library, adds 'filter_coverage': 'unknown'.
    """
    start_name = record.get("start", "")
    end_name   = record.get("end",   "")

    if start_name not in lib or end_name not in lib:
        record["filter_coverage"] = "unknown"
        return record

    a = lib[start_name]
    b = lib[end_name]

    # ── 2-adic filter ──────────────────────────────────────────────────────
    adic = adic_compatibility(a, b)
    record["adic_incompatibility"]   = float(adic["incompatibility"])
    record["adic_sparsity_change"]   = int(adic["sparsity_change"])
    record["adic_valuation_gap"]     = float(adic["valuation_gap"])
    record["adic_is_type7"]          = bool(adic["is_type7_candidate"])

    # ── Spectral warmup filter ─────────────────────────────────────────────
    sw = spectral_type7_prefilter(a, b)
    record["spectral_low_freq_dist"] = float(sw["low_freq_dist"])
    record["spectral_flag_main"]     = bool(sw["flag_main_effect"])
    record["spectral_is_type7"]      = bool(sw["is_type7_candidate"])

    # ── Dual confirmation ──────────────────────────────────────────────────
    dual = bool(adic["is_type7_candidate"] and sw["is_type7_candidate"])
    record["dual_confirmed_type7"] = dual

    # Coverage summary
    if adic["is_type7_candidate"] and sw["is_type7_candidate"]:
        record["filter_coverage"] = "both"
    elif adic["is_type7_candidate"]:
        record["filter_coverage"] = "adic_only"
    elif sw["is_type7_candidate"]:
        record["filter_coverage"] = "spectral_only"
    else:
        record["filter_coverage"] = "neither"

    # ── SLERP: geodesic step evenness ─────────────────────────────────────
    # For Type 7 candidates the SLERP path exists geometrically (SLERP is
    # always valid) but is less informative. We compute it for all records.
    try:
        slerp_res = run_slerp(a, b, n_steps=6)
        record["slerp_wass_var"]  = float(slerp_res["wass_var"])
        record["slerp_theta_deg"] = float(slerp_res["theta_deg"])
    except Exception:
        record["slerp_wass_var"]  = None
        record["slerp_theta_deg"] = None

    # -- Gap metrics (band-1, band-2, residual, string tension) ---------------
    # Uses 9-point linear-interpolation relay chain. Valid for wht_residual and
    # string_tension: both measure band-1 distance from lerp-midpoint to
    # endpoints, which is relay-method-independent.
    n_lerp = 9
    images = [a + t * (b - a) for t in np.linspace(0, 1, n_lerp)]
    record["gap_go_map_b1"]     = float(go_map_score(a, b))
    record["gap_go_map_b2"]     = float(go_map_score_b2(a, b))
    wht_res = wht_residual_spectrum(images, a, b)
    st      = string_tension(images, a, b)
    record["gap_residual_norm"] = float(wht_res["residual_norm"])
    record["gap_sigma_norm"]    = float(st["sigma_norm"])
    record["gap_sigma_regime"]  = str(st["regime"])
    # Combined Type 7 score: high when go_map_b1 low AND sigma high
    record["gap_type7_score"]   = float(1.0 - record["gap_go_map_b1"] *
                                         (1.0 - record["gap_sigma_norm"]))

    return record


def main():
    print("=" * 70)
    print("TAXONOMY RECLASSIFICATION — 2-adic + Spectral + SLERP enrichment")
    print("=" * 70)

    # Load taxonomy
    taxonomy = json.loads(TAXONOMY_IN.read_text())
    print(f"\nLoaded {len(taxonomy)} records from {TAXONOMY_IN}")

    # Build function library (same one the explorer uses)
    lib = build_function_library()
    print(f"Function library: {len(lib)} named functions")

    # Check coverage
    all_names = set()
    for r in taxonomy:
        all_names.add(r.get("start", ""))
        all_names.add(r.get("end",   ""))
    known    = all_names & set(lib.keys())
    unknown  = all_names - set(lib.keys())
    print(f"Names in taxonomy: {len(all_names)}  |  "
          f"known={len(known)}  |  unknown={len(unknown)}")
    if unknown:
        print(f"  Unknown names: {sorted(unknown)[:10]}{'...' if len(unknown) > 10 else ''}")

    # Enrich each record
    t0 = time.perf_counter()
    enriched = [enrich_record(dict(r), lib) for r in taxonomy]
    elapsed  = time.perf_counter() - t0
    print(f"\nEnrichment complete in {elapsed:.2f}s  "
          f"({len(enriched) / elapsed:.0f} records/sec)")

    # ── Summary statistics ─────────────────────────────────────────────────
    with_filters = [r for r in enriched if r.get("filter_coverage") not in ("unknown", None)]
    n = len(with_filters)

    if n == 0:
        print("\nNo records had known function names — check library coverage.")
        return

    n_both     = sum(1 for r in with_filters if r["filter_coverage"] == "both")
    n_adic     = sum(1 for r in with_filters if r["filter_coverage"] == "adic_only")
    n_spectral = sum(1 for r in with_filters if r["filter_coverage"] == "spectral_only")
    n_neither  = sum(1 for r in with_filters if r["filter_coverage"] == "neither")
    n_dual     = sum(1 for r in with_filters if r["dual_confirmed_type7"])

    print(f"\n{'-'*60}")
    print(f"TYPE 7 DETECTION SUMMARY ({n} records with known functions)")
    print(f"{'-'*60}")
    print(f"  Dual-confirmed Type 7  (both flags): {n_dual:4d}  "
          f"({100*n_dual/n:.1f}%)")
    print(f"  2-adic flag only:                    {n_adic:4d}  "
          f"({100*n_adic/n:.1f}%)")
    print(f"  Spectral flag only:                  {n_spectral:4d}  "
          f"({100*n_spectral/n:.1f}%)")
    print(f"  Neither flagged (valid morphs):      {n_neither:4d}  "
          f"({100*n_neither/n:.1f}%)")

    # Distribution of incompatibility scores
    incompat = [r["adic_incompatibility"] for r in with_filters]
    print(f"\n  2-adic incompatibility: "
          f"min={min(incompat):.3f}  "
          f"max={max(incompat):.3f}  "
          f"mean={np.mean(incompat):.3f}  "
          f"median={np.median(incompat):.3f}")

    sparsity_changes = [r["adic_sparsity_change"] for r in with_filters]
    print(f"  WHT sparsity change:    "
          f"min={min(sparsity_changes)}  "
          f"max={max(sparsity_changes)}  "
          f"mean={np.mean(sparsity_changes):.2f}")

    # SLERP statistics
    theta_vals = [r["slerp_theta_deg"] for r in with_filters
                  if r.get("slerp_theta_deg") is not None]
    if theta_vals:
        print(f"\n  SLERP rotation angles:  "
              f"min={min(theta_vals):.1f}°  "
              f"max={max(theta_vals):.1f}°  "
              f"mean={np.mean(theta_vals):.1f}°")
        wv_vals = [r["slerp_wass_var"] for r in with_filters
                   if r.get("slerp_wass_var") is not None]
        print(f"  SLERP wass_var:         "
              f"min={min(wv_vals):.4f}  "
              f"max={max(wv_vals):.4f}  "
              f"mean={np.mean(wv_vals):.4f}")

    # Show top-10 most incompatible morphs
    top10 = sorted(with_filters, key=lambda r: r["adic_incompatibility"], reverse=True)[:10]
    print(f"\nTOP 10 MOST INCOMPATIBLE MORPHS (highest 2-adic score):")
    print(f"  {'Start':25s} {'End':25s}  incompat  sp_chg  theta°  dual")
    for r in top10:
        print(f"  {r['start']:25s} {r['end']:25s}  "
              f"{r['adic_incompatibility']:7.3f}  "
              f"{r['adic_sparsity_change']:5d}  "
              f"{r.get('slerp_theta_deg', 0.0):6.1f}  "
              f"{'YES' if r['dual_confirmed_type7'] else 'no'}")

    # Show top-10 valid morphs (lowest incompatibility, high TPI)
    valid = [r for r in with_filters if not r["dual_confirmed_type7"]]
    if valid:
        top_valid = sorted(valid,
                           key=lambda r: r.get("topological_protection_index", 0),
                           reverse=True)[:10]
        print(f"\nTOP 10 VALID MORPHS BY TPI (most topologically protected):")
        print(f"  {'Start':25s} {'End':25s}  incompat   TPI   theta°")
        for r in top_valid:
            print(f"  {r['start']:25s} {r['end']:25s}  "
                  f"{r['adic_incompatibility']:7.3f}  "
                  f"{r.get('topological_protection_index', 0.0):5.3f}  "
                  f"{r.get('slerp_theta_deg', 0.0):6.1f}")

    # Save enriched taxonomy
    TAXONOMY_OUT.write_text(json.dumps(enriched, indent=2))
    print(f"\nSaved enriched taxonomy -> {TAXONOMY_OUT}  ({len(enriched)} records)")
    print(f"New fields per record: adic_incompatibility, adic_sparsity_change, "
          f"adic_valuation_gap, adic_is_type7,\n"
          f"  spectral_low_freq_dist, spectral_flag_main, spectral_is_type7, "
          f"dual_confirmed_type7, filter_coverage,\n"
          f"  slerp_wass_var, slerp_theta_deg")


if __name__ == "__main__":
    main()
