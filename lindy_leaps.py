"""Lindy LEAPS — layer debasement pricing on top of the Lindy IR scan.

Reads the Lindy IR ranking (from lindy_scan.py cache or runs it fresh),
then pulls LEAPS chains for the top names and scores by debasement edge.

The final output is a combined table: Lindy trend quality + LEAPS edge.
This is the canonical "what should I buy?" output.

Pipeline:
    1. Load Lindy IR results (run lindy_scan.py first, or --fresh)
    2. For each top-N name, pull LEAPS chains via TradingView
    3. Compute debasement fair value using per-name beta from the universe
    4. Score and rank by debasement edge
    5. Output combined table: Lindy metrics + best LEAPS pick

CLI:
    python3 lindy_leaps.py                   # default: top 20
    python3 lindy_leaps.py --top 30
    python3 lindy_leaps.py --m2 0.06         # lower M2 assumption
    python3 lindy_leaps.py --fresh           # re-run lindy_scan first
    python3 lindy_leaps.py --json
    python3 lindy_leaps.py --use-ir-beta     # derive beta from IR slope instead of universe
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Documents/projects/darkfield"))

from chain_analysis import analyze_ticker, DEFAULT_M2, DEFAULT_R, DEFAULT_DELTA_RANGE
from tv_options import TVOptions

LINDY_CACHE = os.path.join(os.path.dirname(__file__), "lindy_ir_cache.json")


def load_lindy_results() -> list[dict]:
    """Load cached Lindy IR results. Run lindy_scan.py first."""
    if not os.path.exists(LINDY_CACHE):
        return []
    with open(LINDY_CACHE) as f:
        return json.load(f)


def scan_leaps(
    lindy_results: list[dict],
    top_n: int = 20,
    m2: float = DEFAULT_M2,
    r: float = DEFAULT_R,
    use_ir_beta: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """Pull LEAPS chains for top Lindy names and compute debasement edge.

    Args:
        lindy_results: output of lindy_scan.scan_lindy().
        top_n: how many Lindy names to pull chains for.
        m2: M2 growth assumption.
        r: risk-free rate.
        use_ir_beta: if True, derive beta from 10yr slope / m2 instead of
            universe-assigned beta. More empirical, less thematic.
        verbose: print progress.

    Returns:
        List of combined dicts: Lindy metrics + best LEAPS pick per ticker.
    """
    tv = TVOptions()
    combined: list[dict] = []

    for entry in lindy_results[:top_n]:
        ticker = entry["ticker"]
        tv_sym = entry.get("tv_symbol", "")
        if not tv_sym or ":" not in tv_sym:
            if verbose:
                print(f"{ticker:6s} — no TV symbol, skipping")
            continue

        # Determine beta
        if use_ir_beta:
            # Derive from longest-window slope
            for sl_key in ["sl_max", "sl_20y", "sl_15y", "sl_10y", "sl_5y", "sl_2y"]:
                sl = entry.get(sl_key)
                if sl is not None and sl > 0:
                    beta = (sl / 100) / m2
                    break
            else:
                beta = entry.get("beta", 1.5)
        else:
            beta = entry.get("beta", 1.5)

        try:
            rows = analyze_ticker(
                symbol=tv_sym, beta=beta, m2=m2, r=r,
                delta_range=DEFAULT_DELTA_RANGE, tv=tv,
            )
        except Exception as e:
            if verbose:
                print(f"{ticker:6s} ERR: {str(e)[:50]}")
            continue

        if not rows:
            if verbose:
                print(f"{ticker:6s} — no eligible LEAPS")
            # Still include in output with null LEAPS data
            combined.append({
                **entry,
                "beta_used": beta,
                "has_leaps": False,
            })
            continue

        best = max(rows, key=lambda r_: r_["score"])
        if verbose:
            print(f"{ticker:6s} β={beta:.1f}  K={best['strike']:.0f} DTE={best['dte']}  "
                  f"edge=${best['edge_usd']:+.2f} ({best['edge_pct']:+.1f}%)  "
                  f"IV={best['iv']*100:.0f}% BE={best['be_g_pct']:+.1f}%")

        combined.append({
            # Lindy metrics
            "ticker": ticker,
            "tv_symbol": tv_sym,
            "total_years": entry.get("total_years"),
            "lindy_ir": entry.get("lindy_ir"),
            "min_ir": entry.get("min_ir"),
            "avg_ir": entry.get("avg_ir"),
            "min_slope": entry.get("min_slope"),
            "n_windows": entry.get("n_windows"),
            # LEAPS metrics
            "has_leaps": True,
            "beta_used": round(beta, 2),
            "spot": best["spot"],
            "strike": best["strike"],
            "dte": best["dte"],
            "exp": best["exp"],
            "ask": best["ask"],
            "mid": best["mid"],
            "iv": best["iv"],
            "delta": best["delta"],
            "debase_fair": best["debase_fair"],
            "edge_usd": best["edge_usd"],
            "edge_pct": best["edge_pct"],
            "be_g_pct": best["be_g_pct"],
            "leaps_score": best["score"],
        })

    return combined


def format_combined_table(results: list[dict]) -> str:
    """Format combined Lindy + LEAPS results as text table."""
    lines = []
    hdr = (f"{'#':>3s} {'tick':6s} {'yrs':>5s} {'L_IR':>5s} {'minIR':>5s} "
           f"{'β':>4s} {'DTE':>4s} {'K':>7s} {'ask':>7s} {'MID':>7s} "
           f"{'DB':>7s} {'EDGE$':>7s} {'EDGE%':>6s} {'IV':>3s} {'BE%':>5s}")
    lines.append(hdr)
    for i, r in enumerate(results):
        if not r.get("has_leaps"):
            lines.append(
                f"{i+1:3d} {r['ticker']:6s} {r.get('total_years', 0):5.1f} "
                f"{r.get('lindy_ir', 0):5.2f} {r.get('min_ir', 0):5.2f} "
                f"{'':>4s} {'— no LEAPS available':40s}"
            )
            continue
        lines.append(
            f"{i+1:3d} {r['ticker']:6s} {r.get('total_years', 0):5.1f} "
            f"{r.get('lindy_ir', 0):5.2f} {r.get('min_ir', 0):5.2f} "
            f"{r['beta_used']:4.1f} {r['dte']:4d} {r['strike']:7.1f} "
            f"{r['ask']:7.2f} {r['mid']:7.2f} {r['debase_fair']:7.2f} "
            f"{r['edge_usd']:+7.2f} {r['edge_pct']:+6.1f} {r['iv']*100:3.0f} "
            f"{r['be_g_pct']:+5.1f}"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Lindy LEAPS — debasement pricing on the most Lindy compounders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--top", type=int, default=20,
                   help="Number of Lindy names to pull chains for (default 20)")
    p.add_argument("--m2", type=float, default=DEFAULT_M2,
                   help=f"M2 growth assumption (default {DEFAULT_M2})")
    p.add_argument("--rfr", type=float, default=DEFAULT_R,
                   help=f"Risk-free rate (default {DEFAULT_R})")
    p.add_argument("--use-ir-beta", action="store_true",
                   help="Derive beta from historical slope instead of sector assignment")
    p.add_argument("--fresh", action="store_true",
                   help="Re-run lindy_scan before pulling chains")
    p.add_argument("--min-slope", type=float, default=12.0,
                   help="Min slope gate for lindy_scan (default 12)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    # Step 1: get Lindy results
    if args.fresh or not os.path.exists(LINDY_CACHE):
        if not args.json:
            print("Running Lindy scan first...\n")
        from lindy_scan import scan_lindy
        from build_universe import build_sp900
        universe = build_sp900(verbose=not args.json)
        lindy_results = scan_lindy(
            universe=universe,
            min_slope=args.min_slope,
            top_n=0,
            verbose=not args.json,
        )
        with open(LINDY_CACHE, "w") as f:
            json.dump(lindy_results, f, indent=2)
    else:
        lindy_results = load_lindy_results()
        if not args.json:
            print(f"Loaded {len(lindy_results)} Lindy results from cache")

    if not lindy_results:
        print("No Lindy results. Run: python3 lindy_scan.py")
        return 1

    # Step 2: pull LEAPS and score
    if not args.json:
        print(f"\nPulling LEAPS chains for top {args.top}...\n")

    combined = scan_leaps(
        lindy_results=lindy_results,
        top_n=args.top,
        m2=args.m2,
        r=args.rfr,
        use_ir_beta=args.use_ir_beta,
        verbose=not args.json,
    )

    # Sort by LEAPS edge score (names with chains first)
    with_leaps = [r for r in combined if r.get("has_leaps")]
    without_leaps = [r for r in combined if not r.get("has_leaps")]
    with_leaps.sort(key=lambda r: r.get("leaps_score", 0), reverse=True)
    final = with_leaps + without_leaps

    if args.json:
        print(json.dumps({
            "m2": args.m2, "rfr": args.rfr,
            "use_ir_beta": args.use_ir_beta,
            "count": len(final),
            "with_leaps": len(with_leaps),
            "results": final,
        }, indent=2))
    else:
        print(f"\n=== LINDY × LEAPS — sorted by debasement edge score ===")
        print(f"M2={args.m2*100:.0f}%  r={args.rfr*100:.1f}%  "
              f"beta={'from IR slope' if args.use_ir_beta else 'from sector'}\n")
        print(format_combined_table(final))

    return 0


if __name__ == "__main__":
    sys.exit(main())
