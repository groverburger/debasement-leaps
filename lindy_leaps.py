"""Lindy LEAPS — layer debasement pricing on top of the Lindy R² scan.

Reads the Lindy ranking (from lindy_scan.py cache or runs it fresh),
pulls ALL LEAPS chains for the top names, and ranks by % reward / capital
at risk:

    reward_pct = (debase_fair - ask) / ask × 100

This is the honest metric — you PAY the ask, not the mid. The output is
the canonical "what should I buy?" list.

Pipeline:
    1. Load Lindy R² results (run lindy_scan.py first, or --fresh)
    2. For each top-N Lindy name, pull ALL options (all strikes, all DTEs)
    3. Compute debasement fair value using empirical beta from slope
    4. Score each option by reward_pct (return on capital at risk)
    5. Pick best option per ticker, sort by reward_pct

CLI:
    python3 lindy_leaps.py                   # default: top 20 Lindy names
    python3 lindy_leaps.py --top 30
    python3 lindy_leaps.py --m2 0.06         # conservative M2
    python3 lindy_leaps.py --fresh           # re-run lindy_scan first
    python3 lindy_leaps.py --json
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Documents/projects/darkfield"))

from pricing import bs_call, debase_fair_value, breakeven_annualized_growth
from tv_options import TVOptions

LINDY_CACHE = os.path.join(os.path.dirname(__file__), "lindy_cache.json")
DEFAULT_M2 = 0.07
DEFAULT_R = 0.045
MIN_DTE = 150
DELTA_MIN = 0.25
DELTA_MAX = 0.80


def load_lindy_results() -> list[dict]:
    """Load cached Lindy R² results. Run lindy_scan.py first."""
    if not os.path.exists(LINDY_CACHE):
        return []
    with open(LINDY_CACHE) as f:
        return json.load(f)


def _derive_beta(entry: dict, m2: float) -> float:
    """Derive beta from the longest available historical slope."""
    for sl_key in ["sl_max", "sl_20y", "sl_15y", "sl_10y", "sl_5y", "sl_2y"]:
        sl = entry.get(sl_key)
        if sl is not None and sl > 0:
            return (sl / 100) / m2
    return entry.get("beta", 1.5)


def pull_all_options(
    tv: TVOptions,
    tv_sym: str,
    beta: float,
    m2: float = DEFAULT_M2,
    r: float = DEFAULT_R,
) -> list[dict]:
    """Pull every call option across all DTEs ≥ MIN_DTE.

    Returns list of dicts with debasement pricing and reward_pct.
    """
    try:
        spot = tv.get_underlying_price(tv_sym)
        if not spot or spot <= 0:
            return []
        exps = tv.get_expirations(tv_sym)
    except Exception:
        return []

    mu = beta * m2
    today = datetime.date.today()
    today_int = int(today.strftime("%Y%m%d"))
    rows: list[dict] = []

    for e in sorted(exps):
        if e <= today_int:
            continue
        try:
            ed = datetime.date(int(str(e)[:4]), int(str(e)[4:6]), int(str(e)[6:8]))
        except Exception:
            continue
        dte = (ed - today).days
        if dte < MIN_DTE:
            continue

        try:
            chain = tv.get_chain(tv_sym, e, spot=spot)
        except Exception:
            continue

        T = dte / 365.0
        for o in chain.options:
            if o.option_type != "call":
                continue
            if not o.mid or o.mid <= 0:
                continue
            if not o.iv or o.iv <= 0:
                continue
            if not o.ask or o.ask <= 0:
                continue
            delta = o.delta or 0.0
            if delta < DELTA_MIN or delta > DELTA_MAX:
                continue

            K = o.strike
            ask = o.ask
            mid = o.mid
            bid = o.bid or 0.0
            iv = o.iv
            db = debase_fair_value(spot, K, T, r, iv, mu)
            be = breakeven_annualized_growth(spot, K, T, mid)

            # The canonical metric: % reward on capital at risk
            reward_pct = (db - ask) / ask * 100 if ask > 0 else 0.0
            # Also keep mid-based edge for reference
            edge_pct = (db - mid) / mid * 100 if mid > 0 else 0.0

            rows.append({
                "spot": spot,
                "exp": e,
                "dte": dte,
                "strike": K,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "iv": iv,
                "delta": delta,
                "debase_fair": db,
                "reward_pct": reward_pct,
                "edge_pct": edge_pct,
                "be_g_pct": be * 100,
            })

    return rows


def scan_leaps(
    lindy_results: list[dict],
    top_n: int = 20,
    m2: float = DEFAULT_M2,
    r: float = DEFAULT_R,
    verbose: bool = True,
) -> list[dict]:
    """Pull all LEAPS for top Lindy names, return best per ticker by reward_pct."""
    tv = TVOptions()
    combined: list[dict] = []

    for entry in lindy_results[:top_n]:
        ticker = entry["ticker"]
        tv_sym = entry.get("tv_symbol", "")
        if not tv_sym or ":" not in tv_sym:
            if verbose:
                print(f"{ticker:6s} — no TV symbol, skipping")
            continue

        beta = _derive_beta(entry, m2)

        options = pull_all_options(tv, tv_sym, beta, m2, r)

        if not options:
            if verbose:
                print(f"{ticker:6s} — no eligible options")
            combined.append({
                "ticker": ticker,
                "lindy_rank": lindy_results.index(entry) + 1,
                "lindy_score": entry.get("lindy_score"),
                "avg_r2": entry.get("avg_r2"),
                "min_r2": entry.get("min_r2"),
                "total_years": entry.get("total_years"),
                "beta_used": round(beta, 1),
                "has_leaps": False,
                "n_options": 0,
            })
            continue

        # Best option by reward_pct
        best = max(options, key=lambda o: o["reward_pct"])

        if verbose:
            print(f"{ticker:6s} β={beta:.1f}  {len(options):3d} opts  "
                  f"best: K={best['strike']:.0f} DTE={best['dte']}  "
                  f"ask=${best['ask']:.2f} → reward={best['reward_pct']:+.1f}%  "
                  f"IV={best['iv']*100:.0f}% BE={best['be_g_pct']:+.1f}%")

        combined.append({
            "ticker": ticker,
            "lindy_rank": lindy_results.index(entry) + 1,
            "lindy_score": entry.get("lindy_score"),
            "avg_r2": entry.get("avg_r2"),
            "min_r2": entry.get("min_r2"),
            "total_years": entry.get("total_years"),
            "has_leaps": True,
            "beta_used": round(beta, 1),
            "n_options": len(options),
            "spot": best["spot"],
            "strike": best["strike"],
            "dte": best["dte"],
            "exp": best["exp"],
            "ask": best["ask"],
            "bid": best["bid"],
            "mid": best["mid"],
            "iv": best["iv"],
            "delta": best["delta"],
            "debase_fair": best["debase_fair"],
            "reward_pct": best["reward_pct"],
            "edge_pct": best["edge_pct"],
            "be_g_pct": best["be_g_pct"],
        })

    return combined


def format_combined_table(results: list[dict]) -> str:
    """Format combined Lindy + LEAPS results, sorted by reward_pct."""
    lines = []
    hdr = (f"{'#':>3s} {'tick':6s} {'L#':>3s} {'LINDY':>6s} {'avgR²':>5s} {'yrs':>5s} "
           f"{'β':>4s} {'DTE':>4s} {'K':>7s} {'ask':>7s} {'DB':>7s} "
           f"{'RWD%':>7s} {'IV':>3s} {'Δ':>4s} {'BE%':>5s}")
    lines.append(hdr)
    for i, r in enumerate(results):
        if not r.get("has_leaps"):
            lines.append(
                f"{i+1:3d} {r['ticker']:6s} {r.get('lindy_rank',0):3d} "
                f"{r.get('lindy_score', 0):6.3f} {r.get('avg_r2', 0):5.3f} "
                f"{r.get('total_years', 0):5.1f} "
                f"{'':>4s} {'— no LEAPS available':40s}"
            )
            continue
        lines.append(
            f"{i+1:3d} {r['ticker']:6s} {r.get('lindy_rank',0):3d} "
            f"{r.get('lindy_score', 0):6.3f} {r.get('avg_r2', 0):5.3f} "
            f"{r.get('total_years', 0):5.1f} "
            f"{r['beta_used']:4.1f} {r['dte']:4d} {r['strike']:7.1f} "
            f"{r['ask']:7.2f} {r['debase_fair']:7.2f} "
            f"{r['reward_pct']:+7.1f} {r['iv']*100:3.0f} "
            f"{r['delta']:4.2f} {r['be_g_pct']:+5.1f}"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Lindy LEAPS — best debasement options on the most Lindy compounders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--top", type=int, default=20,
                   help="Number of Lindy names to pull chains for (default 20)")
    p.add_argument("--m2", type=float, default=DEFAULT_M2,
                   help=f"M2 growth assumption (default {DEFAULT_M2})")
    p.add_argument("--rfr", type=float, default=DEFAULT_R,
                   help=f"Risk-free rate (default {DEFAULT_R})")
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

    # Step 2: pull ALL options and score by reward_pct
    if not args.json:
        print(f"\nPulling ALL chains for top {args.top} Lindy names...\n")

    combined = scan_leaps(
        lindy_results=lindy_results,
        top_n=args.top,
        m2=args.m2,
        r=args.rfr,
        verbose=not args.json,
    )

    # Sort by reward_pct (names with LEAPS first)
    with_leaps = [r for r in combined if r.get("has_leaps")]
    without_leaps = [r for r in combined if not r.get("has_leaps")]
    with_leaps.sort(key=lambda r: r.get("reward_pct", 0), reverse=True)
    final = with_leaps + without_leaps

    if args.json:
        print(json.dumps({
            "m2": args.m2, "rfr": args.rfr,
            "metric": "reward_pct = (debase_fair - ask) / ask",
            "count": len(final),
            "with_leaps": len(with_leaps),
            "results": final,
        }, indent=2))
    else:
        print(f"\n=== LINDY × LEAPS — ranked by % reward / capital at risk ===")
        print(f"RWD% = (debase_fair - ask) / ask — return on money you actually spend")
        print(f"M2={args.m2*100:.0f}%  r={args.rfr*100:.1f}%  "
              f"beta=derived from historical slope\n")
        print(format_combined_table(final))

    return 0


if __name__ == "__main__":
    sys.exit(main())
