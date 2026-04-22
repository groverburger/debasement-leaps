"""Lindy LEAPS — two lenses on the same options.

For each of the top Lindy compounders, pulls ALL options chains and
computes fair value under two independent frames:

1. LINDY lens: "if the historical trend continues at its observed rate,
   what is this option worth?" Uses min_slope (the Lindy floor — most
   conservative rate the stock has ever compounded at) as drift.
   No assumptions about M2 or beta. Pure observed data.

2. DEBASEMENT lens: "if M2 grows at X% and this stock has beta Y to M2,
   what is this option worth?" Uses β × M2 as drift.
   Requires two assumptions (M2 rate + beta).

Both produce a reward_pct = (fair_value - ask) / ask — return on capital
at risk. R² provides the confidence bound on the Lindy estimate.

CLI:
    python3 lindy_leaps.py                   # default: top 20 Lindy names
    python3 lindy_leaps.py --top 30
    python3 lindy_leaps.py --m2 0.06         # conservative M2 for debasement lens
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


def pull_all_options(
    tv: TVOptions,
    tv_sym: str,
    min_slope: float,
    beta: float,
    m2: float = DEFAULT_M2,
    r: float = DEFAULT_R,
) -> list[dict]:
    """Pull every call option across all DTEs ≥ MIN_DTE.

    Computes fair value under BOTH lenses for each option.
    """
    try:
        spot = tv.get_underlying_price(tv_sym)
        if not spot or spot <= 0:
            return []
        exps = tv.get_expirations(tv_sym)
    except Exception:
        return []

    mu_lindy = min_slope / 100   # Lindy lens: use the observed floor rate
    mu_debase = beta * m2        # Debasement lens: β × M2
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

            # Lindy lens: fair value if trend continues at min_slope
            lindy_fair = debase_fair_value(spot, K, T, r, iv, mu_lindy)
            lindy_rwd = (lindy_fair - ask) / ask * 100 if ask > 0 else 0.0

            # Debasement lens: fair value if β × M2 drift
            debase_fair = debase_fair_value(spot, K, T, r, iv, mu_debase)
            debase_rwd = (debase_fair - ask) / ask * 100 if ask > 0 else 0.0

            be = breakeven_annualized_growth(spot, K, T, mid)

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
                "lindy_fair": lindy_fair,
                "lindy_rwd": lindy_rwd,
                "debase_fair": debase_fair,
                "debase_rwd": debase_rwd,
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
    """Pull all LEAPS for top Lindy names, return best per ticker."""
    tv = TVOptions()
    combined: list[dict] = []

    for entry in lindy_results[:top_n]:
        ticker = entry["ticker"]
        tv_sym = entry.get("tv_symbol", "")
        if not tv_sym or ":" not in tv_sym:
            if verbose:
                print(f"{ticker:6s} — no TV symbol, skipping")
            continue

        min_slope = entry.get("min_slope", 12.0)

        # Derive beta from longest slope
        for sl_key in ["sl_max", "sl_20y", "sl_15y", "sl_10y", "sl_5y", "sl_2y"]:
            sl = entry.get(sl_key)
            if sl is not None and sl > 0:
                beta = (sl / 100) / m2
                break
        else:
            beta = entry.get("beta", 1.5)

        options = pull_all_options(tv, tv_sym, min_slope, beta, m2, r)

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
                "min_slope": min_slope,
                "beta_used": round(beta, 1),
                "has_leaps": False,
            })
            continue

        # Best option by LINDY reward (the primary lens)
        best = max(options, key=lambda o: o["lindy_rwd"])

        if verbose:
            print(f"{ticker:6s} slope={min_slope:+.1f}%  β={beta:.1f}  {len(options):3d} opts  "
                  f"K={best['strike']:.0f} DTE={best['dte']}  "
                  f"ask=${best['ask']:.2f}  "
                  f"lindy={best['lindy_rwd']:+.0f}%  "
                  f"debase={best['debase_rwd']:+.0f}%  "
                  f"IV={best['iv']*100:.0f}%")

        combined.append({
            "ticker": ticker,
            "lindy_rank": lindy_results.index(entry) + 1,
            "lindy_score": entry.get("lindy_score"),
            "avg_r2": entry.get("avg_r2"),
            "min_r2": entry.get("min_r2"),
            "total_years": entry.get("total_years"),
            "min_slope": min_slope,
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
            "lindy_fair": best["lindy_fair"],
            "lindy_rwd": best["lindy_rwd"],
            "debase_fair": best["debase_fair"],
            "debase_rwd": best["debase_rwd"],
            "be_g_pct": best["be_g_pct"],
        })

    return combined


def format_combined_table(results: list[dict]) -> str:
    """Format combined Lindy + LEAPS results."""
    lines = []
    hdr = (f"{'#':>3s} {'tick':6s} {'L#':>3s} {'LINDY':>6s} {'R²':>5s} {'yrs':>5s} "
           f"{'DTE':>4s} {'K':>7s} {'ask':>7s} "
           f"{'L_fair':>7s} {'L_rwd%':>7s} "
           f"{'D_fair':>7s} {'D_rwd%':>7s} "
           f"{'IV':>3s} {'Δ':>4s} {'BE%':>5s}")
    lines.append(hdr)
    for i, r in enumerate(results):
        if not r.get("has_leaps"):
            lines.append(
                f"{i+1:3d} {r['ticker']:6s} {r.get('lindy_rank',0):3d} "
                f"{r.get('lindy_score', 0):6.3f} {r.get('avg_r2', 0):5.3f} "
                f"{r.get('total_years', 0):5.1f} "
                f"{'— no LEAPS available':50s}"
            )
            continue
        lines.append(
            f"{i+1:3d} {r['ticker']:6s} {r.get('lindy_rank',0):3d} "
            f"{r.get('lindy_score', 0):6.3f} {r.get('avg_r2', 0):5.3f} "
            f"{r.get('total_years', 0):5.1f} "
            f"{r['dte']:4d} {r['strike']:7.1f} {r['ask']:7.2f} "
            f"{r['lindy_fair']:7.2f} {r['lindy_rwd']:+7.1f} "
            f"{r['debase_fair']:7.2f} {r['debase_rwd']:+7.1f} "
            f"{r['iv']*100:3.0f} {r['delta']:4.2f} {r['be_g_pct']:+5.1f}"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Lindy LEAPS — two lenses on the best compounders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--m2", type=float, default=DEFAULT_M2,
                   help=f"M2 growth assumption for debasement lens (default {DEFAULT_M2})")
    p.add_argument("--rfr", type=float, default=DEFAULT_R)
    p.add_argument("--fresh", action="store_true",
                   help="Re-run lindy_scan before pulling chains")
    p.add_argument("--min-slope", type=float, default=12.0)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

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

    if not args.json:
        print(f"\nPulling ALL chains for top {args.top} Lindy names...\n")

    combined = scan_leaps(
        lindy_results=lindy_results,
        top_n=args.top,
        m2=args.m2,
        r=args.rfr,
        verbose=not args.json,
    )

    with_leaps = [r for r in combined if r.get("has_leaps")]
    without_leaps = [r for r in combined if not r.get("has_leaps")]
    with_leaps.sort(key=lambda r: r.get("lindy_rwd", 0), reverse=True)
    final = with_leaps + without_leaps

    if args.json:
        print(json.dumps({
            "m2": args.m2, "rfr": args.rfr,
            "count": len(final),
            "with_leaps": len(with_leaps),
            "results": final,
        }, indent=2))
    else:
        print(f"\n=== LINDY × LEAPS — two lenses ===")
        print(f"L_rwd% = if Lindy trend continues at min_slope (pure observed data)")
        print(f"D_rwd% = if M2={args.m2*100:.0f}% × historical β (debasement thesis)")
        print(f"Both = (fair_value - ask) / ask — return on capital at risk\n")
        print(format_combined_table(final))

    return 0


if __name__ == "__main__":
    sys.exit(main())
