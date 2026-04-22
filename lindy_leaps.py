"""Lindy LEAPS — deviation-adjusted LEAPS edge on the best compounders.

For each of the top Lindy compounders, pulls ALL options chains and
computes the LEAPS edge under two lenses, PLUS a deviation adjustment
that accounts for where spot sits relative to the long-term trend line.

The deviation adjustment:
  If a stock is BELOW its regression line, the Lindy thesis says it will
  revert — meaning the effective drift during the LEAPS period is HIGHER
  than the raw slope (you get the trend + the snap-back). If ABOVE, the
  drift is LOWER (you're paying a premium for entry).

  effective_drift = slope + (min_R² × -deviation / T)

  min_R² scales the confidence: R²=0.97 → 97% reversion credit.
  The built-in stop-loss of LEAPS (premium = max loss) makes this safe
  to include — you can't lose more than the premium if reversion fails.

Ranking layers:
  lindy_scan.py → ranks STOCKS by trend quality (CASY #1, timeless)
  lindy_leaps.py → ranks TRADES by edge (point-in-time, deviation-adjusted)

CLI:
    python3 lindy_leaps.py                   # default: top 20 Lindy names
    python3 lindy_leaps.py --top 30
    python3 lindy_leaps.py --m2 0.06         # conservative M2 for debasement lens
    python3 lindy_leaps.py --fresh           # re-run lindy_scan first
    python3 lindy_leaps.py --charts          # generate edge visualizations
    python3 lindy_leaps.py --json
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys

import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.expanduser("~/Documents/projects/darkfield"))

from pricing import debase_fair_value, breakeven_annualized_growth
from tv_options import TVOptions

LINDY_CACHE = os.path.join(os.path.dirname(__file__), "lindy_cache.json")
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")
DEFAULT_M2 = 0.07
DEFAULT_R = 0.045
MIN_DTE = 150
DELTA_MIN = 0.25
DELTA_MAX = 0.80


def load_lindy_results() -> list[dict]:
    if not os.path.exists(LINDY_CACHE):
        return []
    with open(LINDY_CACHE) as f:
        return json.load(f)


def compute_trend_deviation(ticker: str) -> dict | None:
    """Compute where spot sits relative to the max-history log-linear regression.

    Uses period="max" to match the Lindy scan's methodology — the regression
    is fit on the full available history, same data the Lindy score is based on.
    """
    data = yf.download(ticker, period="max", interval="1d",
                       auto_adjust=True, progress=False)
    if data.empty or len(data) < 504:
        return None
    c = data["Close"].values.flatten()
    n = len(c)
    x = np.arange(n)
    y = np.log(c)
    slope_daily, intercept = np.polyfit(x, y, 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    ann_slope = slope_daily * 252
    trend_today = math.exp(intercept + slope_daily * (n - 1))
    spot = float(c[-1])
    deviation = (spot / trend_today - 1)  # positive = above trend
    return {
        "slope_daily": slope_daily,
        "intercept": intercept,
        "ann_slope": ann_slope,
        "r2_5y": r2,
        "n_bars": n,
        "trend_today": trend_today,
        "spot": spot,
        "deviation": deviation,
    }


def pull_all_options(
    tv: TVOptions,
    tv_sym: str,
    min_slope: float,
    min_r2: float,
    deviation: float,
    beta: float,
    m2: float = DEFAULT_M2,
    r: float = DEFAULT_R,
) -> list[dict]:
    """Pull every call option across all DTEs ≥ MIN_DTE.

    Computes fair value under three approaches:
    1. lindy_fair: raw slope from spot (no deviation adjustment)
    2. adj_fair: deviation-adjusted (effective_drift = slope + R² × -dev/T)
    3. debase_fair: β × M2 drift
    """
    try:
        spot = tv.get_underlying_price(tv_sym)
        if not spot or spot <= 0:
            return []
        exps = tv.get_expirations(tv_sym)
    except Exception:
        return []

    mu_lindy = min_slope / 100
    mu_debase = beta * m2
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
        # Deviation-adjusted drift: slope + R²-scaled reversion
        reversion_per_yr = min_r2 * (-deviation) / T
        mu_adjusted = mu_lindy + reversion_per_yr

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

            lindy_fair = debase_fair_value(spot, K, T, r, iv, mu_lindy)
            lindy_rwd = (lindy_fair - ask) / ask * 100 if ask > 0 else 0.0

            adj_fair = debase_fair_value(spot, K, T, r, iv, mu_adjusted)
            adj_rwd = (adj_fair - ask) / ask * 100 if ask > 0 else 0.0

            debase_fair = debase_fair_value(spot, K, T, r, iv, mu_debase)
            debase_rwd = (debase_fair - ask) / ask * 100 if ask > 0 else 0.0

            be = breakeven_annualized_growth(spot, K, T, mid)

            rows.append({
                "spot": spot, "exp": e, "dte": dte, "strike": K,
                "bid": bid, "ask": ask, "mid": mid, "iv": iv, "delta": delta,
                "lindy_fair": lindy_fair, "lindy_rwd": lindy_rwd,
                "adj_fair": adj_fair, "adj_rwd": adj_rwd,
                "debase_fair": debase_fair, "debase_rwd": debase_rwd,
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
    """Pull all LEAPS for top Lindy names, return best per ticker by adj_rwd."""
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
        min_r2 = entry.get("min_r2", 0.80)

        # Derive beta from longest slope
        for sl_key in ["sl_max", "sl_20y", "sl_15y", "sl_10y", "sl_5y", "sl_2y"]:
            sl = entry.get(sl_key)
            if sl is not None and sl > 0:
                beta = (sl / 100) / m2
                break
        else:
            beta = entry.get("beta", 1.5)

        # Compute trend deviation
        trend = compute_trend_deviation(ticker)
        if trend is None:
            deviation = 0.0
            trend_today = None
        else:
            deviation = trend["deviation"]
            trend_today = trend["trend_today"]

        options = pull_all_options(
            tv, tv_sym, min_slope, min_r2, deviation, beta, m2, r)

        base_entry = {
            "ticker": ticker,
            "lindy_rank": lindy_results.index(entry) + 1,
            "lindy_score": entry.get("lindy_score"),
            "avg_r2": entry.get("avg_r2"),
            "min_r2": min_r2,
            "total_years": entry.get("total_years"),
            "min_slope": min_slope,
            "beta_used": round(beta, 1),
            "deviation": round(deviation * 100, 1),
            "trend_today": round(trend_today, 2) if trend_today else None,
        }

        if not options:
            if verbose:
                print(f"{ticker:6s} — no eligible options")
            combined.append({**base_entry, "has_leaps": False})
            continue

        # Best option by ADJUSTED reward (primary ranking metric)
        best = max(options, key=lambda o: o["adj_rwd"])

        dev_str = f"dev={deviation*100:+.1f}%"
        if verbose:
            print(f"{ticker:6s} slope={min_slope:+.1f}%  {dev_str:>11s}  "
                  f"{len(options):3d} opts  K={best['strike']:.0f} DTE={best['dte']}  "
                  f"ask=${best['ask']:.2f}  "
                  f"adj={best['adj_rwd']:+.0f}%  "
                  f"raw={best['lindy_rwd']:+.0f}%  "
                  f"IV={best['iv']*100:.0f}%")

        combined.append({
            **base_entry,
            "has_leaps": True,
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
            "adj_fair": best["adj_fair"],
            "adj_rwd": best["adj_rwd"],
            "debase_fair": best["debase_fair"],
            "debase_rwd": best["debase_rwd"],
            "be_g_pct": best["be_g_pct"],
        })

    return combined


def format_combined_table(results: list[dict]) -> str:
    lines = []
    hdr = (f"{'#':>3s} {'tick':6s} {'L#':>3s} {'LINDY':>6s} {'R²':>5s} {'yrs':>5s} "
           f"{'dev%':>6s} {'DTE':>4s} {'K':>7s} {'ask':>7s} "
           f"{'adj_rwd':>7s} {'raw_rwd':>7s} {'db_rwd':>7s} "
           f"{'IV':>3s} {'Δ':>4s} {'BE%':>5s}")
    lines.append(hdr)
    for i, r in enumerate(results):
        if not r.get("has_leaps"):
            lines.append(
                f"{i+1:3d} {r['ticker']:6s} {r.get('lindy_rank',0):3d} "
                f"{r.get('lindy_score', 0):6.3f} {r.get('avg_r2', 0):5.3f} "
                f"{r.get('total_years', 0):5.1f} "
                f"{r.get('deviation', 0):+6.1f} "
                f"{'— no LEAPS available':50s}"
            )
            continue
        lines.append(
            f"{i+1:3d} {r['ticker']:6s} {r.get('lindy_rank',0):3d} "
            f"{r.get('lindy_score', 0):6.3f} {r.get('avg_r2', 0):5.3f} "
            f"{r.get('total_years', 0):5.1f} "
            f"{r.get('deviation', 0):+6.1f} "
            f"{r['dte']:4d} {r['strike']:7.1f} {r['ask']:7.2f} "
            f"{r['adj_rwd']:+7.1f} {r['lindy_rwd']:+7.1f} {r['debase_rwd']:+7.1f} "
            f"{r['iv']*100:3.0f} {r['delta']:4.2f} {r['be_g_pct']:+5.1f}"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Lindy LEAPS — deviation-adjusted edge on the best compounders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--m2", type=float, default=DEFAULT_M2)
    p.add_argument("--rfr", type=float, default=DEFAULT_R)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--min-slope", type=float, default=12.0)
    p.add_argument("--charts", action="store_true",
                   help="Generate edge visualization PNGs into charts/")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if args.fresh or not os.path.exists(LINDY_CACHE):
        if not args.json:
            print("Running Lindy scan first...\n")
        from lindy_scan import scan_lindy
        from build_universe import build_sp900
        universe = build_sp900(verbose=not args.json)
        lindy_results = scan_lindy(
            universe=universe, min_slope=args.min_slope,
            top_n=0, verbose=not args.json,
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
        m2=args.m2, r=args.rfr,
        verbose=not args.json,
    )

    with_leaps = [r for r in combined if r.get("has_leaps")]
    without_leaps = [r for r in combined if not r.get("has_leaps")]
    with_leaps.sort(key=lambda r: r.get("adj_rwd", 0), reverse=True)
    final = with_leaps + without_leaps

    if args.json:
        print(json.dumps({
            "m2": args.m2, "rfr": args.rfr,
            "metric": "adj_rwd = deviation-adjusted Lindy reward",
            "count": len(final),
            "results": final,
        }, indent=2))
    else:
        print(f"\n=== LINDY × LEAPS — deviation-adjusted edge ===")
        print(f"adj_rwd = slope + R²-scaled mean-reversion from spot to trend")
        print(f"raw_rwd = slope only (no deviation adjustment)")
        print(f"dev%    = spot vs 5yr regression (negative = below trend = bonus)\n")
        print(format_combined_table(final))

    # Generate charts
    if args.charts:
        os.makedirs(CHARTS_DIR, exist_ok=True)
        from visualize_edge import plot_edge
        print(f"\nGenerating charts into {CHARTS_DIR}/...")
        for r in final:
            if not r.get("has_leaps"):
                continue
            tv_sym = None
            for entry in lindy_results:
                if entry["ticker"] == r["ticker"]:
                    tv_sym = entry.get("tv_symbol")
                    break
            if tv_sym:
                save = os.path.join(CHARTS_DIR, f"{r['ticker'].lower()}_edge.png")
                try:
                    plot_edge(tv_sym, r=args.rfr, save_path=save)
                except Exception as e:
                    print(f"  {r['ticker']}: chart failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
