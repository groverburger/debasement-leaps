"""Lindy trend scanner — find the straightest log-linear compounders.

Scans the full S&P 500 + MidCap 400 universe (~900 names) for stocks that
look like straight lines on a log chart, weighted by how many years they've
been doing it.

Metric: Information Ratio (IR) = annualized_slope / annualized_residual_noise
This is the Sharpe ratio of the trend — signal/noise, independent of window
length. We compute IR across every available window (2y, 3y, 5y, 7y, 10y,
15y, 20y, max) and rank by:

    Lindy IR = min(IR across all windows) × ln(1 + max_years_of_data)

Slope is a gate (default ≥12%), not a ranking factor. Once a stock is "fast
enough for LEAPS," we only care how straight and how long.

Pipeline:
    1. Build/load S&P 900 universe (from Wikipedia via build_universe.py)
    2. Download max daily history via yfinance
    3. Compute IR across all windows
    4. Gate on min_slope, rank by Lindy IR
    5. Output top N

CLI:
    python3 lindy_scan.py                    # default: top 30
    python3 lindy_scan.py --top 50
    python3 lindy_scan.py --min-slope 15     # stricter gate
    python3 lindy_scan.py --json             # machine-readable
    python3 lindy_scan.py --refresh          # re-scrape universe from Wikipedia

Then layer on LEAPS pricing:
    python3 lindy_leaps.py                   # reads lindy_scan output, pulls chains
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from build_universe import build_sp900

CACHE_FILE = os.path.join(os.path.dirname(__file__), "lindy_ir_cache.json")
WINDOWS = [("2y", 504), ("3y", 756), ("5y", 1260), ("7y", 1764),
           ("10y", 2520), ("15y", 3780), ("20y", 5040), ("max", None)]


def compute_ir(prices: np.ndarray) -> tuple[float, float, float, float]:
    """Compute Information Ratio for a price series.

    Returns (annualized_slope%, IR, R², annualized_noise%).
    IR = ann_slope / ann_residual_noise — the Sharpe of the trend.
    """
    n = len(prices)
    if n < 60:
        return 0.0, 0.0, 0.0, 0.0
    x = np.arange(n)
    y = np.log(prices)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residuals = y - predicted
    daily_resid_std = float(np.std(np.diff(residuals)))
    ann_noise = daily_resid_std * np.sqrt(252)
    ann_slope = slope * 252
    ir = ann_slope / ann_noise if ann_noise > 0 else 0.0
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    return float(ann_slope * 100), float(ir), r2, float(ann_noise * 100)


def scan_lindy(
    universe: dict[str, tuple[str, float, str]],
    min_slope: float = 12.0,
    top_n: int = 30,
    verbose: bool = True,
) -> list[dict]:
    """Run the full Lindy IR scan.

    Args:
        universe: ticker → (tv_symbol, beta, theme) dict.
        min_slope: minimum annualized slope (%) across ALL windows. Gate only.
        top_n: return this many results.
        verbose: print progress.

    Returns:
        List of result dicts sorted by lindy_ir descending.
    """
    tickers = sorted(universe.keys())
    if verbose:
        print(f"Downloading max daily history for {len(tickers)} tickers...")

    data = yf.download(tickers, period="max", interval="1d",
                       auto_adjust=True, progress=False,
                       group_by="ticker", threads=True)

    results: list[dict] = []
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[t].dropna()
            else:
                df = data.dropna()
            c = df["Close"].values.flatten()
            n = len(c)
            if n < 504:
                continue
            last = float(c[-1])
            if last < 5:
                continue
            total_years = n / 252

            windows: dict[str, dict] = {}
            for label, bars in WINDOWS:
                actual_bars = bars if bars is not None else n
                if n < actual_bars:
                    continue
                sl, ir, r2, noise = compute_ir(c[-actual_bars:])
                if sl > 0:
                    windows[label] = {
                        "slope": sl, "ir": ir, "r2": r2, "noise": noise,
                        "years": actual_bars / 252,
                    }

            if not windows:
                continue

            irs = [w["ir"] for w in windows.values()]
            slopes = [w["slope"] for w in windows.values()]
            min_sl = min(slopes)
            if min_sl < min_slope:
                continue

            min_ir = min(irs)
            avg_ir = float(np.mean(irs))
            max_years = max(w["years"] for w in windows.values())
            lindy_ir = min_ir * math.log(1 + max_years)

            # Carry TV symbol and beta from universe
            tv_sym, beta, theme = universe.get(t, ("", 1.5, "unknown"))

            row: dict = {
                "ticker": t,
                "tv_symbol": tv_sym,
                "beta": beta,
                "theme": theme,
                "last": last,
                "total_years": round(total_years, 1),
                "n_windows": len(windows),
                "min_ir": round(min_ir, 3),
                "avg_ir": round(avg_ir, 3),
                "min_slope": round(min_sl, 1),
                "max_years": round(max_years, 1),
                "lindy_ir": round(lindy_ir, 3),
            }
            for label in ["2y", "3y", "5y", "10y", "15y", "20y", "max"]:
                if label in windows:
                    row[f"ir_{label}"] = round(windows[label]["ir"], 3)
                    row[f"sl_{label}"] = round(windows[label]["slope"], 1)
                    row[f"r2_{label}"] = round(windows[label]["r2"], 3)
                else:
                    row[f"ir_{label}"] = None
                    row[f"sl_{label}"] = None
                    row[f"r2_{label}"] = None

            results.append(row)
        except Exception:
            pass

    results.sort(key=lambda x: x["lindy_ir"], reverse=True)
    if verbose:
        print(f"{len(results)} names passed the {min_slope}% slope gate")
    return results[:top_n] if top_n else results


def format_lindy_table(results: list[dict]) -> str:
    """Format results as a text table."""
    lines = []
    hdr = (f"{'#':>3s} {'tick':6s} {'yrs':>5s} {'#w':>3s} {'lindy_ir':>8s} "
           f"{'minIR':>5s} {'avgIR':>5s} {'ir_2y':>5s} {'ir_5y':>5s} {'ir_10y':>6s} "
           f"{'ir_20y':>6s} {'ir_max':>6s} {'min_sl':>6s}")
    lines.append(hdr)
    for i, r in enumerate(results):
        def fmt(v): return f"{v:5.2f}" if v is not None else "   — "
        def fmt6(v): return f"{v:6.2f}" if v is not None else "    — "
        lines.append(
            f"{i+1:3d} {r['ticker']:6s} {r['total_years']:5.1f} {r['n_windows']:3d} "
            f"{r['lindy_ir']:8.3f} {r['min_ir']:5.2f} {r['avg_ir']:5.2f} "
            f"{fmt(r.get('ir_2y'))} {fmt(r.get('ir_5y'))} {fmt6(r.get('ir_10y'))} "
            f"{fmt6(r.get('ir_20y'))} {fmt6(r.get('ir_max'))} {r['min_slope']:+6.1f}%"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Lindy trend scanner — find the straightest log-linear compounders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--top", type=int, default=30,
                   help="Number of results to return (default 30)")
    p.add_argument("--min-slope", type=float, default=12.0,
                   help="Minimum annualized slope %% gate (default 12)")
    p.add_argument("--refresh", action="store_true",
                   help="Re-scrape universe from Wikipedia before scanning")
    p.add_argument("--no-cache", action="store_true",
                   help="Don't read/write the IR cache")
    p.add_argument("--json", action="store_true",
                   help="Output JSON instead of text table")
    args = p.parse_args()

    universe = build_sp900(refresh=args.refresh, verbose=not args.json)

    results = scan_lindy(
        universe=universe,
        min_slope=args.min_slope,
        top_n=0,  # get all, then slice
        verbose=not args.json,
    )

    # Cache full results
    if not args.no_cache:
        with open(CACHE_FILE, "w") as f:
            json.dump(results, f, indent=2)
        if not args.json:
            print(f"Cached {len(results)} results to {CACHE_FILE}")

    top = results[:args.top]

    if args.json:
        print(json.dumps({
            "min_slope_gate": args.min_slope,
            "total_passed": len(results),
            "returned": len(top),
            "results": top,
        }, indent=2))
    else:
        print(f"\nLindy IR = min(IR) × ln(1 + years_of_data)")
        print(f"IR = ann_slope / ann_residual_noise (Sharpe of the trend)")
        print(f"Gate: min slope ≥ {args.min_slope}% across ALL windows\n")
        print(format_lindy_table(top))

    return 0


if __name__ == "__main__":
    sys.exit(main())
