"""Debasement-LEAPS scanner — smooth multi-year trend edition.

Two-stage pipeline:
  1. yfinance pre-filter on a compounder-heavy universe. Filter thresholds
     are CLI-tunable (see --help).
  2. For survivors: pull LEAPS chains, compute debasement fair value,
     rank by composite edge score.

CLI:
    python3 scan_smooth.py                             # default text output
    python3 scan_smooth.py --top 40
    python3 scan_smooth.py --m2 0.08 --json
    python3 scan_smooth.py --min-above-200 80 --max-dd -25 --max-atr 5.5

For a single-ticker drill-down, use inspect.py.
"""
from __future__ import annotations

import argparse
import json
import pandas as pd
import yfinance as yf

from chain_analysis import (
    analyze_ticker,
    format_rows_table,
    DEFAULT_M2,
    DEFAULT_R,
    DEFAULT_DELTA_RANGE,
)
from tv_options import TVOptions


# (tv_symbol, beta_to_m2, theme)
UNIVERSE: dict[str, tuple[str, float, str]] = {
    # --- Boring compounders / toll-road financials ---
    "V":    ("NYSE:V",        1.5, "payment_rail"),
    "MA":   ("NYSE:MA",       1.5, "payment_rail"),
    "SPGI": ("NYSE:SPGI",     1.5, "data_monopoly"),
    "MCO":  ("NYSE:MCO",      1.5, "data_monopoly"),
    "MSCI": ("NYSE:MSCI",     1.5, "data_monopoly"),
    "FICO": ("NYSE:FICO",     1.5, "data_monopoly"),
    "COST": ("NASDAQ:COST",   1.3, "consumer_moat"),
    "WM":   ("NYSE:WM",       1.2, "infra_service"),
    "RSG":  ("NYSE:RSG",      1.2, "infra_service"),
    "CTAS": ("NASDAQ:CTAS",   1.3, "infra_service"),
    "ROL":  ("NYSE:ROL",      1.2, "infra_service"),
    "CPRT": ("NASDAQ:CPRT",   1.3, "auction_moat"),
    # --- Insurance smooth compounders ---
    "PGR":  ("NYSE:PGR",      1.2, "insurance"),
    "AJG":  ("NYSE:AJG",      1.3, "insurance_broker"),
    "BRO":  ("NYSE:BRO",      1.3, "insurance_broker"),
    "WRB":  ("NYSE:WRB",      1.3, "insurance"),
    "ACGL": ("NASDAQ:ACGL",   1.3, "insurance"),
    "TRV":  ("NYSE:TRV",      1.2, "insurance"),
    # --- Asset managers (AUM scales with debasement) ---
    "BLK":  ("NYSE:BLK",      2.0, "asset_mgr"),
    "KKR":  ("NYSE:KKR",      2.5, "alt_mgr"),
    "BX":   ("NYSE:BX",       2.5, "alt_mgr"),
    "APO":  ("NYSE:APO",      2.5, "alt_mgr"),
    "BAM":  ("NYSE:BAM",      2.0, "alt_mgr"),
    # --- Industrial compounders / real capex ---
    "ETN":  ("NYSE:ETN",      1.8, "electrification"),
    "PH":   ("NYSE:PH",       1.5, "industrial"),
    "ITW":  ("NYSE:ITW",      1.5, "industrial"),
    "ROP":  ("NASDAQ:ROP",    1.5, "industrial_sw"),
    "APH":  ("NYSE:APH",      1.8, "connectors"),
    "TDG":  ("NYSE:TDG",      1.5, "defense_aero"),
    "HEI":  ("NYSE:HEI",      1.5, "defense_aero"),
    "LIN":  ("NASDAQ:LIN",    1.3, "industrial_gas"),
    # --- Grid / electrification from prior scans ---
    "PWR":  ("NYSE:PWR",      1.5, "grid_infra"),
    "GEV":  ("NYSE:GEV",      1.8, "grid_infra"),
    "MTZ":  ("NYSE:MTZ",      1.5, "grid_infra"),
    "FIX":  ("NYSE:FIX",      1.8, "hvac_datactr"),
    "JBL":  ("NYSE:JBL",      2.0, "contract_mfg"),
    "KEYS": ("NYSE:KEYS",     1.8, "test_mmt"),
    # --- Healthcare smooth compounders ---
    "LLY":  ("NYSE:LLY",      1.5, "pharma_glp1"),
    "ISRG": ("NASDAQ:ISRG",   1.8, "med_device"),
    "TMO":  ("NYSE:TMO",      1.5, "life_sci_tools"),
    "DHR":  ("NYSE:DHR",      1.5, "life_sci_tools"),
    "VRTX": ("NASDAQ:VRTX",   1.5, "biotech_monopoly"),
    "REGN": ("NASDAQ:REGN",   1.5, "biotech"),
    "IDXX": ("NASDAQ:IDXX",   1.3, "vet_diagnostics"),
    # --- Tech capex ---
    "ASML": ("NASDAQ:ASML",   2.0, "semi_cap"),
    "AMZN": ("NASDAQ:AMZN",   1.8, "megacap"),
    "ORCL": ("NYSE:ORCL",     1.8, "megacap"),
    "WDC":  ("NASDAQ:WDC",    2.0, "storage"),
    # --- Hard asset / debasement-direct ---
    "GDX":  ("AMEX:GDX",      2.0, "gold_miners"),
    "CCJ":  ("NYSE:CCJ",      2.2, "uranium"),
    "FTI":  ("NYSE:FTI",      1.8, "energy_svc"),
    "LIT":  ("AMEX:LIT",      2.0, "battery_etf"),
    # --- Already-held thesis names ---
    "HOOD": ("NASDAQ:HOOD",   3.0, "crypto_beta"),
    "C":    ("NYSE:C",        1.2, "money_ctr_bank"),
}


def atr_pct(df: pd.DataFrame, n: int = 14) -> float:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float((tr.rolling(n).mean() / c).iloc[-1])


def screen_smoothness(
    tickers: list[str],
    min_ret_2y: float = 15.0,
    min_above_200: float = 75.0,
    max_dd: float = -28.0,
    max_atr: float = 6.0,
    max_dist_hi: float = 15.0,
    verbose: bool = True,
) -> dict[str, tuple[bool, dict]]:
    """yfinance pre-filter for smooth multi-year trends.

    Returns dict ticker -> (passes, metrics). Metrics always include a
    `smooth_score` (higher = smoother). Falls back to (False, {reason})
    when data unavailable.
    """
    if verbose:
        print(f"Pre-filtering {len(tickers)} names via yfinance...")
    data = yf.download(tickers, period="3y", interval="1d",
                       auto_adjust=True, progress=False, group_by="ticker", threads=True)
    results: dict[str, tuple[bool, dict]] = {}
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[t].dropna()
            else:
                df = data.dropna()
            if len(df) < 504:
                results[t] = (False, {"reason": "insufficient_history"})
                continue
            c = df["Close"]
            last = float(c.iloc[-1])
            ma200 = c.rolling(200).mean()
            last504 = c.iloc[-504:]
            ma200_504 = ma200.iloc[-504:]
            above200_pct = float((last504 > ma200_504).mean()) * 100
            rmax = last504.cummax()
            dd = float((last504 / rmax - 1).min() * 100)
            ret_504 = (last / float(last504.iloc[0]) - 1) * 100
            ret_252 = (last / float(c.iloc[-252]) - 1) * 100
            hi252 = float(df["High"].iloc[-252:].max())
            dist_hi = (hi252 - last) / last * 100
            atr = atr_pct(df, 14) * 100
            passes = (
                ret_504 > min_ret_2y and
                above200_pct >= min_above_200 and
                dd >= max_dd and
                atr < max_atr and
                dist_hi < max_dist_hi
            )
            smooth = above200_pct + max(0, dd + 22) - atr * 3 + min(ret_504 / 10, 20)
            results[t] = (passes, {
                "last": last, "ret_504": ret_504, "ret_252": ret_252,
                "above200_pct": above200_pct, "max_dd": dd,
                "dist_hi": dist_hi, "atr_pct": atr, "smooth_score": smooth,
            })
        except Exception as e:
            results[t] = (False, {"reason": str(e)[:80]})
    return results


def scan_smooth_universe(
    universe: dict[str, tuple[str, float, str]],
    m2: float = DEFAULT_M2,
    r: float = DEFAULT_R,
    dte_targets: tuple[int, ...] = (400, 550, 700),
    delta_range: tuple[float, float] = DEFAULT_DELTA_RANGE,
    screen_kwargs: dict | None = None,
    verbose: bool = True,
) -> tuple[list[dict], dict[str, tuple[bool, dict]]]:
    """Full pipeline: smoothness screen + chain analysis for survivors.

    Returns (rows, screen_results).
    """
    screen_kwargs = screen_kwargs or {}
    screen = screen_smoothness(list(universe.keys()), verbose=verbose, **screen_kwargs)
    survivors = [t for t, (p, _) in screen.items() if p]

    if verbose:
        print(f"\n=== {len(survivors)}/{len(universe)} passed smooth-trend filter ===")
        print(f"{'tick':5s} {'ret2y%':>7s} {'above200%':>10s} {'maxdd%':>7s} "
              f"{'atr%':>5s} {'dist_hi%':>8s} {'smooth':>7s}")
        for t, m in sorted([(t, screen[t][1]) for t in survivors],
                           key=lambda x: x[1].get("smooth_score", 0), reverse=True):
            print(f"{t:5s} {m['ret_504']:7.1f} {m['above200_pct']:10.1f} "
                  f"{m['max_dd']:7.1f} {m['atr_pct']:5.2f} {m['dist_hi']:8.2f} "
                  f"{m['smooth_score']:7.1f}")

    tv = TVOptions()
    rows: list[dict] = []
    if verbose and survivors:
        print(f"\n=== Pulling chains for survivors ===")
    for t in survivors:
        sym, beta, theme = universe[t]
        try:
            r_ = analyze_ticker(symbol=sym, beta=beta, m2=m2, r=r,
                                dte_targets=dte_targets, delta_range=delta_range, tv=tv)
            for row in r_:
                row["ticker"] = t
                row["theme"] = theme
            rows.extend(r_)
            if verbose and r_:
                best = max(r_, key=lambda x: x["score"])
                print(f"{t:5s} β={beta:.1f}  n={len(r_):3d}  best: K={best['strike']:.0f} "
                      f"DTE={best['dte']}  edge=${best['edge_usd']:+6.2f} "
                      f"({best['edge_pct']:+.1f}%)  IV={best['iv']*100:.0f}%")
            elif verbose:
                print(f"{t:5s} — no eligible LEAPS strikes")
        except Exception as e:
            if verbose:
                print(f"{t:5s} ERR: {str(e)[:60]}")
    return rows, screen


def main() -> int:
    p = argparse.ArgumentParser(description="Smooth-trend debasement-LEAPS scanner.")
    p.add_argument("--m2", type=float, default=DEFAULT_M2)
    p.add_argument("--rfr", type=float, default=DEFAULT_R)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--min-ret-2y", type=float, default=15.0,
                   help="Minimum 2-year return %% (default 15)")
    p.add_argument("--min-above-200", type=float, default=75.0,
                   help="Minimum %% of last 504 days above 200MA (default 75)")
    p.add_argument("--max-dd", type=float, default=-28.0,
                   help="Maximum allowed max drawdown %% over 504d (default -28)")
    p.add_argument("--max-atr", type=float, default=6.0,
                   help="Maximum ATR%% 14d (default 6)")
    p.add_argument("--max-dist-hi", type=float, default=15.0,
                   help="Max %% from 52w high (default 15)")
    p.add_argument("--delta-min", type=float, default=DEFAULT_DELTA_RANGE[0])
    p.add_argument("--delta-max", type=float, default=DEFAULT_DELTA_RANGE[1])
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    screen_kwargs = {
        "min_ret_2y": args.min_ret_2y,
        "min_above_200": args.min_above_200,
        "max_dd": args.max_dd,
        "max_atr": args.max_atr,
        "max_dist_hi": args.max_dist_hi,
    }

    rows, screen = scan_smooth_universe(
        universe=UNIVERSE,
        m2=args.m2,
        r=args.rfr,
        delta_range=(args.delta_min, args.delta_max),
        screen_kwargs=screen_kwargs,
        verbose=not args.json,
    )

    if not rows:
        if args.json:
            print(json.dumps({"error": "no rows after filter", "count": 0, "rows": []}))
        else:
            print("no rows")
        return 1

    rows.sort(key=lambda r_: r_["score"], reverse=True)
    top_rows = rows[:args.top]
    by_tick = {}
    for r_ in rows:
        t = r_["ticker"]
        if t not in by_tick or r_["score"] > by_tick[t]["score"]:
            by_tick[t] = r_
    best_per_tick = sorted(by_tick.values(), key=lambda r_: r_["score"], reverse=True)

    if args.json:
        print(json.dumps({
            "m2": args.m2, "rfr": args.rfr,
            "filter": screen_kwargs,
            "count": len(rows),
            "survivors": [t for t, (p, _) in screen.items() if p],
            "top": top_rows,
            "best_per_ticker": best_per_tick,
        }, indent=2))
        return 0

    print(f"\n=== TOP {args.top} BY COMPOSITE SCORE ===")
    print(format_rows_table(top_rows))
    print(f"\n=== BEST PICK PER TICKER ===")
    print(format_rows_table(best_per_tick))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
