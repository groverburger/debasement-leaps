"""Debasement-LEAPS scanner — broad universe.

Runs debasement analysis across a curated universe of real-capex, semi, AI,
crypto-beta, commodity, and megacap tech names. Each name has a per-name
beta-to-M2 estimate.

CLI:
    python3 scan_broad.py                   # default text output, top 30
    python3 scan_broad.py --top 50
    python3 scan_broad.py --m2 0.08 --json  # JSON for downstream tooling
    python3 scan_broad.py --tickers PWR,HOOD,GDX

For a single-ticker drill-down, use inspect.py instead.
"""
from __future__ import annotations

import argparse
import json

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
    # Energy / grid infra — real capex beneficiaries
    "PWR":  ("NYSE:PWR",      1.5, "grid_infra"),
    "GEV":  ("NYSE:GEV",      1.8, "grid_infra"),
    "MTZ":  ("NYSE:MTZ",      1.5, "grid_infra"),
    "FIX":  ("NYSE:FIX",      1.8, "hvac_datactr"),
    "FTI":  ("NYSE:FTI",      1.8, "energy_svc"),
    "JBL":  ("NYSE:JBL",      2.0, "contract_mfg"),
    "KEYS": ("NYSE:KEYS",     1.8, "test_mmt"),
    "CW":   ("NYSE:CW",       1.5, "defense"),
    # Semis / AI
    "SMH":  ("AMEX:SMH",      2.0, "semis_etf"),
    "WDC":  ("NASDAQ:WDC",    2.0, "storage"),
    "CIEN": ("NYSE:CIEN",     2.2, "optical"),
    "FN":   ("NYSE:FN",       2.5, "photonics"),
    "CRDO": ("NASDAQ:CRDO",   2.5, "ai_network"),
    "ALAB": ("NASDAQ:ALAB",   2.5, "ai_chip"),
    "MRVL": ("NASDAQ:MRVL",   2.0, "semis"),
    # Megacap tech
    "AMZN": ("NASDAQ:AMZN",   1.8, "megacap"),
    "ORCL": ("NYSE:ORCL",     1.8, "megacap"),
    # Crypto proxies / high-beta financials
    "HOOD": ("NASDAQ:HOOD",   3.0, "crypto_beta"),
    "COIN": ("NASDAQ:COIN",   3.5, "crypto_exch"),
    "MSTR": ("NASDAQ:MSTR",   4.0, "btc_proxy"),
    # Traditional financials
    "C":    ("NYSE:C",        1.2, "money_ctr_bank"),
    "WBS":  ("NYSE:WBS",      1.2, "regional_bank"),
    # Commodities / hard asset
    "GDX":  ("AMEX:GDX",      2.0, "gold_miners"),
    "CCJ":  ("NYSE:CCJ",      2.2, "uranium"),
    "LEU":  ("AMEX:LEU",      2.5, "uranium"),
    "LIT":  ("AMEX:LIT",      2.0, "battery_etf"),
}


def scan_universe(
    universe: dict[str, tuple[str, float, str]],
    m2: float = DEFAULT_M2,
    r: float = DEFAULT_R,
    dte_targets: tuple[int, ...] = (300, 400, 500),
    delta_range: tuple[float, float] = DEFAULT_DELTA_RANGE,
    progress: bool = True,
) -> list[dict]:
    """Run analyze_ticker() across a universe and tag results with ticker/theme."""
    tv = TVOptions()
    all_rows: list[dict] = []
    for ticker, (sym, beta, theme) in universe.items():
        try:
            rows = analyze_ticker(
                symbol=sym, beta=beta, m2=m2, r=r,
                dte_targets=dte_targets, delta_range=delta_range, tv=tv,
            )
            for row in rows:
                row["ticker"] = ticker
                row["theme"] = theme
            all_rows.extend(rows)
            if progress and rows:
                best = max(rows, key=lambda r_: r_["score"])
                print(f"{ticker:6s} spot={best['spot']:8.2f}  β={beta:.1f}  "
                      f"n={len(rows):3d}  best: K={best['strike']:.1f} DTE={best['dte']}  "
                      f"edge=${best['edge_usd']:+6.2f} ({best['edge_pct']:+.1f}%)  "
                      f"IV={best['iv']*100:.0f}%")
            elif progress:
                print(f"{ticker:6s} — no eligible LEAPS")
        except Exception as e:
            if progress:
                print(f"{ticker:6s} ERR: {e}")
    return all_rows


def main() -> int:
    p = argparse.ArgumentParser(description="Debasement-LEAPS scanner over a broad universe.")
    p.add_argument("--m2", type=float, default=DEFAULT_M2)
    p.add_argument("--rfr", type=float, default=DEFAULT_R)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--tickers", type=str, default=None,
                   help="Comma-separated subset of universe to scan (e.g. PWR,HOOD,GDX)")
    p.add_argument("--delta-min", type=float, default=DEFAULT_DELTA_RANGE[0])
    p.add_argument("--delta-max", type=float, default=DEFAULT_DELTA_RANGE[1])
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",")}
        universe = {t: v for t, v in UNIVERSE.items() if t in wanted}
        missing = wanted - set(universe.keys())
        if missing and not args.json:
            print(f"(skipping unknown tickers: {sorted(missing)})")
    else:
        universe = UNIVERSE

    all_rows = scan_universe(
        universe=universe,
        m2=args.m2,
        r=args.rfr,
        delta_range=(args.delta_min, args.delta_max),
        progress=not args.json,
    )
    if not all_rows:
        if args.json:
            print(json.dumps({"error": "no rows", "count": 0, "rows": []}))
        else:
            print("no rows")
        return 1

    all_rows.sort(key=lambda r_: r_["score"], reverse=True)
    top_rows = all_rows[:args.top]
    by_tick = {}
    for r in all_rows:
        t = r["ticker"]
        if t not in by_tick or r["score"] > by_tick[t]["score"]:
            by_tick[t] = r
    best_per_tick = sorted(by_tick.values(), key=lambda r_: r_["score"], reverse=True)

    if args.json:
        print(json.dumps({
            "m2": args.m2, "rfr": args.rfr,
            "count": len(all_rows),
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
