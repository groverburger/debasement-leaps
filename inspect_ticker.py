"""Single-ticker debasement-LEAPS inspector (CLI).

Agent-friendly: pass a symbol and a beta; get back ranked strikes showing
market price, debasement fair value, $ and % edge.

Examples:
    python3 inspect.py NYSE:PWR --beta 1.5
    python3 inspect.py NASDAQ:HOOD --beta 3.0 --top 8
    python3 inspect.py NYSE:TRV --beta 1.2 --m2 0.08 --json
    python3 inspect.py NASDAQ:COST --beta 1.3 --min-dte 400 --top 5 --json

Output is a ranked list (by composite score: edge_pct × √T × IV_penalty).
Use --json for machine-readable output suitable for downstream tooling.
"""
from __future__ import annotations

import argparse
import json
import sys

from chain_analysis import (
    analyze_ticker,
    format_rows_table,
    DEFAULT_M2,
    DEFAULT_R,
    DEFAULT_DELTA_RANGE,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Debasement-LEAPS inspector for a single ticker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("symbol",
                   help="TradingView-style symbol, e.g. NYSE:PWR, NASDAQ:HOOD, AMEX:LIT")
    p.add_argument("--beta", type=float, required=True,
                   help="Asset's beta to M2 expansion (e.g. 1.5 grid infra, 3.0 crypto beta)")
    p.add_argument("--m2", type=float, default=DEFAULT_M2,
                   help=f"Annualized M2 growth assumption (default {DEFAULT_M2})")
    p.add_argument("--rfr", type=float, default=DEFAULT_R,
                   help=f"Risk-free rate (default {DEFAULT_R})")
    p.add_argument("--min-dte", type=int, default=None,
                   help="Minimum days-to-expiration filter")
    p.add_argument("--max-dte", type=int, default=None,
                   help="Maximum days-to-expiration filter")
    p.add_argument("--delta-min", type=float, default=DEFAULT_DELTA_RANGE[0],
                   help=f"Minimum delta (default {DEFAULT_DELTA_RANGE[0]})")
    p.add_argument("--delta-max", type=float, default=DEFAULT_DELTA_RANGE[1],
                   help=f"Maximum delta (default {DEFAULT_DELTA_RANGE[1]})")
    p.add_argument("--top", type=int, default=10, help="Number of rows to return")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text table")
    args = p.parse_args()

    rows = analyze_ticker(
        symbol=args.symbol,
        beta=args.beta,
        m2=args.m2,
        r=args.rfr,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        delta_range=(args.delta_min, args.delta_max),
    )
    rows.sort(key=lambda r_: r_["score"], reverse=True)
    rows = rows[:args.top]

    if not rows:
        msg = {"error": f"no eligible calls for {args.symbol}"}
        if args.json:
            print(json.dumps(msg))
        else:
            print(msg["error"])
        return 1

    if args.json:
        print(json.dumps({
            "symbol": args.symbol,
            "beta": args.beta,
            "m2": args.m2,
            "rfr": args.rfr,
            "mu": args.beta * args.m2,
            "count": len(rows),
            "rows": rows,
        }, indent=2))
    else:
        r0 = rows[0]
        print(f"{args.symbol}  spot={r0['spot']:.2f}  β={args.beta}  "
              f"μ={r0['mu']*100:.1f}%  r={args.rfr*100:.1f}%  "
              f"returning top {len(rows)}\n")
        print(format_rows_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
