"""Shared chain-analysis helpers used by the TUI, scanners, and inspect CLI.

The primary entry point for programmatic callers (agents, notebooks, other
scripts) is:

    analyze_ticker(symbol, beta, ...) -> list[dict]

Each result dict is a single call option enriched with:
    symbol, spot, beta, mu              – context
    exp, dte, strike                    – contract identity
    bid, ask, mid, iv, delta            – market quote
    bs_fair                             – standard Black-Scholes fair value
    debase_fair                         – BS fair value under debasement drift
    edge_usd   = debase_fair - mid      – $ edge per contract
    edge_pct   = edge_usd / mid * 100   – % edge
    be_g_pct                            – annualized breakeven growth rate
    score                               – edge_pct × √T × iv_penalty

Use best_strikes() for a ranked short-list.
"""
from __future__ import annotations

import os
import sys
import math
import datetime
from typing import Optional

sys.path.insert(0, os.path.expanduser("~/Documents/projects/darkfield"))
from tv_options import TVOptions

from pricing import bs_call, debase_fair_value, breakeven_annualized_growth


DEFAULT_M2 = 0.07
DEFAULT_R = 0.045
DEFAULT_DTE_TARGETS = (300, 450, 600)
DEFAULT_DELTA_RANGE = (0.40, 0.75)


def _pick_expirations(future: list[tuple[int, int]],
                      dte_targets: tuple[int, ...]) -> list[tuple[int, int]]:
    picks: list[tuple[int, int]] = []
    for want in dte_targets:
        best = min(future, key=lambda x: abs(x[1] - want))
        if best not in picks:
            picks.append(best)
    return picks


def analyze_ticker(
    symbol: str,
    beta: float,
    m2: float = DEFAULT_M2,
    r: float = DEFAULT_R,
    dte_targets: tuple[int, ...] = DEFAULT_DTE_TARGETS,
    delta_range: tuple[float, float] = DEFAULT_DELTA_RANGE,
    min_dte: Optional[int] = None,
    max_dte: Optional[int] = None,
    tv: Optional[TVOptions] = None,
) -> list[dict]:
    """Pull LEAPS-style chains for a single underlying and compute debasement edge.

    Args:
        symbol: TradingView-style symbol, e.g. "NYSE:PWR", "NASDAQ:HOOD".
        beta: Asset's beta to M2 expansion (e.g. 1.5 for grid infra, 3.0 for
            crypto-beta equities).
        m2: Annualized M2 growth assumption (default 0.07 = 7%/yr).
        r: Risk-free rate (default 0.045 = 4.5%).
        dte_targets: Tuple of target DTEs — we pick the closest available
            expiration to each. Default (300, 450, 600).
        delta_range: (min, max) delta filter. Default (0.40, 0.75).
        min_dte / max_dte: Optional explicit DTE gates applied after the
            target-based expiration pick.
        tv: Optional TVOptions client. A new one is created if None.

    Returns:
        List of row dicts (see module docstring). Empty if data is missing.
    """
    if tv is None:
        tv = TVOptions()

    try:
        spot = tv.get_underlying_price(symbol)
        if not spot or spot <= 0:
            return []
        exps = tv.get_expirations(symbol)
    except Exception:
        return []

    today = datetime.date.today()
    today_int = int(today.strftime("%Y%m%d"))
    future = []
    for e in exps:
        if e <= today_int:
            continue
        try:
            ed = datetime.date(int(str(e)[:4]), int(str(e)[4:6]), int(str(e)[6:8]))
        except Exception:
            continue
        dte = (ed - today).days
        if min_dte is not None and dte < min_dte:
            continue
        if max_dte is not None and dte > max_dte:
            continue
        future.append((e, dte))
    if not future:
        return []

    picks = _pick_expirations(future, dte_targets)
    mu = beta * m2
    dmin, dmax = delta_range

    rows: list[dict] = []
    for exp, dte in picks:
        try:
            chain = tv.get_chain(symbol, exp, spot=spot)
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
            delta = o.delta or 0.0
            if delta < dmin or delta > dmax:
                continue
            K = o.strike
            mid = o.mid
            iv = o.iv
            bs = bs_call(spot, K, T, r, iv, q=0.0)
            dbf = debase_fair_value(spot, K, T, r, iv, mu)
            edge_usd = dbf - mid
            edge_pct = edge_usd / mid * 100 if mid > 0 else 0.0
            be = breakeven_annualized_growth(spot, K, T, mid) * 100
            iv_penalty = max(0.3, 1.0 - max(0.0, iv - 0.8) * 2)
            score = edge_pct * math.sqrt(T) * iv_penalty
            rows.append({
                "symbol": symbol,
                "spot": spot,
                "beta": beta,
                "mu": mu,
                "exp": exp,
                "dte": dte,
                "strike": K,
                "bid": o.bid or 0.0,
                "ask": o.ask or 0.0,
                "mid": mid,
                "iv": iv,
                "delta": delta,
                "bs_fair": bs,
                "debase_fair": dbf,
                "edge_usd": edge_usd,
                "edge_pct": edge_pct,
                "be_g_pct": be,
                "score": score,
            })
    return rows


def best_strikes(
    symbol: str,
    beta: float,
    n: int = 5,
    **kwargs,
) -> list[dict]:
    """Return the top N rows from analyze_ticker, sorted by composite score."""
    rows = analyze_ticker(symbol, beta, **kwargs)
    rows.sort(key=lambda r_: r_["score"], reverse=True)
    return rows[:n]


def format_rows_table(rows: list[dict], limit: Optional[int] = None) -> str:
    """Render rows as a fixed-width ASCII table (for CLI output)."""
    if not rows:
        return "(no rows)"
    if limit is not None:
        rows = rows[:limit]
    hdr = (f"{'sym':14s} {'DTE':>4s} {'K':>7s} {'spot':>7s} {'ask':>7s} "
           f"{'MID':>7s} {'DB':>7s} {'EDGE$':>7s} {'EDGE%':>6s} {'IV':>4s} "
           f"{'Δ':>4s} {'BE%':>6s} {'score':>6s}")
    out = [hdr]
    for r in rows:
        out.append(
            f"{r['symbol']:14s} {r['dte']:4d} {r['strike']:7.2f} "
            f"{r['spot']:7.2f} {r['ask']:7.2f} {r['mid']:7.2f} "
            f"{r['debase_fair']:7.2f} {r['edge_usd']:+7.2f} {r['edge_pct']:+6.1f} "
            f"{r['iv']*100:4.0f} {r['delta']:4.2f} {r['be_g_pct']:+6.1f} {r['score']:6.1f}"
        )
    return "\n".join(out)
