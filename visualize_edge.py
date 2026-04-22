"""Visualize the LEAPS edge: historical trend vs IV uncertainty cone.

For a given symbol, plots:
  1. Past year of daily price data (log scale)
  2. The Lindy slope projected forward to LEAPS expiry (the drift the market isn't pricing)
  3. The IV cone: ±1σ and ±2σ from the risk-neutral forward (what the market IS pricing)
  4. The gap between the Lindy line and the IV cone center IS the edge

The visual makes the thesis intuitive: the stock has been climbing at X%/yr
along a straight line, but the options are priced as if it will grow at 4.5%/yr
with Y% uncertainty. The divergence between the trend line and the cone center
is the unpriced drift.

CLI:
    python3 visualize_edge.py NYSE:TJX --beta 2.4
    python3 visualize_edge.py NASDAQ:FAST --beta 2.4 --save fast_edge.png
    python3 visualize_edge.py NYSE:COR --beta 2.1 --interactive
"""
from __future__ import annotations

import argparse
import datetime
import math
import os
import sys

import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.expanduser("~/Documents/projects/darkfield"))
from tv_options import TVOptions
from pricing import debase_fair_value


def get_best_leaps(tv: TVOptions, symbol: str, spot: float):
    """Find the longest-DTE call with delta 0.35-0.55 (near ATM-ish)."""
    exps = tv.get_expirations(symbol)
    today = datetime.date.today()
    today_int = int(today.strftime("%Y%m%d"))

    best = None
    for e in sorted(exps, reverse=True):  # longest first
        if e <= today_int:
            continue
        try:
            ed = datetime.date(int(str(e)[:4]), int(str(e)[4:6]), int(str(e)[6:8]))
        except:
            continue
        dte = (ed - today).days
        if dte < 150:
            continue
        try:
            chain = tv.get_chain(symbol, e, spot=spot)
        except:
            continue
        calls = [o for o in chain.options
                 if o.option_type == "call" and o.iv and o.iv > 0
                 and o.delta and 0.35 <= o.delta <= 0.55
                 and o.mid and o.mid > 0]
        if calls:
            # Pick closest to delta 0.45
            pick = min(calls, key=lambda o: abs((o.delta or 0) - 0.45))
            best = {
                "exp": e, "dte": dte, "exp_date": ed,
                "strike": pick.strike, "iv": pick.iv,
                "delta": pick.delta, "mid": pick.mid, "ask": pick.ask,
            }
            break  # take the longest available
    return best


def compute_lindy_slope(ticker: str, lookback_years: int = 5):
    """Compute annualized log-linear slope and R² from daily data."""
    data = yf.download(ticker, period=f"{lookback_years}y", interval="1d",
                       auto_adjust=True, progress=False)
    if data.empty or len(data) < 252:
        return None
    c = data["Close"].values.flatten()
    x = np.arange(len(c))
    y = np.log(c)
    slope, intercept = np.polyfit(x, y, 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    ann_slope = slope * 252
    return {
        "dates": data.index,
        "prices": c,
        "slope_daily": slope,
        "intercept": intercept,
        "ann_slope": ann_slope,
        "r2": r2,
    }


def plot_edge(symbol: str, beta: float, m2: float = 0.07, r: float = 0.045,
              save_path: str | None = None, interactive: bool = False):
    """Generate the edge visualization."""
    ticker = symbol.split(":")[-1] if ":" in symbol else symbol

    # Get Lindy slope
    lindy = compute_lindy_slope(ticker, lookback_years=5)
    if lindy is None:
        print(f"Insufficient data for {ticker}")
        return

    # Get options data
    tv = TVOptions()
    spot = tv.get_underlying_price(symbol)
    if not spot:
        print(f"Can't get spot for {symbol}")
        return

    leaps = get_best_leaps(tv, symbol, spot)
    if leaps is None:
        print(f"No eligible LEAPS for {symbol}")
        return

    print(f"{ticker}: spot=${spot:.2f}, slope={lindy['ann_slope']*100:.1f}%/yr, R²={lindy['r2']:.3f}")
    print(f"  LEAPS: K={leaps['strike']:.0f}, DTE={leaps['dte']}, IV={leaps['iv']*100:.0f}%, "
          f"Δ={leaps['delta']:.2f}, ask=${leaps['ask']:.2f}")

    # --- Build the plot ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_yscale("log")

    dates = lindy["dates"]
    prices = lindy["prices"]

    # Only show last year of history for clarity
    one_year_ago = dates[-1] - datetime.timedelta(days=365)
    mask = dates >= one_year_ago
    hist_dates = dates[mask]
    hist_prices = prices[mask.values] if hasattr(mask, 'values') else prices[mask]

    # Plot historical prices
    ax.plot(hist_dates, hist_prices, color="#2196F3", linewidth=1.5, label="Price", zorder=3)

    # --- Forward projection dates ---
    today = dates[-1]
    exp_date = leaps["exp_date"]
    n_forward_days = leaps["dte"]
    forward_dates = [today + datetime.timedelta(days=i) for i in range(n_forward_days + 1)]
    forward_t = np.array([i / 252 for i in range(n_forward_days + 1)])  # in years

    # --- Lindy trend line (past + projected forward) ---
    # Fit line on last 2 years for the visual slope
    n_2y = min(504, len(prices))
    recent = prices[-n_2y:]
    x_recent = np.arange(n_2y)
    slope_2y, intercept_2y = np.polyfit(x_recent, np.log(recent), 1)
    ann_slope_2y = slope_2y * 252

    # Draw trend line through history and forward
    trend_start_idx = len(prices) - n_2y
    trend_dates_hist = dates[trend_start_idx:]
    trend_vals_hist = np.exp(intercept_2y + slope_2y * np.arange(n_2y))

    # Forward projection from last price at the historical slope
    lindy_forward = spot * np.exp(lindy["ann_slope"] * forward_t)
    ax.plot(forward_dates, lindy_forward, color="#4CAF50", linewidth=2.5,
            linestyle="--", label=f"Lindy slope ({lindy['ann_slope']*100:.1f}%/yr, R²={lindy['r2']:.2f})",
            zorder=4)

    # Also show trend line through recent history
    ax.plot(trend_dates_hist, trend_vals_hist, color="#4CAF50", linewidth=1,
            alpha=0.4, zorder=2)

    # --- Risk-neutral forward (what market prices) ---
    rf_forward = spot * np.exp(r * forward_t)
    ax.plot(forward_dates, rf_forward, color="#FF5722", linewidth=2,
            linestyle=":", label=f"Risk-neutral drift ({r*100:.1f}%/yr — what options price)",
            zorder=4)

    # --- IV cone (±1σ, ±2σ) ---
    iv = leaps["iv"]
    sigma_t = iv * np.sqrt(forward_t)  # cumulative vol

    # Upper and lower bounds centered on risk-neutral forward
    cone_1u = rf_forward * np.exp(+1 * sigma_t)
    cone_1d = rf_forward * np.exp(-1 * sigma_t)
    cone_2u = rf_forward * np.exp(+2 * sigma_t)
    cone_2d = rf_forward * np.exp(-2 * sigma_t)

    ax.fill_between(forward_dates, cone_2d, cone_2u,
                     color="#FF5722", alpha=0.08, label=f"IV ±2σ ({iv*100:.0f}% vol)")
    ax.fill_between(forward_dates, cone_1d, cone_1u,
                     color="#FF5722", alpha=0.15, label=f"IV ±1σ")

    # --- Mark the strike ---
    ax.axhline(y=leaps["strike"], color="#9C27B0", linewidth=1, linestyle="-.",
               alpha=0.6, label=f"Strike ${leaps['strike']:.0f}")

    # --- Mark the edge at expiry ---
    lindy_at_exp = spot * np.exp(lindy["ann_slope"] * leaps["dte"] / 252)
    rf_at_exp = spot * np.exp(r * leaps["dte"] / 252)

    # Annotate the gap
    mid_y = math.sqrt(lindy_at_exp * rf_at_exp)  # geometric midpoint for log scale
    edge_pct = (lindy_at_exp / rf_at_exp - 1) * 100
    ax.annotate("",
                xy=(exp_date, lindy_at_exp),
                xytext=(exp_date, rf_at_exp),
                arrowprops=dict(arrowstyle="<->", color="#E91E63", lw=2))
    ax.annotate(f"  THE EDGE\n  {edge_pct:.0f}% unpriced drift",
                xy=(exp_date, mid_y),
                fontsize=11, fontweight="bold", color="#E91E63",
                ha="left", va="center")

    # --- Formatting ---
    ax.set_title(f"{ticker} — LEAPS Edge Visualization\n"
                 f"Slope={lindy['ann_slope']*100:.1f}%/yr  R²={lindy['r2']:.3f}  "
                 f"IV={iv*100:.0f}%  DTE={leaps['dte']}  K=${leaps['strike']:.0f}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (log scale)")
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(hist_dates[0], exp_date + datetime.timedelta(days=30))

    # Price labels on right axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))

    plt.tight_layout()

    # Save
    if save_path is None:
        save_path = f"{ticker.lower()}_edge.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")

    if interactive:
        plt.show()
    plt.close()


def main():
    p = argparse.ArgumentParser(
        description="Visualize the LEAPS edge: Lindy slope vs IV cone.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("symbol", help="TradingView symbol, e.g. NYSE:TJX")
    p.add_argument("--beta", type=float, default=None,
                   help="Beta (for annotation only; slope is computed from data)")
    p.add_argument("--m2", type=float, default=0.07)
    p.add_argument("--rfr", type=float, default=0.045)
    p.add_argument("--save", type=str, default=None, help="Output filename")
    p.add_argument("--interactive", action="store_true", help="Show plot window")
    args = p.parse_args()

    plot_edge(
        symbol=args.symbol,
        beta=args.beta or 1.5,
        m2=args.m2,
        r=args.rfr,
        save_path=args.save,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()
