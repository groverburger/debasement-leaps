"""Visualize the LEAPS edge: historical trend vs IV uncertainty cone.

For a given symbol, plots:
  1. Past year of daily price data (log scale)
  2. ONE continuous regression line through history AND projected forward
     (the Lindy trend — where the stock "should be" if the trend holds)
  3. Spot's deviation from the trend line (overextended / underextended)
  4. The IV cone: ±1σ and ±2σ from the risk-neutral forward
  5. The gap between the trend line at expiry and the IV cone center = the edge

The visual makes the thesis intuitive: the stock has been climbing along a
straight line for years, but options are priced as if it will grow at 4.5%/yr.
The divergence between the trend line and the cone center is the unpriced drift.

CLI:
    python3 visualize_edge.py NYSE:TJX
    python3 visualize_edge.py NASDAQ:FAST --save fast_edge.png
    python3 visualize_edge.py NYSE:COR --interactive
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


def get_best_leaps(tv: TVOptions, symbol: str, spot: float):
    """Find the longest-DTE call with delta 0.35-0.55."""
    exps = tv.get_expirations(symbol)
    today = datetime.date.today()
    today_int = int(today.strftime("%Y%m%d"))

    for e in sorted(exps, reverse=True):
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
            pick = min(calls, key=lambda o: abs((o.delta or 0) - 0.45))
            return {
                "exp": e, "dte": dte, "exp_date": ed,
                "strike": pick.strike, "iv": pick.iv,
                "delta": pick.delta, "mid": pick.mid, "ask": pick.ask,
            }
    return None


def plot_edge(symbol: str, r: float = 0.045,
              save_path: str | None = None, interactive: bool = False):
    """Generate the edge visualization."""
    ticker = symbol.split(":")[-1] if ":" in symbol else symbol

    # Get price history — 5 years for regression, show last year
    hist = yf.download(ticker, period="5y", interval="1d",
                       auto_adjust=True, progress=False)
    if hist.empty or len(hist) < 504:
        print(f"Insufficient data for {ticker}")
        return

    all_dates = hist.index
    all_prices = hist["Close"].values.flatten()
    n = len(all_prices)

    # Fit ONE regression on the full 5yr history
    x_all = np.arange(n)
    y_all = np.log(all_prices)
    slope_daily, intercept = np.polyfit(x_all, y_all, 1)
    ann_slope = slope_daily * 252
    r2 = float(np.corrcoef(x_all, y_all)[0, 1] ** 2)

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

    # Where the trend line says the stock "should be" today
    trend_today = math.exp(intercept + slope_daily * (n - 1))
    deviation = (spot / trend_today - 1) * 100
    dev_label = "above" if deviation > 0 else "below"

    print(f"{ticker}: spot=${spot:.2f}, trend=${trend_today:.2f} "
          f"({deviation:+.1f}% {dev_label} trend)")
    print(f"  slope={ann_slope*100:.1f}%/yr, R²={r2:.3f}")
    print(f"  LEAPS: K={leaps['strike']:.0f}, DTE={leaps['dte']}, "
          f"IV={leaps['iv']*100:.0f}%, Δ={leaps['delta']:.2f}, ask=${leaps['ask']:.2f}")

    # --- Build the plot ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_yscale("log")

    # Show last year of price history
    one_year_bars = min(252, n)
    hist_dates = all_dates[-one_year_bars:]
    hist_prices = all_prices[-one_year_bars:]
    ax.plot(hist_dates, hist_prices, color="#2196F3", linewidth=1.5,
            label="Price", zorder=3)

    # --- ONE continuous regression line: history + forward ---
    today_dt = all_dates[-1]
    exp_date = leaps["exp_date"]
    dte = leaps["dte"]

    # Regression line through visible history
    x_hist_vis = np.arange(n - one_year_bars, n)
    trend_hist = np.exp(intercept + slope_daily * x_hist_vis)
    ax.plot(hist_dates, trend_hist, color="#4CAF50", linewidth=2,
            alpha=0.6, zorder=2)

    # Same regression line projected forward (continuous from the same fit)
    x_forward = np.arange(n, n + dte + 1)
    trend_forward = np.exp(intercept + slope_daily * x_forward)
    forward_dates = [today_dt + datetime.timedelta(days=int(i * 365 / 252))
                     for i in range(dte + 1)]
    # Approximate: map trading days to calendar days
    cal_days_per_trade_day = 365 / 252
    forward_cal_dates = [today_dt + datetime.timedelta(days=int(i * cal_days_per_trade_day))
                         for i in range(len(x_forward))]

    ax.plot(forward_cal_dates, trend_forward, color="#4CAF50", linewidth=2.5,
            linestyle="--",
            label=f"Lindy trend ({ann_slope*100:.1f}%/yr, R²={r2:.2f})",
            zorder=4)

    # Mark spot's deviation from trend
    ax.plot(today_dt, spot, "o", color="#2196F3", markersize=8, zorder=5)
    ax.plot(today_dt, trend_today, "o", color="#4CAF50", markersize=8, zorder=5)
    if abs(deviation) > 2:
        ax.annotate(f"  spot {deviation:+.1f}%\n  vs trend",
                    xy=(today_dt, spot), fontsize=9, color="#2196F3",
                    ha="left", va="bottom" if deviation > 0 else "top",
                    fontweight="bold")

    # --- Risk-neutral forward (from SPOT, not from trend) ---
    forward_t = np.array([i / 252 for i in range(len(x_forward))])
    rf_forward = spot * np.exp(r * forward_t)
    ax.plot(forward_cal_dates, rf_forward, color="#FF5722", linewidth=2,
            linestyle=":",
            label=f"Risk-neutral ({r*100:.1f}%/yr — what options price)",
            zorder=4)

    # --- IV cone (±1σ, ±2σ) from risk-neutral ---
    iv = leaps["iv"]
    sigma_t = iv * np.sqrt(forward_t)
    cone_1u = rf_forward * np.exp(+1 * sigma_t)
    cone_1d = rf_forward * np.exp(-1 * sigma_t)
    cone_2u = rf_forward * np.exp(+2 * sigma_t)
    cone_2d = rf_forward * np.exp(-2 * sigma_t)

    ax.fill_between(forward_cal_dates, cone_2d, cone_2u,
                     color="#FF5722", alpha=0.08, label=f"IV ±2σ ({iv*100:.0f}% vol)")
    ax.fill_between(forward_cal_dates, cone_1d, cone_1u,
                     color="#FF5722", alpha=0.15, label=f"IV ±1σ")

    # --- Strike ---
    ax.axhline(y=leaps["strike"], color="#9C27B0", linewidth=1, linestyle="-.",
               alpha=0.5, label=f"Strike ${leaps['strike']:.0f}")

    # --- Edge annotation at expiry ---
    trend_at_exp = float(trend_forward[-1])
    rf_at_exp = float(rf_forward[-1])
    edge_pct = (trend_at_exp / rf_at_exp - 1) * 100

    mid_y = math.sqrt(trend_at_exp * rf_at_exp)
    ax.annotate("",
                xy=(forward_cal_dates[-1], trend_at_exp),
                xytext=(forward_cal_dates[-1], rf_at_exp),
                arrowprops=dict(arrowstyle="<->", color="#E91E63", lw=2.5))
    ax.annotate(f"  THE EDGE\n  {edge_pct:.0f}% unpriced drift",
                xy=(forward_cal_dates[-1], mid_y),
                fontsize=11, fontweight="bold", color="#E91E63",
                ha="left", va="center")

    # --- Formatting ---
    title_line2 = (f"Slope={ann_slope*100:.1f}%/yr  R²={r2:.3f}  "
                   f"IV={iv*100:.0f}%  DTE={dte}  K=${leaps['strike']:.0f}")
    if abs(deviation) > 1:
        title_line2 += f"  Spot {deviation:+.1f}% vs trend"
    ax.set_title(f"{ticker} — LEAPS Edge Visualization\n{title_line2}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (log scale)")
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(hist_dates[0], forward_cal_dates[-1] + datetime.timedelta(days=30))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))

    plt.tight_layout()

    if save_path is None:
        save_path = f"{ticker.lower()}_edge.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")

    if interactive:
        plt.show()
    plt.close()

    return {
        "ticker": ticker, "spot": spot, "trend_today": trend_today,
        "deviation_pct": deviation, "slope": ann_slope, "r2": r2,
        "trend_at_exp": trend_at_exp, "rf_at_exp": rf_at_exp,
        "edge_pct": edge_pct,
    }


def main():
    p = argparse.ArgumentParser(
        description="Visualize the LEAPS edge: Lindy trend vs IV cone.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("symbol", help="TradingView symbol, e.g. NYSE:TJX")
    p.add_argument("--rfr", type=float, default=0.045)
    p.add_argument("--save", type=str, default=None, help="Output filename")
    p.add_argument("--interactive", action="store_true", help="Show plot window")
    args = p.parse_args()

    plot_edge(
        symbol=args.symbol,
        r=args.rfr,
        save_path=args.save,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()
