"""Visualize the LEAPS edge: two cones + trend line.

Shows two uncertainty cones:
  1. RED cone (IV): the market's risk-neutral distribution, centered on
     spot × e^(rT), spread = IV. What the options market prices.
  2. GREEN cone (Lindy): the regression's prediction interval, centered on
     the trend line, spread from residual std (driven by R²). What the
     Lindy model predicts, with uncertainty.

The non-overlapping region between the cones IS the edge, visually.
High R² → tight green cone → high confidence. Low IV → tight red cone →
cheap to capture. Both tight + far apart = maximum edge.

CLI:
    python3 visualize_edge.py NYSE:TJX
    python3 visualize_edge.py NYSE:COR --save cor_edge.png
    python3 visualize_edge.py NASDAQ:CASY --interactive
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
    """Generate the two-cone edge visualization."""
    ticker = symbol.split(":")[-1] if ":" in symbol else symbol

    # Pull max history for Lindy years, 5yr for regression
    hist_max = yf.download(ticker, period="max", interval="1d",
                           auto_adjust=True, progress=False)
    hist_5y = yf.download(ticker, period="5y", interval="1d",
                          auto_adjust=True, progress=False)
    if hist_5y.empty or len(hist_5y) < 504:
        print(f"Insufficient data for {ticker}")
        return

    total_years = len(hist_max) / 252 if not hist_max.empty else 0

    all_dates = hist_5y.index
    all_prices = hist_5y["Close"].values.flatten()
    n = len(all_prices)

    # Fit regression on 5yr history
    x_all = np.arange(n)
    y_all = np.log(all_prices)
    slope_daily, intercept = np.polyfit(x_all, y_all, 1)
    ann_slope = slope_daily * 252
    r2 = float(np.corrcoef(x_all, y_all)[0, 1] ** 2)

    # Residual std (drives the Lindy cone width)
    predicted = intercept + slope_daily * x_all
    residuals = y_all - predicted
    resid_std_daily = float(np.std(np.diff(residuals)))
    resid_std_ann = resid_std_daily * np.sqrt(252)

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

    trend_today = math.exp(intercept + slope_daily * (n - 1))
    deviation = (spot / trend_today - 1) * 100

    iv = leaps["iv"]
    dte = leaps["dte"]
    drift_in_sigma = (ann_slope - r) * math.sqrt(dte / 365) / iv

    print(f"{ticker}: spot=${spot:.2f}, trend=${trend_today:.2f} "
          f"({deviation:+.1f}% vs trend)")
    print(f"  slope={ann_slope*100:.1f}%/yr, R²={r2:.3f}, "
          f"Lindy={total_years:.0f}yr, resid_σ={resid_std_ann*100:.1f}%/yr")
    print(f"  LEAPS: K={leaps['strike']:.0f}, DTE={dte}, "
          f"IV={iv*100:.0f}%, Δ={leaps['delta']:.2f}, "
          f"drift={drift_in_sigma:.2f}σ above risk-neutral")

    # --- Build the plot ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_yscale("log")

    # Show last year of price history
    one_year_bars = min(252, n)
    hist_dates = all_dates[-one_year_bars:]
    hist_prices = all_prices[-one_year_bars:]
    ax.plot(hist_dates, hist_prices, color="#2196F3", linewidth=1.5,
            label="Price", zorder=3)

    # --- Continuous regression line: history + forward ---
    today_dt = all_dates[-1]
    exp_date = leaps["exp_date"]

    # History portion
    x_hist_vis = np.arange(n - one_year_bars, n)
    trend_hist = np.exp(intercept + slope_daily * x_hist_vis)
    ax.plot(hist_dates, trend_hist, color="#4CAF50", linewidth=2,
            alpha=0.6, zorder=2)

    # Forward portion
    x_forward = np.arange(n, n + dte + 1)
    trend_forward = np.exp(intercept + slope_daily * x_forward)
    cal_days_per_td = 365 / 252
    forward_cal_dates = [today_dt + datetime.timedelta(days=int(i * cal_days_per_td))
                         for i in range(len(x_forward))]
    forward_t = np.array([i / 252 for i in range(len(x_forward))])

    ax.plot(forward_cal_dates, trend_forward, color="#4CAF50", linewidth=2.5,
            linestyle="--",
            label=f"Lindy trend ({ann_slope*100:.1f}%/yr, R²={r2:.2f})",
            zorder=4)

    # --- GREEN CONE: Lindy prediction interval (R²-driven) ---
    # Width comes from regression residual std, growing with sqrt(forward time)
    lindy_sigma_t = resid_std_ann * np.sqrt(forward_t)
    lindy_center = trend_forward  # centered on the regression line
    lindy_1u = lindy_center * np.exp(+1 * lindy_sigma_t)
    lindy_1d = lindy_center * np.exp(-1 * lindy_sigma_t)
    lindy_2u = lindy_center * np.exp(+2 * lindy_sigma_t)
    lindy_2d = lindy_center * np.exp(-2 * lindy_sigma_t)

    ax.fill_between(forward_cal_dates, lindy_2d, lindy_2u,
                     color="#4CAF50", alpha=0.06,
                     label=f"Lindy ±2σ (R²={r2:.2f}, resid {resid_std_ann*100:.0f}%)")
    ax.fill_between(forward_cal_dates, lindy_1d, lindy_1u,
                     color="#4CAF50", alpha=0.12,
                     label=f"Lindy ±1σ")

    # --- Mark spot deviation ---
    ax.plot(today_dt, spot, "o", color="#2196F3", markersize=8, zorder=5)
    ax.plot(today_dt, trend_today, "o", color="#4CAF50", markersize=8, zorder=5)
    if abs(deviation) > 2:
        ax.annotate(f"  spot {deviation:+.1f}%\n  vs trend",
                    xy=(today_dt, spot), fontsize=9, color="#2196F3",
                    ha="left", va="bottom" if deviation > 0 else "top",
                    fontweight="bold")

    # --- Risk-neutral forward (from spot) ---
    rf_forward = spot * np.exp(r * forward_t)
    ax.plot(forward_cal_dates, rf_forward, color="#FF5722", linewidth=2,
            linestyle=":",
            label=f"Risk-neutral ({r*100:.1f}%/yr)",
            zorder=4)

    # --- RED CONE: IV distribution (market's view) ---
    iv_sigma_t = iv * np.sqrt(forward_t)
    iv_1u = rf_forward * np.exp(+1 * iv_sigma_t)
    iv_1d = rf_forward * np.exp(-1 * iv_sigma_t)
    iv_2u = rf_forward * np.exp(+2 * iv_sigma_t)
    iv_2d = rf_forward * np.exp(-2 * iv_sigma_t)

    ax.fill_between(forward_cal_dates, iv_2d, iv_2u,
                     color="#FF5722", alpha=0.06,
                     label=f"IV ±2σ ({iv*100:.0f}% vol)")
    ax.fill_between(forward_cal_dates, iv_1d, iv_1u,
                     color="#FF5722", alpha=0.12,
                     label=f"IV ±1σ")

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
    ax.annotate(f"  EDGE: {edge_pct:.0f}%\n  drift = {drift_in_sigma:.1f}σ",
                xy=(forward_cal_dates[-1], mid_y),
                fontsize=11, fontweight="bold", color="#E91E63",
                ha="left", va="center")

    # --- Title with Lindy years ---
    title_line1 = f"{ticker} — LEAPS Edge: Two Cones"
    title_line2 = (f"Slope={ann_slope*100:.1f}%/yr  R²={r2:.3f}  "
                   f"Lindy={total_years:.0f}yr  "
                   f"IV={iv*100:.0f}%  DTE={dte}  K=${leaps['strike']:.0f}")
    if abs(deviation) > 1:
        title_line2 += f"  Spot {deviation:+.1f}% vs trend"
    ax.set_title(f"{title_line1}\n{title_line2}",
                 fontsize=13, fontweight="bold")

    # --- Info box with key metrics ---
    info_text = (
        f"Lindy: {total_years:.0f}yr history, {ann_slope*100:.1f}%/yr, R²={r2:.3f}\n"
        f"Green cone: resid σ = {resid_std_ann*100:.0f}%/yr (from R²)\n"
        f"Red cone: IV = {iv*100:.0f}%/yr (from options market)\n"
        f"Drift = {drift_in_sigma:.2f}σ above risk-neutral"
    )
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
            fontsize=9, fontfamily="monospace",
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="gray", alpha=0.85))

    # --- Formatting ---
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (log scale)")
    ax.legend(loc="upper left", fontsize=9)
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
        "resid_std_ann": resid_std_ann,
        "iv": iv, "drift_in_sigma": drift_in_sigma,
        "trend_at_exp": trend_at_exp, "rf_at_exp": rf_at_exp,
        "edge_pct": edge_pct, "total_years": total_years,
    }


def main():
    p = argparse.ArgumentParser(
        description="Visualize the LEAPS edge: Lindy cone vs IV cone.",
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
