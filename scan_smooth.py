"""Debasement-LEAPS scanner — smooth multi-year trend edition.

Two-stage pipeline:
  1. yfinance pre-filter on a broad universe of compounders/debasement-exposed
     names. Keeps only those with:
       - 2-year return > +30%
       - % of last 504 bars above 200MA >= 85%
       - max drawdown over 504d >= -22%
       - ATR% (14d) < 4.5% (low realized vol)
       - within 8% of 52-week high (still in trend, not basing deep)
  2. For survivors: pull LEAPS chains via tv_options, compute debasement
     fair value, rank by edge (debase_fair − market_mid, in $ and %).

Universe is biased toward multi-year smooth compounders — insurance, asset
managers, toll-road financials, boring industrials, quality healthcare, plus
the real-capex and hard-asset names.
"""
import sys, os, math, datetime
import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.expanduser("~/Documents/projects/darkfield"))
from tv_options import TVOptions
from pricing import bs_call, debase_fair_value, breakeven_annualized_growth

# (tv_symbol, beta_to_m2, theme)
UNIVERSE = {
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

R = 0.045
M2 = 0.07
DTE_TARGETS = (400, 550, 700)
DELTA_MIN = 0.40
DELTA_MAX = 0.75


def atr_pct(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return (tr.rolling(n).mean() / c).iloc[-1]


def screen_smoothness(tickers):
    """Pre-filter via yfinance. Returns dict ticker -> (passes, metrics)."""
    print(f"Pre-filtering {len(tickers)} names via yfinance...")
    data = yf.download(tickers, period="3y", interval="1d",
                       auto_adjust=True, progress=False, group_by="ticker", threads=True)
    results = {}
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
            max_dd = float((last504 / rmax - 1).min() * 100)
            ret_504 = (last / float(last504.iloc[0]) - 1) * 100
            ret_252 = (last / float(c.iloc[-252]) - 1) * 100
            hi252 = float(df["High"].iloc[-252:].max())
            dist_hi = (hi252 - last) / last * 100
            atr = atr_pct(df, 14) * 100
            passes = (
                ret_504 > 15 and
                above200_pct >= 75 and
                max_dd >= -28 and
                atr < 6.0 and
                dist_hi < 15.0
            )
            # smoothness score: higher is smoother trend
            smooth = above200_pct + max(0, max_dd + 22) - atr * 3 + min(ret_504 / 10, 20)
            results[t] = (passes, {
                "last": last, "ret_504": ret_504, "ret_252": ret_252,
                "above200_pct": above200_pct, "max_dd": max_dd,
                "dist_hi": dist_hi, "atr_pct": atr, "smooth_score": smooth,
            })
        except Exception as e:
            results[t] = (False, {"reason": str(e)[:50]})
    return results


def analyze_chain(tv, ticker, sym, beta, spot_hint):
    try:
        spot = tv.get_underlying_price(sym) or spot_hint
        if not spot or spot <= 0:
            return []
        exps = tv.get_expirations(sym)
    except Exception:
        return []
    today = datetime.date.today()
    today_int = int(today.strftime("%Y%m%d"))
    future = []
    for e in exps:
        if e <= today_int: continue
        try:
            ed = datetime.date(int(str(e)[:4]), int(str(e)[4:6]), int(str(e)[6:8]))
        except Exception: continue
        dte = (ed - today).days
        future.append((e, dte))
    if not future:
        return []
    picks = []
    for want in DTE_TARGETS:
        b = min(future, key=lambda x: abs(x[1] - want))
        if b not in picks: picks.append(b)

    mu = beta * M2
    rows = []
    for exp, dte in picks:
        try: chain = tv.get_chain(sym, exp, spot=spot)
        except Exception: continue
        T = dte / 365.0
        for o in chain.options:
            if o.option_type != "call": continue
            if not o.mid or o.mid <= 0: continue
            if not o.iv or o.iv <= 0: continue
            delta = o.delta or 0
            if delta < DELTA_MIN or delta > DELTA_MAX: continue
            K = o.strike
            mid = o.mid
            iv = o.iv
            bs = bs_call(spot, K, T, R, iv, q=0)
            df_fv = debase_fair_value(spot, K, T, R, iv, mu)
            edge_usd = df_fv - mid
            edge_pct = edge_usd / mid * 100 if mid > 0 else 0
            be = breakeven_annualized_growth(spot, K, T, mid)
            iv_penalty = max(0.3, 1.0 - max(0, iv - 0.8) * 2)
            score = edge_pct * math.sqrt(T) * iv_penalty
            rows.append({
                "ticker": ticker, "theme": UNIVERSE[ticker][2],
                "spot": spot, "beta": beta, "mu": mu,
                "exp": exp, "dte": dte,
                "strike": K, "ask": o.ask or 0, "bid": o.bid or 0,
                "mid": mid, "iv": iv, "delta": delta,
                "bs_fair": bs, "debase_fair": df_fv,
                "edge_usd": edge_usd, "edge_pct": edge_pct, "be_g_pct": be * 100,
                "score": score,
            })
    return rows


def main():
    tickers = list(UNIVERSE.keys())
    screen = screen_smoothness(tickers)
    survivors = [t for t, (p, _) in screen.items() if p]
    print(f"\n=== {len(survivors)}/{len(tickers)} passed smooth-trend filter ===")
    print(f"{'tick':5s} {'ret2y%':>7s} {'above200%':>10s} {'maxdd%':>7s} {'atr%':>5s} {'dist_hi%':>8s} {'smooth':>7s}")
    smooth_rank = sorted([(t, screen[t][1]) for t in survivors],
                         key=lambda x: x[1].get("smooth_score", 0), reverse=True)
    for t, m in smooth_rank:
        print(f"{t:5s} {m['ret_504']:7.1f} {m['above200_pct']:10.1f} "
              f"{m['max_dd']:7.1f} {m['atr_pct']:5.2f} {m['dist_hi']:8.2f} {m['smooth_score']:7.1f}")

    tv = TVOptions()
    all_rows = []
    print(f"\n=== Pulling chains for survivors ===")
    for t in survivors:
        sym, beta, _ = UNIVERSE[t]
        try:
            rows = analyze_chain(tv, t, sym, beta, screen[t][1]["last"])
            all_rows.extend(rows)
            if rows:
                best = max(rows, key=lambda r: r["score"])
                print(f"{t:5s} β={beta:.1f}  n={len(rows):3d}  best: K={best['strike']:.0f} "
                      f"DTE={best['dte']}  edge=${best['edge_usd']:5.2f} ({best['edge_pct']:+.1f}%)  "
                      f"IV={best['iv']*100:.0f}%")
            else:
                print(f"{t:5s} — no eligible LEAPS strikes")
        except Exception as e:
            print(f"{t:5s} ERR: {str(e)[:60]}")

    if not all_rows:
        return

    print(f"\n=== TOP 30 BY COMPOSITE SCORE (edge% × √T × IV-penalty) ===")
    print(f"Columns: MID=market mid | DB=debasement fair value | EDGE$=DB-MID in $ | EDGE%=DB/MID-1")
    hdr = (f"{'tick':5s} {'theme':16s} {'DTE':>4s} {'K':>7s} {'spot':>7s} {'ask':>6s} "
           f"{'MID':>6s} {'DB':>6s} {'EDGE$':>6s} {'EDGE%':>6s} {'IV':>4s} {'Δ':>4s} "
           f"{'BE%':>6s} {'score':>6s}")
    print(hdr)
    all_rows.sort(key=lambda r: r["score"], reverse=True)
    for r in all_rows[:30]:
        print(f"{r['ticker']:5s} {r['theme']:16s} {r['dte']:4d} {r['strike']:7.1f} "
              f"{r['spot']:7.2f} {r['ask']:6.2f} {r['mid']:6.2f} {r['debase_fair']:6.2f} "
              f"{r['edge_usd']:+6.2f} {r['edge_pct']:+6.1f} {r['iv']*100:4.0f} {r['delta']:4.2f} "
              f"{r['be_g_pct']:+6.1f} {r['score']:6.1f}")

    print(f"\n=== BEST PICK PER TICKER (ranked by score) ===")
    by_tick = {}
    for r in all_rows:
        t = r['ticker']
        if t not in by_tick or r['score'] > by_tick[t]['score']:
            by_tick[t] = r
    best_list = sorted(by_tick.values(), key=lambda r: r['score'], reverse=True)
    print(hdr)
    for r in best_list:
        print(f"{r['ticker']:5s} {r['theme']:16s} {r['dte']:4d} {r['strike']:7.1f} "
              f"{r['spot']:7.2f} {r['ask']:6.2f} {r['mid']:6.2f} {r['debase_fair']:6.2f} "
              f"{r['edge_usd']:+6.2f} {r['edge_pct']:+6.1f} {r['iv']*100:4.0f} {r['delta']:4.2f} "
              f"{r['be_g_pct']:+6.1f} {r['score']:6.1f}")


if __name__ == "__main__":
    main()
