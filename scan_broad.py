"""Debasement-LEAPS scanner.

For each name in the universe:
  1. Pull LEAPS chain (~9-15 month DTE)
  2. Filter calls to reasonable-delta range (0.40-0.75)
  3. Compute "debasement fair value" under per-name beta × M2 growth drift
  4. Rank by divergence from market mid

Per-name beta reflects the asset's historical sensitivity to money-supply
expansion. Crypto proxies carry the highest beta; boring infra names lower.
Edit UNIVERSE to add/remove names or adjust betas.
"""
import sys, os, math, datetime
sys.path.insert(0, os.path.expanduser("~/Documents/projects/darkfield"))
from tv_options import TVOptions
from pricing import bs_call, debase_fair_value, breakeven_annualized_growth

# (tv_symbol, beta_to_m2, theme)
UNIVERSE = {
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

# Assumptions
R = 0.045          # risk-free rate
M2 = 0.07          # M2 expansion rate
DTE_TARGETS = (300, 400, 500)  # target LEAPS DTEs (~10, 13, 16 months)
DELTA_MIN = 0.40
DELTA_MAX = 0.75



def analyze(tv, ticker, sym, beta):
    try:
        spot = tv.get_underlying_price(sym)
        if not spot or spot <= 0:
            return []
        exps = tv.get_expirations(sym)
    except Exception as e:
        return []

    today = datetime.date.today()
    today_int = int(today.strftime("%Y%m%d"))
    future = []
    for e in exps:
        if e <= today_int: continue
        try:
            ed = datetime.date(int(str(e)[:4]), int(str(e)[4:6]), int(str(e)[6:8]))
        except Exception:
            continue
        dte = (ed - today).days
        future.append((e, dte))
    if not future:
        return []

    # Pick closest to each DTE target
    picks = []
    for want in DTE_TARGETS:
        best = min(future, key=lambda x: abs(x[1] - want))
        if best not in picks:
            picks.append(best)

    mu = beta * M2
    rows = []
    for exp, dte in picks:
        try:
            chain = tv.get_chain(sym, exp, spot=spot)
        except Exception:
            continue
        T = dte / 365.0
        for o in chain.options:
            if o.option_type != "call": continue
            if o.mid is None or o.mid <= 0: continue
            if o.iv is None or o.iv <= 0: continue
            delta = o.delta or 0
            if delta < DELTA_MIN or delta > DELTA_MAX: continue
            K = o.strike
            mid = o.mid
            iv = o.iv
            bs = bs_call(spot, K, T, R, iv, q=0)
            df = debase_fair_value(spot, K, T, R, iv, mu)
            diff = (df - mid) / mid * 100 if mid > 0 else 0
            be = breakeven_annualized_growth(spot, K, T, mid) * 100
            # Composite score: divergence × sqrt(DTE/365) × reasonable-IV penalty
            # Penalize very high IV (>80%) because drift edge gets eaten
            iv_penalty = max(0.3, 1.0 - max(0, iv - 0.8) * 2)
            score = diff * math.sqrt(T) * iv_penalty
            rows.append({
                "ticker": ticker, "theme": UNIVERSE[ticker][2],
                "spot": spot, "beta": beta, "mu": mu,
                "exp": exp, "dte": dte,
                "strike": K, "ask": o.ask or 0, "bid": o.bid or 0,
                "mid": mid, "iv": iv, "delta": delta,
                "bs_fair": bs, "debase_fair": df,
                "diff_pct": diff, "be_g_pct": be,
                "score": score,
            })
    return rows


def main():
    tv = TVOptions()
    all_rows = []
    for ticker, (sym, beta, theme) in UNIVERSE.items():
        try:
            rows = analyze(tv, ticker, sym, beta)
            all_rows.extend(rows)
            if rows:
                best = max(rows, key=lambda r: r["score"])
                print(f"{ticker:6s} spot={best['spot']:8.2f}  β={beta:.1f}  "
                      f"n={len(rows)}  best: K={best['strike']:.1f} DTE={best['dte']} "
                      f"diff={best['diff_pct']:+.1f}%  IV={best['iv']*100:.0f}%")
            else:
                print(f"{ticker:6s} — no eligible LEAPS")
        except Exception as e:
            print(f"{ticker:6s} ERR: {e}")

    if not all_rows:
        print("no rows")
        return

    print(f"\n=== TOP 30 BY COMPOSITE SCORE (divergence × sqrt(T) × IV-penalty) ===")
    all_rows.sort(key=lambda r: r["score"], reverse=True)
    hdr = f"{'tick':5s} {'theme':14s} {'DTE':>4s} {'K':>7s} {'spot':>7s} {'ask':>6s} {'mid':>6s} {'IV':>4s} {'Δ':>4s} {'diff%':>6s} {'BE%':>6s} {'score':>6s}"
    print(hdr)
    for r in all_rows[:30]:
        print(f"{r['ticker']:5s} {r['theme']:14s} {r['dte']:4d} {r['strike']:7.1f} "
              f"{r['spot']:7.2f} {r['ask']:6.2f} {r['mid']:6.2f} {r['iv']*100:4.0f} "
              f"{r['delta']:4.2f} {r['diff_pct']:+6.1f} {r['be_g_pct']:+6.1f} {r['score']:6.1f}")

    print(f"\n=== BEST PER TICKER (top by ticker, sorted by score) ===")
    by_tick = {}
    for r in all_rows:
        t = r['ticker']
        if t not in by_tick or r['score'] > by_tick[t]['score']:
            by_tick[t] = r
    best_list = sorted(by_tick.values(), key=lambda r: r['score'], reverse=True)
    print(hdr)
    for r in best_list:
        print(f"{r['ticker']:5s} {r['theme']:14s} {r['dte']:4d} {r['strike']:7.1f} "
              f"{r['spot']:7.2f} {r['ask']:6.2f} {r['mid']:6.2f} {r['iv']*100:4.0f} "
              f"{r['delta']:4.2f} {r['diff_pct']:+6.1f} {r['be_g_pct']:+6.1f} {r['score']:6.1f}")


if __name__ == "__main__":
    main()
