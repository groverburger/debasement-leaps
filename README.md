# debasement-leaps

Tools for finding LEAPS that are cheap under a monetary-debasement thesis.

## The thesis

Standard option pricing (Black-Scholes) assumes the underlying drifts at the
risk-free rate `r`. If you believe the broad money supply expands at rate `m`
per year and your asset has beta `b` to that expansion, the true expected
drift is `μ = m·b` — typically well above `r`.

Re-pricing calls with drift `μ` (via BS with `q = r − μ`, equivalent to
Garman-Kohlhagen) produces a *debasement fair value*. Calls where market mid
lags the debasement fair value are cheap under your thesis.

See [THEORY.md](./THEORY.md) for the deeper argument: why dealers' standard
delta-hedging framework systematically under-captures drift on LEAPS of
high-β-to-M2 names, and why that's the structural source of the edge.

## Files

| File | Purpose |
|---|---|
| `THEORY.md` | Why this edge exists — dealer hedging friction under drift asymmetry. |
| `pricing.py` | Shared BS + debasement fair-value + breakeven math. |
| `chain_analysis.py` | Programmatic API: `analyze_ticker()`, `best_strikes()`, `format_rows_table()`. |
| `inspect_ticker.py` | **CLI** — single-ticker inspector. Start here for agent calls. |
| `scan_broad.py` | **CLI** — broad universe scan (crypto beta, semis, infra, commodities). |
| `scan_smooth.py` | **CLI** — smooth-trend scan: yfinance pre-filter + LEAPS chain analysis. |
| `debasement_leaps.py` | Textual TUI for interactive exploration (F5 to refetch). |

## CLI reference (agent-friendly)

All CLIs support `--json` for machine-readable output. The text table and the
JSON rows share the same schema (see *Row schema* below).

### Single-ticker inspector — `inspect_ticker.py`

```bash
# Text table, default assumptions
python3 inspect_ticker.py NYSE:PWR --beta 1.5

# JSON, explicit assumptions
python3 inspect_ticker.py NASDAQ:HOOD --beta 3.0 --top 5 --json

# Tune M2, risk-free rate, DTE gate, delta range
python3 inspect_ticker.py NYSE:TRV --beta 1.2 --m2 0.08 --rfr 0.04 \
    --min-dte 400 --max-dte 800 --delta-min 0.45 --delta-max 0.70
```

### Broad universe scan — `scan_broad.py`

```bash
# Full scan, top 30
python3 scan_broad.py --top 30

# Subset of tickers, JSON output
python3 scan_broad.py --tickers PWR,HOOD,GDX --json

# Higher M2 assumption
python3 scan_broad.py --m2 0.09
```

### Smooth-trend scan — `scan_smooth.py`

```bash
# Default filters (multi-year smooth compounders only)
python3 scan_smooth.py

# Loosen smoothness filters, JSON output
python3 scan_smooth.py --min-above-200 70 --max-dd -35 --max-atr 7 --json

# Tighten everything for only the cleanest trends
python3 scan_smooth.py --min-above-200 90 --max-dd -15 --max-atr 3
```

Filter flags (`scan_smooth.py`):
- `--min-ret-2y` — minimum 2-year return % (default 15)
- `--min-above-200` — minimum % of last 504d above 200MA (default 75)
- `--max-dd` — max allowed drawdown over 504d (default -28)
- `--max-atr` — max 14d ATR % (default 6)
- `--max-dist-hi` — max % below 52w high (default 15)

## Python API (for notebooks / agents running in-process)

```python
from chain_analysis import analyze_ticker, best_strikes

# All eligible calls, enriched with debasement fair value + edge
rows = analyze_ticker(
    symbol="NASDAQ:HOOD",
    beta=3.0,
    m2=0.07,          # 7% M2 growth
    r=0.045,          # risk-free rate
    min_dte=300,
    max_dte=700,
    delta_range=(0.40, 0.75),
)

# Or just the top N by composite score
top = best_strikes("NYSE:PWR", beta=1.5, n=10)
for row in top:
    print(f"{row['strike']:.0f}@{row['dte']}d  mid=${row['mid']:.2f}  "
          f"debase=${row['debase_fair']:.2f}  edge={row['edge_pct']:+.1f}%")
```

### Row schema

Every result row has these keys:

| Field | Type | Meaning |
|---|---|---|
| `symbol` | str | Input symbol, e.g. `"NYSE:PWR"` |
| `spot` | float | Underlying spot price at scan time |
| `beta` | float | Per-name M2 beta used |
| `mu` | float | Implied drift = `beta × m2` |
| `exp` | int | Expiration YYYYMMDD |
| `dte` | int | Days to expiration |
| `strike` | float | Strike price |
| `bid`, `ask`, `mid` | float | Market quote |
| `iv` | float | Implied vol (decimal, e.g. 0.43 = 43%) |
| `delta` | float | Delta |
| `bs_fair` | float | Standard Black-Scholes fair value (drift = r) |
| `debase_fair` | float | BS fair value under debasement drift (drift = μ) |
| `edge_usd` | float | `debase_fair − mid`, $ per contract |
| `edge_pct` | float | `(debase_fair / mid − 1) × 100`, % |
| `be_g_pct` | float | Annualized breakeven growth rate % |
| `score` | float | Composite: `edge_pct × √T × IV_penalty` |

Universe-scan rows additionally carry `ticker` (short name) and `theme`.

### Low-level pricing primitives

```python
from pricing import bs_call, debase_fair_value, breakeven_annualized_growth

bs_call(S=100, K=100, T=1.0, r=0.045, sigma=0.30, q=0.0)
debase_fair_value(S=100, K=100, T=1.0, r=0.045, sigma=0.30, mu=0.105)
breakeven_annualized_growth(S=100, K=105, T=1.0, premium=8.5)
```

## Per-name beta reference

Rough anchors for populating the `beta` input:

| Asset class | β to M2 |
|---|---|
| Gold | ~1.0 |
| Insurance compounders (TRV, AJG, BRO) | 1.0–1.3 |
| Consumer moats (COST, CTAS, WM) | 1.2–1.3 |
| Toll-road financials (V, MA, SPGI, MSCI) | 1.5 |
| Industrial compounders (ETN, PH, ITW, ROP) | 1.5–1.8 |
| Grid infra (PWR, GEV, MTZ) | 1.5–1.8 |
| Megacap tech (AMZN, ORCL) | 1.8–2.0 |
| Semis / semi-cap (SMH, ASML, WDC) | 2.0 |
| Asset managers (BLK, KKR, BX) | 2.0–2.5 |
| Photonics / AI networking (CRDO, ALAB) | 2.5 |
| Crypto-beta equities (HOOD, COIN) | 3.0–3.5 |
| BTC proxies (MSTR) | 4.0+ |

Edit `UNIVERSE` in the scanners freely — the β assumption dominates the result.

## Dependencies

- `~/Documents/projects/darkfield/tv_options.py` — TradingView options client.
- Python: `scipy`, `yfinance`, `pandas`, `numpy`, `textual`, `requests`.

## Math reference

- BS call with dividend yield: `C = S·e^(-qT)·N(d₁) − K·e^(-rT)·N(d₂)`
- Debasement fair value: set `q = r − μ`
- Edge: `debase_fair − mid`
- Edge %: `(debase_fair − mid) / mid × 100`
- Breakeven growth: `g = ((K + premium) / S)^(1/T) − 1`
- Composite score: `edge_pct × √T × iv_penalty` where `iv_penalty = max(0.3, 1 − max(0, IV − 0.8) × 2)`
