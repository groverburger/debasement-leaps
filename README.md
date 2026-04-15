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

Edge is shown in both dollars (`debase_fair − mid`) and percent.
Breakeven growth rate (the annualized return the underlying must achieve
for the long call to break even at expiry) is shown as a sanity floor.

See [THEORY.md](./THEORY.md) for the deeper argument: why dealers' standard
delta-hedging framework systematically under-captures drift on LEAPS of
high-β-to-M2 names, and why that's the structural source of the edge these
scanners surface.

## Files

| File | Purpose |
|---|---|
| `THEORY.md` | Why this edge exists — dealer hedging friction under drift asymmetry. |
| `pricing.py` | Shared BS + debasement fair-value + breakeven math. |
| `debasement_leaps.py` | **Textual TUI** — interactive per-ticker chain inspector. Tweak M2 growth, beta, risk-free rate live; press `F5` to refetch. |
| `scan_broad.py` | **Broad universe scanner** — runs debasement analysis across a curated list of real-capex, crypto-beta, commodity, and semi names. Output sorted by composite score. |
| `scan_smooth.py` | **Smooth-trend scanner** — pre-filters a compounder-heavy universe via yfinance for multi-year smooth uptrends (high % above 200MA, shallow DD, low ATR, near 52w high), then pulls LEAPS chains only for survivors. Biased toward insurance / toll-road / boring-industrial compounders. |

## Run

```bash
pip install -r requirements.txt

# TUI
python3 debasement_leaps.py

# Broad scan (crypto beta, semis, infra, etc.)
python3 scan_broad.py

# Smooth-trend scan (TRV, COST, PH, and similar) — recommended starting point
python3 scan_smooth.py
```

## Per-name beta

Both scanners carry per-name beta estimates in their `UNIVERSE` dict. These
reflect each asset's historical sensitivity to monetary expansion. Rough
anchors:

| Asset class | β to M2 |
|---|---|
| Gold | ~1.0 |
| Insurance compounders (TRV, AJG, BRO) | 1.0–1.3 |
| Consumer moats (COST, CTAS, WM) | 1.2–1.3 |
| Toll-road financials (V, MA, SPGI, MSCI) | 1.5 |
| Industrial compounders (ETN, PH, ITW, ROP) | 1.5–1.8 |
| Megacap tech (AMZN, ORCL) | 1.8–2.0 |
| Semis / semi-cap (SMH, ASML, WDC) | 2.0 |
| Asset managers (BLK, KKR, BX) | 2.0–2.5 |
| Photonics / AI networking (CRDO, ALAB) | 2.5 |
| Crypto-beta equities (HOOD, COIN) | 3.0–3.5 |
| BTC proxies (MSTR) | 4.0+ |

Edit freely — the assumption dominates the result.

## Dependencies

- `~/Documents/projects/darkfield/tv_options.py` — TradingView options client.
- `yfinance` for price history (smooth-trend pre-filter).

## Workflow

1. Start with `scan_smooth.py` — it surfaces high-quality low-IV LEAPS setups.
2. For anything interesting, open the TUI and sweep across strikes/expirations
   with your preferred `M2 growth` and per-name `beta`.
3. For higher-beta / theme-driven ideas (crypto proxies, semis), use
   `scan_broad.py` instead — it doesn't pre-filter on smoothness.

## Math reference

- BS call with dividend yield: `C = S·e^(-qT)·N(d₁) − K·e^(-rT)·N(d₂)`
- Debasement fair value: set `q = r − μ`
- Edge: `debase_fair − mid`
- Edge %: `(debase_fair − mid) / mid`
- Breakeven growth: `g = ((K + premium) / S)^(1/T) − 1`
