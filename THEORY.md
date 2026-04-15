# Why debasement LEAPS have a structural edge

This note captures the theoretical justification for why the `Diff%` / `EDGE%`
column in our scans isn't just an academic artifact — it's estimating real
dollar edge a LEAPS buyer has over a dealer pricing under standard
Black-Scholes assumptions.

## The question

Options sellers typically make money when realized volatility is lower than
implied volatility (σ_realized < σ_IV). The dealer sells at σ_IV, delta-hedges,
and pockets the gap.

But BS assumes risk-neutral drift = r. If the underlying persistently drifts
at μ > r (debasement regime on a high-β asset), the delta-hedging assumption
is violated. How does this square with the standard "dealers win on IV > RV"
framing?

## How dealers actually make money on LEAPS

On long-dated options, dealers aren't primarily being paid for
realized-vs-implied vol. They're being paid for:

- **Vega risk** — if IV moves 5 points against them, that's $X per contract.
- **Skew and term-structure shifts** — long-dated surfaces move with macro
  regime.
- **Pin/gamma risk near expiry** — but that's years away, heavily discounted.
- **Capital cost** of carrying a long-dated inventory position.

So LEAPS premium is less "vol risk premium" and more
**"inventory + vega + funding" premium**. Short-dated books collect vol
premium; long-dated books collect carry and vega premium.

## Where the debasement insight bites

The dealer's hedging model assumes risk-neutral drift = r. Under that measure,
`E[S_T] = S_0 · e^(rT)`. Delta hedging is calibrated to that expectation.

If realized drift is μ > r *persistently*, then even a perfectly-hedged dealer
**systematically under-hedges the upside of calls they sold**. They end up
buying shares too slowly as the stock rallies, eating the convexity cost on
every path.

In BS theory this is invisible because BS is internally consistent under its
own measure. In reality, for long-duration options on high-drift assets, the
measure doesn't match the world. The drift escapes through delta/gamma
mismatch that vega-hedging can't capture.

## How it squares

- Short-dated books: dealers sell at σ_IV, hope σ_realized is lower, collect
  vol premium on average. This works because realized drift over a few weeks
  is dominated by noise, not by μ-vs-r asymmetry.
- Long-dated books on high-β-to-M2 names: the realized distribution is skewed
  upward vs. lognormal-at-r. Sellers don't get their usual edge because the
  upside tail is fatter than BS assumes. Their edge is mostly vega and carry,
  not realized-vol-vs-implied-vol.
- The `Diff%` / `EDGE%` column quantifies the gap between the dealer's
  measure (drift = r) and the thesis (drift = μ). That gap is what the
  LEAPS buyer is trying to capture.

## The uncomfortable implication

If the debasement thesis is right, dealers who systematically sell LEAPS on
high-β-to-M2 names over a multi-year debasement regime **lose money
systematically** — not because their vol forecast was wrong, but because
they were pricing in a drift assumption that doesn't match reality.

Historical support: long-dated OTM calls on trending indices (SPX, QQQ)
have tended to be underpriced relative to realized terminal distributions
during secular bull markets (1980s–2000s, 2010s, post-2020). Buying LEAPS
on secular winners has been an edge, not a loss.

The dealer's protection is partly statistical (thousands of positions,
diversification across betas) and partly structural (they hedge vega, not
pure delta, and roll aggressively). In a persistent debasement regime,
that protection is weaker than in a mean-reverting one.

## Practical takeaway

The edge on debasement LEAPS comes from three stacked effects:

1. **Drift asymmetry** (μ > r thesis)
2. **Delta-hedging friction** on the dealer's side (they under-hedge
   upside on high-drift paths)
3. **Fat right tail** from monetary expansion that BS lognormal misses

This is why debasement LEAPS on long-term-trending names (PWR, TRV, COST,
GDX, HOOD) are structurally attractive if the macro thesis is correct.
The buyer isn't just buying options — they're taking the other side of
a dealer whose pricing model doesn't include the thesis drift.

## The risk

If the regime shifts — deflation, hard landing, Fed aggressively tightens
without debasing — the drift flips, the edge disappears, **and** you eat
the vol premium you paid on entry. Debasement LEAPS win in the regime
they're named for; they lose in the regime that invalidates them.

This is why:
- Per-name beta matters (wrong beta ≠ wrong regime, but wrong beta with
  a correct regime still loses)
- Low IV matters (smaller vol premium to eat if the thesis is wrong)
- Diversification across themes matters (grid infra ≠ crypto beta ≠
  insurance float — each fails differently under different regime shifts)
