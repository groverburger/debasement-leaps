"""Debasement-LEAPS TUI — price calls under a debasement drift assumption.

Thesis: market option pricing assumes ~zero real drift (risk-neutral). If you
believe M2 expands at rate m and the underlying captures it with beta b, then
the *true* expected drift is μ = m·b, not r. Re-pricing calls with that drift
(via Black-Scholes with a negative dividend yield q = r - μ) produces a
"debasement fair value" that is typically higher than market mid.

Divergence = (debase_fair - mid) / mid × 100.  Large positive divergence
= option is cheap under the debasement thesis.

Controls (sidebar):
    Ticker        TradingView-style symbol, e.g. NYSE:PWR, NASDAQ:HOOD, AMEX:LIT
    M2 growth %   Annualized monetary expansion rate you expect forward
    Beta          Asset's beta to M2 growth (gold≈1, S&P≈1.5, BTC≈4-5, tech≈2)
    Risk-free %   Discount rate used for present-valuing the strike
    Min DTE       Filter out short-dated; default 120 to focus on LEAPS

Press [F5] to refetch the chain after changing inputs.
"""
import sys
import os
import datetime
from dataclasses import dataclass

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Input, Label, DataTable, Static
from textual.binding import Binding

sys.path.insert(0, os.path.expanduser("~/Documents/projects/darkfield"))
from tv_options import TVOptions
from pricing import bs_call, debase_fair_value, breakeven_annualized_growth


# ---------------------------------------------------------------------------
# Data row
# ---------------------------------------------------------------------------

@dataclass
class Row:
    exp: int
    dte: int
    strike: float
    mid: float
    bid: float
    ask: float
    iv: float
    delta: float
    bs_fair: float
    debase_fair: float
    diff_pct: float
    breakeven_g: float


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class DebasementApp(App):
    CSS = """
    Screen { layout: horizontal; }
    #sidebar {
        width: 34;
        padding: 1 2;
        border-right: solid $primary;
    }
    #sidebar Label { margin-top: 1; color: $text-muted; }
    #sidebar Input { margin-bottom: 0; }
    #main { width: 1fr; }
    #status { height: 3; padding: 1 2; color: $warning; }
    DataTable { height: 1fr; }
    .good-divergence { color: $success; }
    """

    BINDINGS = [
        Binding("f5", "refresh", "Refresh chain"),
        Binding("ctrl+r", "refresh", "Refresh chain"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.tv = TVOptions()
        self.rows: list[Row] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            Vertical(
                Label("Ticker (exchange:symbol)"),
                Input(value="NYSE:PWR", id="ticker"),
                Label("M2 growth (%/yr)"),
                Input(value="7.0", id="m2"),
                Label("Asset beta to M2"),
                Input(value="1.5", id="beta"),
                Label("Risk-free rate (%)"),
                Input(value="4.5", id="rfr"),
                Label("Min DTE"),
                Input(value="120", id="min_dte"),
                Static("", id="status"),
                id="sidebar",
            ),
            Vertical(
                DataTable(id="chain_table", cursor_type="row", zebra_stripes=True),
                id="main",
            ),
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#chain_table", DataTable)
        table.add_columns(
            "Exp", "DTE", "Strike", "Bid", "Ask", "Mid", "IV",
            "Δ", "BS Fair", "Debase Fair", "Diff%", "BE g%/yr",
        )
        self.set_status("Press F5 to load chain.")

    # --- actions ---

    def action_refresh(self) -> None:
        self.refresh_chain()

    def set_status(self, msg: str) -> None:
        self.query_one("#status", Static).update(msg)

    def _get_float(self, widget_id: str, default: float) -> float:
        try:
            return float(self.query_one(f"#{widget_id}", Input).value)
        except Exception:
            return default

    def _get_int(self, widget_id: str, default: int) -> int:
        try:
            return int(self.query_one(f"#{widget_id}", Input).value)
        except Exception:
            return default

    def refresh_chain(self) -> None:
        symbol = self.query_one("#ticker", Input).value.strip()
        if ":" not in symbol:
            self.set_status("Symbol must be EXCHANGE:TICKER e.g. NYSE:PWR")
            return

        m2_pct = self._get_float("m2", 7.0)
        beta = self._get_float("beta", 1.5)
        rfr_pct = self._get_float("rfr", 4.5)
        min_dte = self._get_int("min_dte", 120)

        mu = (m2_pct / 100.0) * beta          # debasement drift
        r = rfr_pct / 100.0                    # risk-free rate (continuous approx)

        self.set_status(f"Fetching {symbol}…")
        try:
            spot = self.tv.get_underlying_price(symbol)
            if not spot or spot <= 0:
                self.set_status(f"No spot for {symbol} (wrong exchange prefix?)")
                return
            exps = self.tv.get_expirations(symbol)
        except Exception as e:
            self.set_status(f"Fetch failed: {e}")
            return

        today = datetime.date.today()
        today_int = int(today.strftime("%Y%m%d"))
        rows: list[Row] = []
        considered = 0

        for e in sorted(exps):
            if e <= today_int:
                continue
            try:
                exp_date = datetime.date(int(str(e)[:4]), int(str(e)[4:6]), int(str(e)[6:8]))
            except Exception:
                continue
            dte = (exp_date - today).days
            if dte < min_dte:
                continue
            try:
                chain = self.tv.get_chain(symbol, e, spot=spot)
            except Exception:
                continue
            T = dte / 365.0
            for o in chain.options:
                if o.option_type != "call":
                    continue
                if o.strike < spot * 0.6 or o.strike > spot * 1.6:
                    continue
                mid = o.mid or 0.0
                iv = o.iv or 0.0
                if mid <= 0 or iv <= 0:
                    continue
                considered += 1
                bs_fair = bs_call(spot, o.strike, T, r, iv, q=0.0)
                df_fair = debase_fair_value(spot, o.strike, T, r, iv, mu)
                diff_pct = (df_fair - mid) / mid * 100 if mid > 0 else 0.0
                be_g = breakeven_annualized_growth(spot, o.strike, T, mid) * 100
                rows.append(Row(
                    exp=e, dte=dte, strike=o.strike, mid=mid,
                    bid=o.bid or 0.0, ask=o.ask or 0.0,
                    iv=iv, delta=o.delta or 0.0,
                    bs_fair=bs_fair, debase_fair=df_fair,
                    diff_pct=diff_pct, breakeven_g=be_g,
                ))

        rows.sort(key=lambda r_: r_.diff_pct, reverse=True)
        self.rows = rows
        self._repaint_table(spot, mu, r)
        if rows:
            top = rows[0]
            self.set_status(
                f"{symbol} spot={spot:.2f}  μ={mu*100:.1f}%  r={r*100:.1f}%  "
                f"{len(rows)}/{considered} calls.  Best: K={top.strike:.0f} "
                f"DTE={top.dte}  +{top.diff_pct:.1f}%"
            )
        else:
            self.set_status(f"{symbol} spot={spot:.2f} — no eligible calls ≥ {min_dte}d DTE")

    def _repaint_table(self, spot: float, mu: float, r: float) -> None:
        table = self.query_one("#chain_table", DataTable)
        table.clear()
        if not self.rows:
            return
        # highlight top decile by divergence
        threshold = self.rows[max(0, len(self.rows) // 10 - 1)].diff_pct
        for row in self.rows:
            highlight = row.diff_pct >= threshold and row.diff_pct > 0
            strike_txt = f"{row.strike:.2f}"
            diff_txt = f"{row.diff_pct:+.1f}"
            if highlight:
                strike_txt = f"[bold $success]{strike_txt}[/]"
                diff_txt = f"[bold $success]{diff_txt}[/]"
            table.add_row(
                str(row.exp),
                str(row.dte),
                strike_txt,
                f"{row.bid:.2f}",
                f"{row.ask:.2f}",
                f"{row.mid:.2f}",
                f"{row.iv*100:.1f}",
                f"{row.delta:.2f}",
                f"{row.bs_fair:.2f}",
                f"{row.debase_fair:.2f}",
                diff_txt,
                f"{row.breakeven_g:+.1f}",
            )


if __name__ == "__main__":
    DebasementApp().run()
