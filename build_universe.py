"""Build a comprehensive universe programmatically from S&P index constituents.

Scrapes S&P 500 and S&P MidCap 400 from Wikipedia (~900 names), assigns
per-sector beta-to-M2 estimates, and resolves TradingView exchange prefixes.

Usage:
    # As a library
    from build_universe import build_sp900
    universe = build_sp900()  # dict[str, tuple[str, float, str]]

    # As a CLI (builds + caches to sp900_cache.json)
    python3 build_universe.py
    python3 build_universe.py --refresh   # force re-scrape

    # Use with scanner
    python3 scan_smooth.py --sp1500

Cache is stored as sp900_cache.json. The scrape hits Wikipedia once;
subsequent runs load from cache unless --refresh is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd
import requests
import yfinance as yf

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
}


def _read_wiki_table(url: str) -> list[pd.DataFrame]:
    """Read HTML tables from Wikipedia with proper User-Agent."""
    import io
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))

CACHE_FILE = os.path.join(os.path.dirname(__file__), "sp900_cache.json")

# Beta-to-M2 by GICS sector. These are broad defaults; the hand-curated
# universe_large.py has per-name overrides for names we know well.
SECTOR_BETA: dict[str, float] = {
    "Consumer Staples":         1.2,
    "Utilities":                1.2,
    "Real Estate":              1.3,
    "Health Care":              1.5,
    "Consumer Discretionary":   1.5,
    "Industrials":              1.5,
    "Financials":               1.5,
    "Communication Services":   1.5,
    "Materials":                1.8,
    "Energy":                   1.8,
    "Information Technology":   2.0,
}
DEFAULT_BETA = 1.5

# TradingView exchange prefix mapping from yfinance exchange codes
EXCHANGE_MAP: dict[str, str] = {
    "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "NMQ": "NASDAQ",
    "NYQ": "NYSE", "NYS": "NYSE",
    "PCX": "AMEX", "ASE": "AMEX", "BTS": "AMEX",
}


def scrape_sp500() -> pd.DataFrame:
    """Scrape S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _read_wiki_table(url)
    df = tables[0]
    df = df.rename(columns={"Symbol": "ticker", "GICS Sector": "sector",
                             "GICS Sub-Industry": "sub_industry",
                             "Security": "name"})
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # BRK.B → BRK-B
    df["index"] = "SP500"
    return df[["ticker", "name", "sector", "sub_industry", "index"]]


def scrape_sp400() -> pd.DataFrame:
    """Scrape S&P MidCap 400 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    tables = _read_wiki_table(url)
    df = tables[0]
    # Column names vary; normalize
    cols = df.columns.tolist()
    rename = {}
    for c in cols:
        cl = c.lower()
        if "symbol" in cl or "ticker" in cl:
            rename[c] = "ticker"
        elif "company" in cl or "security" in cl or "name" in cl:
            rename[c] = "name"
        elif "sector" in cl and "sub" not in cl:
            rename[c] = "sector"
        elif "sub" in cl and "industry" in cl:
            rename[c] = "sub_industry"
    df = df.rename(columns=rename)
    if "ticker" not in df.columns:
        # fallback: first column is usually ticker
        df = df.rename(columns={df.columns[0]: "ticker", df.columns[1]: "name"})
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    if "sub_industry" not in df.columns:
        df["sub_industry"] = ""
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df["index"] = "SP400"
    return df[["ticker", "name", "sector", "sub_industry", "index"]]


def resolve_exchange(ticker: str) -> str | None:
    """Resolve TradingView exchange prefix via yfinance info."""
    try:
        info = yf.Ticker(ticker).info
        exch = info.get("exchange", "")
        prefix = EXCHANGE_MAP.get(exch)
        if prefix:
            return f"{prefix}:{ticker.replace('-', '.')}"
        # Fallback: try common prefixes
        return None
    except Exception:
        return None


def build_sp900(refresh: bool = False, resolve_exchanges: bool = False,
                verbose: bool = True) -> dict[str, tuple[str, float, str]]:
    """Build universe dict from S&P 500 + S&P 400.

    Returns dict: ticker → (tv_symbol, beta, sector_tag)

    If resolve_exchanges is False (default for speed), uses a heuristic:
    most large/mid caps are NYSE or NASDAQ. The scanner will try both
    when pulling chains.
    """
    if not refresh and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cached = json.load(f)
        if verbose:
            print(f"Loaded {len(cached)} tickers from cache ({CACHE_FILE})")
        # Convert list values back to tuples
        return {k: tuple(v) for k, v in cached.items()}

    if verbose:
        print("Scraping S&P 500 from Wikipedia...")
    sp500 = scrape_sp500()
    if verbose:
        print(f"  got {len(sp500)} tickers")

    if verbose:
        print("Scraping S&P MidCap 400 from Wikipedia...")
    try:
        sp400 = scrape_sp400()
        if verbose:
            print(f"  got {len(sp400)} tickers")
    except Exception as e:
        if verbose:
            print(f"  S&P 400 scrape failed: {e}; continuing with S&P 500 only")
        sp400 = pd.DataFrame(columns=sp500.columns)

    combined = pd.concat([sp500, sp400], ignore_index=True)
    combined = combined.drop_duplicates(subset="ticker", keep="first")
    if verbose:
        print(f"Combined: {len(combined)} unique tickers")

    universe: dict[str, tuple[str, float, str]] = {}
    skipped = []

    for _, row in combined.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker or len(ticker) > 6:
            continue

        sector = str(row.get("sector", "Unknown"))
        sub = str(row.get("sub_industry", ""))
        beta = SECTOR_BETA.get(sector, DEFAULT_BETA)

        # Sector-specific beta refinements
        sub_lower = sub.lower()
        if "insurance" in sub_lower:
            beta = 1.3
        elif "bank" in sub_lower:
            beta = 1.3
        elif "asset management" in sub_lower or "capital markets" in sub_lower:
            beta = 2.0
        elif "semiconductor" in sub_lower:
            beta = 2.2
        elif "gold" in sub_lower or "precious" in sub_lower:
            beta = 2.0
        elif "oil" in sub_lower or "gas" in sub_lower or "petroleum" in sub_lower:
            beta = 1.8
        elif "utility" in sub_lower or "electric" in sub_lower:
            beta = 1.3
        elif "reit" in sub_lower or "real estate" in sub_lower:
            beta = 1.3
        elif "data processing" in sub_lower or "financial exchanges" in sub_lower:
            beta = 1.5
        elif "pharma" in sub_lower or "biotech" in sub_lower:
            beta = 1.5

        # Theme tag from sector
        tag = sector.lower().replace(" ", "_")
        if sub:
            # Use sub-industry for more specific tag
            tag = sub.lower().replace(" ", "_").replace("/", "_")[:30]

        # TV symbol: heuristic assignment (resolver is too slow for 900 names)
        # Most S&P names are NYSE or NASDAQ. Use ticker characteristics:
        tv_sym = f"NYSE:{ticker.replace('-', '.')}"
        # Known NASDAQ patterns
        if len(ticker) >= 4 or ticker in {"META", "GOOG", "GOOGL", "AMZN",
                "AAPL", "MSFT", "NVDA", "AVGO", "COST", "NFLX", "ADBE",
                "CSCO", "INTC", "QCOM", "AMGN", "GILD", "ISRG", "REGN",
                "VRTX", "IDXX", "MCHP", "LRCX", "KLAC", "ASML", "SNPS",
                "CDNS", "NXPI", "MRVL", "SWKS", "CRWD", "PANW", "DDOG",
                "ZS", "FTNT", "WDAY", "VEEV", "SNOW", "PLTR", "HUBS",
                "MNDY", "TEAM", "INTU", "ADSK", "CPRT", "ORLY", "ROST",
                "SBUX", "ODFL", "SAIA", "FAST", "PAYX", "CTAS", "CME",
                "TMUS", "CHTR", "EA", "TTWO", "LULU", "ON", "TXN",
                "ADI", "PCAR", "PDD", "BKNG", "ABNB", "EXPE", "MAR",
                "EBAY", "PYPL", "ADP", "VRSK"}:
            tv_sym = f"NASDAQ:{ticker.replace('-', '.')}"

        universe[ticker.replace("-", ".")] = (tv_sym, beta, tag)

    if verbose:
        print(f"Built universe: {len(universe)} names")
        sectors = {}
        for _, (_, b, t) in universe.items():
            s = t.split("_")[0] if "_" in t else t
            sectors[s] = sectors.get(s, 0) + 1
        print("Sector distribution:")
        for s, count in sorted(sectors.items(), key=lambda x: -x[1])[:15]:
            print(f"  {s}: {count}")

    # Cache
    with open(CACHE_FILE, "w") as f:
        json.dump({k: list(v) for k, v in universe.items()}, f, indent=2)
    if verbose:
        print(f"Cached to {CACHE_FILE}")

    return universe


def main():
    p = argparse.ArgumentParser(description="Build S&P 900 universe for debasement scanning")
    p.add_argument("--refresh", action="store_true", help="Force re-scrape from Wikipedia")
    args = p.parse_args()
    universe = build_sp900(refresh=args.refresh)
    print(f"\nTotal: {len(universe)} names ready for scanning")
    print(f"Run: python3 scan_smooth.py --sp1500")


if __name__ == "__main__":
    main()
