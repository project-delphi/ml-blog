"""Fetch four years of daily closes for META, AAPL, AMZN, NFLX, GOOGL, NVDA, AZN.

Writes posts/why-so-many-matrix-factorizations/prices.csv. Run this script
yourself when the window needs refreshing; the post reads the committed CSV
and does not hit the network at render.

Nasdaq.com historical quotes are the collector (exchange prints). Close/Last
on that endpoint is split-adjusted; the 10-for-1 NVDA split of 2024-06-10 is
already in the series (June 2024 prints near $121, not $1,210).
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

TICKERS = ["META", "AAPL", "AMZN", "NFLX", "GOOGL", "NVDA", "AZN"]
START = "2022-09-01"
END = "2026-09-01"
OUT = Path(__file__).resolve().parent.parent / "prices.csv"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def fetch_nasdaq(ticker: str) -> dict[str, float]:
    url = (
        f"https://api.nasdaq.com/api/quote/{ticker}/historical"
        f"?assetclass=stocks&fromdate={START}&todate={END}&limit=9999"
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode())
    rows = payload["data"]["tradesTable"]["rows"]
    out = {}
    for row in rows:
        day = datetime.strptime(row["date"], "%m/%d/%Y").date().isoformat()
        raw = row["close"].replace("$", "").replace(",", "")
        out[day] = float(raw)
    if len(out) < 200:
        raise RuntimeError(f"{ticker}: only {len(out)} rows")
    return out


def main() -> int:
    series = {t: fetch_nasdaq(t) for t in TICKERS}
    dates = sorted(set.intersection(*(set(s) for s in series.values())))
    dates = [d for d in dates if START <= d <= END]
    if not dates:
        raise RuntimeError("no common trading dates")
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", *TICKERS])
        for d in dates:
            w.writerow([d, *[f"{series[t][d]:.6f}" for t in TICKERS]])
    print(f"wrote {OUT} ({len(dates)} rows, {dates[0]} .. {dates[-1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
