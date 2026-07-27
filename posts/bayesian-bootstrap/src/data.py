"""Download, clean and cache the two return series, and load MNIST.

Provenance and their caveats, both of which appear in the post:

S&P 500 — the Shiller monthly series.  Each monthly level is the *average of
that month's daily closes*, not a month-end close, so monthly volatility is
slightly understated.

IBM — daily closes from the plotly five-year S&P constituents file.  These are
*unadjusted* closes, hence price returns: dividends are excluded, which
understates total return by roughly 3-4%/yr over this window.

If a download fails this module raises with the URL.  It never falls back to
synthetic data — a post that silently invents its own returns is worse than one
that does not build.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray

Floats = NDArray[np.float64]

SP500_URL: Final[str] = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
)
STOCKS_URL: Final[str] = (
    "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"
)

_HERE: Final[Path] = Path(__file__).resolve().parent.parent
RAW: Final[Path] = _HERE / "data" / "raw"
CACHE: Final[Path] = _HERE / "data" / "cache"

# The head-to-head window: same months for both series, so the comparison is
# apples-to-apples.  n = 59 each.
HEAD_START: Final[str] = "2013-03"
HEAD_END: Final[str] = "2018-01"
# The long window, used only to show the Efron/Bayes gap vanishing.
LONG_START: Final[str] = "1990-02"
# The short window, used only to show the gap biting.
SHORT_N: Final[int] = 12


class DownloadError(RuntimeError):
    """Raised when a required data file cannot be retrieved."""


def _fetch(url: str, dest: Path, timeout: int = 180) -> Path:
    """Download ``url`` to ``dest`` unless it is already there.

    Args:
        url: Source URL.
        dest: Destination path.
        timeout: Seconds before giving up.

    Returns:
        The destination path.

    Raises:
        DownloadError: On any network or HTTP failure, naming the URL.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DownloadError(
            f"Could not download {url}\n"
            f"This build needs network access; there is deliberately no "
            f"synthetic fallback.\nUnderlying error: {exc}"
        ) from exc
    dest.write_bytes(response.content)
    return dest


def _monthly_returns_sp500() -> pd.Series:
    """Monthly percentage changes in the Shiller S&P 500 level, indexed by period."""
    path = _fetch(SP500_URL, RAW / "sp500_shiller.csv")
    frame = pd.read_csv(path, parse_dates=["Date"])
    level = frame.set_index("Date")["SP500"].astype(float).sort_index()
    returns = level.pct_change(fill_method=None).dropna() * 100.0
    returns.index = returns.index.to_period("M")
    returns.name = "sp500"
    return returns


def _monthly_returns_ibm() -> pd.Series:
    """Monthly percentage price changes for IBM, indexed by period.

    Daily closes are resampled to month-end and differenced.  The final month of
    the source file (2018-02) is incomplete — the data stop on the 7th — so it
    is dropped rather than reported as a partial month.
    """
    path = _fetch(STOCKS_URL, RAW / "all_stocks_5yr.csv")
    frame = pd.read_csv(path, parse_dates=["date"], usecols=["date", "close", "Name"])
    ibm = frame.loc[frame["Name"] == "IBM"].set_index("date")["close"].sort_index()
    if ibm.empty:
        raise DownloadError(f"No rows with Name == 'IBM' in {path} (from {STOCKS_URL})")
    month_end = ibm.resample("ME").last()
    # Drop the incomplete trailing month.
    last_day = ibm.index.max()
    if last_day.day < 20:
        month_end = month_end.iloc[:-1]
    returns = month_end.pct_change(fill_method=None).dropna() * 100.0
    returns.index = returns.index.to_period("M")
    returns.name = "ibm"
    return returns


@dataclass(frozen=True)
class Returns:
    """The cleaned return series and the three analysis windows.

    Attributes:
        sp500: Full monthly S&P 500 percentage changes.
        ibm: Full monthly IBM percentage price changes.
        retrieved: ISO date the raw files were downloaded.
    """

    sp500: pd.Series
    ibm: pd.Series
    retrieved: str

    def head_to_head(self) -> tuple[Floats, Floats]:
        """The n = 59 window, 2013-03 to 2018-01, for both series.

        Returns:
            ``(sp500, ibm)`` as plain float arrays in date order.
        """
        lo, hi = pd.Period(HEAD_START), pd.Period(HEAD_END)
        sp = self.sp500[(self.sp500.index >= lo) & (self.sp500.index <= hi)]
        ib = self.ibm[(self.ibm.index >= lo) & (self.ibm.index <= hi)]
        if len(sp) != len(ib):
            raise ValueError(
                f"head-to-head window is not aligned: "
                f"S&P has {len(sp)} months, IBM has {len(ib)}"
            )
        return sp.to_numpy(np.float64), ib.to_numpy(np.float64)

    def long_window(self) -> Floats:
        """S&P 500 from 1990-02 to the end of the series."""
        series = self.sp500[self.sp500.index >= pd.Period(LONG_START)]
        return series.to_numpy(np.float64)

    def short_window(self) -> Floats:
        """The most recent ``SHORT_N`` months of the S&P 500."""
        return self.sp500.to_numpy(np.float64)[-SHORT_N:]

    def window_labels(self) -> dict[str, str]:
        """Human-readable date ranges for the three windows, for tables."""
        long_series = self.sp500[self.sp500.index >= pd.Period(LONG_START)]
        short_index = self.sp500.index[-SHORT_N:]
        fmt = "%b %Y"
        return {
            "short": f"{short_index[0].strftime(fmt)} - "
            f"{short_index[-1].strftime(fmt)}",
            "head": f"{pd.Period(HEAD_START).strftime(fmt)} - "
            f"{pd.Period(HEAD_END).strftime(fmt)}",
            "long": f"{long_series.index[0].strftime(fmt)} - "
            f"{long_series.index[-1].strftime(fmt)}",
        }


def load_returns(refresh: bool = False) -> Returns:
    """Load both return series, using the parquet cache when it exists.

    Args:
        refresh: Re-download and rebuild the cache even if it is present.

    Returns:
        A :class:`Returns` bundle.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    parquet = CACHE / "returns.parquet"
    meta_path = CACHE / "returns_meta.json"

    if parquet.exists() and meta_path.exists() and not refresh:
        frame = pd.read_parquet(parquet)
        meta = json.loads(meta_path.read_text())
        index = pd.PeriodIndex(frame["month"].astype(str), freq="M")
        sp = pd.Series(frame["sp500"].to_numpy(), index=index, name="sp500").dropna()
        ib = pd.Series(frame["ibm"].to_numpy(), index=index, name="ibm").dropna()
        return Returns(sp500=sp, ibm=ib, retrieved=meta["retrieved"])

    sp = _monthly_returns_sp500()
    ib = _monthly_returns_ibm()
    retrieved = _dt.date.today().isoformat()

    merged = pd.DataFrame({"sp500": sp}).join(pd.DataFrame({"ibm": ib}), how="outer")
    merged.insert(0, "month", merged.index.astype(str))
    merged.reset_index(drop=True).to_parquet(parquet, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "retrieved": retrieved,
                "sp500_url": SP500_URL,
                "stocks_url": STOCKS_URL,
                "sp500_months": int(sp.size),
                "ibm_months": int(ib.size),
            },
            indent=2,
        )
    )
    return Returns(sp500=sp, ibm=ib, retrieved=retrieved)


def annualised_vol(returns: Floats) -> float:
    """Annualised volatility in percent from monthly percentage returns.

    Args:
        returns: Monthly percentage returns.

    Returns:
        ``sd * sqrt(12)`` in percent.  The Shiller series over 1990-present
        should give about 12.3%; the post asserts this as a download check.
    """
    return float(np.std(returns, ddof=1) * np.sqrt(12.0))


def describe(name: str, returns: Floats, index: pd.PeriodIndex) -> dict[str, object]:
    """One row of the sample-description table.

    Args:
        name: Series label.
        returns: Monthly percentage returns.
        index: Matching period index, used to date the best and worst months.

    Returns:
        A mapping of column name to value.
    """
    best, worst = int(np.argmax(returns)), int(np.argmin(returns))
    return {
        "series": name,
        "n": returns.size,
        "mean %": returns.mean(),
        "sd %": returns.std(ddof=1),
        "median %": float(np.median(returns)),
        "best": f"{returns[best]:+.3f} ({index[best].strftime('%b %Y')})",
        "worst": f"{returns[worst]:+.3f} ({index[worst].strftime('%b %Y')})",
        "months < -5%": int((returns <= -5.0).sum()),
    }
