"""The speculation index — how much of the price is momentum.

Three terms, each on 0-1, combined as a **geometric** mean and reported on
0-100::

    vol      realized volatility against a reference
    volume   on-chain USD transaction volume, recent against trailing-year
    band     how far above the liquidity-implied band the price sits

Why geometric and not an average
--------------------------------

Because speculation is a conjunction, and an average would let any one term
carry the index alone. Volatility without volume and without an extended price
is an ordinary bad week. Volume without volatility is settlement. A price above
the implied band in a quiet, thin market is the money supply having moved, not
the crowd. Each of the three is unremarkable by itself and the state the index
names only exists when all three are present — which is precisely the property a
geometric mean has and an arithmetic one does not.

The consequence is that the index is **zero for long stretches**, because any
term at zero zeroes the product. That is the correct answer and not a defect to
be smoothed away: bitcoin is in the state this index describes during a handful
of windows in its history, and an index that read 35 through 2019 would be
measuring the fact that bitcoin exists.

Degradation
-----------

A term whose input is absent is dropped and the geometric mean is taken over
what remains, with the count published. This is not the same as scoring it zero:
"we could not read the on-chain volume" and "on-chain volume is flat" are
different statements, and only the second is evidence about speculation. The
engine reports the reduced term count through a signal and a confidence penalty
rather than quietly publishing a two-term index under a three-term name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("findynamics.engines.crypto.speculation")

#: Names of the three terms, in the order they are reported.
TERMS = ("vol", "volume", "band")


@dataclass(frozen=True)
class SpeculationRules:
    """Scaling for each term, from ``config/engines/crypto.yaml``."""

    #: Annualized realized volatility (percent) that maps the vol term to 1.0.
    #: 120% is roughly bitcoin's 2017 and 2021 peaks — calibrated on the asset,
    #: because a scale shared with equities would peg this term at 1.0 forever.
    vol_reference_pct: float = 120.0
    #: Trailing window for the volume baseline, in days.
    volume_baseline_days: int = 365
    #: Recent window the baseline is compared against, in days.
    volume_recent_days: int = 90
    #: Excess of recent-over-baseline volume that maps the volume term to 1.0.
    #: 0.5 = on-chain dollar volume running 50% above its trailing year.
    volume_span: float = 0.5
    #: Band half-widths above the implied level that map the band term to 1.0.
    band_span: float = 1.0

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> SpeculationRules:
        raw = params.get("speculation") or {}
        if not isinstance(raw, dict):
            raise ValueError("engines/crypto.yaml: 'speculation' must be a mapping")
        defaults = cls()
        return cls(
            vol_reference_pct=float(raw.get("vol_reference_pct", defaults.vol_reference_pct)),
            volume_baseline_days=int(
                raw.get("volume_baseline_days", defaults.volume_baseline_days)
            ),
            volume_recent_days=int(raw.get("volume_recent_days", defaults.volume_recent_days)),
            volume_span=float(raw.get("volume_span", defaults.volume_span)),
            band_span=float(raw.get("band_span", defaults.band_span)),
        )


@dataclass(frozen=True)
class SpeculationResult:
    """The index and the terms it was built from."""

    #: 0-100 per date. NaN where no term could be computed.
    index: pd.Series
    #: Term name -> its 0-1 series. Absent terms are simply not keyed.
    terms: dict[str, pd.Series]

    @property
    def empty(self) -> bool:
        return self.index.empty or not self.index.notna().any()

    @property
    def term_count(self) -> int:
        return len(self.terms)

    @property
    def missing_terms(self) -> tuple[str, ...]:
        return tuple(name for name in TERMS if name not in self.terms)

    def latest(self) -> float | None:
        if self.index.empty:
            return None
        clean = self.index.dropna()
        return None if clean.empty else float(clean.iloc[-1])

    def latest_terms(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, series in self.terms.items():
            clean = series.dropna()
            if not clean.empty:
                out[f"speculation_{name}"] = round(float(clean.iloc[-1]), 6)
        return out


def volume_trend(volume: pd.Series, rules: SpeculationRules) -> pd.Series:
    """Recent on-chain dollar volume against its trailing year, as a ratio.

    Both windows are trailing and right-closed. A ratio rather than a z-score
    because the level of on-chain volume has grown by four orders of magnitude
    since 2010 and an expanding z-score of it would be measuring that growth
    rather than this year's activity.
    """
    clean = volume.dropna()
    clean = clean[clean > 0]
    if clean.empty:
        return pd.Series(dtype=float)
    recent = clean.rolling(
        rules.volume_recent_days, min_periods=rules.volume_recent_days // 2
    ).mean()
    baseline = clean.rolling(
        rules.volume_baseline_days, min_periods=rules.volume_baseline_days // 2
    ).mean()
    return recent / baseline.replace(0.0, np.nan)


def compute(
    realized_vol: pd.Series,
    volume: pd.Series | None,
    band_excess: pd.Series | None,
    rules: SpeculationRules,
    *,
    index: pd.Index | None = None,
) -> SpeculationResult:
    """The 0-100 index over a daily date index.

    Each input is optional except the volatility, without which there is no
    index at all: a market with no measurable volatility is not one anybody is
    speculating in.
    """
    spine = index if index is not None else realized_vol.index
    terms: dict[str, pd.Series] = {}

    vol = realized_vol.reindex(spine)
    if vol.notna().any():
        terms["vol"] = (vol / max(rules.vol_reference_pct, 1e-9)).clip(0.0, 1.0)

    if volume is not None and volume.notna().any():
        ratio = volume_trend(volume, rules).reindex(spine)
        if ratio.notna().any():
            # (ratio - 1) / span: flat volume scores 0, `span` above trend scores 1.
            terms["volume"] = ((ratio - 1.0) / max(rules.volume_span, 1e-9)).clip(0.0, 1.0)

    if band_excess is not None and band_excess.notna().any():
        excess = band_excess.reindex(spine)
        if excess.notna().any():
            terms["band"] = (excess / max(rules.band_span, 1e-9)).clip(0.0, 1.0)

    if not terms:
        return SpeculationResult(index=pd.Series(dtype=float, index=spine), terms={})

    stacked = pd.concat(terms.values(), axis=1)
    # Geometric mean via logs, so a term at exactly 0 takes the product to 0
    # rather than to a NaN. Dates where any present term is missing stay NaN:
    # a partial product is a different index from the one being published.
    with np.errstate(divide="ignore"):
        logged = np.log(stacked.clip(lower=0.0))
    combined = np.exp(logged.mean(axis=1))
    combined = combined.where(stacked.notna().all(axis=1))
    # log(0) is -inf, so exp(mean) is 0 — which is the intended answer, but the
    # intermediate can surface as -inf on an all-zero row.
    combined = combined.replace([np.inf, -np.inf], 0.0)

    missing = [name for name in TERMS if name not in terms]
    if missing:
        log.info(
            "crypto speculation index: running on %d of 3 terms (missing %s)",
            len(terms),
            ", ".join(missing),
        )

    return SpeculationResult(index=(combined * 100.0).clip(0.0, 100.0), terms=terms)


__all__ = [
    "TERMS",
    "SpeculationResult",
    "SpeculationRules",
    "compute",
    "volume_trend",
]
