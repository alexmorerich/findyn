"""FinEquity fixtures: the committed price snapshot and world builders.

The snapshot is real, for the same reason the money and rates suites use real
ones — the interesting assertions here are about *which series wins a role* and
*what a transform does to actual index history*, and neither can be validated
against a series invented until the test passed.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from findynamics.core.artifacts import ArtifactStore
from findynamics.core.config import SeriesConfig
from findynamics.core.contracts.state import WorldState
from findynamics.data.accessor import PandasPITAccessor
from findynamics.engines.equity.engine import EquityEngine
from tests.conftest import FIXTURE_DIR

#: Role -> series id, matching config/series.yaml engines.equity.series.
PRIMARY = "FRED:SP500"
REGIME_PROXY = "FRED:NASDAQ100"
BACKFILL = "STOOQ:^SPX"
DEEP_HISTORY = "SHILLER:NOMINAL_PRICE"

#: The snapshot's newest observation. Fixed rather than "today" so the suite does
#: not change behaviour as the calendar moves.
SNAPSHOT_AS_OF = date(2026, 7, 30)


@pytest.fixture(scope="session")
def equity_observations() -> pd.DataFrame:
    """The committed equity price snapshot, as a PIT frame."""
    frame = pd.read_csv(FIXTURE_DIR / "equity_prices.csv")
    for column in ("obs_date", "release_date", "revision_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


@pytest.fixture
def equity_engine(config: SeriesConfig, artifacts: ArtifactStore) -> EquityEngine:
    """The engine on the shipped configuration, with a throwaway artifact store."""
    return EquityEngine(config, artifacts)


def world_from(
    observations: pd.DataFrame,
    as_of: date = SNAPSHOT_AS_OF,
    factors: dict | None = None,
) -> WorldState:
    """A WorldState over ``observations`` clamped to ``as_of``."""
    return WorldState(
        as_of=as_of,
        factors=factors or {},
        series=PandasPITAccessor(observations, as_of),
    )


def price_rows(
    series_id: str,
    values: dict[date, float],
    *,
    lag_days: int = 1,
) -> list[dict]:
    """Observation rows for one price series, each released ``lag_days`` later."""
    return [
        {
            "series_id": series_id,
            "obs_date": pd.Timestamp(day),
            "release_date": pd.Timestamp(day) + pd.Timedelta(days=lag_days),
            "revision_date": pd.Timestamp(day) + pd.Timedelta(days=lag_days),
            "value": float(value),
        }
        for day, value in values.items()
    ]


def synthetic_path(
    series_id: str,
    *,
    start: date = date(2010, 1, 4),
    days: int = 1500,
    drift: float = 0.0003,
    seed: int = 7,
    volatility: float = 0.01,
) -> pd.DataFrame:
    """A geometric random walk on business days — for transform-level tests.

    Synthetic where a closed form or a known property is the assertion (the FFD
    weights, the filter's causality), real where the answer depends on what
    markets actually did.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, periods=days)
    steps = rng.normal(drift, volatility, size=days)
    level = 1000.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(price_rows(series_id, dict(zip(index.date, level, strict=True))))
