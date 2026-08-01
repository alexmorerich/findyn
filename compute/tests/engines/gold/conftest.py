"""FinGold fixtures: the real driver snapshot and synthetic jump-diffusion paths.

Split by what each kind of input can prove, the same way the money suite is.

* **The real snapshot** (``gold_daily.csv``) is for the regime. Which years count
  as a crisis bid and which as a rate headwind is not a question a made-up series
  can answer — you can always invent one that crosses whichever line you wrote —
  so those assertions run against 1968-2026 of the actual LBMA fix and the actual
  drivers, walk-forward.
* **Synthetic jump diffusions** are for the detector. There the jump dates are
  *known by construction*, which is the one thing real data cannot offer: nobody
  publishes the true set of gold's jumps, so a detector tested only against
  history can be scored on plausibility and never on correctness.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.core.artifacts import ArtifactStore
from findynamics.core.config import SeriesConfig
from findynamics.core.contracts.state import WorldState
from findynamics.data.accessor import PandasPITAccessor
from findynamics.engines.gold.engine import GoldEngine
from tests.conftest import FIXTURE_DIR

#: Role -> series id, matching config/series.yaml engines.gold.series.
PRICE = "LBMA:GOLD_PM"
NOMINAL_10Y = "FRED:DGS10"
BREAKEVEN_10Y = "FRED:T10YIE"
CPI = "FRED:CPIAUCSL"
USD_INDEX = "FRED:DTWEXBGS"
USD_INDEX_LEGACY = "FRED:DTWEXM"
LIQUIDITY_STRESS = "FRED:NFCI"
EQUITY_PROXY = "FRED:NASDAQ100"
EQUITY_RII = "ENGINE:equity.rii"

#: Role map as the engine resolves it, for testing the model modules directly.
SERIES_IDS = {
    "price": PRICE,
    "nominal_10y": NOMINAL_10Y,
    "breakeven_10y": BREAKEVEN_10Y,
    "cpi": CPI,
    "usd_index": USD_INDEX,
    "usd_index_legacy": USD_INDEX_LEGACY,
    "liquidity_stress": LIQUIDITY_STRESS,
    "equity_proxy": EQUITY_PROXY,
    "equity_rii": EQUITY_RII,
}

#: The snapshot's newest observation. Fixed rather than "today" so the suite does
#: not change behaviour as the calendar moves.
SNAPSHOT_AS_OF = date(2026, 7, 31)


@pytest.fixture(scope="session")
def gold_observations() -> pd.DataFrame:
    """The committed gold driver snapshot, as a PIT frame."""
    frame = pd.read_csv(FIXTURE_DIR / "gold_daily.csv")
    for column in ("obs_date", "release_date", "revision_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


@pytest.fixture
def gold_engine(config: SeriesConfig, artifacts: ArtifactStore) -> GoldEngine:
    """The engine on the shipped configuration, with a throwaway artifact store."""
    return GoldEngine(config, artifacts)


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


def wide_frame(observations: pd.DataFrame, as_of: date = SNAPSHOT_AS_OF) -> pd.DataFrame:
    """The wide PIT frame the driver panel is built from."""
    return PandasPITAccessor(observations, as_of).wide()


def observation_rows(
    series_id: str,
    values: dict[date, float] | pd.Series,
    *,
    lag_days: int = 1,
) -> list[dict]:
    """Observation rows for one series, each released ``lag_days`` later."""
    items = values.items()
    return [
        {
            "series_id": series_id,
            "obs_date": pd.Timestamp(day),
            "release_date": pd.Timestamp(day) + pd.Timedelta(days=lag_days),
            "revision_date": pd.Timestamp(day) + pd.Timedelta(days=lag_days),
            "value": float(value),
        }
        for day, value in items
    ]


def jump_diffusion(
    *,
    days: int = 2000,
    start: date = date(2000, 1, 3),
    volatility: float = 0.009,
    drift: float = 0.0002,
    jump_size: float = 0.09,
    n_jumps: int = 12,
    seed: int = 20260801,
) -> tuple[pd.Series, list[pd.Timestamp]]:
    """A Merton jump diffusion, and the dates the jumps were planted on.

    Returns ``(log_returns, jump_dates)``. The jumps are placed on a fixed grid
    rather than drawn from a Poisson process so that every run of the test looks
    at the same dates: a detector's recall is the assertion, and a fixture whose
    jump count varies with the seed makes a flaky test out of a deterministic
    property.

    ``jump_size`` is 10x the daily diffusive standard deviation. That is a real
    gold jump, not a convenient one — 15 April 2013 was -9.3% against a
    contemporaneous daily sigma near 1%.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, periods=days)
    returns = rng.normal(drift, volatility, size=days)

    # Placed clear of the ends so the local volatility window is fully formed on
    # both sides of every planted jump; a jump inside the burn-in would be
    # missed for a reason that has nothing to do with the detector.
    positions = np.linspace(400, days - 100, n_jumps).astype(int)
    signs = np.where(rng.random(n_jumps) < 0.5, -1.0, 1.0)
    for position, sign in zip(positions, signs, strict=True):
        returns[position] += sign * jump_size

    series = pd.Series(returns, index=index, name="log_return")
    return series, [index[p] for p in positions]
