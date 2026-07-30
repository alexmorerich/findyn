"""FinMoney fixtures: the real money-market snapshot and synthetic rate paths.

The suite is deliberately split by what each kind of input can prove.

* **Synthetic constant-rate paths** are for the account arithmetic. A closed form
  exists, so the assertion can be an exact equality against a number worked out
  by hand rather than against whatever the code happened to produce.
* **The real snapshot** (``money_daily.csv``) is for the regimes. Thresholds are
  the one thing a synthetic fixture cannot validate — you can always make up a
  series that crosses whichever line you wrote — so those are asserted against
  September 2019 and March 2020, plus September 2024 as a negative control.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from findynamics.core.config import SeriesConfig
from findynamics.core.contracts.state import WorldState
from findynamics.data.accessor import PandasPITAccessor
from findynamics.engines.money.engine import MoneyEngine
from tests.conftest import FIXTURE_DIR

#: Role -> series id, matching config/series.yaml engines.money.series.
SOFR = "FRED:SOFR"
DTB3 = "FRED:DTB3"
BILL_3M = "FRED:DGS3MO"
RRP = "FRED:RRPONTSYD"

NS_IDS = {
    "level": "ENGINE:rates.ns_level",
    "slope": "ENGINE:rates.ns_slope",
    "curvature": "ENGINE:rates.ns_curvature",
    "lambda": "ENGINE:rates.ns_lambda",
}


@pytest.fixture(scope="session")
def money_observations() -> pd.DataFrame:
    """The committed daily money-market snapshot, as a PIT frame."""
    frame = pd.read_csv(FIXTURE_DIR / "money_daily.csv")
    for column in ("obs_date", "release_date", "revision_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


@pytest.fixture
def money_engine(config: SeriesConfig, artifacts) -> MoneyEngine:
    """The engine on the shipped configuration."""
    return MoneyEngine(config, artifacts)


def world_from(observations: pd.DataFrame, as_of: date, factors: dict | None = None) -> WorldState:
    """A WorldState over ``observations`` clamped to ``as_of``."""
    return WorldState(
        as_of=as_of,
        factors=factors or {},
        series=PandasPITAccessor(observations, as_of),
    )


def rate_rows(
    series_id: str,
    values: dict[date, float],
    *,
    lag_days: int = 1,
) -> list[dict]:
    """Observation rows for one series, each released ``lag_days`` later."""
    return [
        {
            "series_id": series_id,
            "obs_date": pd.Timestamp(day),
            "release_date": pd.Timestamp(day + timedelta(days=lag_days)),
            "revision_date": pd.Timestamp(day + timedelta(days=lag_days)),
            "value": float(value),
        }
        for day, value in values.items()
    ]


def constant_rate_frame(
    rate_pct: float,
    *,
    start: date = date(2020, 1, 1),
    days: int = 365,
    series_id: str = SOFR,
    step_days: int = 1,
) -> pd.DataFrame:
    """One series holding ``rate_pct`` on every date — the closed-form case.

    Calendar days, not trading days: the accrual is calendar-based, so a
    consecutive-day path is the one whose result is a single ``exp`` call.
    """
    values = {start + timedelta(days=i * step_days): rate_pct for i in range(days)}
    return pd.DataFrame(rate_rows(series_id, values))


def curve_rows(
    factors: dict[str, float],
    days: list[date],
    *,
    published_on: date | None = None,
) -> list[dict]:
    """Published NS factor rows, as the ``engine_output`` provider would deliver.

    One publication date for every observation date, which is what a daily run
    republishing a history window genuinely produces.
    """
    rows: list[dict] = []
    for day in days:
        release = published_on or (day + timedelta(days=1))
        for role, value in factors.items():
            rows.append(
                {
                    "series_id": NS_IDS[role],
                    "obs_date": pd.Timestamp(day),
                    "release_date": pd.Timestamp(max(release, day)),
                    "revision_date": pd.Timestamp(max(release, day)),
                    "value": float(value),
                }
            )
    return rows
