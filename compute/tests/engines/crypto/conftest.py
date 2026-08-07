"""FinCrypto fixtures: the real price/on-chain snapshot and synthetic paths.

Split by what each kind of input can prove, the same way the gold suite is.

* **The real snapshot** (``crypto_daily.csv``) is for the regime. Which windows
  count as a frenzy and which as a winter is not a question a made-up series can
  answer — you can always invent one that crosses whichever line you wrote — so
  those assertions run against the actual 2010-2026 record — stitched from a
  daily-average history leg and a close leg, exactly as the engine sees it.
* **Synthetic series** are for the estimators. The liquidity beta is asserted
  against data built with a **known** beta, which is the one thing real data
  cannot offer: nobody publishes bitcoin's true sensitivity to the money supply,
  so an estimator tested only against history can be scored on plausibility and
  never on correctness.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.core.artifacts import ArtifactStore
from findynamics.core.config import SeriesConfig
from findynamics.core.contracts.state import WorldState
from findynamics.data.accessor import PandasPITAccessor
from findynamics.engines.crypto import prices as prices_mod
from findynamics.engines.crypto.engine import CryptoEngine
from tests.conftest import FIXTURE_DIR

#: Role -> series id, matching config/series.yaml engines.crypto.series.
PRICE = "STOOQ:BTCUSD"
PRICE_FALLBACK = "YAHOO:BTC-USD"
PRICE_HISTORY = "BLOCKCHAIN:MARKET_PRICE"
M2 = "FRED:M2SL"
CENTRAL_BANK_ASSETS = "FRED:WALCL"
TX_VOLUME_USD = "BLOCKCHAIN:TX_VOLUME_USD"
ACTIVE_ADDRESSES = "BLOCKCHAIN:N_UNIQUE_ADDRESSES"
TRANSACTIONS = "BLOCKCHAIN:N_TRANSACTIONS"
HASH_RATE = "BLOCKCHAIN:HASH_RATE"

SERIES_IDS = {
    "price": PRICE,
    "price_fallback": PRICE_FALLBACK,
    "price_history": PRICE_HISTORY,
    "m2": M2,
    "central_bank_assets": CENTRAL_BANK_ASSETS,
    "tx_volume_usd": TX_VOLUME_USD,
    "active_addresses": ACTIVE_ADDRESSES,
    "transactions": TRANSACTIONS,
    "hash_rate": HASH_RATE,
}

#: The snapshot's newest observation. Fixed rather than "today" so the suite does
#: not change behaviour as the calendar moves.
SNAPSHOT_AS_OF = date(2026, 8, 5)


@pytest.fixture(scope="session")
def crypto_observations() -> pd.DataFrame:
    """The committed bitcoin price and on-chain snapshot, as a PIT frame."""
    frame = pd.read_csv(FIXTURE_DIR / "crypto_daily.csv")
    for column in ("obs_date", "release_date", "revision_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


@pytest.fixture
def crypto_engine(config: SeriesConfig, artifacts: ArtifactStore) -> CryptoEngine:
    """The engine on the shipped configuration, with a throwaway artifact store."""
    return CryptoEngine(config, artifacts)


@pytest.fixture
def crypto_only_config(config: SeriesConfig) -> SeriesConfig:
    """The shipped config with crypto **enabled** and every other engine off.

    For the quarantine tests, which have to show that being enabled is not enough
    to reach the portfolio layer. Only crypto is left on so the assertion can be
    an exact list rather than a membership check — and so the test does not
    depend on which other engine packages happen to have been imported.
    """
    engines = {
        name: replace(entry, enabled=(name == "crypto")) for name, entry in config.engines.items()
    }
    return replace(config, engines=engines)


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


def price_series(observations: pd.DataFrame, as_of: date = SNAPSHOT_AS_OF) -> pd.Series:
    """The **stitched** daily record, exactly as the engine resolves it.

    Deliberately not "the Yahoo close": the engine reads a record spliced from a
    daily-average history role and a close role, and a regime assertion run
    against only one leg would be verifying a series the engine does not use.
    Everything about the acceptance windows has to be asserted on the real spine.
    """
    frame = PandasPITAccessor(observations, as_of).wide()
    return prices_mod.build(frame, SERIES_IDS).price


def close_leg(observations: pd.DataFrame) -> pd.Series:
    """Just the close role, for tests about the splice rather than about the model."""
    rows = observations[observations["series_id"] == PRICE_FALLBACK]
    return rows.set_index("obs_date")["value"].sort_index()


def history_leg(observations: pd.DataFrame) -> pd.Series:
    """Just the daily-average history role."""
    rows = observations[observations["series_id"] == PRICE_HISTORY]
    return rows.set_index("obs_date")["value"].sort_index()


def observation_rows(
    series_id: str,
    values: dict[date, float] | pd.Series,
    *,
    lag_days: int = 0,
) -> list[dict]:
    """Observation rows for one series, each released ``lag_days`` later."""
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


def known_beta_series(
    *,
    beta: float,
    months: int = 180,
    alpha: float = 0.0,
    noise: float = 0.0,
    liquidity_growth: float = 0.005,
    start: date = date(2010, 1, 31),
    seed: int = 20260805,
) -> tuple[pd.Series, pd.Series]:
    """Monthly returns generated from a liquidity path with a **known** beta.

    Returns ``(monthly_log_returns, liquidity_level)``. The level grows at
    ``liquidity_growth`` per month plus a random wobble, and the return is
    ``alpha + beta * dLog(level) + noise`` — so the regression has one right
    answer and the test can assert on it rather than on plausibility.

    ``noise`` of 0 makes the fit exact, which is what pins the estimator's
    arithmetic. A non-zero value is what shows it still recovers the coefficient
    when the relationship is not deterministic.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=months, freq="ME")

    growth = liquidity_growth + rng.normal(0.0, liquidity_growth / 2.0, size=months)
    level = pd.Series(1000.0 * np.exp(np.cumsum(growth)), index=index, name="liquidity")

    d_log = np.log(level).diff()
    returns = alpha + beta * d_log
    if noise > 0:
        returns = returns + pd.Series(rng.normal(0.0, noise, size=months), index=index)

    return returns.rename("ret"), level
