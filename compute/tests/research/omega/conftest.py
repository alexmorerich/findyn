"""Fixtures for the KK-Ω suite.

Real data where the answer depends on what markets actually did, synthetic where
a closed form or a known property is the assertion — the split
``tests/engines/equity/conftest.py`` documents, and the reason its snapshot is
real too. A latent stress coordinate that is only ever exercised on a random
walk has never been asked whether it finds 2008.

Nothing here reaches the network. The committed fixture
``tests/fixtures/equity_prices.csv`` carries every series ``omega.yaml``
configures.
"""

from __future__ import annotations

import copy
from datetime import date

import pandas as pd
import pytest

from findynamics.data.accessor import PandasPITAccessor
from findynamics.research.omega import OmegaConfig, OmegaEngine, load_omega_config
from findynamics.research.omega.contracts import OmegaSpec
from tests.conftest import FIXTURE_DIR

#: The committed snapshot's newest observation. Fixed rather than "today" so the
#: suite does not change behaviour as the calendar moves.
SNAPSHOT_AS_OF = date(2026, 7, 31)

#: Cutoffs the leakage tests truncate at: a calm stretch, the GFC, and a recent
#: window. Named because the assertion is only meaningful if the truncation
#: lands somewhere the data is doing something different in each case.
TRUNCATION_CUTOFFS: dict[str, date] = {
    "calm 2006": date(2006, 6, 30),
    "2008 GFC": date(2008, 10, 31),
    "recent 2024": date(2024, 6, 28),
}


@pytest.fixture(scope="session")
def omega_observations() -> pd.DataFrame:
    """The committed equity price snapshot, as a PIT frame."""
    frame = pd.read_csv(FIXTURE_DIR / "equity_prices.csv")
    for column in ("obs_date", "release_date", "revision_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


@pytest.fixture
def omega_config() -> OmegaConfig:
    """The shipped research config, loaded fresh so a test can copy and edit it."""
    return load_omega_config()


@pytest.fixture
def omega_engine(omega_config: OmegaConfig) -> OmegaEngine:
    return OmegaEngine(config=omega_config)


def accessor_at(observations: pd.DataFrame, as_of: date = SNAPSHOT_AS_OF) -> PandasPITAccessor:
    """A PIT accessor over ``observations`` clamped to ``as_of``."""
    return PandasPITAccessor(observations, as_of)


def with_params(config: OmegaConfig, **blocks: dict) -> OmegaConfig:
    """A deep copy of ``config`` with parameter blocks merged in.

    Deep, because the blocks are nested dicts and a shallow copy would let one
    test's override leak into the next through the shared fixture.
    """
    params = copy.deepcopy(config.params)
    for name, override in blocks.items():
        block = params.setdefault(name, {})
        if isinstance(block, dict):
            block.update(override)
        else:
            params[name] = override
    return OmegaConfig(enabled=config.enabled, experimental=config.experimental, params=params)


def observation_rows(
    series_id: str,
    values: dict[date, float],
    *,
    lag_days: int = 0,
) -> list[dict]:
    """PIT rows for one synthetic series, each released ``lag_days`` later."""
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


def price_frame(
    values: list[float],
    *,
    series_id: str = "SYNTH:PRICE",
    start: date = date(2000, 1, 3),
    lag_days: int = 0,
) -> pd.DataFrame:
    """A synthetic daily price series as a PIT frame, on business days."""
    index = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame(
        observation_rows(
            series_id,
            dict(zip(index.date, values, strict=True)),
            lag_days=lag_days,
        )
    )


def price_only_config(config: OmegaConfig, series_id: str = "SYNTH:PRICE") -> OmegaConfig:
    """``config`` pointed at one synthetic price series and nothing else.

    Used by the transform unit tests, where the assertion is about what
    ``realized_vol`` does to a constant series rather than about which columns
    survive. The macro roles are cleared rather than left pointing at FRED ids
    the synthetic frame does not carry — an unconfigured role and an absent
    series take the same path, but only one of them says so in the config.
    """
    params = copy.deepcopy(config.params)
    params["series"] = {"price": series_id}
    params["features"] = {**params.get("features", {}), "start": None}
    return OmegaConfig(enabled=config.enabled, experimental=config.experimental, params=params)


def first_vintage(observations: pd.DataFrame) -> pd.DataFrame:
    """Every observation as **first published**, with later revisions removed.

    The committed fixture carries real vintages: ``FRED:NASDAQ100`` has 1,195
    observation dates published more than once and 657 whose value actually
    changed, and ``FRED:DGS10`` and ``FRED:T10Y3M`` have a handful each. That is
    the point-in-time system working — an information set in 2024 sees the
    numbers FRED had published by 2024, not the ones it publishes today.

    It also makes a cross-information-set comparison ambiguous: two runs at
    different cutoffs differ both because one saw more *dates* and because one
    saw a later *vintage* of the same date, and only the first would be
    lookahead. This helper removes the second cause so the leakage tests can
    assert bit-identity and mean it.
    """
    ordered = observations.sort_values(["series_id", "obs_date", "release_date"], kind="stable")
    return ordered.drop_duplicates(subset=["series_id", "obs_date"], keep="first").reset_index(
        drop=True
    )


def dummy_spec(
    columns: tuple[str, ...] = ("realized_vol",),
    *,
    n_observations: int = 1_000,
) -> OmegaSpec:
    """A minimal valid :class:`OmegaSpec`, for tests about the *dynamics*.

    The dynamics do not read the spec's numbers — they carry it onto the path so
    a stored artifact says which projection produced the Ω it describes — so a
    test about Ω̇ should not have to fit a PCA to get one.
    """
    width = len(columns)
    return OmegaSpec(
        estimator="pca",
        columns=tuple(sorted(columns)),
        dropped=(),
        mean=(0.0,) * width,
        scale=(1.0,) * width,
        loadings=(1.0,) * width,
        explained_variance_ratio=1.0,
        sign_reference=sorted(columns)[0],
        fit_start=date(2000, 1, 3),
        fit_end=date(2012, 1, 1),
        n_observations=n_observations,
        model_version="test",
    )
