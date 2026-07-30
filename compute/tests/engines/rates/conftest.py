"""Rates-engine fixtures and synthetic-curve builders.

``treasury_observations``, ``config`` and ``artifacts`` come from the top-level
conftest; the replay suite needs them too.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from findynamics.core.artifacts import ArtifactStore
from findynamics.core.contracts.state import WorldState
from findynamics.data.accessor import PandasPITAccessor
from findynamics.engines.rates.engine import RatesEngine
from findynamics.engines.rates.nelson_siegel import DEFAULT_LAMBDA, design_matrix

TENOR_IDS = {
    "FRED:DGS1MO": 0.08333,
    "FRED:DGS3MO": 0.25,
    "FRED:DGS6MO": 0.5,
    "FRED:DGS1": 1.0,
    "FRED:DGS2": 2.0,
    "FRED:DGS3": 3.0,
    "FRED:DGS5": 5.0,
    "FRED:DGS7": 7.0,
    "FRED:DGS10": 10.0,
    "FRED:DGS20": 20.0,
    "FRED:DGS30": 30.0,
}


@pytest.fixture
def monthly_engine(config, artifacts: ArtifactStore) -> RatesEngine:
    """Engine configured for the monthly fixture's cadence.

    The shipped thresholds are in trading days (252 ≈ a year); the snapshot is
    month-end, so the trend window becomes 12 periods for the same meaning.
    Overriding the window rather than the rules keeps the classifier under test.
    """
    engine = RatesEngine(config, artifacts)
    params = monthly_params(config)
    engine._config = _with_params(config, params)
    return engine


def monthly_params(config) -> dict:
    """The shipped rates params retuned from trading days to calendar months."""
    params = dict(config.engines["rates"].params)
    params["regime"] = {**(params.get("regime") or {}), "trend_days": 12}
    params["risk"] = {**(params.get("risk") or {}), "vol_periods_per_year": 12}
    params["outputs"] = {"history_days": 40 * 365}
    return params


def _with_params(config, params):
    """Copy of ``config`` with the rates engine's params replaced."""
    from dataclasses import replace

    engine_config = replace(config.engines["rates"], params=params)
    return replace(config, engines={**config.engines, "rates": engine_config})


def world_from(observations: pd.DataFrame, as_of: date) -> WorldState:
    """A WorldState over ``observations`` clamped to ``as_of``, no factors."""
    return WorldState(as_of=as_of, factors={}, series=PandasPITAccessor(observations, as_of))


def synthetic_curve_frame(
    betas: list[tuple[float, float, float]],
    *,
    start: date = date(2020, 1, 1),
    lambda_: float = DEFAULT_LAMBDA,
    tenors: dict[str, float] | None = None,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Observation frame generated from known Nelson-Siegel factors.

    One curve per calendar day from ``start``, each released ``lag_days`` later,
    so a PIT cutoff behaves exactly as it would on real data.
    """
    resolved = tenors or TENOR_IDS
    maturities = np.array(list(resolved.values()))
    loadings = design_matrix(maturities, lambda_)

    rows = []
    for offset, (b0, b1, b2) in enumerate(betas):
        obs_date = start + timedelta(days=offset)
        release = obs_date + timedelta(days=lag_days)
        yields = loadings @ np.array([b0, b1, b2])
        for series_id, value in zip(resolved, yields, strict=True):
            rows.append(
                {
                    "series_id": series_id,
                    "obs_date": pd.Timestamp(obs_date),
                    "release_date": pd.Timestamp(release),
                    "revision_date": pd.Timestamp(release),
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)
