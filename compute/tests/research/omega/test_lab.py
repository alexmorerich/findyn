"""The Lab artifact: its schema, and the byte-identity the committed file needs.

The artifact is a **file**, not an API response, and it is committed. Both of
those put an obligation on it: it has to say what it is without the reader
finding the docstring, and two runs on one snapshot have to produce the same
bytes or every regeneration is a spurious diff.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.research.omega.lab import (
    CHART_POINTS,
    DISCLAIMER,
    LAB_SERIES,
    build_artifact,
    decimate,
    write_artifact,
)
from tests.research.omega.conftest import with_params
from tests.research.omega.test_walk_forward import FAST  # noqa: F401


def _series(n: int, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=n).cumsum(), index=pd.bdate_range(date(2000, 1, 3), periods=n))


def test_a_short_series_is_emitted_whole():
    values = _series(100)

    points = decimate(values, points=CHART_POINTS)

    assert len(points) == 100
    assert points[0][0] == "2000-01-03"
    assert all(isinstance(iso, str) and isinstance(value, float) for iso, value in points)


def test_decimation_respects_the_cap_and_keeps_the_span():
    values = _series(20_000)

    points = decimate(values, points=500)

    # One per bucket plus the two endpoints, before de-duplication.
    assert 500 <= len(points) <= 502
    assert points[0][0] == values.index[0].date().isoformat()
    assert points[-1][0] == values.index[-1].date().isoformat()
    assert [p[0] for p in points] == sorted(p[0] for p in points)


def test_decimation_keeps_a_one_day_spike():
    """The property the serving decimator exists for, on the case it exists for.

    A single-day crash that survives 20,000 rows of stride sampling is the
    difference between a chart of the market and a chart of the sampling.
    """
    values = _series(20_000)
    values.iloc[9_000] = 500.0

    points = decimate(values, points=200)

    assert max(value for _, value in points) == pytest.approx(500.0)


def test_non_finite_values_are_dropped_rather_than_emitted_as_null():
    """A gap is the message; a ``null`` would have to be filtered anyway."""
    values = _series(300)
    values.iloc[100:150] = np.nan

    points = decimate(values, points=CHART_POINTS)

    assert len(points) == 250
    assert all(np.isfinite(value) for _, value in points)


def test_an_empty_series_yields_an_empty_array():
    assert decimate(pd.Series(dtype=float)) == []


@pytest.mark.slow
def test_the_artifact_carries_its_schema_and_regenerates_byte_identically(
    omega_observations, tmp_path
):
    """Two runs, one snapshot, identical bytes — and the disclaimer travels."""
    from findynamics.research.omega import load_omega_config
    from findynamics.research.omega.backtest import walk_forward
    from findynamics.research.omega.evaluate import evaluate

    config = with_params(load_omega_config(), walk_forward=FAST)
    truncated = omega_observations[omega_observations["obs_date"] <= pd.Timestamp(2014, 12, 31)]
    result = walk_forward(truncated, config)
    artifact = build_artifact(result, evaluate(result, redundancy=0.0))

    assert set(artifact) == {
        "disclaimer",
        "generated",
        "model_version",
        "config_hash",
        "spec",
        "verdicts",
        "series",
        "diagnostics",
    }
    assert artifact["disclaimer"] == DISCLAIMER
    assert "no causal relationship is implied" in artifact["disclaimer"]
    assert set(artifact["series"]) <= set(LAB_SERIES)
    assert artifact["spec"]["columns"]
    assert artifact["verdicts"]
    # No wall clock: `generated` is the information set, not the moment of
    # writing, which is what lets two runs a week apart agree.
    assert artifact["generated"] == result.panel.index[-1].date().isoformat()

    first = write_artifact(artifact, tmp_path / "omega.json")
    second_body = json.dumps(
        build_artifact(result, evaluate(result, redundancy=0.0)),
        sort_keys=True,
        indent=2,
        separators=(",", ": "),
    )
    assert first.read_text() == second_body + "\n"
    # And it round-trips, because the dashboard parses it.
    assert json.loads(first.read_text())["model_version"] == result.model_version


def test_the_committed_artifact_matches_the_schema_the_dashboard_expects():
    """The file in ``dashboard/public/research/`` is a contract with the Lab."""
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3].parent
        / "dashboard"
        / "public"
        / "research"
        / "omega.json"
    )
    if not path.exists():
        pytest.skip("lab artifact not generated in this working tree")

    artifact = json.loads(path.read_text())

    assert artifact["disclaimer"] == DISCLAIMER
    assert isinstance(artifact["spec"]["columns"], list)
    assert isinstance(artifact["spec"]["loadings"], dict)
    for name, points in artifact["series"].items():
        assert name in LAB_SERIES
        assert len(points) <= CHART_POINTS + 2
        for iso, value in points:
            date.fromisoformat(iso)
            assert isinstance(value, float)
