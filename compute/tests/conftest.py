"""Shared test fixtures.

The resilience layer is built around injected time, sleep and HTTP, so tests
drive real backoff and breaker transitions without spending wall-clock seconds
or touching the network.

Nothing in the suite reaches the network or reads a live API key: model tests
run on synthetic Nelson-Siegel curves or on the committed month-end Treasury
snapshot described in ``tests/fixtures/README.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from findynamics.core.artifacts import ArtifactStore
from findynamics.core.config import load_series_config
from findynamics.data.providers.resilience import Response

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def treasury_observations() -> pd.DataFrame:
    """The committed month-end Treasury curve snapshot, as a PIT frame."""
    frame = pd.read_csv(FIXTURE_DIR / "treasury_monthly.csv")
    for column in ("obs_date", "release_date", "revision_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


@pytest.fixture
def config():
    """The shipped configuration, loaded fresh so a test can copy and edit it."""
    return load_series_config()


@pytest.fixture
def artifacts(tmp_path) -> ArtifactStore:
    """Artifact store under tmp_path, so no test reads or writes a real refit."""
    return ArtifactStore(tmp_path / "artifacts")


class FakeClock:
    """Monotonic clock advanced explicitly, or by the sleeps under test."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingSleeper:
    """Records requested delays and advances the clock instead of blocking."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        self.clock.advance(seconds)

    @property
    def total(self) -> float:
        return sum(self.delays)


class FakeFetcher:
    """Replays a scripted sequence of responses or exceptions."""

    def __init__(self, script: list[Response | Exception] | Response | Exception) -> None:
        self.script = script if isinstance(script, list) else [script]
        self.calls: list[tuple[str, Mapping[str, Any] | None]] = []

    def __call__(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Response:
        self.calls.append((url, params))
        # The last scripted item repeats, so "always fails" needs one entry.
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def call_count(self) -> int:
        return len(self.calls)


def ok(body: str | bytes, status: int = 200, **headers: str) -> Response:
    content = body.encode() if isinstance(body, str) else body
    return Response(status_code=status, content=content, headers=headers)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sleeper(clock: FakeClock) -> RecordingSleeper:
    return RecordingSleeper(clock)
