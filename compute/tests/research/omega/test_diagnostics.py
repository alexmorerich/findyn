"""The panels, and the two sentences a threshold is allowed to write.

The redundancy verdict and the unit-root caveat are both emitted by a constant
rather than by a judgement call. That is deliberate: the person running this on
the day the numbers arrive does not get to decide whether the inconvenient
sentence is worth writing, and these tests are what keep it that way.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.research.omega import OmegaEngine, render_report
from findynamics.research.omega.diagnostics import (
    REDUNDANT_RHO,
    UNIT_ROOT_P,
    DiagnosticsInput,
    correlations,
    equity_reference,
    redundancy_panel,
    stationarity_panel,
)
from tests.research.omega.conftest import accessor_at


def _series(values, start: date = date(2000, 1, 3)) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    """A near-integrated series — ``phi`` close to 1 without being a random walk."""
    values = np.zeros(n)
    shocks = rng.normal(size=n)
    for i in range(1, n):
        values[i] = phi * values[i - 1] + shocks[i]
    return values


def test_correlations_report_rank_and_linear_side_by_side():
    """A high Spearman with a low Pearson says the two agree on order, not scale."""
    rng = np.random.default_rng(2)
    base = _series(rng.normal(size=1_000).cumsum())
    latent = {"omega": base}
    published = {"monotone": np.exp(base / 5.0), "unrelated": _series(rng.normal(size=1_000))}

    table = correlations(latent, published)

    assert table.loc["monotone", "omega_rho"] == pytest.approx(1.0, abs=1e-9)
    assert table.loc["monotone", "omega_r"] < 0.999
    assert abs(table.loc["unrelated", "omega_rho"]) < 0.2
    assert table.loc["monotone", "omega_n"] == 1_000


def test_a_short_overlap_is_reported_as_unavailable_rather_than_as_a_number():
    """A rank correlation over eighty dates is a number, not evidence."""
    latent = {"omega": _series(np.arange(100.0))}
    published = {"velocity": _series(np.arange(100.0))}

    table = correlations(latent, published)

    assert np.isnan(table.loc["velocity", "omega_rho"])
    assert table.loc["velocity", "omega_n"] == 100.0


def test_the_redundancy_panel_names_a_re_derivation_of_the_instability_index():
    """The sentence the panel is required to write, emitted by a threshold.

    ``K`` here is a monotone transform of the RII, so it *is* the RII under
    another name. The renderer must say so at the top of the panel and must not
    soften it.
    """
    rng = np.random.default_rng(4)
    rii = _series(np.abs(rng.normal(size=2_000).cumsum()))
    latent = {"curvature": rii * 3.0 + 1.0}
    published = {"rii": rii}

    lines = redundancy_panel(latent, published)

    assert "**K is a re-derivation of the existing instability index.**" in lines
    assert abs(correlations(latent, published).loc["rii", "curvature_rho"]) >= REDUNDANT_RHO


def test_the_redundancy_panel_names_any_other_re_derivation_too():
    """Not only ``K`` versus the RII — the rule is general."""
    rng = np.random.default_rng(6)
    jerk = _series(np.abs(rng.normal(size=2_000)))
    latent = {"omega": jerk * 2.0}
    published = {"jerk_z": jerk}

    lines = redundancy_panel(latent, published)

    assert any("is a re-derivation of `jerk_z`" in line for line in lines)


def test_the_redundancy_panel_says_so_when_nothing_is_redundant():
    """And is careful that "not redundant" is not "useful"."""
    rng = np.random.default_rng(8)
    latent = {"omega": _series(rng.normal(size=2_000))}
    published = {"rii": _series(rng.normal(size=2_000))}

    lines = redundancy_panel(latent, published)

    body = "\n".join(lines)
    assert f"|ρ_s| ≥ {REDUNDANT_RHO}" in body
    assert "only KK3 can test the second" in body
    assert "re-derivation of" not in body.split("| Published")[0].replace(
        "is a re-derivation of something", ""
    )


def _reported_p(lines: list[str], name: str) -> float:
    """The p-value the panel printed, read back out of its own table."""
    for line in lines:
        if line.startswith(f"| `{name}` |"):
            return float(line.split("|")[3].strip())
    raise AssertionError(f"{name} is not in the table:\n" + "\n".join(lines))


@pytest.mark.parametrize(
    "label,build",
    [
        ("near-integrated", lambda rng, n: _ar1(rng, n, 0.9995)),
        ("white noise", lambda rng, n: rng.normal(size=n)),
    ],
)
def test_the_unit_root_sentence_appears_exactly_when_the_threshold_is_crossed(label, build):
    """The renderer's contract, tested without depending on a draw.

    The obvious version of this test — build a random walk, assert the caveat
    appears — fails on roughly one seed in twenty, because a 5% test rejects a
    true unit root 5% of the time. That is not a flaw in the series, it is what
    the test *means*, and "fix" it by trying seeds until one passes and the test
    is choosing its own answer.

    So the assertion is the invariant instead: the sentence is present **iff**
    the p-value the panel itself printed exceeds :data:`UNIT_ROOT_P`. That holds
    for every input, and both branches are exercised — a near-integrated series
    normally trips it and white noise normally does not.
    """
    rng = np.random.default_rng(10)
    values = _series(build(rng, 2_000))

    lines = stationarity_panel({"coupling": values})
    body = "\n".join(lines)
    flagged = "A unit root cannot be ruled out in `coupling`" in body

    assert flagged == (_reported_p(lines, "coupling") > UNIT_ROOT_P), label
    assert flagged != (f"Every series rejects a unit root at p < {UNIT_ROOT_P}" in body)
    if flagged:
        assert "has to say what it did about that" in body


def test_white_noise_clears_the_unit_root_test():
    """The other branch, on a series where the answer is not in doubt.

    White noise rejects at p far below any threshold, so this one does not
    depend on the seed in the way a random walk does.
    """
    rng = np.random.default_rng(12)
    noise = _series(rng.normal(size=2_000))

    lines = stationarity_panel({"omega": noise})

    assert _reported_p(lines, "omega") < 1e-6
    assert f"Every series rejects a unit root at p < {UNIT_ROOT_P}" in "\n".join(lines)


def test_the_stationarity_reading_is_a_pure_function_of_the_series():
    """Fixed lag, not ``autolag``.

    ``autolag`` picks the lag from the data, which is deterministic but makes
    the reported statistic depend on a choice nothing else in the report can
    see. Two runs must print the same table.
    """
    rng = np.random.default_rng(14)
    values = _series(rng.normal(size=1_500))

    assert stationarity_panel({"omega": values}) == stationarity_panel({"omega": values})


@pytest.mark.slow
def test_the_equity_reference_publishes_what_the_engine_publishes(omega_observations):
    """The comparison set is the engine's real output, not a re-derivation of it.

    Runs the whole stack — Kalman MLE, FFD search, HMM fit, RII — which is the
    expensive part of any diagnostics run and the reason this is marked slow.
    """
    reference = equity_reference(accessor_at(omega_observations))
    published = reference.published()

    assert {"velocity", "acceleration", "jerk_z", "rii"} <= set(published)
    assert len(reference.rii.components) == 7
    assert reference.rii.missing == ()
    assert reference.posteriors.shape[1] == 5
    for name, series in published.items():
        assert series.notna().any(), f"{name} is entirely empty"


@pytest.mark.slow
def test_the_report_renders_every_panel_and_says_it_is_in_sample(omega_observations):
    """End to end on the committed fixture, no network and no API keys."""
    report = render_report(
        OmegaEngine.from_shipped_config().analyze(accessor_at(omega_observations))
    )

    for heading in (
        "## Ω",
        "## Coupling `C`",
        "## Curvature `K`",
        "## Redundancy against what the engine already publishes",
        "## Stationarity",
    ):
        assert heading in report
    assert "**Everything below is in-sample.**" in report
    assert "inspired by Kaluza–Klein" in report
    # The framing the track forbids, checked here rather than only in KK5.
    for forbidden in ("markets have a fifth dimension", "proves", "the market's true state"):
        assert forbidden not in report


@pytest.mark.slow
def test_the_report_is_byte_identical_across_two_runs(omega_observations):
    """It is an artifact, and KK5 regenerates it from stored CSVs."""
    engine = OmegaEngine.from_shipped_config()
    accessor = accessor_at(omega_observations)

    assert render_report(engine.analyze(accessor)) == render_report(engine.analyze(accessor))


def test_the_renderer_works_without_an_equity_reference(omega_observations):
    """The redundancy panel is skipped, not faked, when there is nothing to compare."""
    engine = OmegaEngine.from_shipped_config()
    reference = equity_reference(accessor_at(omega_observations))
    data = engine.analyze(accessor_at(omega_observations), reference=reference)
    without = DiagnosticsInput(
        path=data.path, couplings=data.couplings, curvature=data.curvature, reference=None
    )

    report = render_report(without)

    assert "## Redundancy" not in report
    assert "## Stationarity" in report
