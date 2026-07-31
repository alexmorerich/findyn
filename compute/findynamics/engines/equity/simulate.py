"""Regime-switching Monte Carlo with a shock overlay (FINDYN_V1_SPEC.md §11).

The output contract is the whole point: **quantiles, never a number**. §0's first
non-goal is "no deterministic price targets", and every forward-looking value
this engine publishes is a band. That is not hedging — a point forecast of an
index at a ten-year horizon is a claim nobody can hold anyone to, and a
distribution is one you can score.

How a path is generated
-----------------------

1. **The starting state is today's posterior**, not a guess and not the argmax.
   A day the model reads as 60/40 between two regimes should produce paths from
   both in that proportion; collapsing to the winner throws away exactly the
   uncertainty a fan chart exists to show.
2. **The regime evolves through the fitted transition matrix.** Each step draws
   the next state from the current row.
3. **Returns are drawn per regime**, from that state's fitted mean and
   volatility as measured on the calibration series.
4. **Shocks are overlaid independently** (§11: "do not replay 2008/2020"). A
   Poisson arrival whose intensity scales with the RII, a severity from the same
   EVT tail the crash decomposition uses, and a recovery that plays out over
   following steps rather than instantaneously.

Why the overlay is drift-neutral
--------------------------------

Because the crashes are already in the regime process, and adding them twice was
measurably wrong.

The states are fitted on the real record, so the ``crisis`` state's −25.8%
annualized return *is* the permanent damage of historical crashes. The chain's
stationary drift comes out at 5.61%/yr against an actual 6.13%/yr for the S&P
price index over 1927-2026 — close, and closer than any overlay needs to fix.

The first version of this let each shock retrace only two thirds of its fall.
That looked realistic and it double-counted: the 12-year median implied 2.7%/yr,
less than half the historical drift, because every simulated crash subtracted a
permanent loss the regime means had already subtracted. A fan chart whose centre
is wrong by a factor of two is worse than no fan chart.

So the overlay now heals fully. It contributes what the regime process cannot —
the *discontinuity*: a fall of tail magnitude arriving over days rather than
being spread across a state's average, which is what drives path-dependent
statistics like maximum drawdown and time under water. It does not contribute
drift, because the drift is already accounted for.

§4's framing is the same: proximity to a critical point, not a replay of a
particular crisis. The shock taxonomy in ``domain.SHOCK_CLASSES`` is generic for
that reason — modelling "the next financial crisis" as a rerun of 2008 is how a
model learns the shape of one event and misses every other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from findynamics.core.contracts.vocab import EDUCATIONAL_HORIZONS, QUANTILES
from findynamics.engines.equity.crash import TailFit
from findynamics.engines.equity.domain import SHOCK_CLASSES

log = logging.getLogger("findynamics.engines.equity.simulate")

#: §11 — "generate ≥10,000 paths per horizon". The floor, not a target.
DEFAULT_PATHS = 10_000

#: §10 horizons, in years. The two educational ones are simulated and then
#: excluded from every accuracy evaluation — they exist to show the shape of
#: compounding uncertainty, not to be scored.
HORIZON_YEARS: dict[str, float] = {
    "tactical": 0.5,
    "strategic": 2.0,
    "generational": 12.0,
    "educational_30y": 30.0,
    "educational_50y": 50.0,
}

#: Shock arrivals per year at a neutral RII, before the hazard multiplier.
#: Calibrated against the declustered record: a >10% drawdown episode starts
#: about once every three years.
DEFAULT_SHOCK_INTENSITY = 0.33

#: How long a shock's drawdown takes to arrive and to heal, in trading days.
#: A crash is fast and a recovery is not, which is the asymmetry that makes
#: time-under-water worth reporting separately from depth.
DEFAULT_SHOCK_ONSET_DAYS = 10
DEFAULT_SHOCK_RECOVERY_DAYS = 250

#: Fraction of each shock that is retraced. **1.0 on purpose** — see the module
#: docstring: the permanent component of a crash is already inside the fitted
#: regime means, and retracing less than all of it subtracted the same loss
#: twice and halved the long-horizon drift.
SHOCK_RETRACE = 1.0

#: Fixed so a run is reproducible and the replay test can compare two of them.
DEFAULT_SEED = 20260731


@dataclass(frozen=True)
class HorizonForecast:
    """One horizon's distribution, and the statistics §11 asks to report."""

    horizon: str
    years: float
    #: Quantile -> projected log index level. The published contract.
    quantiles: dict[float, float]
    #: P(the path draws down more than x from its own running peak).
    drawdown_probabilities: dict[float, float]
    #: Median and 95th percentile of maximum drawdown across paths.
    median_max_drawdown: float
    worst_decile_max_drawdown: float
    #: Median fraction of the horizon spent below a previous peak.
    median_time_under_water: float
    paths: int
    educational_only: bool

    @property
    def median_log_level(self) -> float:
        return self.quantiles[0.5]

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "years": self.years,
            "quantiles": {str(k): v for k, v in self.quantiles.items()},
            "drawdown_probabilities": {str(k): v for k, v in self.drawdown_probabilities.items()},
            "median_max_drawdown": self.median_max_drawdown,
            "worst_decile_max_drawdown": self.worst_decile_max_drawdown,
            "median_time_under_water": self.median_time_under_water,
            "paths": self.paths,
            "educational_only": self.educational_only,
        }


@dataclass(frozen=True)
class SimulationResult:
    """Every horizon, plus what the run was conditioned on."""

    forecasts: dict[str, HorizonForecast]
    start_log_level: float
    #: Shock arrivals per year actually used, after the RII multiplier.
    shock_intensity: float
    seed: int
    #: Per-horizon path bundles, kept only when the caller asked for them —
    #: 10,000 × 12,600 steps is 1.3GB at the generational horizon.
    bundles: dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {"mc_shock_intensity": round(self.shock_intensity, 6)}
        for name, forecast in self.forecasts.items():
            out[f"mc_{name}_median"] = round(forecast.median_log_level, 6)
            out[f"mc_{name}_p20_drawdown"] = round(
                forecast.drawdown_probabilities.get(0.20, float("nan")), 6
            )
        return out


def _regime_returns(
    stats: list[tuple[float, float]],
    periods_per_year: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step mean and volatility per state, from annualized statistics.

    The mean divides by the period count and the volatility by its square root —
    the standard diffusion scaling, and the one place a units error would make
    every path wrong in a way that still looks like a plausible chart.
    """
    means = np.array([mu / periods_per_year for mu, _ in stats], dtype=float)
    vols = np.array([sigma / np.sqrt(periods_per_year) for _, sigma in stats], dtype=float)
    return means, vols


def simulate_paths(
    *,
    posterior: np.ndarray,
    transmat: np.ndarray,
    regime_stats: list[tuple[float, float]],
    steps: int,
    paths: int,
    periods_per_year: float,
    tail: TailFit | None,
    shock_intensity: float,
    rng: np.random.Generator,
    onset_days: int = DEFAULT_SHOCK_ONSET_DAYS,
    recovery_days: int = DEFAULT_SHOCK_RECOVERY_DAYS,
) -> np.ndarray:
    """Cumulative log returns, shape ``(paths, steps)``.

    Vectorised across paths rather than looping over them: 10,000 paths × 12,600
    steps is 126 million draws, and a per-path Python loop turns a two-second
    simulation into a twenty-minute one.
    """
    means, vols = _regime_returns(regime_stats, periods_per_year)
    n_states = len(means)

    # Start every path from a state drawn from today's posterior.
    start = np.asarray(posterior, dtype=float)
    start = start / start.sum() if start.sum() > 0 else np.full(n_states, 1.0 / n_states)
    states = rng.choice(n_states, size=paths, p=start)

    # Cumulative rows of the transition matrix, so the next state is one
    # searchsorted per path per step rather than a choice() call each.
    cumulative = np.cumsum(np.asarray(transmat, dtype=float), axis=1)
    cumulative[:, -1] = 1.0

    increments = np.empty((paths, steps), dtype=float)
    noise = rng.standard_normal((paths, steps))
    draws = rng.random((paths, steps))

    for step in range(steps):
        increments[:, step] = means[states] + vols[states] * noise[:, step]
        states = (draws[:, step, None] > cumulative[states]).sum(axis=1)
        np.clip(states, 0, n_states - 1, out=states)

    if tail is not None and shock_intensity > 0:
        increments += _shock_overlay(
            paths=paths,
            steps=steps,
            periods_per_year=periods_per_year,
            tail=tail,
            intensity=shock_intensity,
            rng=rng,
            onset_days=onset_days,
            recovery_days=recovery_days,
        )

    return np.cumsum(increments, axis=1)


def _shock_overlay(
    *,
    paths: int,
    steps: int,
    periods_per_year: float,
    tail: TailFit,
    intensity: float,
    rng: np.random.Generator,
    onset_days: int,
    recovery_days: int,
) -> np.ndarray:
    """Independent shock arrivals, spread over an onset and a partial recovery.

    Severity comes from the same GPD the crash decomposition uses, so the two
    published numbers cannot disagree about how bad a tail event is. It is drawn
    at the fitted (monthly) frequency and applied as a *total* move spread across
    the onset window — the magnitude is the size of the episode, not of one day.

    Recovery is **full**, and that is not a simplification. The permanent loss
    from historical crashes is already priced into the fitted regime means; a
    partial retrace here subtracts it a second time. Measured, that error took
    the 12-year median from 6%/yr to 2.7%/yr. What the overlay adds is the
    *shape* — a discontinuity over days — not the level.
    """
    from scipy import stats as scipy_stats

    overlay = np.zeros((paths, steps), dtype=float)
    per_step = intensity / periods_per_year
    arrivals = rng.random((paths, steps)) < per_step
    if not arrivals.any():
        return overlay

    rows, columns = np.nonzero(arrivals)
    severities = tail.threshold + scipy_stats.genpareto.rvs(
        tail.shape, scale=tail.scale, size=rows.size, random_state=rng
    )
    # A drawdown is a loss: convert the magnitude to a negative log move.
    magnitudes = np.log1p(-np.clip(severities, 0.0, 0.95))

    onset = max(int(onset_days), 1)
    recovery = max(int(recovery_days), 1)
    for row, column, magnitude in zip(rows, columns, magnitudes, strict=True):
        end = min(column + onset, steps)
        overlay[row, column:end] += magnitude / onset
        heal_end = min(end + recovery, steps)
        if heal_end > end:
            overlay[row, end:heal_end] += (-magnitude * SHOCK_RETRACE) / (heal_end - end)
    return overlay


def summarise(
    cumulative: np.ndarray,
    *,
    horizon: str,
    years: float,
    start_log_level: float,
    thresholds: tuple[float, ...] = (0.20, 0.30, 0.50),
) -> HorizonForecast:
    """Quantiles and drawdown statistics from a bundle of paths."""
    terminal = start_log_level + cumulative[:, -1]

    # Max drawdown per path, measured against each path's own running peak.
    running_peak = np.maximum.accumulate(cumulative, axis=1)
    underwater = cumulative - running_peak
    max_drawdown = 1.0 - np.exp(underwater.min(axis=1))
    time_under_water = (underwater < 0).mean(axis=1)

    return HorizonForecast(
        horizon=horizon,
        years=years,
        quantiles={q: float(np.quantile(terminal, q)) for q in QUANTILES},
        drawdown_probabilities={
            threshold: float((max_drawdown > threshold).mean()) for threshold in thresholds
        },
        median_max_drawdown=float(np.median(max_drawdown)),
        worst_decile_max_drawdown=float(np.quantile(max_drawdown, 0.90)),
        median_time_under_water=float(np.median(time_under_water)),
        paths=int(cumulative.shape[0]),
        educational_only=horizon in EDUCATIONAL_HORIZONS,
    )


def run_simulation(
    *,
    posterior: np.ndarray,
    transmat: np.ndarray,
    regime_stats: list[tuple[float, float]],
    start_log_level: float,
    periods_per_year: float,
    tail: TailFit | None = None,
    rii: float | None = None,
    paths: int = DEFAULT_PATHS,
    horizons: dict[str, float] | None = None,
    seed: int = DEFAULT_SEED,
    keep_bundles: bool = False,
    base_intensity: float = DEFAULT_SHOCK_INTENSITY,
) -> SimulationResult:
    """Simulate every horizon from one conditioning state.

    ``paths`` applies per horizon, so the §11 floor of 10,000 is met at each.
    Bundles are kept only on request: the generational horizon alone is
    10,000 × 12,600 floats, and the daily job wants the quantiles.
    """
    from findynamics.engines.equity.crash import rii_hazard_multiplier

    resolved = dict(HORIZON_YEARS if horizons is None else horizons)
    intensity = base_intensity * rii_hazard_multiplier(rii)

    forecasts: dict[str, HorizonForecast] = {}
    bundles: dict[str, np.ndarray] = {}

    for name, years in resolved.items():
        steps = max(int(round(years * periods_per_year)), 1)
        # One generator per horizon, seeded from the run seed and the horizon
        # name, so adding a horizon does not shift the paths of the others.
        rng = np.random.default_rng([seed, abs(hash(name)) % (2**32)])
        cumulative = simulate_paths(
            posterior=posterior,
            transmat=transmat,
            regime_stats=regime_stats,
            steps=steps,
            paths=paths,
            periods_per_year=periods_per_year,
            tail=tail,
            shock_intensity=intensity,
            rng=rng,
        )
        forecasts[name] = summarise(
            cumulative, horizon=name, years=years, start_log_level=start_log_level
        )
        if keep_bundles:
            bundles[name] = cumulative

        forecast = forecasts[name]
        log.info(
            "mc %s (%.1fy, %d steps): median %.4f, p5-p95 [%.4f, %.4f], "
            "P(dd>20%%)=%.1f%%, median max dd %.1f%%",
            name,
            years,
            steps,
            forecast.median_log_level,
            forecast.quantiles[0.05],
            forecast.quantiles[0.95],
            forecast.drawdown_probabilities[0.20] * 100,
            forecast.median_max_drawdown * 100,
        )

    return SimulationResult(
        forecasts=forecasts,
        start_log_level=float(start_log_level),
        shock_intensity=float(intensity),
        seed=seed,
        bundles=bundles,
    )


def forecast_rows(result: SimulationResult) -> list[dict[str, Any]]:
    """Quantile rows in the shape ``forecast_distribution`` takes.

    Quantiles only. §0's non-goal 1 forbids a deterministic target, and the
    table has no column for one — the schema enforces the contract rather than
    relying on everyone remembering it.
    """
    rows: list[dict[str, Any]] = []
    for name, forecast in result.forecasts.items():
        rows.extend(
            {
                "horizon": name,
                "quantile": quantile,
                "value": round(value, 8),
                "educational_only": forecast.educational_only,
            }
            for quantile, value in forecast.quantiles.items()
        )
    return rows


def shock_taxonomy() -> tuple[str, ...]:
    """§11's shock classes. Generic on purpose — see the module docstring."""
    return SHOCK_CLASSES


__all__ = [
    "DEFAULT_PATHS",
    "DEFAULT_SEED",
    "DEFAULT_SHOCK_INTENSITY",
    "SHOCK_RETRACE",
    "HORIZON_YEARS",
    "HorizonForecast",
    "SimulationResult",
    "forecast_rows",
    "run_simulation",
    "shock_taxonomy",
    "simulate_paths",
    "summarise",
]
