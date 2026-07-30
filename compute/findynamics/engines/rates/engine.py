"""FinRates — the interest-rate engine (Phase 1).

Pipeline: point-in-time Treasury quotes -> ragged yield curve per date ->
Nelson-Siegel factors -> rule-based regime -> :class:`AssetState`.

What this engine does **not** do is forecast rates. ``expected_return`` is not a
prediction that the 10y will return that much; it is what a 10y constant-maturity
position earns over the next year *if the curve does not move*, which is a
statement about today's curve and nothing else. Every other output is a state, a
score or a signal.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from findynamics.core.artifacts import ArtifactStore
from findynamics.core.config import SeriesConfig, get_series_config
from findynamics.core.contracts.state import AssetState, EngineOutput, Signal, WorldState
from findynamics.core.engine import AssetEngine
from findynamics.core.registry import register_engine
from findynamics.engines.rates import curve as curve_mod
from findynamics.engines.rates import nelson_siegel as ns
from findynamics.engines.rates import regime as regime_mod
from findynamics.engines.rates.curve import YieldCurve
from findynamics.engines.rates.domain import RATE_REGIMES, regime_code

log = logging.getLogger("findynamics.engines.rates")

MODEL_VERSION = "rates-1.0.0"

#: Trading days per year, for annualizing the level-factor volatility.
TRADING_DAYS = 252


@dataclass(frozen=True)
class RatesAnalysis:
    """Everything one run derives from the curve history, computed once."""

    lambda_: float
    #: Per-date factors: ns_level, ns_slope, ns_curvature, ns_rmse, n_tenors, spread.
    factors: pd.DataFrame
    #: Per-date regime label, causally classified.
    regimes: pd.Series
    #: Newest usable curve in the information set.
    latest: YieldCurve
    #: Fit of :attr:`latest`.
    fit: ns.NelsonSiegelFit
    rules: regime_mod.RegimeRules
    n_configured_tenors: int


def factor_history(
    frame: pd.DataFrame,
    *,
    lambda_: float,
    min_tenors: int,
    short_years: float,
    long_years: float,
) -> pd.DataFrame:
    """Fit Nelson-Siegel to every date in ``frame`` and tabulate the factors.

    One independent cross-sectional fit per date — no smoothing across dates, no
    state carried forward. That is what makes the whole history replayable: the
    value on any row depends only on that row's quotes and the frozen lambda.
    """
    rows: list[dict[str, float]] = []
    index: list[pd.Timestamp] = []

    maturities = np.asarray(frame.columns, dtype=float)
    for key, values in zip(frame.index, frame.to_numpy(dtype=float), strict=True):
        fit = ns.fit_curve(maturities, values, lambda_=lambda_, min_tenors=min_tenors)
        if fit is None:
            continue
        index.append(key)
        rows.append(
            {
                "ns_level": fit.level,
                "ns_slope": fit.slope,
                "ns_curvature": fit.curvature,
                "ns_rmse": fit.rmse,
                "n_tenors": float(fit.n_tenors),
                "spread": float(fit.yield_at(long_years)) - float(fit.yield_at(short_years)),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["ns_level", "ns_slope", "ns_curvature", "ns_rmse", "n_tenors", "spread"],
            index=pd.DatetimeIndex([], name="obs_date"),
        )
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="obs_date"))


def par_bond_duration(yield_pct: float, years: float) -> float:
    """Modified duration of a par bond, the standard closed form ``(1-(1+y)^-n)/y``.

    Used as the duration of a constant-maturity position, which is what the
    CMT series describe: a par bond of that maturity, refreshed continuously.
    At a zero yield the expression is 0/0 and the limit is ``n``.
    """
    if years <= 0:
        return 0.0
    y = yield_pct / 100.0
    if abs(y) < 1e-9:
        return float(years)
    duration = (1.0 - (1.0 + y) ** -years) / y
    return float(duration) if math.isfinite(duration) else float(years)


def carry_and_rolldown(
    fit: ns.NelsonSiegelFit,
    *,
    position_years: float,
    horizon_years: float,
) -> tuple[float, float]:
    """Expected 12m total return of a constant-maturity position, decomposed.

    A 10y position held one year is a 9y bond by then. If the curve is unchanged,
    two things happen and nothing else:

    * **carry** — it earns its coupon, approximated by the par yield ``y(n)``;
    * **rolldown** — it is repriced off the 9y point instead of the 10y one, a
      yield change of ``y(n-h) - y(n)``, worth ``-D * dy`` in price.

    So ``return ~= y(n) + D(n-h) * [y(n) - y(n-h)]``, with ``D`` the modified
    duration of the shorter par bond. Returned as (carry, rolldown) in percent.
    Both legs are kept apart because on an inverted curve they have opposite
    signs and the total alone hides that.
    """
    start = float(fit.yield_at(position_years))
    end_maturity = max(position_years - horizon_years, 1e-6)
    end = float(fit.yield_at(end_maturity))
    duration = par_bond_duration(end, end_maturity)
    return start, duration * (start - end)


@register_engine
class RatesEngine(AssetEngine):
    """Nelson-Siegel curve factors and rate regimes."""

    name: ClassVar[str] = "rates"
    version: ClassVar[str] = MODEL_VERSION

    def __init__(
        self,
        config: SeriesConfig | None = None,
        artifacts: ArtifactStore | None = None,
    ) -> None:
        self._config = config or get_series_config()
        self._artifacts = artifacts or ArtifactStore()
        self._cache: tuple[object, RatesAnalysis] | None = None

    # -- configuration ----------------------------------------------------

    @property
    def params(self) -> dict[str, Any]:
        engine = self._config.engines.get(self.name)
        return dict(engine.params) if engine else {}

    @property
    def tenors(self) -> dict[str, float]:
        return curve_mod.tenor_map(self.params)

    @property
    def breakeven_series(self) -> str | None:
        value = self.params.get("breakeven_series")
        return str(value) if value else None

    def required_series(self) -> tuple[str, ...]:
        ids = list(self.tenors)
        breakeven = self.breakeven_series
        if breakeven:
            ids.append(breakeven)
        return tuple(sorted(ids))

    def _ns_params(self) -> dict[str, Any]:
        raw = self.params.get("nelson_siegel") or {}
        return raw if isinstance(raw, dict) else {}

    @property
    def min_tenors(self) -> int:
        return int(self._ns_params().get("min_tenors", ns.MIN_TENORS))

    @property
    def lambda_(self) -> float:
        """The frozen shape parameter: whatever the last refit chose, else config.

        Falling back to the configured default rather than refitting on the spot
        is the point — a daily run must never move lambda, because that would
        silently rebase every factor history it publishes.
        """
        stored = self._artifacts.load(self.name).get("lambda")
        if isinstance(stored, int | float) and float(stored) > 0:
            return float(stored)
        return float(self._ns_params().get("lambda_default", ns.DEFAULT_LAMBDA))

    # -- analysis ---------------------------------------------------------

    def analyze(self, world: WorldState) -> RatesAnalysis | None:
        """Curve history, factors and regimes for one information set.

        Memoized on the accessor identity: ``predict`` and ``outputs`` are called
        back to back with the same world, and fitting 25 years of curves twice
        per run is pure waste.
        """
        if self._cache is not None and self._cache[0] is world.series:
            return self._cache[1]

        tenors = self.tenors
        rules = regime_mod.RegimeRules.from_params(self.params)
        frame = curve_mod.curve_frame(world.series, tenors)
        if frame.empty:
            log.warning("rates: no curve observations knowable at %s", world.as_of)
            return None

        lambda_ = self.lambda_
        factors = factor_history(
            frame,
            lambda_=lambda_,
            min_tenors=self.min_tenors,
            short_years=rules.spread_short_years,
            long_years=rules.spread_long_years,
        )
        if factors.empty:
            log.warning("rates: no date had enough tenors to fit as of %s", world.as_of)
            return None

        latest = curve_mod.latest_curve(frame, min_tenors=self.min_tenors)
        fit = (
            ns.fit_curve(
                latest.maturities, latest.yields, lambda_=lambda_, min_tenors=self.min_tenors
            )
            if latest is not None
            else None
        )
        if latest is None or fit is None:
            return None

        analysis = RatesAnalysis(
            lambda_=lambda_,
            factors=factors,
            regimes=regime_mod.classify_history(factors, rules),
            latest=latest,
            fit=fit,
            rules=rules,
            n_configured_tenors=len(tenors),
        )
        self._cache = (world.series, analysis)
        return analysis

    # -- AssetEngine ------------------------------------------------------

    def fit(self, world: WorldState) -> None:
        """Select and freeze lambda over the expanding window (monthly cadence).

        Expanding, not rolling: the training set is every curve knowable at
        ``world.as_of`` (§14.1 rule 4). The chosen value is written to the
        artifact store and picked up by every subsequent ``predict``.
        """
        tenors = self.tenors
        frame = curve_mod.curve_frame(world.series, tenors)
        if frame.empty:
            log.warning("rates.fit: nothing to fit on as of %s", world.as_of)
            return

        maturities = np.asarray(frame.columns, dtype=float)
        curves = [
            (maturities, row)
            for row in frame.to_numpy(dtype=float)
            if int(np.isfinite(row).sum()) >= self.min_tenors
        ]
        grid = self._ns_params().get("lambda_grid") or [ns.DEFAULT_LAMBDA]
        chosen, scores = ns.select_lambda(curves, grid, min_tenors=self.min_tenors)

        self._artifacts.save(
            self.name,
            {
                "lambda": chosen,
                "lambda_grid_rmse": {str(k): v for k, v in sorted(scores.items())},
                "fitted_as_of": world.as_of.isoformat(),
                "training_curves": len(curves),
                "model_version": self.version,
            },
        )
        self._cache = None
        log.info(
            "rates.fit: lambda=%.3f over %d curves through %s", chosen, len(curves), world.as_of
        )

    def predict(self, world: WorldState) -> AssetState:
        """Today's curve state. Pure function of (frozen lambda, world)."""
        analysis = self.analyze(world)
        if analysis is None:
            raise InsufficientCurveDataError(
                f"rates: no usable yield curve knowable at {world.as_of}; "
                "backfill the DGS series before running the engine"
            )

        fit = analysis.fit
        rules = analysis.rules
        inputs = regime_mod.build_inputs(
            analysis.factors, pd.Timestamp(analysis.latest.as_of), rules
        )
        if inputs is None:
            raise InsufficientCurveDataError(f"rates: cannot classify the curve at {world.as_of}")

        label = regime_mod.classify(inputs, rules)
        carry, rolldown = self._carry_and_rolldown(fit)
        duration, level_vol, risk_score = self._risk(analysis, fit)
        real_rate = self._real_rate(world, fit)

        components = {
            "ns_level": round(fit.level, 6),
            "ns_slope": round(fit.slope, 6),
            "ns_curvature": round(fit.curvature, 6),
            "ns_rmse": round(fit.rmse, 6),
            "ns_lambda": round(analysis.lambda_, 6),
            "spread_10y_3m": round(inputs.spread, 6),
            "n_tenors": float(fit.n_tenors),
            "duration_years": round(duration, 6),
            "carry_pct": round(carry, 6),
            "rolldown_pct": round(rolldown, 6),
            "level_vol_annual_pct": round(level_vol, 6),
        }
        if real_rate is not None:
            components["real_rate_10y"] = round(real_rate, 6)

        return AssetState(
            asset=self.name,
            as_of=analysis.latest.as_of,
            regime=label,
            # Decimal fraction, annualized: 0.043 is 4.3%.
            expected_return=round((carry + rolldown) / 100.0, 8),
            risk_score=risk_score,
            confidence=self._confidence(analysis, fit),
            signals=self._signals(analysis, inputs, real_rate),
            model_version=self.version,
            components=components,
        )

    def outputs(self, world: WorldState) -> tuple[EngineOutput, ...]:
        """Per-date NS factors and regime codes over the configured window."""
        analysis = self.analyze(world)
        if analysis is None:
            return ()

        window = int((self.params.get("outputs") or {}).get("history_days", 1825))
        cutoff = pd.Timestamp(analysis.latest.as_of) - pd.Timedelta(days=window)
        factors = analysis.factors.loc[cutoff:]
        regimes = analysis.regimes.loc[cutoff:] if not analysis.regimes.empty else analysis.regimes

        rows: list[EngineOutput] = []
        for metric in ("ns_level", "ns_slope", "ns_curvature", "ns_rmse"):
            column = factors[metric]
            rows.extend(
                EngineOutput(
                    asset=self.name,
                    metric=metric,
                    as_of=key.date(),
                    value=round(float(value), 6),
                )
                for key, value in column.items()
                if np.isfinite(value)
            )

        # engine_output stores REALs, so the regime travels as its index in the
        # vocabulary with the label itself in `meta` for anything reading it.
        rows.extend(
            EngineOutput(
                asset=self.name,
                metric="regime_code",
                as_of=key.date(),
                value=float(regime_code(label)),
                meta={"regime": label},
            )
            for key, label in regimes.items()
            if isinstance(label, str)
        )
        return tuple(rows)

    # -- internals --------------------------------------------------------

    def _carry_and_rolldown(self, fit: ns.NelsonSiegelFit) -> tuple[float, float]:
        params = self.params.get("expected_return") or {}
        return carry_and_rolldown(
            fit,
            position_years=float(params.get("position_years", 10.0)),
            horizon_years=float(params.get("horizon_years", 1.0)),
        )

    def _risk(self, analysis: RatesAnalysis, fit: ns.NelsonSiegelFit) -> tuple[float, float, float]:
        """Duration risk as (duration, annualized level vol, 0-100 score).

        The score is the position's annualized return volatility — duration times
        rate volatility — expressed against a configured reference. Duration
        alone would call 2020 and 2022 equally risky at the same maturity, which
        is exactly backwards.

        The annualization factor is configuration, not a constant: the engine
        must not assume its input arrives daily. Scaling month-end observations
        by sqrt(252) would overstate volatility by a factor of about 4.6.
        """
        params = self.params.get("risk") or {}
        reference = float(params.get("vol_reference_pct", 12.0))
        window = int(params.get("vol_window_days", TRADING_DAYS))
        fallback = float(params.get("vol_fallback_pct", 0.9))
        periods_per_year = float(params.get("vol_periods_per_year", TRADING_DAYS))

        position_years = float(
            (self.params.get("expected_return") or {}).get("position_years", 10.0)
        )
        duration = par_bond_duration(float(fit.yield_at(position_years)), position_years)

        changes = analysis.factors["ns_level"].diff().dropna().tail(window)
        if len(changes) >= 20:
            level_vol = float(changes.std(ddof=0) * math.sqrt(periods_per_year))
        else:
            level_vol = fallback
        if not math.isfinite(level_vol):
            level_vol = fallback

        score = 100.0 * min(max(duration * level_vol / max(reference, 1e-9), 0.0), 1.0)
        return duration, level_vol, round(score, 4)

    def _confidence(self, analysis: RatesAnalysis, fit: ns.NelsonSiegelFit) -> float:
        """How much of the curve the fit actually explains, times how much of it we saw.

        Both halves matter: a tight fit through five tenors is a weaker statement
        than a tight fit through eleven, and a full cross-section fitted badly
        means the curve is not Nelson-Siegel-shaped that day.
        """
        floor = float(self._ns_params().get("rmse_confidence_floor", 0.35))
        quality = 1.0 - min(fit.rmse / max(floor, 1e-9), 1.0)
        coverage = min(fit.n_tenors / max(analysis.n_configured_tenors, 1), 1.0)
        return round(min(max(quality * coverage, 0.0), 1.0), 4)

    def _real_rate(self, world: WorldState, fit: ns.NelsonSiegelFit) -> float | None:
        """10y nominal minus 10y breakeven, or ``None`` before TIPS data exists."""
        series_id = self.breakeven_series
        if not series_id:
            return None
        breakeven = world.series.value(series_id)
        if breakeven is None:
            return None
        return float(fit.yield_at(10.0)) - breakeven

    def _signals(
        self,
        analysis: RatesAnalysis,
        inputs: regime_mod.RegimeInputs,
        real_rate: float | None,
    ) -> tuple[Signal, ...]:
        """Directional reads. ``direction`` is +1 supportive, -1 adverse, 0 neutral."""
        rules = analysis.rules
        signals = [
            Signal(
                name="curve_inversion",
                value=round(inputs.spread, 6),
                direction=(
                    -1
                    if inputs.spread < rules.inverted_below
                    else (0 if abs(inputs.spread) < rules.flat_below else 1)
                ),
                note=(
                    f"fitted {rules.spread_long_years:g}y minus "
                    f"{rules.spread_short_years:g}y spread, percentage points"
                ),
            )
        ]

        if inputs.spread_delta is not None:
            signals.append(
                Signal(
                    name="term_premium_trend",
                    value=round(inputs.spread_delta, 6),
                    direction=_sign(inputs.spread_delta, rules.re_steepening_min_delta),
                    note=f"change in the spread over {rules.trend_days} trading days",
                )
            )

        if inputs.level_delta is not None:
            signals.append(
                Signal(
                    name="level_trend",
                    value=round(inputs.level_delta, 6),
                    # Rising rates are a headwind, so the sign is inverted.
                    direction=-_sign(inputs.level_delta, 0.25),
                    note=f"change in the NS level over {rules.trend_days} trading days",
                )
            )

        if real_rate is not None:
            signals.append(
                Signal(
                    name="real_rate_10y",
                    value=round(real_rate, 6),
                    direction=-_sign(real_rate, 0.5),
                    note="10y nominal minus 10y breakeven inflation, percentage points",
                )
            )
        return tuple(signals)


class InsufficientCurveDataError(RuntimeError):
    """The information set contains no curve this engine can speak about."""


def _sign(value: float, deadband: float) -> int:
    """-1/0/+1 with a neutral band, so noise does not read as a signal."""
    if value > deadband:
        return 1
    if value < -deadband:
        return -1
    return 0


__all__ = [
    "MODEL_VERSION",
    "RATE_REGIMES",
    "InsufficientCurveDataError",
    "RatesAnalysis",
    "RatesEngine",
    "carry_and_rolldown",
    "factor_history",
    "par_bond_duration",
]
