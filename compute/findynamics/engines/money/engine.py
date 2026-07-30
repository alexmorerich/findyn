"""FinMoney — the risk-free / time-value engine (Phase 2).

Pipeline: point-in-time short-rate path -> money-market account -> trailing carry
and discount factors -> liquidity state -> :class:`AssetState`.

**No ML, no regression, nothing fitted.** ``fit`` is a genuine no-op and says so.
That is not a gap to be filled later: the numeraire is arithmetic on observed
rates, and a model here would be a model of something already measured. Every
number this engine publishes is either an observation, an integral of
observations, or a threshold comparison whose thresholds are in yaml.

Its job is to be the thing the other engines discount against. FinRates says what
the curve is; FinMoney says what a dollar is worth, which is the question a
present value asks. It reads FinRates' curve to answer it past a year, but as
published data through ``WorldState.series`` — never by importing it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from findynamics.core.artifacts import ArtifactStore
from findynamics.core.config import SeriesConfig, get_series_config
from findynamics.core.contracts.state import AssetState, EngineOutput, Signal, WorldState
from findynamics.core.contracts.vocab import DISCOUNT_HORIZONS
from findynamics.core.engine import AssetEngine
from findynamics.core.registry import register_engine
from findynamics.engines.money import account as account_mod
from findynamics.engines.money import discount as discount_mod
from findynamics.engines.money import liquidity as liquidity_mod
from findynamics.engines.money.account import InsufficientRatePathError, RatePath
from findynamics.engines.money.discount import DiscountCurve, NelsonSiegelFactors
from findynamics.engines.money.domain import (
    CARRY_WINDOWS,
    MONEY_METRICS,
    MONEY_REGIMES,
    liquidity_code,
)

log = logging.getLogger("findynamics.engines.money")

MODEL_VERSION = "money-1.0.0"

#: Roles this engine resolves against ``engines.money.series`` in series.yaml.
#: Naming roles rather than series ids is what lets the short rate be re-pointed
#: at a different series without touching Python.
REQUIRED_ROLES = ("short_rate_primary", "bill_3m", "rrp_volume")
OPTIONAL_ROLES = (
    "short_rate_fallback",
    "ns_level",
    "ns_slope",
    "ns_curvature",
    "ns_lambda",
)

#: Horizons published per date to ``engine_output``. The full grid lives in
#: ``AssetState.components``; these three are the ones the dashboard charts.
PUBLISHED_DISCOUNT_HORIZONS = ("1y", "3y", "10y")


@dataclass(frozen=True)
class MoneyAnalysis:
    """Everything one run derives from the information set, computed once."""

    path: RatePath
    #: Cumulative value of one unit invested at ``path.start``.
    wealth: pd.Series
    #: Per-date liquidity classifier inputs.
    liquidity_inputs: pd.DataFrame
    #: Per-date liquidity state, causally classified.
    liquidity: pd.Series
    rules: liquidity_mod.LiquidityRules
    #: Newest published NS factors in the information set, if any.
    curve: NelsonSiegelFactors | None
    #: Per-date NS factors, for the discount-factor histories.
    curve_history: pd.DataFrame
    #: Newest date the engine can speak about.
    as_of: date

    @property
    def latest_key(self) -> pd.Timestamp:
        return pd.Timestamp(self.as_of)


@register_engine
class MoneyEngine(AssetEngine):
    """Money-market account, discount factors and liquidity state."""

    name: ClassVar[str] = "money"
    version: ClassVar[str] = MODEL_VERSION

    def __init__(
        self,
        config: SeriesConfig | None = None,
        artifacts: ArtifactStore | None = None,
    ) -> None:
        self._config = config or get_series_config()
        # Accepted and unused: this engine fits nothing, and the uniform
        # constructor is what lets the registry build any engine the same way.
        self._artifacts = artifacts or ArtifactStore()
        self._cache: tuple[object, MoneyAnalysis | None] | None = None

    # -- configuration ----------------------------------------------------

    @property
    def params(self) -> dict[str, Any]:
        engine = self._config.engines.get(self.name)
        return dict(engine.params) if engine else {}

    @property
    def series_ids(self) -> dict[str, str]:
        """Role -> series id, from ``engines.money.series`` in series.yaml."""
        configured = self._config.engine_series(self.name)
        missing = [role for role in REQUIRED_ROLES if role not in configured]
        if missing:
            raise ValueError(
                f"series.yaml engines.money.series is missing required role(s) {missing}; "
                f"expected all of {list(REQUIRED_ROLES)}"
            )
        return {
            role: spec.id
            for role, spec in configured.items()
            if role in REQUIRED_ROLES or role in OPTIONAL_ROLES
        }

    def required_series(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.series_ids.values())))

    @property
    def curve_ids(self) -> dict[str, str]:
        """The four NS roles, under the names ``discount.py`` expects.

        Empty when the curve series are not configured at all, which is how a
        deployment can deliberately run the numeraire without the rates read-back.
        """
        ids = self.series_ids
        mapping = {
            "level": ids.get("ns_level"),
            "slope": ids.get("ns_slope"),
            "curvature": ids.get("ns_curvature"),
            "lambda": ids.get("ns_lambda"),
        }
        if any(value is None for value in mapping.values()):
            return {}
        return {role: value for role, value in mapping.items() if value is not None}

    def _block(self, name: str) -> dict[str, Any]:
        raw = self.params.get(name) or {}
        return raw if isinstance(raw, dict) else {}

    @property
    def day_count_basis(self) -> float:
        return float(self._block("account").get("day_count_basis", account_mod.DAY_COUNT_BASIS))

    @property
    def history_days(self) -> int:
        return int(self._block("outputs").get("history_days", 1825))

    # -- analysis ---------------------------------------------------------

    def analyze(self, world: WorldState) -> MoneyAnalysis | None:
        """Rate path, wealth index, liquidity and curve for one information set.

        Memoized on accessor identity: ``predict`` and ``outputs`` are called back
        to back with the same world, and compounding the whole path twice per run
        is pure waste.
        """
        if self._cache is not None and self._cache[0] is world.series:
            return self._cache[1]

        analysis = self._analyze(world)
        self._cache = (world.series, analysis)
        return analysis

    def _analyze(self, world: WorldState) -> MoneyAnalysis | None:
        ids = self.series_ids
        account_params = self._block("account")

        frame = world.series.wide(self.required_series())
        if frame.empty:
            log.warning("money: no observations knowable at %s", world.as_of)
            return None

        path = account_mod.splice_rate_path(
            frame,
            primary=ids["short_rate_primary"],
            fallback=ids.get("short_rate_fallback"),
            fallback_is_discount=bool(account_params.get("fallback_is_discount", True)),
            bill_term_days=int(account_params.get("bill_term_days", account_mod.BILL_TERM_DAYS)),
        )
        if path is None:
            log.warning("money: no short-rate observation knowable at %s", world.as_of)
            return None

        wealth = account_mod.wealth_index(
            path,
            basis=self.day_count_basis,
            max_gap_days=int(
                account_params.get("max_accrual_gap_days", account_mod.MAX_ACCRUAL_GAP_DAYS)
            ),
        )

        rules = liquidity_mod.LiquidityRules.from_params(self.params)
        liquidity_inputs = liquidity_mod.build_inputs(
            frame,
            rules,
            bill_id=ids["bill_3m"],
            overnight_id=ids["short_rate_primary"],
            rrp_id=ids["rrp_volume"],
        )
        liquidity = liquidity_mod.classify_history(liquidity_inputs, rules)

        curve_ids = self.curve_ids
        curve = discount_mod.read_curve_factors(world.series, curve_ids) if curve_ids else None
        curve_history = (
            discount_mod.curve_factor_history(world.series, curve_ids)
            if curve_ids
            else pd.DataFrame()
        )

        return MoneyAnalysis(
            path=path,
            wealth=wealth,
            liquidity_inputs=liquidity_inputs,
            liquidity=liquidity,
            rules=rules,
            curve=curve,
            curve_history=curve_history,
            # The rate path is the spine: the state describes the newest date on
            # which cash actually earned something knowable.
            as_of=path.end,
        )

    # -- AssetEngine ------------------------------------------------------

    def fit(self, world: WorldState) -> None:
        """Nothing to fit — deliberately, not yet.

        The money-market account has no free parameters: the day count is a
        convention, the liquidity thresholds are configuration, and the rate path
        is observed. A refit is a no-op and pretending otherwise by writing an
        artifact would suggest the daily run depends on one.
        """
        log.info(
            "money.fit: no parameters to fit (the numeraire is arithmetic on observed "
            "rates; liquidity thresholds are configuration) — nothing written for %s",
            world.as_of,
        )

    def predict(self, world: WorldState) -> AssetState:
        """Today's numeraire state. Pure function of the world."""
        analysis = self.analyze(world)
        if analysis is None:
            raise InsufficientRatePathError(
                f"money: no short-rate path knowable at {world.as_of}; backfill "
                "FRED:SOFR and FRED:DTB3 before running the engine"
            )

        inputs = self._latest_inputs(analysis)
        regime = liquidity_mod.classify(inputs, analysis.rules)

        short_rate_pct = analysis.path.latest
        curve = discount_mod.build_curve(short_rate_pct, analysis.curve)
        carry = self._carry(analysis)

        components = self._components(analysis, curve, carry, inputs, regime)
        return AssetState(
            asset=self.name,
            as_of=analysis.as_of,
            regime=regime,
            # Decimal fraction, annualized: 0.0431 is 4.31%. This is what cash is
            # earning right now, not a forecast of what it will earn.
            expected_return=round(short_rate_pct / 100.0, 8),
            risk_score=self._risk_score(inputs),
            confidence=self._confidence(analysis, inputs, curve),
            signals=self._signals(world, analysis, inputs, carry, curve),
            model_version=self.version,
            components=components,
        )

    def outputs(self, world: WorldState) -> tuple[EngineOutput, ...]:
        """Per-date wealth index, carry, discount factors and liquidity codes."""
        analysis = self.analyze(world)
        if analysis is None:
            return ()

        cutoff = analysis.latest_key - pd.Timedelta(days=self.history_days)
        rows: list[EngineOutput] = []

        wealth = analysis.wealth.loc[cutoff:]
        # The base is where the *current* accrual segment started, which after a
        # data gap is not the start of the path. Publishing the path start there
        # would caption the chart with a date the dollar was not invested on.
        base_date = account_mod.accrual_base(analysis.wealth) or analysis.path.start
        base = base_date.isoformat()
        rows.extend(
            EngineOutput(
                asset=self.name,
                metric="wealth_index",
                as_of=key.date(),
                value=round(float(value), 8),
                # The base date travels with the series: a wealth index is
                # meaningless without knowing when the dollar was invested.
                meta={"base": base},
            )
            for key, value in wealth.items()
            if np.isfinite(value)
        )

        short_rate = analysis.path.rates.loc[cutoff:]
        sources = analysis.path.sources.loc[cutoff:]
        rows.extend(
            EngineOutput(
                asset=self.name,
                metric="short_rate",
                as_of=key.date(),
                value=round(float(value), 6),
                meta={"source": str(sources.get(key, ""))},
            )
            for key, value in short_rate.items()
            if np.isfinite(value)
        )

        rows.extend(self._carry_outputs(analysis, cutoff))
        rows.extend(self._discount_outputs(analysis, cutoff))

        if "spread" in analysis.liquidity_inputs:
            spread = analysis.liquidity_inputs["spread"].loc[cutoff:].dropna()
            rows.extend(
                EngineOutput(
                    asset=self.name,
                    metric="bill_sofr_spread",
                    as_of=key.date(),
                    value=round(float(value), 6),
                )
                for key, value in spread.items()
                if np.isfinite(value)
            )

        # engine_output stores REALs, so the state travels as its index in the
        # vocabulary with the label itself in `meta` for anything reading it.
        liquidity = analysis.liquidity.loc[cutoff:] if not analysis.liquidity.empty else pd.Series()
        rows.extend(
            EngineOutput(
                asset=self.name,
                metric="liquidity_code",
                as_of=key.date(),
                value=float(liquidity_code(label)),
                meta={"liquidity": label},
            )
            for key, label in liquidity.items()
            if isinstance(label, str)
        )
        return tuple(rows)

    # -- internals --------------------------------------------------------

    def _latest_inputs(self, analysis: MoneyAnalysis) -> liquidity_mod.LiquidityInputs:
        """Classifier inputs on the newest date that actually carries a spread.

        Walks backwards rather than taking the last row outright, for the same
        reason ``latest_curve`` does in the rates engine — and here the need is
        not an edge case but the *normal* production state. SOFR carries a
        one-day publication lag and the constant-maturity bill effectively two,
        so on almost every run the newest row of the rate path has an overnight
        rate and no bill to compare it with. Taking that row would leave the
        liquidity state resting on reverse-repo alone every single day, with the
        engine's primary input sitting one row above, unused.

        Bounded by ``max_spread_staleness_days``: beyond that the spread is not
        describing today's funding market, and reverse-repo alone with a lower
        confidence is the more honest answer than a week-old dislocation read.
        """
        frame = analysis.liquidity_inputs
        if frame.empty:
            return liquidity_mod.LiquidityInputs(
                as_of=analysis.latest_key,
                spread=None,
                spread_mean=None,
                bill_change=None,
                overnight_change=None,
                rrp_level=None,
                rrp_ratio=None,
            )

        usable = frame.loc[: analysis.latest_key]
        if usable.empty:
            usable = frame.iloc[:1]

        key = usable.index[-1]
        if "spread_mean" in usable.columns:
            priced = usable[usable["spread_mean"].notna()]
            max_stale = int(
                (self.params.get("liquidity") or {}).get("max_spread_staleness_days", 5)
            )
            if not priced.empty and (analysis.latest_key - priced.index[-1]).days <= max_stale:
                key = priced.index[-1]

        return liquidity_mod.inputs_on(usable.loc[key], key)

    def _carry(self, analysis: MoneyAnalysis) -> dict[str, float | None]:
        """Trailing realized carry per configured window, annualized decimals."""
        return {
            metric: account_mod.realized_carry(
                analysis.wealth,
                days,
                asof=analysis.latest_key,
                basis=self.day_count_basis,
            )
            for metric, days in CARRY_WINDOWS.items()
        }

    def _carry_outputs(self, analysis: MoneyAnalysis, cutoff: pd.Timestamp) -> list[EngineOutput]:
        """Carry histories: each date's trailing carry, computed as of that date."""
        rows: list[EngineOutput] = []
        keys = [key for key in analysis.wealth.index if key >= cutoff]
        for metric, days in CARRY_WINDOWS.items():
            for key in keys:
                value = account_mod.realized_carry(
                    analysis.wealth, days, asof=key, basis=self.day_count_basis
                )
                if value is None or not math.isfinite(value):
                    continue
                rows.append(
                    EngineOutput(
                        asset=self.name,
                        metric=metric,
                        as_of=key.date(),
                        value=round(value, 8),
                    )
                )
        return rows

    def _discount_outputs(
        self, analysis: MoneyAnalysis, cutoff: pd.Timestamp
    ) -> list[EngineOutput]:
        """Discount-factor histories on the three charted horizons.

        Per date, the curve is that date's published NS factors where they exist
        and the flat short rate where they do not — with the source recorded, so a
        chart can show where the fitted curve starts rather than blending the two
        silently.
        """
        rows: list[EngineOutput] = []
        history = analysis.curve_history
        for key, short_rate in analysis.path.rates.loc[cutoff:].items():
            if not np.isfinite(short_rate):
                continue
            factors = self._curve_on(history, key)
            curve = discount_mod.build_curve(
                float(short_rate),
                factors,
                horizons=PUBLISHED_DISCOUNT_HORIZONS,
            )
            for horizon in PUBLISHED_DISCOUNT_HORIZONS:
                value = curve.factor(horizon)
                if value is None or not math.isfinite(value):
                    continue
                rows.append(
                    EngineOutput(
                        asset=self.name,
                        metric=f"discount_{horizon}",
                        as_of=key.date(),
                        value=round(value, 8),
                        meta={"curve_source": curve.sources.get(horizon, "short_rate")},
                    )
                )
        return rows

    @staticmethod
    def _curve_on(history: pd.DataFrame, key: pd.Timestamp) -> NelsonSiegelFactors | None:
        """Published factors on ``key`` exactly — never carried across dates.

        Reusing an older date's factors for a date the rates engine did not
        publish would put a curve that was never fitted into the history.
        """
        if history.empty or key not in history.index:
            return None
        row = history.loc[key]
        return NelsonSiegelFactors(
            level=float(row["level"]),
            slope=float(row["slope"]),
            curvature=float(row["curvature"]),
            lambda_=float(row["lambda_"]),
        )

    def _risk_score(self, inputs: liquidity_mod.LiquidityInputs) -> float:
        """Near zero by construction, and here is the scale it is near zero on.

        The shared 0-100 axis is calibrated on the other engines: 100 is roughly a
        10-year Treasury position through 2022. Overnight cash has no duration and
        no credit, so on that axis it is 0 — not "low", zero. The only thing that
        can go wrong with a money-market account is that the money market itself
        stops working, and that risk is real but bounded: it costs basis points of
        access, not tens of percent of principal.

        So the score is the funding dislocation, scaled to a configured reference
        and capped at ``ceiling`` (default 10). The cap is the honest part: however
        bad repo gets, cash is still not as risky as duration, and an uncapped
        scaling would have let September 2019 print a number that invited the
        portfolio layer to treat cash as an equity-like exposure for one day.
        """
        params = self._block("risk")
        ceiling = float(params.get("ceiling", 10.0))
        reference = float(params.get("dislocation_reference_pp", 1.0))

        if inputs.spread is None:
            return 0.0
        # Only a negative spread is a dislocation; bills cheap to repo is not.
        dislocation = max(-inputs.spread, 0.0)
        score = ceiling * min(dislocation / max(reference, 1e-9), 1.0)
        return round(min(max(score, 0.0), ceiling), 4)

    def _confidence(
        self,
        analysis: MoneyAnalysis,
        inputs: liquidity_mod.LiquidityInputs,
        curve: DiscountCurve,
    ) -> float:
        """How much of the state rests on its intended inputs rather than fallbacks.

        Three independent deductions, because they degrade three different
        outputs: the short rate itself (a spliced path is a weaker numeraire than
        an overnight one), the liquidity read (no spread means the regime rests on
        reverse-repo alone), and the long end (no published curve means the 30y
        factor is a flat extrapolation).
        """
        params = self._block("confidence")
        confidence = 1.0

        primary = self.series_ids["short_rate_primary"]
        if analysis.path.latest_source != primary:
            confidence -= float(params.get("spliced_short_rate_penalty", 0.25))
        if not inputs.has_spread:
            confidence -= float(params.get("no_spread_penalty", 0.30))
        if not curve.has_curve:
            confidence -= float(params.get("no_curve_penalty", 0.20))

        return round(min(max(confidence, 0.0), 1.0), 4)

    def _signals(
        self,
        world: WorldState,
        analysis: MoneyAnalysis,
        inputs: liquidity_mod.LiquidityInputs,
        carry: dict[str, float | None],
        curve: DiscountCurve,
    ) -> tuple[Signal, ...]:
        """Directional reads. ``direction`` is +1 supportive, -1 adverse, 0 neutral."""
        signals: list[Signal] = []

        nominal = carry.get("carry_12m")
        if nominal is None:
            nominal = analysis.path.latest / 100.0
        signals.append(self._real_carry_signal(world, nominal, carry.get("carry_12m") is not None))

        if inputs.spread is not None:
            rules = analysis.rules
            signals.append(
                Signal(
                    name="bill_sofr_spread",
                    value=round(inputs.spread, 6),
                    # Bills rich to repo is the adverse read: it is what both a
                    # funding squeeze and a flight into bills look like.
                    direction=(
                        -1
                        if inputs.spread <= rules.tightening_spread_pp
                        else (1 if inputs.spread >= 0.0 else 0)
                    ),
                    note=(
                        "3m constant-maturity bill minus the overnight rate, percentage "
                        "points; negative means bills are rich to repo"
                    ),
                )
            )

        if inputs.rrp_ratio is not None and inputs.rrp_level is not None:
            material = inputs.rrp_level >= analysis.rules.rrp_material_bn
            signals.append(
                Signal(
                    name="rrp_buffer_trend",
                    value=round(inputs.rrp_ratio, 6),
                    direction=(
                        0
                        if not material
                        else (
                            -1
                            if inputs.rrp_ratio <= analysis.rules.tightening_rrp_ratio
                            else (1 if inputs.rrp_ratio >= 1.0 else 0)
                        )
                    ),
                    note=(
                        f"reverse-repo {analysis.rules.rrp_fast_window}-day average over its "
                        f"{analysis.rules.rrp_slow_window}-day average; below 1.0 the buffer is "
                        "draining"
                        + (
                            ""
                            if material
                            else f" (uninformative below ${analysis.rules.rrp_material_bn:g}bn)"
                        )
                    ),
                )
            )

        if not curve.has_curve:
            signals.append(
                Signal(
                    name="curve_source_degraded",
                    value=1.0,
                    direction=0,
                    note=(
                        "no published Nelson-Siegel curve in the information set; horizons "
                        "beyond one year are discounted at the flat short rate"
                    ),
                )
            )
        return tuple(signals)

    def _real_carry_signal(self, world: WorldState, nominal: float, from_window: bool) -> Signal:
        """Nominal carry, oriented by what the shared inflation factor is saying.

        The spec asks for "nominal carry minus inflation factor score direction",
        and the wording hides a type error worth being explicit about: the
        inflation factor is a 0-100 percentile on the risk-supportive axis, not a
        rate, so it cannot be subtracted from a carry. What it can do is decide
        whether the carry is worth having.

        So the *value* is the nominal carry — a real number, in decimal — and the
        *direction* is the real read: positive carry with benign inflation is
        supportive, positive carry with inflation running is not. Anyone wanting a
        true real rate in percentage points should take it from FinRates'
        ``real_rate_10y``, which is built from a breakeven and means what it says.
        """
        params = self._block("real_carry")
        benign_above = float(params.get("inflation_benign_above", 55.0))
        hostile_below = float(params.get("inflation_hostile_below", 45.0))

        score = world.factor_score("inflation")
        if nominal <= 0.0:
            direction = -1
        elif score is None:
            direction = 0
        elif score >= benign_above:
            direction = 1
        elif score < hostile_below:
            direction = -1
        else:
            direction = 0

        basis = "trailing 12m realized carry" if from_window else "current short rate"
        inflation = "unavailable" if score is None else f"{score:.1f}/100"
        return Signal(
            name="real_carry",
            value=round(nominal, 8),
            direction=direction,  # type: ignore[arg-type]
            note=(
                f"{basis}, annualized decimal; direction is the read after inflation "
                f"(shared inflation factor {inflation}, where 100 is most benign)"
            ),
        )

    def _components(
        self,
        analysis: MoneyAnalysis,
        curve: DiscountCurve,
        carry: dict[str, float | None],
        inputs: liquidity_mod.LiquidityInputs,
        regime: str,
    ) -> dict[str, float]:
        """The full explainability trace, including the whole discount curve."""
        components: dict[str, float] = {
            "short_rate_pct": round(analysis.path.latest, 6),
            "wealth_index": round(float(analysis.wealth.iloc[-1]), 8),
            "rate_path_days": float(len(analysis.path)),
            "primary_rate_share": round(
                analysis.path.share_from(self.series_ids["short_rate_primary"]), 6
            ),
            "liquidity_code": float(liquidity_code(regime)),
        }

        for metric, value in carry.items():
            if value is not None and math.isfinite(value):
                components[metric] = round(value, 8)

        for horizon in DISCOUNT_HORIZONS:
            value = curve.factor(horizon)
            if value is not None and math.isfinite(value):
                components[f"discount_{horizon}"] = round(value, 8)

        if curve.curve is not None:
            components["ns_level"] = round(curve.curve.level, 6)
            components["ns_slope"] = round(curve.curve.slope, 6)
            components["ns_curvature"] = round(curve.curve.curvature, 6)
            components["ns_lambda"] = round(curve.curve.lambda_, 6)

        components.update(liquidity_mod.explain(inputs, analysis.rules))
        return components


__all__ = [
    "CARRY_WINDOWS",
    "MODEL_VERSION",
    "MONEY_METRICS",
    "MONEY_REGIMES",
    "PUBLISHED_DISCOUNT_HORIZONS",
    "InsufficientRatePathError",
    "MoneyAnalysis",
    "MoneyEngine",
]
