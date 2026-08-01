"""FinGold — the hard-asset / crisis-protection engine (Phase 4).

Pipeline: point-in-time driver panel -> Markov regime posterior -> jump intensity
and crisis premium -> hedge score -> :class:`AssetState`.

**There is no valuation step and there never will be.** Gold has no cash flow, so
a discounted value of it is not a hard number to compute — it is undefined. What
this engine models is the demand for a monetary asset with no issuer, through the
four things that move it: the real interest rate, the dollar, financial stress and
the instability of the asset it is held against.

``expected_return`` is the weakest number this engine publishes and it is labelled
as such everywhere it appears. It is the regime-conditional historical mean
monthly return, annualized — that is, "when gold has been in this state before,
it went up about this much". That is a fact about the past tense of a 600-month
sample, not a forecast, and ``confidence`` is capped accordingly. The honest
outputs here are the regime, the hedge score and the crisis premium.

FinEquity's instability index arrives as **data**, through ``WorldState.series``
under ``ENGINE:equity.rii`` — never by importing the equity engine
(``01-target-architecture.md`` §3 rule 2, enforced by ``lint-imports``). Absent
it, the panel is one driver short and the engine says so instead of stopping.
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
from findynamics.core.contracts.state import (
    AssetState,
    EngineOutput,
    RegimeProbability,
    Signal,
    WorldState,
)
from findynamics.core.engine import AssetEngine, StateUnavailable
from findynamics.core.registry import register_engine
from findynamics.engines.gold import drivers as drivers_mod
from findynamics.engines.gold import hedge as hedge_mod
from findynamics.engines.gold import jumps as jumps_mod
from findynamics.engines.gold import regime as regime_mod
from findynamics.engines.gold.domain import (
    GOLD_METRICS,
    GOLD_REGIMES,
    posterior_metric,
    regime_code,
)

log = logging.getLogger("findynamics.engines.gold")

MODEL_VERSION = "gold-1.0.0"

#: Trading days per year — annualizing the realized volatility and the drift.
TRADING_DAYS = 252
MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class GoldAnalysis:
    """Everything one run derives from the information set, computed once."""

    panel: drivers_mod.DriverPanel
    jumps: jumps_mod.JumpResult
    #: 0-1 per date. NaN before the detector can speak.
    crisis_premium: pd.Series
    hedge: hedge_mod.HedgeResult
    #: ``None`` when no fit is stored or the history is too short to filter.
    regime: regime_mod.RegimeView | None
    #: Regime posterior carried onto the daily index.
    regime_daily: pd.DataFrame
    #: Annualized realized volatility of daily gold returns, per date.
    realized_vol: pd.Series
    #: Regime-conditional mean annualized return, from the fitted sample.
    conditional_return: dict[str, float]
    rules: GoldRules
    as_of: date

    @property
    def latest_key(self) -> pd.Timestamp:
        return pd.Timestamp(self.as_of)

    @property
    def has_regime(self) -> bool:
        return self.regime is not None and not self.regime.empty


@dataclass(frozen=True)
class GoldRules:
    """The four rule blocks, loaded once per run."""

    drivers: drivers_mod.DriverRules
    regime: regime_mod.RegimeRules
    jumps: jumps_mod.JumpRules
    hedge: hedge_mod.HedgeRules

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> GoldRules:
        return cls(
            drivers=drivers_mod.DriverRules.from_params(params),
            regime=regime_mod.RegimeRules.from_params(params),
            jumps=jumps_mod.JumpRules.from_params(params),
            hedge=hedge_mod.HedgeRules.from_params(params),
        )


@register_engine
class GoldEngine(AssetEngine):
    """Regime, hedge score and crisis premium for gold."""

    name: ClassVar[str] = "gold"
    version: ClassVar[str] = MODEL_VERSION

    def __init__(
        self,
        config: SeriesConfig | None = None,
        artifacts: ArtifactStore | None = None,
    ) -> None:
        self._config = config or get_series_config()
        self._artifacts = artifacts or ArtifactStore()
        self._cache: tuple[object, GoldAnalysis | None] | None = None

    # -- configuration ----------------------------------------------------

    @property
    def params(self) -> dict[str, Any]:
        engine = self._config.engines.get(self.name)
        return dict(engine.params) if engine else {}

    @property
    def series_ids(self) -> dict[str, str]:
        """Role -> series id, from ``engines.gold.series`` in series.yaml."""
        configured = self._config.engine_series(self.name)
        missing = [role for role in drivers_mod.REQUIRED_ROLES if role not in configured]
        if missing:
            raise ValueError(
                f"series.yaml engines.gold.series is missing required role(s) {missing}; "
                f"expected all of {list(drivers_mod.REQUIRED_ROLES)}"
            )
        known = set(drivers_mod.REQUIRED_ROLES) | set(drivers_mod.OPTIONAL_ROLES)
        return {role: spec.id for role, spec in configured.items() if role in known}

    def required_series(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.series_ids.values())))

    @property
    def rules(self) -> GoldRules:
        return GoldRules.from_params(self.params)

    def _block(self, name: str) -> dict[str, Any]:
        raw = self.params.get(name) or {}
        return raw if isinstance(raw, dict) else {}

    @property
    def history_days(self) -> int:
        outputs = self._block("outputs")
        key = "backfill_history_days" if self.full_history else "history_days"
        default = 40000 if self.full_history else 3650
        return int(outputs.get(key, default))

    def stored_fit(self) -> regime_mod.MarkovFit | None:
        """The frozen chain from the last refit, or ``None``.

        ``None`` is an ordinary state, not an error: before the first
        ``monthly_refit`` there is no chain, and the engine still publishes its
        drivers, jump intensity and price history.
        """
        payload = self._artifacts.load(self.name).get("regime")
        if not isinstance(payload, dict):
            return None
        return regime_mod.MarkovFit.from_payload(payload)

    # -- analysis ---------------------------------------------------------

    def analyze(self, world: WorldState) -> GoldAnalysis | None:
        """Drivers, regime, jumps and hedge for one information set.

        Memoized on accessor identity: ``predict``, ``outputs`` and
        ``regime_states`` are called back to back with the same world, and
        filtering fifty years of monthly returns three times per run is waste.
        """
        if self._cache is not None and self._cache[0] is world.series:
            return self._cache[1]
        analysis = self._analyze(world)
        self._cache = (world.series, analysis)
        return analysis

    def _analyze(self, world: WorldState) -> GoldAnalysis | None:
        rules = self.rules
        ids = self.series_ids

        frame = world.series.wide(self.required_series())
        if frame.empty:
            log.warning("gold: no observations knowable at %s", world.as_of)
            return None

        panel = drivers_mod.build_panel(frame, ids, rules.drivers)
        if panel.empty:
            log.warning("gold: no gold price knowable at %s", world.as_of)
            return None

        daily_index = panel.daily.index
        gold_returns = panel.daily["log_price"].diff()

        detected = jumps_mod.detect(gold_returns, rules.jumps)
        premium_params = self._block("crisis")
        premium = jumps_mod.crisis_premium(
            detected.intensity.reindex(daily_index),
            panel.daily["z_stress"],
            intensity_reference=float(premium_params.get("intensity_reference", 6.0)),
            stress_weight=float(premium_params.get("stress_weight", 0.5)),
        )

        regime_view, conditional = self._regime(panel, rules)
        regime_daily = (
            regime_view.daily(daily_index)
            if regime_view is not None
            else pd.DataFrame(index=daily_index, columns=list(GOLD_REGIMES), dtype=float)
        )

        support = (
            regime_daily["hedge_bid"].fillna(0.0) + regime_daily["crisis_bid"].fillna(0.0)
            if not regime_daily.empty and regime_daily.notna().any().any()
            else pd.Series(np.nan, index=daily_index)
        )
        equity_id = ids.get("equity_proxy")
        equity = frame[equity_id] if equity_id and equity_id in frame else None
        hedge_result = hedge_mod.compute(gold_returns, equity, support, rules.hedge)

        realized_vol = gold_returns.rolling(TRADING_DAYS, min_periods=TRADING_DAYS // 4).std() * (
            math.sqrt(TRADING_DAYS) * 100.0
        )

        return GoldAnalysis(
            panel=panel,
            jumps=detected,
            crisis_premium=premium,
            hedge=hedge_result,
            regime=regime_view,
            regime_daily=regime_daily,
            realized_vol=realized_vol,
            conditional_return=conditional,
            rules=rules,
            # The newest date gold actually fixed at, which after the one-day
            # publication lag is normally the day before the cutoff.
            as_of=daily_index[-1].date(),
        )

    def _regime(
        self, panel: drivers_mod.DriverPanel, rules: GoldRules
    ) -> tuple[regime_mod.RegimeView | None, dict[str, float]]:
        """Filter the frozen chain, and take the regime-conditional means."""
        model_fit = self.stored_fit()
        if model_fit is None:
            log.info("gold: no stored regime fit; run monthly_refit before expecting a state")
            return None, {}

        try:
            view = regime_mod.posterior(panel.monthly, model_fit, rules.regime)
        except regime_mod.RegimeUnavailable as err:
            log.warning("gold: cannot filter the stored regime fit — %s", err)
            return None, {}
        if view.empty:
            return None, {}

        return view, self._conditional_returns(panel.monthly, view)

    @staticmethod
    def _conditional_returns(
        monthly: pd.DataFrame, view: regime_mod.RegimeView
    ) -> dict[str, float]:
        """Posterior-weighted mean monthly return per regime, annualized.

        A weighted mean rather than a mean over hard-labelled months, because the
        posterior is the thing the model actually believes and a 0.51 assignment
        should not count as much as a 0.99 one.
        """
        returns = monthly["ret"].reindex(view.posterior.index)
        out: dict[str, float] = {}
        for name in GOLD_REGIMES:
            weights = view.posterior[name]
            usable = weights[returns.notna()]
            total = float(usable.sum())
            if total <= 1e-9:
                continue
            monthly_mean = float((usable * returns[returns.notna()]).sum() / total)
            # Monthly log return in percent -> annualized simple return.
            out[name] = round(math.expm1(monthly_mean / 100.0 * MONTHS_PER_YEAR), 8)
        return out

    # -- AssetEngine ------------------------------------------------------

    def fit(self, world: WorldState) -> None:
        """Refit the Markov chain on the expanding window (monthly cadence).

        Expanding, never rolling: the training set is every month knowable at
        ``world.as_of`` (§14.1 rule 4). The fitted parameters are written to the
        artifact store and frozen there until the next refit — a daily run must
        never move them, or every published posterior would be conditioned on a
        different model from the one before it.
        """
        rules = self.rules
        frame = world.series.wide(self.required_series())
        if frame.empty:
            log.warning("gold.fit: nothing knowable at %s", world.as_of)
            return

        panel = drivers_mod.build_panel(frame, self.series_ids, rules.drivers)
        if panel.empty:
            log.warning("gold.fit: no gold price knowable at %s", world.as_of)
            return

        try:
            model_fit = regime_mod.fit(panel.monthly, rules.regime)
        except regime_mod.RegimeUnavailable as err:
            log.warning("gold.fit: not fitting — %s", err)
            return

        self._artifacts.save(
            self.name,
            {
                "regime": model_fit.to_payload(),
                "fitted_as_of": world.as_of.isoformat(),
                "model_version": self.version,
            },
        )
        self._cache = None
        log.info(
            "gold.fit: %d months through %s (llf %.1f)",
            model_fit.n_observations,
            model_fit.fitted_through,
            model_fit.log_likelihood,
        )

    def predict(self, world: WorldState) -> AssetState:
        """Today's gold state. Pure function of (frozen chain, world)."""
        analysis = self.analyze(world)
        if analysis is None:
            raise StateUnavailable(
                f"gold: no price history knowable at {world.as_of}; backfill "
                "LBMA:GOLD_PM before running the engine"
            )
        if not analysis.has_regime:
            raise StateUnavailable(
                "gold: no fitted regime model in the artifact store, so there is no "
                "regime to publish. The per-date outputs are still published; run "
                "jobs.monthly_refit to fit the chain."
            )

        assert analysis.regime is not None  # narrowed by has_regime
        label = analysis.regime.label() or GOLD_REGIMES[0]

        return AssetState(
            asset=self.name,
            as_of=analysis.as_of,
            regime=label,
            expected_return=analysis.conditional_return.get(label),
            risk_score=self._risk_score(analysis),
            confidence=self._confidence(analysis),
            signals=self._signals(world, analysis, label),
            model_version=self.version,
            components=self._components(world, analysis, label),
        )

    def outputs(self, world: WorldState) -> tuple[EngineOutput, ...]:
        """Per-date price, drivers, hedge score, jumps and regime posteriors."""
        analysis = self.analyze(world)
        if analysis is None:
            return ()

        cutoff = analysis.latest_key - pd.Timedelta(days=self.history_days)
        rows: list[EngineOutput] = []

        daily = analysis.panel.daily
        for metric, series in (
            ("price", daily["price"]),
            ("real_rate_10y", daily["real_rate"]),
            ("real_rate_change_12m", daily["real_rate_change_12m"]),
            ("usd_trend", daily["usd_trend"]),
            ("stress_score", daily["z_stress"]),
            ("hedge_score", analysis.hedge.score),
            ("jump_intensity", analysis.jumps.intensity),
            ("crisis_premium", analysis.crisis_premium),
        ):
            rows.extend(self._rows(metric, series, cutoff))

        for name in GOLD_REGIMES:
            if name in analysis.regime_daily:
                rows.extend(self._rows(posterior_metric(name), analysis.regime_daily[name], cutoff))

        # engine_output stores REALs, so the winning label travels as its index
        # in the vocabulary with the name itself in `meta`.
        if analysis.has_regime:
            winners = analysis.regime_daily.loc[cutoff:].dropna(how="all")
            rows.extend(
                EngineOutput(
                    asset=self.name,
                    metric="regime_code",
                    as_of=key.date(),
                    value=float(regime_code(str(row.idxmax()))),
                    meta={"regime": str(row.idxmax())},
                )
                for key, row in winners.iterrows()
            )
        return tuple(rows)

    def regime_states(self, world: WorldState) -> tuple[RegimeProbability, ...]:
        """The whole monthly posterior — §7's ``regime_state`` table.

        The distribution, not the winning label: the difference between a 0.95
        crisis call and a 0.4 one is most of what the state is saying, and a
        single label cannot carry it.
        """
        analysis = self.analyze(world)
        if analysis is None or not analysis.has_regime:
            return ()
        assert analysis.regime is not None

        cutoff = analysis.latest_key - pd.Timedelta(days=self.history_days)
        frame = analysis.regime.posterior.loc[cutoff:]
        return tuple(
            RegimeProbability(
                asset=self.name,
                as_of=key.date(),
                regime=name,
                probability=round(float(value), 8),
                model_version=self.version,
            )
            for key, row in frame.iterrows()
            for name, value in row.items()
            if np.isfinite(value)
        )

    # -- internals --------------------------------------------------------

    def _rows(self, metric: str, series: pd.Series, cutoff: pd.Timestamp) -> list[EngineOutput]:
        if series is None or series.empty:
            return []
        window = series.loc[cutoff:].dropna()
        return [
            EngineOutput(
                asset=self.name,
                metric=metric,
                as_of=key.date(),
                value=round(float(value), 8),
            )
            for key, value in window.items()
            if np.isfinite(value)
        ]

    def _risk_score(self, analysis: GoldAnalysis) -> float:
        """0-100 from realized volatility and jump intensity.

        Two terms because gold's risk is not Gaussian and a volatility number
        alone understates it: the 1980 top, the 2013 collapse and March 2020 were
        all jump events whose damage happened in two or three sessions, and a
        252-day standard deviation reports those as a mild elevation weeks after
        the fact. Volatility carries the ordinary risk, the jump intensity
        carries the part that arrives all at once.

        The reference volatility is what maps to 100 on the shared axis. 30%
        annualized is roughly gold in 1980 and in the worst of 2008 — the scale
        is calibrated on the asset, because a 0-100 axis shared with equities
        would put gold's ordinary state near zero and say nothing.
        """
        params = self._block("risk")
        vol_reference = float(params.get("vol_reference_pct", 30.0))
        jump_reference = float(params.get("jump_intensity_reference", 6.0))
        vol_weight = float(params.get("vol_weight", 0.7))
        jump_weight = float(params.get("jump_weight", 0.3))

        vol = analysis.realized_vol.dropna()
        vol_term = (
            min(float(vol.iloc[-1]) / max(vol_reference, 1e-9), 1.0)
            if not vol.empty
            else float(params.get("vol_fallback", 0.5))
        )
        intensity = analysis.jumps.latest_intensity()
        jump_term = min((intensity or 0.0) / max(jump_reference, 1e-9), 1.0)

        score = 100.0 * (vol_weight * vol_term + jump_weight * jump_term)
        return round(min(max(score, 0.0), 100.0), 4)

    def _confidence(self, analysis: GoldAnalysis) -> float:
        """How much of the state rests on its intended inputs.

        Capped well below 1 by construction. The ceiling is the honest part: the
        published ``expected_return`` is a regime-conditional historical mean and
        the regime itself is half a fitted chain and half two configured gates.
        Neither is the kind of thing that earns a confidence of 0.9, and the
        portfolio layer weights by this number.
        """
        params = self._block("confidence")
        confidence = float(params.get("ceiling", 0.7))

        missing = sum(1 for present in analysis.panel.available.values() if not present)
        confidence -= missing * float(params.get("missing_driver_penalty", 0.05))
        if not analysis.hedge.correlation_available:
            confidence -= float(params.get("no_conditional_correlation_penalty", 0.1))
        if analysis.panel.ex_post_share > 0.5:
            # Most of the fit window used realized rather than expected
            # inflation for the real rate. Defensible, and not the same driver.
            confidence -= float(params.get("ex_post_real_rate_penalty", 0.05))
        if analysis.jumps.empty:
            confidence -= float(params.get("no_jump_detector_penalty", 0.05))

        return round(min(max(confidence, 0.0), 1.0), 4)

    def _signals(self, world: WorldState, analysis: GoldAnalysis, label: str) -> tuple[Signal, ...]:
        """Directional reads. ``direction`` is +1 supportive of gold, -1 adverse."""
        signals: list[Signal] = []
        row = analysis.panel.latest()
        thresholds = self._block("signals")

        score = analysis.hedge.latest()
        if score is not None:
            good = float(thresholds.get("hedge_score_good", 60.0))
            poor = float(thresholds.get("hedge_score_poor", 40.0))
            correlation = analysis.hedge.latest_correlation()
            signals.append(
                Signal(
                    name="hedge_score",
                    value=round(score, 4),
                    direction=1 if score >= good else (-1 if score < poor else 0),
                    note=(
                        "0-100: gold's diversification of an equity drawdown, blended with the "
                        + (
                            f"regime posterior. Conditional correlation {correlation:+.2f} over "
                            f"{int(analysis.hedge.drawdown_days.iloc[-1])} drawdown days."
                            if correlation is not None
                            else "regime posterior. No conditional correlation available, so the "
                            "score is the regime term alone."
                        )
                    ),
                )
            )

        if row is not None and np.isfinite(row.get("real_rate_change_12m", np.nan)):
            change = float(row["real_rate_change_12m"])
            material = float(thresholds.get("real_rate_material_pp", 0.25))
            signals.append(
                Signal(
                    name="real_rate_headwind",
                    value=round(change, 6),
                    # Rising real rates are the headwind, so the sign flips: a
                    # positive change is an adverse read for a non-yielding asset.
                    direction=-1 if change > material else (1 if change < -material else 0),
                    note=(
                        f"12-month change in the 10y real rate, percentage points "
                        f"(level {float(row['real_rate']):+.2f}%). Rising real rates raise the "
                        "opportunity cost of holding an asset that yields nothing."
                    ),
                )
            )

        premium = analysis.crisis_premium.dropna()
        if not premium.empty:
            value = float(premium.iloc[-1])
            elevated = float(thresholds.get("crisis_premium_elevated", 0.4))
            intensity = analysis.jumps.latest_intensity() or 0.0
            signals.append(
                Signal(
                    name="crisis_premium",
                    value=round(value, 6),
                    # A crisis bid is supportive of gold and adverse for the
                    # portfolio holding it — this is the gold-side read.
                    direction=1 if value >= elevated else 0,
                    note=(
                        f"0-1, from a jump intensity of {intensity:.1f} detected jumps per year "
                        "lifted by financial stress. Not a probability of a crisis: it is how "
                        "much of one is already being paid for."
                    ),
                )
            )

        if not analysis.panel.available.get("equity_rii", False):
            signals.append(
                Signal(
                    name="equity_rii_absent",
                    value=1.0,
                    direction=0,
                    note=(
                        "FinEquity's instability index is not in the information set, so the "
                        "driver panel is running without the cross-asset instability read. "
                        "Expected before the equity engine's first publish."
                    ),
                )
            )
        return tuple(signals)

    #: Layer 0 factors that read the same forces this engine's drivers do. Named
    #: here rather than in ``core`` because *which* shared factors are relevant to
    #: gold is a fact about gold.
    SHARED_FACTORS: ClassVar[tuple[str, ...]] = ("real_rate", "usd_strength", "liquidity")

    @staticmethod
    def _shared_factors(world: WorldState) -> dict[str, float]:
        """Layer 0's reading of the same forces, published beside the engine's own.

        Not an input to anything, and deliberately so. The shared factors are
        0-100 percentiles on the risk-supportive axis, while the driver panel is
        in percentage points and standard deviations — subtracting one from the
        other is a type error, which is why this engine computes its own drivers
        rather than consuming the scores.

        They are published because they are the cross-check: ``real_rate``,
        ``usd_strength`` and ``liquidity`` are built from overlapping series by an
        independent pipeline, so a run where Layer 0 says conditions are benign
        and this engine says stress is elevated is worth someone looking at. The
        page shows them side by side and the disagreement is the information.
        """
        out: dict[str, float] = {}
        for name in GoldEngine.SHARED_FACTORS:
            score = world.factor_score(name)
            if score is not None and math.isfinite(score):
                out[f"factor_{name}"] = round(float(score), 4)
        return out

    def _components(
        self, world: WorldState, analysis: GoldAnalysis, label: str
    ) -> dict[str, float]:
        """The explainability trace: drivers, regime parts, jumps and hedge."""
        components = analysis.panel.explain()
        components["regime_code"] = float(regime_code(label))
        components.update(self._shared_factors(world))

        if analysis.regime is not None:
            latest = analysis.regime.latest()
            if latest is not None:
                for name, value in latest.items():
                    components[posterior_metric(str(name))] = round(float(value), 6)
            for key, series in (
                ("markov_violent_probability", analysis.regime.violent_probability),
                ("stress_gate", analysis.regime.stress_gate),
                ("carry_gate", analysis.regime.carry_gate),
            ):
                clean = series.dropna()
                if not clean.empty:
                    components[key] = round(float(clean.iloc[-1]), 6)

        score = analysis.hedge.latest()
        if score is not None:
            components["hedge_score"] = round(score, 4)
        correlation = analysis.hedge.latest_correlation()
        if correlation is not None:
            components["conditional_correlation"] = round(correlation, 6)

        intensity = analysis.jumps.latest_intensity()
        if intensity is not None:
            components["jump_intensity"] = round(intensity, 6)
        premium = analysis.crisis_premium.dropna()
        if not premium.empty:
            components["crisis_premium"] = round(float(premium.iloc[-1]), 6)
        vol = analysis.realized_vol.dropna()
        if not vol.empty:
            components["realized_vol_annual_pct"] = round(float(vol.iloc[-1]), 6)

        # Labelled loudly, because a number called `expected_return` on a
        # dashboard is read as a forecast unless something says otherwise.
        conditional = analysis.conditional_return.get(label)
        if conditional is not None:
            components["expected_return_is_historical_mean"] = 1.0
            components["regime_conditional_mean_return"] = round(conditional, 8)

        components["ex_post_real_rate_share"] = round(analysis.panel.ex_post_share, 6)
        return components


__all__ = [
    "GOLD_METRICS",
    "GOLD_REGIMES",
    "MODEL_VERSION",
    "TRADING_DAYS",
    "GoldAnalysis",
    "GoldEngine",
    "GoldRules",
]
