"""FinEquity — the S&P500 dynamic state engine (FINDYN_V1_SPEC.md, Phase 3).

Two halves, and the seam between them is the point.

**The feature path** (sub-milestone A) runs on the *publication* series and is
published two ways: ``derived_features`` carries the model inputs in model units
and is versioned by model, ``engine_output`` carries the same information in the
units a chart wants.

**The regime model** (sub-milestone B) is fitted on the *calibration* series and
applied here. That transfer is the engine's central technical claim and it rests
on the design matrix being dimensionless — :mod:`.regime.design` carries the
argument, ``test_transfer.py`` the evidence, and every state carries the
provenance in ``model_version``.

``predict`` raises :class:`StateUnavailable` rather than fabricating a state
when no fit is stored — a fresh deployment before its first refit. The features
still publish, so ``/equity`` has data while ``/assets/equity/state`` correctly
reports that no state exists.

Layering: this engine reads the money engine's discount factors and risk-free
benchmark as *data* through ``WorldState.series`` (sub-milestone C), never by
import — CI enforces the direction.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from findynamics.core.artifacts import ArtifactStore
from findynamics.core.config import SeriesConfig, get_series_config
from findynamics.core.contracts.state import (
    AssetState,
    DerivedFeature,
    EngineOutput,
    RegimeProbability,
    Signal,
    WorldState,
)
from findynamics.core.engine import AssetEngine, StateUnavailable
from findynamics.core.registry import register_engine
from findynamics.engines.equity import prices as prices_mod
from findynamics.engines.equity.domain import (
    CHART_METRICS,
    KINEMATIC_FEATURES,
    REGIMES,
)
from findynamics.engines.equity.features.kinematics import JERK_LAMP_CODES, jerk_lamp
from findynamics.engines.equity.features.pipeline import (
    FeatureParams,
    FeatureSet,
    FrozenFeatureParams,
    compute_features,
)
from findynamics.engines.equity.prices import PriceRoles, PriceSeries
from findynamics.engines.equity.regime.calibrate import (
    HORIZON_MONTHS,
    TransitionModels,
    build_training_frame,
    fit_all_horizons,
)
from findynamics.engines.equity.regime.design import DesignError, RegimeDesign, build_design
from findynamics.engines.equity.regime.hmm import HmmFit, RegimeModel, fit_hmm

log = logging.getLogger("findynamics.engines.equity")

#: The model version before the calibration tag is appended. The published
#: version is always ``<base>+cal.<slug>`` (see :meth:`EquityEngine.model_version`).
MODEL_VERSION_BASE = "equity-1.0.0"

#: Artifact document name under ``compute/artifacts/``.
ARTIFACT_NAME = "equity"

#: Roles the daily run computes features for. Only the publication path is
#: needed to publish today's features, and the calibration path is 12k
#: observations of Kalman MLE — real money on a run that will not use it.
#: ``fit`` and the sub-milestone B backtest ask for the others explicitly.
DAILY_ROLES: tuple[str, ...] = ("publication",)

#: Every role the pipeline is fitted over.
ALL_ROLES: tuple[str, ...] = ("publication", "calibration", "deep_history")


def enforce_nesting(transitions: pd.DataFrame) -> pd.DataFrame:
    """Make the horizons respect the fact that they are nested events.

    "Enters bear or crisis within 3 months" implies "within 6 months" — the
    3-month window is a subset of the 6-month one, so its probability cannot be
    larger. Three classifiers fitted independently, each with its own isotonic
    map, have no idea about that and routinely violate it: the first live state
    produced 0.892 / 0.875 / 0.916 across 3m / 6m / 12m.

    A reader cannot be expected to ignore that, and there is no interpretation
    under which it is right. A running maximum in horizon order is the minimal
    repair — it never lowers a longer horizon below a shorter one and leaves any
    already-consistent triple untouched.

    It is a repair, not a fix: the underlying miscalibration is still there and
    is recorded in the open-issues note. What this prevents is publishing an
    incoherent triple while that is outstanding.
    """
    if transitions.empty:
        return transitions

    ordered = sorted(
        transitions.columns, key=lambda name: int(name.removeprefix("p_").removesuffix("m"))
    )
    return transitions[ordered].cummax(axis=1)


@dataclass(frozen=True)
class RegimeView:
    """What the fitted regime model says about the publication path."""

    model: RegimeModel
    #: Filtered posteriors, one column per regime in vocabulary order.
    posteriors: pd.DataFrame
    #: Calibrated P(entering bear or crisis) per horizon, columns ``p_3m`` etc.
    transitions: pd.DataFrame
    #: SHAP contributions for the newest date, per horizon.
    contributions: dict[int, pd.Series]
    models: TransitionModels

    @property
    def latest(self) -> pd.Series:
        return self.posteriors.iloc[-1]

    @property
    def regime(self) -> str:
        return str(self.latest.idxmax())

    @property
    def confidence(self) -> float:
        return float(self.latest.max())

    @property
    def entropy(self) -> float:
        """Posterior entropy — the RII's uncertainty term (§3.2), published now
        so sub-milestone C composes it rather than recomputing it."""
        values = np.clip(self.latest.to_numpy(dtype=float), 1e-12, 1.0)
        return float(-(values * np.log(values)).sum())


@dataclass(frozen=True)
class EquityAnalysis:
    """Everything one run derives from the information set, computed once."""

    roles: PriceRoles
    #: Engine role -> its feature path. Keys are a subset of :data:`ALL_ROLES`.
    features: dict[str, FeatureSet]
    model_version: str
    #: ``None`` when no fitted regime model is in the artifact store yet.
    regime: RegimeView | None = None

    @property
    def publication(self) -> FeatureSet:
        return self.features["publication"]

    @property
    def as_of(self) -> date | None:
        return self.publication.as_of


@register_engine
class EquityEngine(AssetEngine):
    """Kinematics now; regimes, instability and forecasts in B and C."""

    name: ClassVar[str] = "equity"
    version: ClassVar[str] = MODEL_VERSION_BASE

    def __init__(
        self,
        config: SeriesConfig | None = None,
        artifacts: ArtifactStore | None = None,
    ) -> None:
        self._config = config or get_series_config()
        self._artifacts = artifacts or ArtifactStore()
        self._cache: tuple[object, tuple[str, ...], EquityAnalysis] | None = None

    # -- configuration ----------------------------------------------------

    @property
    def params(self) -> dict[str, Any]:
        engine = self._config.engines.get(self.name)
        return dict(engine.params) if engine else {}

    def _block(self, name: str) -> dict[str, Any]:
        raw = self.params.get(name) or {}
        return raw if isinstance(raw, dict) else {}

    @property
    def feature_params(self) -> FeatureParams:
        return FeatureParams.from_params(self.params)

    @property
    def min_observations(self) -> int:
        return int(
            self._block("prices").get("min_observations", prices_mod.DEFAULT_MIN_OBSERVATIONS)
        )

    @property
    def history_days(self) -> int:
        """How far back published histories reach. Bounds the write-back.

        Two windows, not one. The nightly run republishes ``history_days``
        because everything older is an unchanged upsert; a backfill run sets
        ``full_history`` and republishes ``backfill_history_days``, which reaches
        1927. Charting the full record needs the second to have run once — and
        needs the first to stay short, or the cron writes a century every night
        to say nothing new.
        """
        block = self._block("outputs")
        if self.full_history:
            return int(block.get("backfill_history_days", 40000))
        return int(block.get("history_days", 1825))

    def required_series(self) -> tuple[str, ...]:
        """Every configured equity price series.

        All four, not just the resolved ones: which roles are *available* is the
        question :mod:`.prices` answers, and it cannot answer it about a series
        the run never tried to load.
        """
        configured = prices_mod.configured_roles(self._config)
        return tuple(sorted({spec.id for spec in configured.values()}))

    def model_version(self, roles: PriceRoles) -> str:
        """``equity-1.0.0+cal.fred_nasdaq100`` — the fit is part of the identity."""
        return f"{MODEL_VERSION_BASE}+{roles.tag}"

    # -- analysis ---------------------------------------------------------

    def analyze(
        self,
        world: WorldState,
        *,
        roles: tuple[str, ...] = DAILY_ROLES,
    ) -> EquityAnalysis:
        """Resolve roles and build the feature paths for ``roles``.

        Memoized on (accessor identity, requested roles): ``predict``,
        ``outputs`` and ``derived_features`` are called back to back with the
        same world, and re-running the filter three times per night is waste.
        """
        if self._cache is not None and self._cache[0] is world.series and self._cache[1] == roles:
            return self._cache[2]

        analysis = self._analyze(world, roles)
        self._cache = (world.series, roles, analysis)
        return analysis

    def _analyze(self, world: WorldState, wanted: tuple[str, ...]) -> EquityAnalysis:
        try:
            resolved = prices_mod.resolve_from(
                world.series, self._config, min_observations=self.min_observations
            )
        except prices_mod.PriceRoleError as err:
            # An information set with no price backbone is thin, not broken. The
            # job layer treats the two differently, and "the backfill has not run
            # yet" belongs on the quiet side of that line — the run should carry
            # on and the endpoint should keep saying it has nothing.
            raise StateUnavailable(str(err)) from err
        stored = self._stored_params()
        params = self.feature_params

        features: dict[str, FeatureSet] = {}
        for role in wanted:
            series: PriceSeries | None = getattr(resolved, role)
            if series is None:
                continue
            path = prices_mod.price_path(world.series, series)
            if path.empty:
                log.warning("equity: %s (%s) has no knowable closes", role, series.series_id)
                continue
            try:
                features[role] = compute_features(
                    path,
                    series,
                    params=params,
                    frozen=stored.get(series.slug),
                )
            except ValueError as err:
                # A role that cannot be built is a degraded feature set, not a
                # failed run — except for the publication path, which is checked
                # below because the engine has nothing to say without it.
                log.warning("equity: %s (%s) features unavailable: %s", role, series.series_id, err)

        if "publication" in wanted and "publication" not in features:
            raise StateUnavailable(
                f"equity: no usable publication price path at {world.as_of}; "
                f"backfill {resolved.publication.series_id} before running the engine"
            )

        return EquityAnalysis(
            roles=resolved,
            features=features,
            model_version=self.model_version(resolved),
            regime=(
                self._regime_view(features["publication"]) if "publication" in features else None
            ),
        )

    def _regime_view(self, features: FeatureSet) -> RegimeView | None:
        """Apply the stored regime model to the publication path.

        This is the inference half of "fit on calibration, infer on publication".
        The parameters came from another series; the design matrix is
        dimensionless so that is sound, and :mod:`.regime.design` carries the
        argument and ``test_transfer.py`` the evidence.

        Returns ``None`` — not an error — when no fit is stored. A daily run that
        happens before the first refit should publish its features and decline
        its state, exactly as sub-milestone A did.
        """
        document = self._artifacts.load(ARTIFACT_NAME)
        stored = document.get("hmm")
        if not isinstance(stored, dict):
            return None

        try:
            model = RegimeModel(HmmFit.from_dict(stored))
            design = self._design(features)
            posteriors = model.posteriors(design)
        except (KeyError, TypeError, ValueError, DesignError) as err:
            log.warning("equity: stored regime model is unusable (%s); publishing no state", err)
            return None

        models = TransitionModels.from_dict(document.get("transitions"))
        transitions = pd.DataFrame(index=posteriors.index)
        contributions: dict[int, pd.Series] = {}

        if models:
            try:
                inputs = build_training_frame(design, posteriors)
                for months, transition_model in sorted(models.models.items()):
                    transitions[f"p_{months}m"] = transition_model.predict(inputs)
                    if not inputs.empty:
                        contributions[months] = transition_model.contributions(inputs.tail(1)).iloc[
                            0
                        ]
                transitions = enforce_nesting(transitions)
            except (KeyError, ValueError) as err:
                log.warning("equity: transition probabilities unavailable (%s)", err)

        return RegimeView(
            model=model,
            posteriors=posteriors,
            transitions=transitions,
            contributions=contributions,
            models=models,
        )

    def _design(self, features: FeatureSet) -> RegimeDesign:
        block = self._block("regime")
        return build_design(
            features,
            vol_months=float(block.get("vol_months", 1.0)),
            vol_baseline_years=float(block.get("vol_baseline_years", 3.0)),
            drawdown_years=float(block.get("drawdown_years", 1.0)),
            trend_months=float(block.get("trend_months", 3.0)),
        )

    def _stored_params(self) -> dict[str, FrozenFeatureParams]:
        """Frozen per-series parameters from the last refit, by series slug."""
        document = self._artifacts.load(ARTIFACT_NAME)
        raw = document.get("series")
        if not isinstance(raw, dict):
            return {}
        stored: dict[str, FrozenFeatureParams] = {}
        for slug, body in raw.items():
            try:
                stored[str(slug)] = FrozenFeatureParams.from_dict(body)
            except (KeyError, TypeError, ValueError) as err:
                log.warning("equity: ignoring unreadable frozen params for %s: %s", slug, err)
        return stored

    # -- AssetEngine ------------------------------------------------------

    def fit(self, world: WorldState) -> None:
        """Expanding-window refit: re-search ``d`` and re-estimate the variances.

        Runs over **every** resolved role, not just the published one. ``d`` is
        frozen per series (§8.2) because two series with different memory have no
        business sharing one, and sub-milestone B fits its regime model on the
        calibration path — which needs its own frozen parameters or the fit and
        the daily inference would be looking at differently-built features.
        """
        analysis = self.analyze(world, roles=ALL_ROLES)

        document: dict[str, Any] = {
            "model_version": analysis.model_version,
            "fitted_at": datetime.now(UTC).isoformat(),
            "as_of": world.as_of.isoformat(),
            "roles": analysis.roles.describe(),
            "series": {
                feature_set.series.slug: feature_set.frozen().as_dict()
                for feature_set in analysis.features.values()
            },
        }

        calibration = analysis.features.get("calibration")
        if calibration is not None:
            hmm, transitions = self._fit_regime(analysis, calibration)
            if hmm is not None:
                document["hmm"] = hmm.as_dict()
                document["transitions"] = transitions.as_dict()

        self._artifacts.save(ARTIFACT_NAME, document)
        log.info(
            "equity.fit: froze parameters for %d series (%s) at %s%s",
            len(analysis.features),
            ", ".join(sorted(document["series"])),
            world.as_of,
            "" if "hmm" not in document else " + regime model",
        )

    def _fit_regime(
        self, analysis: EquityAnalysis, calibration: FeatureSet
    ) -> tuple[HmmFit | None, TransitionModels]:
        """Fit the HMM and the transition classifiers on the calibration series.

        On ``calibration``, never on ``publication``: a five-state model fitted on
        ten years containing one drawdown does not have a crisis state, and the
        transition classifier would have a handful of positive episodes to learn
        from. Which series supplied the parameters is recorded on the fit and
        travels onto every state through ``model_version``.
        """
        block = self._block("regime")
        try:
            design = self._design(calibration)
            hmm = fit_hmm(
                design,
                seed=int(block.get("seed", 20260731)),
                prior_strength=float(block.get("transition_prior", 1000.0)),
                is_proxy=analysis.roles.calibration_is_proxy,
            )
        except (ValueError, DesignError) as err:
            log.warning("equity.fit: no regime model (%s)", err)
            return None, TransitionModels()

        model = RegimeModel(hmm)
        posteriors = model.posteriors(design)
        transitions = fit_all_horizons(
            design,
            posteriors,
            model.states(design),
            fitted_on=design.series.series_id,
            is_proxy=analysis.roles.calibration_is_proxy,
            horizons=tuple(int(h) for h in block.get("horizons", HORIZON_MONTHS)),
        )
        return hmm, transitions

    def predict(self, world: WorldState) -> AssetState:
        """Today's regime state, from the model fitted on the calibration series.

        Declines — rather than fabricating — when no fit is stored yet. That is
        the state of a fresh deployment before its first refit, and a placeholder
        regime on a dashboard is read as a market view.
        """
        analysis = self.analyze(world)
        view = analysis.regime
        if view is None:
            raise StateUnavailable(
                f"equity: features are published ({len(analysis.publication.frame)} rows "
                f"through {analysis.as_of}), but no fitted regime model is stored; "
                "run jobs.monthly_refit before the state can be published"
            )

        as_of = analysis.as_of or world.as_of
        return AssetState(
            asset=self.name,
            as_of=as_of,
            regime=view.regime,
            expected_return=self._expected_return(view),
            risk_score=self._risk_score(view),
            confidence=round(view.confidence, 6),
            signals=self._signals(analysis, view),
            model_version=analysis.model_version,
            components=self._components(analysis, view),
        )

    def _expected_return(self, view: RegimeView) -> float:
        """Posterior-weighted mean return of the fitted states, annualized.

        **Not** the §10 forecast — that is a Monte Carlo distribution median and
        arrives in sub-milestone C. This is the regime-conditional expectation the
        fitted model already implies: each state's realized mean return, weighted
        by today's posterior.

        It inherits the calibration series' returns, so where that is a proxy this
        number describes what the *proxy's* states earned. The caveat travels on
        ``model_version`` and in the ``fitted_on_proxy`` component rather than
        being buried here.
        """
        stats = {s.label: s.mean_return for s in view.model.fit.stats}
        weighted = sum(
            float(view.latest.get(label, 0.0)) * stats.get(label, 0.0) for label in REGIMES
        )
        return round(float(weighted), 8)

    def _risk_score(self, view: RegimeView) -> float:
        """0-100 regime severity — the RII's regime term, not the RII itself.

        Posterior-weighted position on the risk-on ordering, scaled to the shared
        axis: all weight on ``bull_expansion`` is 0, all weight on ``crisis`` is
        100. Sub-milestone C replaces this with the full §3.2 composite, of which
        this is one input; publishing the composite's name for one of its terms
        would be worse than publishing the term and saying so.
        """
        severity = sum(
            float(view.latest.get(label, 0.0)) * rank for rank, label in enumerate(REGIMES)
        )
        return round(min(max(100.0 * severity / (len(REGIMES) - 1), 0.0), 100.0), 4)

    def _signals(self, analysis: EquityAnalysis, view: RegimeView) -> tuple[Signal, ...]:
        """Calibrated transition probabilities plus the proxy caveat."""
        signals: list[Signal] = []

        for months in sorted(view.models.models):
            column = f"p_{months}m"
            if column not in view.transitions.columns:
                continue
            series = view.transitions[column].dropna()
            if series.empty:
                continue
            probability = float(series.iloc[-1])
            base = view.models.models[months].base_rate
            signals.append(
                Signal(
                    name=f"p_transition_{months}m",
                    value=round(probability, 6),
                    # Against the base rate, not against 50%: a 30% chance of a
                    # fresh deterioration inside three months is *reassuring* when
                    # the unconditional rate is 26%, and alarming at 12 months
                    # where it would be well under the 71% base.
                    direction=(
                        -1 if probability > base * 1.15 else (1 if probability < base * 0.85 else 0)
                    ),
                    note=(
                        f"calibrated P(the regime enters bear or crisis within {months}m). "
                        f"Read against the unconditional rate of {base:.0%} on "
                        f"{view.model.fit.fitted_on}, not against 50%. An entry is a "
                        "regime change, not necessarily a market collapse — the engine "
                        "changes regime a few times a year."
                    ),
                )
            )

        signals.append(
            Signal(
                name="posterior_entropy",
                value=round(view.entropy, 6),
                # High entropy is the model saying it does not know, which §3.2
                # treats as instability rather than as neutrality.
                direction=-1 if view.entropy > 1.0 else 0,
                note=(
                    "entropy of the regime posterior; 0 is certainty, "
                    f"{np.log(len(REGIMES)):.2f} is a uniform guess"
                ),
            )
        )

        if analysis.roles.calibration_is_proxy:
            signals.append(
                Signal(
                    name="regime_model_is_proxy_fitted",
                    value=1.0,
                    direction=0,
                    note=(
                        f"every fitted parameter comes from {view.model.fit.fitted_on}, "
                        "not the published index; daily S&P history before 2016 is "
                        "not reachable (see docs/backtests/equity-p3b.md)"
                    ),
                )
            )
        return tuple(signals)

    def _components(self, analysis: EquityAnalysis, view: RegimeView) -> dict[str, float]:
        """The explanation trace: the posterior, the fit, and the top SHAP terms."""
        components: dict[str, float] = {
            f"posterior_{label}": round(float(view.latest.get(label, 0.0)), 6) for label in REGIMES
        }
        components["posterior_entropy"] = round(view.entropy, 6)
        for months, transition_model in sorted(view.models.models.items()):
            # The probability is meaningless without it: 0.89 against a 0.28 base
            # is a warning, 0.89 against a 0.85 base is a shrug.
            components[f"base_rate_{months}m"] = round(transition_model.base_rate, 6)
        components["fitted_on_proxy"] = float(view.model.fit.fitted_on_proxy)
        components["fit_observations"] = float(view.model.fit.observations)

        stats = view.model.fit.stats_for(view.regime)
        if stats is not None:
            components["regime_mean_return"] = round(stats.mean_return, 6)
            components["regime_volatility"] = round(stats.volatility, 6)
            components["regime_persistence_obs"] = round(stats.persistence, 2)

        for months, contributions in sorted(view.contributions.items()):
            # §14.3: top contributions per prediction, by magnitude.
            top = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
            for feature, value in top.head(4).items():
                if math.isfinite(float(value)):
                    components[f"shap_{months}m_{feature}"] = round(float(value), 6)

        return components

    def outputs(self, world: WorldState) -> tuple[EngineOutput, ...]:
        """The publication feature path in chartable units.

        Index points rather than logs, because this is the series the dashboard
        draws. The model's own inputs — logs, annualized rates, z-scores — go to
        :meth:`derived_features` instead, so neither table needs a footnote about
        which units it is in.
        """
        try:
            analysis = self.analyze(world)
        except StateUnavailable as err:
            log.warning("equity: no outputs — %s", err)
            return ()

        features = analysis.publication
        frame = features.frame
        cutoff = self._cutoff(frame.index)

        rows: list[EngineOutput] = []

        # Raw close and filtered level, both in index points, so the chart can
        # overlay them without the page doing arithmetic on model units.
        close = np.exp(features.log_price.loc[cutoff:])
        rows.extend(self._output_rows("price_close", close))
        filtered = np.exp(frame["price_filtered"].loc[cutoff:])
        rows.extend(self._output_rows("price_filtered", filtered))

        for metric in ("velocity", "acceleration", "jerk_z"):
            rows.extend(self._output_rows(metric, frame[metric].loc[cutoff:]))

        # The lamp travels as its code with the label in meta — engine_output
        # stores REALs, and a chart of the code reads upwards as instability rises.
        jerk = frame["jerk_z"].loc[cutoff:].dropna()
        rows.extend(
            EngineOutput(
                asset=self.name,
                metric="jerk_lamp",
                as_of=key.date(),
                value=float(JERK_LAMP_CODES[jerk_lamp(float(value))]),
                meta={"lamp": jerk_lamp(float(value))},
            )
            for key, value in jerk.items()
        )
        rows.extend(self._regime_outputs(analysis))
        return tuple(rows)

    def _regime_outputs(self, analysis: EquityAnalysis) -> list[EngineOutput]:
        """Regime severity and the transition probabilities, for the charts.

        The posterior itself goes to ``regime_state`` (:meth:`regime_states`),
        which is the table the spec gives it and the one ``/regime`` reads. What
        is duplicated here is only the single-number summary a sparkline needs.
        """
        view = analysis.regime
        if view is None:
            return []

        cutoff = self._cutoff(view.posteriors.index)
        rows: list[EngineOutput] = []

        severity = (
            view.posteriors.loc[cutoff:]
            .mul([rank for rank, _ in enumerate(REGIMES)], axis=1)
            .sum(axis=1)
        )
        rows.extend(
            EngineOutput(
                asset=self.name,
                metric="regime_severity",
                as_of=key.date(),
                value=round(float(value), 6),
                meta={"regime": str(view.posteriors.loc[key].idxmax())},
            )
            for key, value in severity.items()
            if math.isfinite(float(value))
        )

        for column in view.transitions.columns:
            rows.extend(
                self._output_rows(
                    column.replace("p_", "p_transition_"),
                    view.transitions[column].loc[cutoff:],
                )
            )
        return rows

    def regime_states(self, world: WorldState) -> tuple[RegimeProbability, ...]:
        """The filtered posterior per date — §7's ``regime_state`` table.

        The whole distribution, not the winning label: §12's contract shows all
        five, and the difference between a 0.95 call and a 0.35 one is most of
        what the state is saying.
        """
        try:
            analysis = self.analyze(world)
        except StateUnavailable as err:
            log.warning("equity: no regime states — %s", err)
            return ()

        view = analysis.regime
        if view is None:
            return ()

        window = view.posteriors.loc[self._cutoff(view.posteriors.index) :]
        return tuple(
            RegimeProbability(
                asset=self.name,
                as_of=key.date(),
                regime=str(label),
                probability=round(float(probability), 8),
                model_version=analysis.model_version,
            )
            for key, row in window.iterrows()
            for label, probability in row.items()
            if math.isfinite(float(probability))
        )

    def derived_features(self, world: WorldState) -> tuple[DerivedFeature, ...]:
        """The model inputs, in model units, versioned by ``model_version``.

        This is what the rule-5 replay test reproduces and what sub-milestone B's
        HMM is fitted on, so it is stored exactly as the model sees it.
        """
        try:
            analysis = self.analyze(world)
        except StateUnavailable as err:
            log.warning("equity: no derived features — %s", err)
            return ()

        frame = analysis.publication.frame
        cutoff = self._cutoff(frame.index)
        window = frame.loc[cutoff:]

        rows: list[DerivedFeature] = []
        for column in window.columns:
            series = window[column].dropna()
            rows.extend(
                DerivedFeature(
                    asset=self.name,
                    feature=column,
                    as_of=key.date(),
                    value=round(float(value), 10),
                    model_version=analysis.model_version,
                )
                for key, value in series.items()
                if math.isfinite(float(value))
            )
        return tuple(rows)

    # -- internals --------------------------------------------------------

    def _cutoff(self, index: pd.Index) -> pd.Timestamp:
        """Oldest date the published histories reach back to."""
        if len(index) == 0:
            return pd.Timestamp.min
        return pd.Timestamp(index[-1]) - pd.Timedelta(days=self.history_days)

    def _output_rows(self, metric: str, series: pd.Series) -> list[EngineOutput]:
        return [
            EngineOutput(
                asset=self.name,
                metric=metric,
                as_of=key.date(),
                value=round(float(value), 8),
            )
            for key, value in series.dropna().items()
            if math.isfinite(float(value))
        ]


__all__ = [
    "ALL_ROLES",
    "ARTIFACT_NAME",
    "CHART_METRICS",
    "DAILY_ROLES",
    "KINEMATIC_FEATURES",
    "MODEL_VERSION_BASE",
    "EquityAnalysis",
    "EquityEngine",
]
