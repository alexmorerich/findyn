"""FinEquity — the S&P500 dynamic state engine (FINDYN_V1_SPEC.md, Phase 3).

Sub-milestone A: the causal feature path. Roles are resolved
(:mod:`.prices`), the Kalman/FFD/kinematics pipeline runs over the publication
series, and the result is published two ways — as ``derived_features`` (the model
inputs, versioned by model) and as ``engine_output`` (the same information in
chartable units).

``predict`` deliberately raises :class:`StateUnavailable` until sub-milestone B.
An ``AssetState`` requires a ``regime``, and the only honest source of one is the
fitted HMM. Publishing a placeholder regime would put a number on the dashboard
that means nothing, and the moment it is on the dashboard someone reads it. So
the engine says "no state yet" in the one way the job layer understands, keeps
publishing the features it *can* stand behind, and ``/assets/equity/state``
keeps answering 501 with its phase tag.

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
    WorldState,
)
from findynamics.core.engine import AssetEngine, StateUnavailable
from findynamics.core.registry import register_engine
from findynamics.engines.equity import prices as prices_mod
from findynamics.engines.equity.domain import (
    CHART_METRICS,
    KINEMATIC_FEATURES,
)
from findynamics.engines.equity.features.kinematics import JERK_LAMP_CODES, jerk_lamp
from findynamics.engines.equity.features.pipeline import (
    FeatureParams,
    FeatureSet,
    FrozenFeatureParams,
    compute_features,
)
from findynamics.engines.equity.prices import PriceRoles, PriceSeries

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


@dataclass(frozen=True)
class EquityAnalysis:
    """Everything one run derives from the information set, computed once."""

    roles: PriceRoles
    #: Engine role -> its feature path. Keys are a subset of :data:`ALL_ROLES`.
    features: dict[str, FeatureSet]
    model_version: str

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
        """How far back published histories reach. Bounds the write-back."""
        return int(self._block("outputs").get("history_days", 1825))

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
            roles=resolved, features=features, model_version=self.model_version(resolved)
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
        self._artifacts.save(ARTIFACT_NAME, document)
        log.info(
            "equity.fit: froze parameters for %d series (%s) at %s",
            len(analysis.features),
            ", ".join(sorted(document["series"])),
            world.as_of,
        )

    def predict(self, world: WorldState) -> AssetState:
        """No state yet — the regime model is sub-milestone B.

        Not a stub and not a failure. ``AssetState.regime`` has no defensible
        value before the HMM exists, and the alternatives are both worse than
        declining: a hard-coded regime is fiction, and a rule-based interim
        regime is a second model nobody asked for that B would delete.

        The feature path still publishes on every run — see :meth:`outputs` and
        :meth:`derived_features` — so ``/equity`` has data while
        ``/assets/equity/state`` correctly reports that no state exists.
        """
        analysis = self.analyze(world)
        raise StateUnavailable(
            f"equity: features are published ({len(analysis.publication.frame)} rows "
            f"through {analysis.as_of}), but the regime model that an AssetState "
            "needs lands in sub-milestone B; no state is published until then"
        )

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
        return tuple(rows)

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
