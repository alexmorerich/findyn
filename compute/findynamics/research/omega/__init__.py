"""KK-Ω — a latent market-state coordinate, as a falsifiable research question.

The equity engine represents an asset as ``(P, v, a, j)``: price, and three
derivatives of the Kalman-filtered log price (``FINDYN_V1_SPEC.md`` §8.1, §8.3).
This package asks whether extending that to
``(P, v, a, j, Ω, Ω̇, Ω̈, C, K)`` carries statistically significant
out-of-sample information about future returns, volatility, regime transitions
and drawdowns.

A fifth-dimensional latent-state representation inspired by Kaluza–Klein
geometry is being tested as a quantitative modelling hypothesis. The line
element ``ds² = g_ij dx^i dx^j + φ²(dΩ + A_i dx^i)²`` is guidance for the
*shape* of the model — an observable block, one latent coordinate, and an
explicit coupling between them — and nothing else is borrowed from it. No claim
is made about physics.

**The question is not "does Ω beat price alone".** ``engines/equity/rii.py``
already publishes a composite built from posterior entropy, confidence deficit,
``|jerk_z|``, vol-of-vol, stock-bond correlation breakdown, credit velocity and
liquidity stress — nearly the same observables Ω is built from. Beating four
derivatives of one series is close to guaranteed and close to worthless. The
question is whether Ω carries information beyond ``(P, v, a, j)`` **and beyond
the RII the engine already publishes**, which is why every model comparison in
this track carries an RII control arm.

The module is allowed to fail. A well-evidenced ``REJECTED`` closes the track
successfully; ``docs/research/kk-omega-design.md`` §9 pre-registers the verdict
rules and §10 says what would make us delete this package.

Two independent gates keep it out of production:

1. the ``Research is quarantined`` import-linter contract, which makes a
   production import of this package a CI failure rather than a code review
   comment; and
2. ``config/research/omega.yaml``, which ships ``enabled: false``.

The first is the real one. The flag is the second line of defence, and reading
it as the only one is how a research module ends up in a daily job.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from findynamics.core.contracts.pit import PITAccessor
from findynamics.research.omega.config import (
    OMEGA_CONFIG_PATH,
    OmegaConfig,
    OmegaConfigError,
    get_omega_config,
    is_enabled,
    load_omega_config,
)
from findynamics.research.omega.contracts import OmegaContractError, OmegaPath, OmegaSpec
from findynamics.research.omega.domain import (
    FEATURE_COLUMNS,
    OMEGA_REGIME_CODES,
    OMEGA_REGIMES,
    PRICE_DERIVED_COLUMNS,
)
from findynamics.research.omega.dynamics import (
    DynamicsParams,
    OmegaDynamicsError,
    omega_dynamics,
    omega_regime,
)
from findynamics.research.omega.estimator import (
    DEFAULT_SEED,
    EstimatorParams,
    OmegaEstimator,
    OmegaFitError,
    PCAOmegaEstimator,
    PriceOnlyOmegaEstimator,
    build_estimator,
)
from findynamics.research.omega.features import (
    FeatureParams,
    OmegaFeatureError,
    available_columns,
    build_feature_frame,
)

log = logging.getLogger("findynamics.research.omega")


@dataclass(frozen=True)
class OmegaEngine:
    """The module's façade: feature block → fitted spec → Ω and its dynamics.

    **This is not an ``AssetEngine`` and must never become one.** It does not
    subclass it, it declares no ``name`` or ``version`` class attribute, it is
    never passed to ``core.registry.register_engine``, and it writes no
    ``asset_state`` / ``engine_output`` / ``derived_features`` row.
    ``core/contracts/vocab.py::ASSETS`` is a closed five-tuple mirrored in
    ``serving/src/domain.ts``; adding a sixth member is a schema change across
    two deployment planes, and this module is not worth one.

    The omission of ``name`` is deliberate rather than incidental:
    ``register_engine`` reads that attribute first, so a copy-paste that tried
    to register this class fails on the attribute it does not have, before it
    can reach the ASSETS check that would also have refused it. The suite
    asserts both.

    ``fit`` and ``transform`` stay separate for the reason the estimator
    protocol keeps them separate: the walk-forward fits on the window ending at
    *t* and transforms only observations after it. :meth:`fit_transform` is the
    convenience for a whole-history diagnostic run and is **in-sample** — it is
    not, and must not be reported as, a result.
    """

    config: OmegaConfig

    @classmethod
    def from_shipped_config(cls) -> OmegaEngine:
        """The engine on ``config/research/omega.yaml`` as committed."""
        return cls(config=get_omega_config())

    @property
    def model_version(self) -> str:
        return EstimatorParams.from_config(self.config).model_version

    @property
    def estimator(self) -> PCAOmegaEstimator:
        """The estimator ``omega.yaml`` names. Rebuilt per access, and cheap.

        Rebuilt rather than cached because it holds only frozen parameter
        objects — there is no fitted state in an estimator here, which is the
        whole point of ``OmegaSpec`` carrying it instead.
        """
        return build_estimator(self.config)

    def features(self, accessor: PITAccessor) -> pd.DataFrame:
        """The causal feature block ``Z_t`` as of the accessor's cutoff."""
        return build_feature_frame(accessor, self.config)

    def fit(self, z: pd.DataFrame) -> OmegaSpec:
        """Freeze a spec on the window ``z``. Nothing outside it is consulted."""
        return self.estimator.fit(z)

    def transform(self, z: pd.DataFrame, spec: OmegaSpec) -> pd.Series:
        """Ω from a frozen spec — a pure function of ``(row, spec)``."""
        return self.estimator.transform(z, spec)

    def fit_transform(self, accessor: PITAccessor) -> OmegaPath:
        """Fit on everything knowable and transform it. **In-sample.**

        The diagnostic path: it answers "what does Ω look like over the record"
        and it cannot answer "does Ω predict anything", because every row was
        used to fit the projection that produced it. KK3's walk-forward is the
        only thing that may produce a reported number.
        """
        z = self.features(accessor)
        spec = self.fit(z)
        omega = self.transform(z, spec)
        return omega_dynamics(
            omega,
            spec,
            periods_per_year=FeatureParams.from_config(self.config).periods_per_year,
            config=self.config,
        )

    def diagnostics(self, path: OmegaPath) -> dict[str, float]:
        """Everything a panel needs about one path, flattened to floats.

        A pure function of the path rather than state left on the engine: KK3
        builds hundreds of these and an engine that remembered its last run
        would make "which run is this" a question the caller has to track.
        """
        out = dict(path.diagnostics)
        out.update({f"loading_{column}": path.spec.loading(column) for column in path.spec.columns})
        scored = int(path.omega_regime.notna().sum())
        for label in OMEGA_REGIMES:
            share = float((path.omega_regime == label).sum()) / scored if scored else 0.0
            out[f"share_{label}"] = share
        return out


__all__ = [
    "DEFAULT_SEED",
    "FEATURE_COLUMNS",
    "OMEGA_CONFIG_PATH",
    "OMEGA_REGIMES",
    "OMEGA_REGIME_CODES",
    "PRICE_DERIVED_COLUMNS",
    "DynamicsParams",
    "EstimatorParams",
    "FeatureParams",
    "OmegaConfig",
    "OmegaConfigError",
    "OmegaContractError",
    "OmegaDynamicsError",
    "OmegaEngine",
    "OmegaEstimator",
    "OmegaFeatureError",
    "OmegaFitError",
    "OmegaPath",
    "OmegaSpec",
    "PCAOmegaEstimator",
    "PriceOnlyOmegaEstimator",
    "available_columns",
    "build_estimator",
    "build_feature_frame",
    "get_omega_config",
    "is_enabled",
    "load_omega_config",
    "omega_dynamics",
    "omega_regime",
]
