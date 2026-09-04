"""How Ω is inferred from ``Z_t``, and why by a PCA.

The MVP estimator is the first principal component of the window-standardized
feature block. It was chosen against a five-state HMM latent state and against
an autoencoder bottleneck on four criteria — deterministic, explainable, cheap,
easy to backtest — and the full comparison is in
``docs/research/kk-omega-design.md`` §4. The short version:

* the autoencoder loses on all four and would need a dependency the track
  forbids in ``dependencies``;
* the **HMM is deferred, not rejected**. ``engines/equity/regime/hmm.py``
  already exists, already solves the determinism problem and already carries a
  documented labelling rule. What it does not give is a *continuous* Ω, and
  Ω̇ / Ω̈ / ``C`` all need one — a state index is not differentiable, and
  dressing one up as a coordinate would make every derivative in this track a
  difference of integers.

The deciding argument is orientation. A PCA loading vector has exactly one
degree of orientation freedom and one documented rule pins it (below). Five HMM
states can permute 120 ways; pinning *that* is a labelling rule with its own
failure modes, and ``regime/hmm.py`` is on its third such rule because the two
more sophisticated ones each failed on a real series. Starting with the
estimator that has one sign to pin rather than 120 permutations is putting the
harder problem after the first out-of-sample result.

The sign problem
----------------

A principal component's sign is arbitrary: ``(loadings, scores)`` and
``(−loadings, −scores)`` are the same decomposition, and which one comes back
depends on the LAPACK driver, the BLAS build and the data. Under walk-forward
refits this is not cosmetic, it is a **signal factory**. If the window ending in
June returns ``+w`` and the window ending in July returns ``−w``, Ω does not
drift across that boundary — it reflects through zero. Ω̇ then records the
largest excursion in its history on the refit date, every refit date, and Ω̈
records a larger one. The result is a spike train aligned to a monthly calendar:
large, regular, robust to winsorization, correlated with anything else that has
monthly structure, and entirely an artifact of a linear-algebra library.

**The pinning rule.** The loading on ``sign_reference`` is forced non-negative;
if it is negative, the whole loading vector is negated. ``sign_reference`` is
the first **surviving** column in the precedence order declared in
``omega.yaml``, and the column actually used is recorded in
``OmegaSpec.sign_reference``.

Two refinements the obvious version misses:

* A column whose loading is *zero* pins nothing — negating a vector does not
  change a zero — so the search skips past any candidate below
  :data:`SIGN_TOLERANCE` rather than accepting it and leaving the sign
  undetermined.
* The precedence list is fixed and ordered in config. "The column with the
  largest absolute loading" is the tempting alternative and it is itself
  unstable: two columns with near-equal loadings would hand the reference back
  and forth between windows and reintroduce the flip through a side door.

Ω therefore **increases with market stress by construction**. That is an
orientation convention and not a finding — the same move ``rii.py`` makes when
it fixes ``direction=+1`` so that 100 means maximally unstable — and the report
must say so wherever Ω's sign is interpreted.

Determinism
-----------

``svd_solver="full"`` and ``threadpool_limits(1)`` around the fit, both for the
reasons ``regime/hmm.py`` documents at length: the randomized solver draws a
random projection and is not bit-reproducible across BLAS builds, and a threaded
floating-point reduction sums its partials in whatever order the threads finish.
A seed alone does not buy reproducibility. The block is at most nine columns
wide, so there is no performance argument on the other side of either choice.

:meth:`PCAOmegaEstimator.transform` projects with ``math.fsum`` rather than a
dot product for the same family of reasons. A BLAS ``ddot`` is deterministic for
a fixed build and thread count and is not guaranteed to be across them, and this
is the function KK3 calls once per walk-forward window on a path that is then
compared bit-for-bit against a truncated re-run. ``math.fsum`` is exactly
rounded on every platform and every Python version, so the comparison tests the
model rather than the linear-algebra library it was linked against. Nine terms
per row makes the cost irrelevant.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from findynamics.research.omega.config import OmegaConfig
from findynamics.research.omega.contracts import OmegaSpec
from findynamics.research.omega.domain import FEATURE_COLUMNS, PRICE_DERIVED_COLUMNS
from findynamics.research.omega.features import FeatureParams, available_columns

log = logging.getLogger("findynamics.research.omega.estimator")

#: Fixed so a refit on the same data lands in the same place. A module constant
#: rather than a config key: changing it is a model change and belongs in a
#: version bump, not in a yaml tweak (``regime/hmm.py::DEFAULT_SEED``).
DEFAULT_SEED = 20260904

#: A loading this close to zero pins no sign. Not an epsilon for float
#: comparison — a genuine "this column is orthogonal to PC1, ask the next one".
SIGN_TOLERANCE = 1e-12


class OmegaFitError(ValueError):
    """Raised when a window cannot support a fit. Named, and never silent.

    Distinct from returning a short Ω, which is the failure this replaces: a
    two-point projection is not a coordinate, and a caller that got one back
    would have no way to tell it from a real one.
    """


class OmegaEstimator(Protocol):
    """The extension point. ``fit`` freezes a spec; ``transform`` applies one.

    Split in two on purpose. The walk-forward fits on the expanding window
    ending at ``t`` and transforms only observations after ``t``, so the two
    halves run at different times on different data and a combined
    ``fit_transform`` would make that separation impossible to express — which
    is exactly the shape of the leak ``test_no_lookahead_guards`` greps for.
    """

    @property
    def name(self) -> str:
        """Estimator id, stamped onto every spec this produces."""
        ...

    def fit(self, z: pd.DataFrame) -> OmegaSpec:
        """Freeze a scaler, a projection and a pinned sign on this window."""
        ...

    def transform(self, z: pd.DataFrame, spec: OmegaSpec) -> pd.Series:
        """Ω from a frozen spec. A pure function of ``(row, spec)``."""
        ...


@dataclass(frozen=True)
class EstimatorParams:
    """The estimator's half of ``omega.yaml``."""

    name: str = "pca"
    sign_reference_precedence: tuple[str, ...] = FEATURE_COLUMNS
    min_fit_observations: int = 504
    model_version: str = "omega-unversioned"

    @classmethod
    def from_config(cls, config: OmegaConfig) -> EstimatorParams:
        block = config.block("estimator")

        precedence = tuple(str(c) for c in (block.get("sign_reference_precedence") or ()))
        unknown = sorted(set(precedence) - set(FEATURE_COLUMNS))
        if unknown:
            raise OmegaFitError(
                f"estimator.sign_reference_precedence names unknown column(s) {unknown}; "
                f"expected a subset of {list(FEATURE_COLUMNS)}"
            )
        if len(set(precedence)) != len(precedence):
            raise OmegaFitError("estimator.sign_reference_precedence contains duplicates")

        minimum = int(block.get("min_fit_observations", 504))
        if minimum < 2:
            raise OmegaFitError(
                f"estimator.min_fit_observations must be >= 2, got {minimum}; a covariance "
                "matrix needs more than one row"
            )

        version = config.params.get("model_version")
        return cls(
            name=str(block.get("name", "pca")),
            sign_reference_precedence=precedence or FEATURE_COLUMNS,
            min_fit_observations=minimum,
            model_version=str(version) if version else "omega-unversioned",
        )


class PCAOmegaEstimator:
    """PC1 of the standardized surviving columns. The MVP.

    ``candidates`` restricts which columns this specification is even willing to
    consider, *before* the availability check runs. That is how
    :class:`PriceOnlyOmegaEstimator` is a different specification rather than
    the same one on a different window.
    """

    def __init__(
        self,
        params: EstimatorParams,
        features: FeatureParams,
        *,
        candidates: tuple[str, ...] = FEATURE_COLUMNS,
        name: str = "pca",
    ) -> None:
        self._params = params
        self._features = features
        self._candidates = tuple(sorted(set(candidates)))
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def candidates(self) -> tuple[str, ...]:
        return self._candidates

    def fit(self, z: pd.DataFrame) -> OmegaSpec:
        """Freeze mean, scale, loadings and the pinned sign on ``z``.

        ``z`` is the fit window and nothing else. Every number this returns is
        computed from rows inside it, which is what makes ``transform`` safe to
        apply to later data (``docs/research/kk-omega-design.md`` §6).
        """
        if z.empty:
            raise OmegaFitError("cannot fit Ω on an empty feature block")

        considered = z[[c for c in z.columns if c in self._candidates]]
        out_of_scope = tuple(sorted(set(z.columns) - set(self._candidates)))
        if considered.empty or not len(considered.columns):
            raise OmegaFitError(
                f"none of this estimator's candidate columns {list(self._candidates)} are in the "
                f"feature block {list(z.columns)}"
            )

        surviving, unavailable = available_columns(
            considered,
            self._features.min_column_observations,
            self._features.min_column_coverage,
        )
        if not surviving:
            raise OmegaFitError(
                f"every candidate column failed the availability check over {len(z)} row(s): "
                f"none reached {self._features.min_column_observations} observations at "
                f"{self._features.min_column_coverage:.0%} coverage"
            )

        matrix = considered[list(surviving)].dropna()
        if len(matrix) < self._params.min_fit_observations:
            raise OmegaFitError(
                f"{len(matrix)} complete row(s) over columns {list(surviving)} is below the "
                f"{self._params.min_fit_observations} this estimator requires; a projection "
                "fitted on less is a statement about its own start-up"
            )

        mean = matrix.mean()
        # Population standard deviation, matching what a StandardScaler would
        # store. ddof is not a free choice here: the scale travels in the spec
        # and is applied to later rows, so it has to be the same convention the
        # PCA's own covariance assumes.
        scale = matrix.std(ddof=0)

        # A column that never moved over the window carries no information and
        # would divide by zero. Dropped and named, exactly like an unavailable
        # one — the contract refuses a non-positive scale outright, so this is
        # the only place the case can be handled.
        constant = tuple(sorted(c for c in surviving if not scale[c] > 0.0))
        if constant:
            log.info(
                "omega estimator: dropping constant column(s) %s over %s → %s",
                ", ".join(constant),
                matrix.index[0].date(),
                matrix.index[-1].date(),
            )
            surviving = tuple(c for c in surviving if c not in constant)
            if not surviving:
                raise OmegaFitError(
                    "every surviving column is constant over the fit window; there is no "
                    "variation for a principal component to describe"
                )
            matrix = considered[list(surviving)].dropna()
            mean = matrix.mean()
            scale = matrix.std(ddof=0)

        standardized = (matrix - mean) / scale

        # threadpool_limits, not just the seed: k-means is not the only
        # OpenMP-parallel reduction in scikit-learn, and an SVD summing its
        # partials in thread-completion order moves the loadings in the last
        # bits — which is the difference between an artifact that reproduces and
        # one that 409s (issue #6, regime/hmm.py).
        with threadpool_limits(limits=1):
            pca = PCA(n_components=1, svd_solver="full", random_state=DEFAULT_SEED)
            pca.fit(standardized.to_numpy(dtype=float))

        loadings = np.asarray(pca.components_[0], dtype=float)
        explained = float(pca.explained_variance_ratio_[0])

        reference, loadings = self._pin_sign(surviving, loadings)

        dropped = tuple(sorted(set(unavailable) | set(constant) | set(out_of_scope)))
        spec = OmegaSpec(
            estimator=self._name,
            columns=tuple(surviving),
            dropped=dropped,
            mean=tuple(float(mean[c]) for c in surviving),
            scale=tuple(float(scale[c]) for c in surviving),
            loadings=tuple(float(v) for v in loadings),
            explained_variance_ratio=min(max(explained, 0.0), 1.0),
            sign_reference=reference,
            fit_start=matrix.index[0].date(),
            fit_end=matrix.index[-1].date(),
            n_observations=len(matrix),
            model_version=self._params.model_version,
        )
        log.info(
            "omega %s: fitted on %d row(s) %s → %s; PC1 explains %.1f%% over %s; "
            "sign pinned on %s; dropped %s",
            self._name,
            spec.n_observations,
            spec.fit_start,
            spec.fit_end,
            spec.explained_variance_ratio * 100.0,
            ", ".join(spec.columns),
            spec.sign_reference,
            ", ".join(spec.dropped) or "nothing",
        )
        return spec

    def _pin_sign(self, columns: tuple[str, ...], loadings: np.ndarray) -> tuple[str, np.ndarray]:
        """Orient the loading vector. Returns the reference column and the vector.

        Walks the configured precedence order and takes the first surviving
        column whose loading is meaningfully non-zero — a zero loading pins
        nothing, because negating a vector leaves a zero where it was.
        """
        for candidate in self._params.sign_reference_precedence:
            if candidate not in columns:
                continue
            value = float(loadings[columns.index(candidate)])
            if abs(value) <= SIGN_TOLERANCE:
                log.info(
                    "omega %s: %s loads at %.3g on PC1, which pins no sign; "
                    "falling through to the next reference in precedence",
                    self._name,
                    candidate,
                    value,
                )
                continue
            return candidate, (-loadings if value < 0.0 else loadings)

        # Every configured candidate was dropped or orthogonal. Falling back to
        # the first surviving column keeps the spec constructible and keeps the
        # orientation *documented*, which is the property that matters — the
        # alternative is an unpinned sign, and that is the failure this whole
        # mechanism exists to prevent.
        fallback = columns[0]
        value = float(loadings[0])
        log.warning(
            "omega %s: none of the configured sign references survived with a non-zero "
            "loading; pinning on %r instead. Ω's orientation is still deterministic but no "
            "longer means 'increases with market stress' — check the loading table before "
            "reading its sign",
            self._name,
            fallback,
        )
        return fallback, (-loadings if value < 0.0 else loadings)

    def transform(self, z: pd.DataFrame, spec: OmegaSpec) -> pd.Series:
        """Ω over ``z``, from a frozen spec. Pure in ``(row, spec)``.

        Reads mean, scale and loadings out of the spec and never recomputes
        them. A row missing any of the spec's columns yields ``NaN`` rather than
        a partial projection: a projection onto eight of nine axes is a
        different coordinate, and publishing it under the same name would make
        Ω discontinuous wherever one input was late.
        """
        missing = sorted(set(spec.columns) - set(z.columns))
        if missing:
            raise OmegaFitError(
                f"the feature block is missing column(s) {missing} that this spec was fitted "
                f"on; it carries {list(z.columns)}"
            )
        if z.empty:
            return pd.Series(dtype=float, index=z.index, name="omega")

        block = z[list(spec.columns)].to_numpy(dtype=float)
        mean = np.asarray(spec.mean, dtype=float)
        scale = np.asarray(spec.scale, dtype=float)
        loadings = np.asarray(spec.loadings, dtype=float)

        standardized = (block - mean) / scale
        terms = standardized * loadings
        # math.fsum per row rather than a dot product: exactly rounded on every
        # platform, so a truncated re-run compares bit-for-bit against the full
        # one (module docstring). Nine terms per row makes the cost irrelevant.
        values = np.array(
            [math.fsum(row) if np.isfinite(row).all() else np.nan for row in terms],
            dtype=float,
        )
        return pd.Series(values, index=z.index, name="omega")


class PriceOnlyOmegaEstimator(PCAOmegaEstimator):
    """PC1 of the price-derived columns only — a **different specification**.

    Exists so the track has a run that reaches back to 1927-12-30 on
    ``YAHOO:^GSPC``, where the rates, liquidity and credit columns do not exist.
    It is reported as its own row in every table and is never merged with the
    full-column specification: the two do not have the same columns, and an
    average of them is a number about nothing.
    """

    def __init__(self, params: EstimatorParams, features: FeatureParams) -> None:
        super().__init__(
            params,
            features,
            candidates=PRICE_DERIVED_COLUMNS,
            name="pca_price_only",
        )


#: Config ``estimator.name`` -> implementation. A mapping rather than a chain of
#: ``if``s so that adding the planned HMM specification is one entry and one
#: class, and so an unknown name fails with the list of known ones.
ESTIMATORS: dict[str, type[PCAOmegaEstimator]] = {
    "pca": PCAOmegaEstimator,
    "pca_price_only": PriceOnlyOmegaEstimator,
}


def build_estimator(config: OmegaConfig) -> PCAOmegaEstimator:
    """The estimator ``omega.yaml`` names."""
    params = EstimatorParams.from_config(config)
    features = FeatureParams.from_config(config)
    try:
        factory = ESTIMATORS[params.name]
    except KeyError:
        raise OmegaFitError(
            f"unknown estimator {params.name!r}; configured: {sorted(ESTIMATORS)}"
        ) from None
    return factory(params, features)


__all__ = [
    "DEFAULT_SEED",
    "ESTIMATORS",
    "SIGN_TOLERANCE",
    "EstimatorParams",
    "OmegaEstimator",
    "OmegaFitError",
    "PCAOmegaEstimator",
    "PriceOnlyOmegaEstimator",
    "build_estimator",
]
