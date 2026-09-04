"""The frozen records KK-Ω passes across its own boundaries.

Two records carry everything: :class:`OmegaSpec`, which is *what a fit froze*,
and :class:`OmegaPath`, which is *the computed path at one information set*.

Why a spec object at all. The walk-forward in KK3 refits Ω at every rebalance
date and transforms forward only, so there is one fitted object per window and
the report has to be able to print their history — which columns survived the
availability check that month, which column pinned the sign, how much variance
PC1 explained. A transform carrying its parameters implicitly inside a fitted
sklearn estimator would make that history unprintable and the run
unreproducible. ``KalmanParams`` and ``FfdFit`` are the precedent, and
:class:`OmegaSpec` round-trips through ``as_dict()`` / ``from_dict()`` for the
same reason they do.

Validation lives in ``__post_init__`` rather than at a write boundary, following
``core/contracts/state.py``: a nonsensical value is cheapest to catch at the
moment it is constructed. Four invariants are load-bearing.

1. **The column and parameter vectors are the same length.** A spec whose
   loadings and columns have drifted apart silently reorders the dot product,
   and every Ω it produces is a projection onto the wrong axes — with no symptom
   at all, because the result is still a plausible-looking float.
2. **The columns are sorted and unique.** Sorted rather than in insertion order
   because the surviving set comes out of an availability check whose iteration
   order is not a modelling decision.
3. **The sign is already pinned.** ``sign_reference`` must be one of
   ``columns`` and its loading must be non-negative. Enforcing it here rather
   than trusting the estimator makes an unpinned spec *unconstructible*, which
   matters because an unpinned sign does not fail — it produces a Ω̇ spike train
   aligned to the refit calendar that reads exactly like a signal
   (``docs/research/kk-omega-design.md`` §5).
4. **No wall clock, ever.** ``fit_start`` / ``fit_end`` / ``n_observations``
   describe the *data*. Anything derived from ``datetime.now()`` makes two
   identical fits serialize to different bytes, which is issue #6 in this
   repository and cost it a monthly job that went red with an unactionable 409.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from findynamics.research.omega.domain import OMEGA_REGIMES


class OmegaContractError(ValueError):
    """Raised when an Ω record violates its documented invariants.

    Its own type rather than ``core.contracts.state.ContractError``: these
    records never cross a production boundary, and a research failure surfacing
    as an engine contract error would be actively misleading in a traceback.
    """


def _check_finite_tuple(values: object, length: int, where: str) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise OmegaContractError(f"{where}: expected a tuple, got {type(values).__name__}")
    if len(values) != length:
        raise OmegaContractError(f"{where}: expected {length} entries, got {len(values)}")
    out: list[float] = []
    for i, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise OmegaContractError(f"{where}[{i}]: expected a number, got {value!r}")
        if not math.isfinite(value):
            raise OmegaContractError(f"{where}[{i}]: must be finite, got {value!r}")
        out.append(float(value))
    return tuple(out)


def _check_names(values: object, where: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise OmegaContractError(f"{where}: expected a tuple, got {type(values).__name__}")
    for name in values:
        if not isinstance(name, str) or not name:
            raise OmegaContractError(f"{where}: entries must be non-empty strings, got {name!r}")
    if len(set(values)) != len(values):
        raise OmegaContractError(f"{where}: contains duplicates: {list(values)}")
    if list(values) != sorted(values):
        raise OmegaContractError(f"{where}: must be sorted, got {list(values)}")
    return values


@dataclass(frozen=True)
class OmegaSpec:
    """What one fit froze. Everything ``transform`` needs and nothing else."""

    #: Estimator id — ``"pca"`` for the MVP, ``"pca_price_only"`` for the
    #: 1927 specification. Recorded rather than inferred so a stored artifact
    #: says which model produced it.
    estimator: str
    #: Feature columns actually used, sorted. The dot-product axes.
    columns: tuple[str, ...]
    #: Columns the availability check dropped, sorted. Never imputed, never
    #: zero-filled — zero on a stress axis means "maximally calm" and absence
    #: means nothing of the kind (``rii.py``'s rule, reused).
    dropped: tuple[str, ...]
    #: Per-column mean over the fit window only.
    mean: tuple[float, ...]
    #: Per-column scale over the fit window only. Strictly positive.
    scale: tuple[float, ...]
    #: PC1 loadings, sign already pinned.
    loadings: tuple[float, ...]
    explained_variance_ratio: float
    #: The column whose loading pins the sign.
    sign_reference: str
    fit_start: date
    fit_end: date
    n_observations: int
    model_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.estimator, str) or not self.estimator:
            raise OmegaContractError(
                f"OmegaSpec.estimator must be non-empty, got {self.estimator!r}"
            )
        if not isinstance(self.model_version, str) or not self.model_version:
            raise OmegaContractError(
                f"OmegaSpec.model_version must be non-empty, got {self.model_version!r}"
            )

        columns = _check_names(self.columns, "OmegaSpec.columns")
        if not columns:
            raise OmegaContractError("OmegaSpec.columns must name at least one column")
        dropped = _check_names(self.dropped, "OmegaSpec.dropped")
        overlap = sorted(set(columns) & set(dropped))
        if overlap:
            raise OmegaContractError(
                f"OmegaSpec: {overlap} are listed as both used and dropped; a column is one or "
                "the other, and a spec that says both cannot be reproduced from its own record"
            )

        width = len(columns)
        _check_finite_tuple(self.mean, width, "OmegaSpec.mean")
        scale = _check_finite_tuple(self.scale, width, "OmegaSpec.scale")
        _check_finite_tuple(self.loadings, width, "OmegaSpec.loadings")
        for name, value in zip(columns, scale, strict=True):
            # A zero scale would divide the whole column by nothing. It means the
            # column was constant over the fit window, which is a degenerate
            # input the estimator must drop rather than standardize.
            if value <= 0.0:
                raise OmegaContractError(
                    f"OmegaSpec.scale[{name!r}] must be > 0, got {value}; a constant column "
                    "carries no information and belongs in `dropped`"
                )

        if isinstance(self.explained_variance_ratio, bool) or not isinstance(
            self.explained_variance_ratio, int | float
        ):
            raise OmegaContractError(
                f"OmegaSpec.explained_variance_ratio must be a number, "
                f"got {self.explained_variance_ratio!r}"
            )
        if not 0.0 <= float(self.explained_variance_ratio) <= 1.0:
            raise OmegaContractError(
                f"OmegaSpec.explained_variance_ratio must be within [0, 1], "
                f"got {self.explained_variance_ratio}"
            )

        if self.sign_reference not in columns:
            raise OmegaContractError(
                f"OmegaSpec.sign_reference {self.sign_reference!r} is not among the used columns "
                f"{list(columns)}; the pinning rule cannot reference a column that was dropped"
            )
        reference_loading = self.loadings[columns.index(self.sign_reference)]
        if reference_loading < 0.0:
            raise OmegaContractError(
                f"OmegaSpec: the loading on the sign reference {self.sign_reference!r} is "
                f"{reference_loading}, which is negative. The sign is pinned before the spec is "
                "built (docs/research/kk-omega-design.md §5); an unpinned spec makes Ω̇ a spike "
                "train aligned to the refit calendar"
            )

        for name, value in (("fit_start", self.fit_start), ("fit_end", self.fit_end)):
            # datetime is a date subclass and carries a time component that would
            # quietly break the information-set convention — the check
            # `core/contracts/state.py::_check_date` makes, for the same reason.
            if type(value) is not date:
                raise OmegaContractError(
                    f"OmegaSpec.{name}: expected datetime.date, got {type(value).__name__}"
                )
        if self.fit_start > self.fit_end:
            raise OmegaContractError(
                f"OmegaSpec: fit_start {self.fit_start} is after fit_end {self.fit_end}"
            )

        if isinstance(self.n_observations, bool) or not isinstance(self.n_observations, int):
            raise OmegaContractError(
                f"OmegaSpec.n_observations must be an int, got {self.n_observations!r}"
            )
        if self.n_observations <= 0:
            raise OmegaContractError(
                f"OmegaSpec.n_observations must be > 0, got {self.n_observations}"
            )

    @property
    def width(self) -> int:
        return len(self.columns)

    def loading(self, column: str) -> float:
        """The loading on one column, by name rather than by position."""
        try:
            return self.loadings[self.columns.index(column)]
        except ValueError:
            raise OmegaContractError(
                f"{column!r} is not one of this spec's columns {list(self.columns)}"
            ) from None

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready. Keys in a fixed order; dates as ISO strings.

        Round-trips exactly, because KK3 stores one of these per walk-forward
        window and the determinism test compares the serialized bytes.
        """
        return {
            "estimator": self.estimator,
            "columns": list(self.columns),
            "dropped": list(self.dropped),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "loadings": list(self.loadings),
            "explained_variance_ratio": self.explained_variance_ratio,
            "sign_reference": self.sign_reference,
            "fit_start": self.fit_start.isoformat(),
            "fit_end": self.fit_end.isoformat(),
            "n_observations": self.n_observations,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> OmegaSpec:
        return cls(
            estimator=str(raw["estimator"]),
            columns=tuple(str(c) for c in raw["columns"]),
            dropped=tuple(str(c) for c in raw["dropped"]),
            mean=tuple(float(v) for v in raw["mean"]),
            scale=tuple(float(v) for v in raw["scale"]),
            loadings=tuple(float(v) for v in raw["loadings"]),
            explained_variance_ratio=float(raw["explained_variance_ratio"]),
            sign_reference=str(raw["sign_reference"]),
            fit_start=date.fromisoformat(str(raw["fit_start"])),
            fit_end=date.fromisoformat(str(raw["fit_end"])),
            n_observations=int(raw["n_observations"]),
            model_version=str(raw["model_version"]),
        )


@dataclass(frozen=True)
class OmegaPath:
    """Ω and everything derived from it, at one information set.

    Every series shares one index — the trading calendar the feature block was
    built on. Checked rather than assumed: two of these get concatenated into a
    frame downstream, and a silent reindex there would align a 2008 Ω̇ against a
    2009 Ω without complaining.
    """

    omega: pd.Series
    #: Filtered slope of Ω, annualized. **Not** ``omega.diff()`` — see
    #: :mod:`findynamics.research.omega.dynamics`.
    omega_velocity: pd.Series
    omega_acceleration: pd.Series
    #: Trailing standard deviation of Ω̇.
    omega_volatility: pd.Series
    #: **Expanding** z-score of Ω, scored like ``jerk_z``.
    omega_zscore: pd.Series
    #: One of :data:`findynamics.research.omega.domain.OMEGA_REGIMES`, or NaN
    #: where the expanding percentile has not filled yet.
    omega_regime: pd.Series
    spec: OmegaSpec
    diagnostics: dict[str, float] = field(default_factory=dict)

    #: The float-valued series, in the order a CSV artifact writes them.
    SERIES_FIELDS = (
        "omega",
        "omega_velocity",
        "omega_acceleration",
        "omega_volatility",
        "omega_zscore",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.spec, OmegaSpec):
            raise OmegaContractError(
                f"OmegaPath.spec must be an OmegaSpec, got {type(self.spec).__name__}"
            )

        reference = self.omega.index
        for name in (*self.SERIES_FIELDS, "omega_regime"):
            series = getattr(self, name)
            if not isinstance(series, pd.Series):
                raise OmegaContractError(
                    f"OmegaPath.{name}: expected a pandas Series, got {type(series).__name__}"
                )
            if not series.index.equals(reference):
                raise OmegaContractError(
                    f"OmegaPath.{name} has a different index from omega "
                    f"({len(series)} rows vs {len(reference)}); every series in a path shares "
                    "one calendar"
                )

        labels = set(self.omega_regime.dropna().unique())
        unknown = sorted(labels - set(OMEGA_REGIMES))
        if unknown:
            raise OmegaContractError(
                f"OmegaPath.omega_regime carries labels outside the vocabulary: {unknown}; "
                f"expected {list(OMEGA_REGIMES)}"
            )

        if not isinstance(self.diagnostics, dict):
            raise OmegaContractError(
                f"OmegaPath.diagnostics must be a mapping, got {type(self.diagnostics).__name__}"
            )
        for key, value in self.diagnostics.items():
            if not isinstance(key, str):
                raise OmegaContractError(f"OmegaPath.diagnostics keys must be str, got {key!r}")
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise OmegaContractError(
                    f"OmegaPath.diagnostics[{key!r}]: expected a number, got {value!r}"
                )
            if not math.isfinite(value):
                raise OmegaContractError(
                    f"OmegaPath.diagnostics[{key!r}]: must be finite, got {value!r}"
                )

    def __len__(self) -> int:
        return len(self.omega)

    def frame(self) -> pd.DataFrame:
        """The path as one frame — what a CSV artifact writes."""
        columns = {name: getattr(self, name) for name in self.SERIES_FIELDS}
        columns["omega_regime"] = self.omega_regime
        frame = pd.DataFrame(columns)
        frame.index.name = "obs_date"
        return frame

    def latest(self) -> dict[str, float]:
        """The newest finite value of every float column."""
        out: dict[str, float] = {}
        for name in self.SERIES_FIELDS:
            usable = getattr(self, name).dropna()
            if not usable.empty:
                out[name] = float(usable.iloc[-1])
        return out


__all__ = ["OmegaContractError", "OmegaPath", "OmegaSpec"]
