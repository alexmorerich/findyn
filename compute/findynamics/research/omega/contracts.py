"""The frozen records KK-Ω passes across its own boundaries. **Scaffold (KK0).**

Two records carry everything: :class:`OmegaSpec`, which is *what a fit froze*,
and :class:`OmegaPath`, which is *the computed path at one information set*.
KK1 implements them; this module ships the error type and the field contract so
that the implementing phase has a target rather than a blank file.

Why a spec object at all. The walk-forward in KK3 refits Ω at every rebalance
date and transforms forward only, so there is one fitted object per window and
the report has to be able to print their history — which columns survived the
availability check that month, which column pinned the sign, how much variance
PC1 explained. A transform that carried its parameters implicitly in a fitted
sklearn object would make that history unprintable and the run unreproducible;
``KalmanParams`` and ``FfdFit`` are the precedent, and ``OmegaSpec`` round-trips
through ``as_dict()`` / ``from_dict()`` for the same reason they do.

The field contract KK1 implements (``docs/research/kk-omega-design.md`` §4)::

    OmegaSpec                       # frozen; validated in __post_init__
        estimator: str              # "pca" for the MVP
        columns: tuple[str, ...]    # feature columns used, in sorted order
        dropped: tuple[str, ...]    # columns dropped for insufficient history
        mean: tuple[float, ...]     # scaler mean, per column, fit-window only
        scale: tuple[float, ...]    # scaler scale, per column, fit-window only
        loadings: tuple[float, ...] # PC1 loadings, sign already pinned (§5)
        explained_variance_ratio: float
        sign_reference: str         # the column whose loading pins the sign
        fit_start: date
        fit_end: date
        n_observations: int
        model_version: str

    OmegaPath                       # frozen; every series shares one index
        omega: pd.Series
        omega_velocity: pd.Series
        omega_acceleration: pd.Series
        omega_volatility: pd.Series
        omega_zscore: pd.Series
        omega_regime: pd.Series     # latent_calm | latent_transition | latent_stress
        spec: OmegaSpec
        diagnostics: dict[str, float]

Three invariants are not negotiable and belong in ``__post_init__`` rather than
at a write boundary, for the reason ``core/contracts/state.py`` gives: a
nonsensical value is cheapest to catch where it is constructed.

1. ``len(columns) == len(mean) == len(scale) == len(loadings)``. A spec whose
   loadings and columns have drifted apart silently reorders the dot product,
   and every Ω it produces is a projection onto the wrong axes.
2. ``sign_reference in columns``. The sign-pinning rule is meaningless if the
   reference column is not one of the ones that survived.
3. No wall-clock field, ever. ``fit_start`` / ``fit_end`` / ``n_observations``
   describe the *data*; anything derived from ``datetime.now()`` makes two
   identical fits serialize to different bytes, which is issue #6 in this
   repository and cost it a monthly job that went red with an unactionable 409.

The sibling scaffolds already annotate against ``OmegaSpec`` inside
``if TYPE_CHECKING:`` blocks, so the names above are forward references until
KK1 lands. That is deliberate: the signatures state what each function will be
handed, which is the part of the contract worth fixing now. Nothing evaluates
those annotations at runtime — every module in this package declares
``from __future__ import annotations``.
"""

from __future__ import annotations


class OmegaContractError(ValueError):
    """Raised when an Ω record violates its documented invariants.

    Its own type rather than ``core.contracts.state.ContractError``: these
    records never cross a production boundary, and a research failure surfacing
    as an engine contract error would be actively misleading in a traceback.
    """


__all__ = ["OmegaContractError"]
