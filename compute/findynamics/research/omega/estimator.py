"""How Ω is inferred from ``Z_t``, and why by a PCA. **Scaffold (KK0).**

The MVP estimator is the first principal component of the window-standardized
feature block. It was chosen against a five-state HMM latent state and against
an autoencoder bottleneck on four criteria — deterministic, explainable, cheap,
easy to backtest — and the full comparison is in
``docs/research/kk-omega-design.md`` §4. The short version:

* the autoencoder loses on all four and would need a dependency the master
  prompt forbids in ``dependencies``;
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
if it is negative, the whole loading vector is negated. ``sign_reference``
defaults to ``realized_vol``. If that column was dropped by the availability
check, the reference falls to the next surviving column in the **fixed
precedence order declared in** ``omega.yaml``, and the column actually used is
recorded in ``OmegaSpec.sign_reference``.

The precedence list is fixed and ordered in config rather than computed. "The
column with the largest absolute loading" is the tempting alternative and it is
itself unstable: two columns with near-equal loadings would hand the reference
back and forth between windows and reintroduce the flip through a side door.

Ω therefore **increases with market stress by construction**. That is an
orientation convention and not a finding, the same move ``rii.py`` makes when it
fixes ``direction=+1`` so 100 means maximally unstable, and the report must say
so wherever Ω's sign is interpreted.

Determinism
-----------

``svd_solver="full"`` and ``threadpool_limits(1)`` around the fit, both for the
reasons ``regime/hmm.py`` documents at length: the randomized solver draws a
random projection and is not bit-reproducible across BLAS builds, and a threaded
floating-point reduction sums its partials in whatever order the threads finish.
A seed alone does not buy reproducibility. The block is nine columns wide, so
there is no performance argument on the other side of either choice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - scaffold
    import pandas as pd

    from findynamics.research.omega.contracts import OmegaSpec

#: Fixed so a refit on the same data lands in the same place. A module constant
#: rather than a config key: changing it is a model change and belongs in a
#: version bump, not in a yaml tweak (``regime/hmm.py::DEFAULT_SEED``).
DEFAULT_SEED = 20260904


class OmegaEstimator(Protocol):
    """The extension point. ``fit`` freezes a spec; ``transform`` applies one.

    Split in two on purpose. The walk-forward fits on the expanding window
    ending at ``t`` and transforms only observations after ``t``, so the two
    halves run at different times on different data and a combined
    ``fit_transform`` would make that separation impossible to express — which
    is exactly the shape of the leak ``test_no_lookahead_guards`` greps for.
    """

    def fit(self, z: pd.DataFrame) -> OmegaSpec:
        """Freeze a scaler, a projection and a pinned sign on this window."""
        ...

    def transform(self, z: pd.DataFrame, spec: OmegaSpec) -> pd.Series:
        """Ω from a frozen spec. A pure function of ``(row, spec)``."""
        ...


class PCAOmegaEstimator:
    """PC1 of the standardized surviving columns. The MVP. **KK1.**"""

    def fit(self, z: pd.DataFrame) -> OmegaSpec:
        raise NotImplementedError("KK1: implement the PCA fit and the sign pinning")

    def transform(self, z: pd.DataFrame, spec: OmegaSpec) -> pd.Series:
        raise NotImplementedError("KK1: implement the frozen-spec transform")


class PriceOnlyOmegaEstimator:
    """PC1 of the price-derived columns only — a **different specification**. **KK1.**

    Exists so the track has a run that reaches back to 1927-12-30 on
    ``YAHOO:^GSPC``, where the rates, liquidity and credit columns do not exist.
    It is reported as its own row in every table and is never merged with the
    full-column specification: the two do not have the same columns, and an
    average of them is a number about nothing.
    """

    def fit(self, z: pd.DataFrame) -> OmegaSpec:
        raise NotImplementedError("KK1: implement the price-only specification")

    def transform(self, z: pd.DataFrame, spec: OmegaSpec) -> pd.Series:
        raise NotImplementedError("KK1: implement the price-only specification")


__all__ = [
    "DEFAULT_SEED",
    "OmegaEstimator",
    "PCAOmegaEstimator",
    "PriceOnlyOmegaEstimator",
]
