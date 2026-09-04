"""The observable block ``Z_t`` that Ω is inferred from. **Scaffold (KK0).**

Nine candidate columns over seven series that are **already declared** in
``config/series.yaml`` — adding a data dependency is out of scope for this
track. The full list, its publication lags and its availability in the committed
fixture are in ``docs/research/kk-omega-design.md`` §3.

Two decisions from the recon shape this module, and both are about availability
rather than about modelling.

**Every column is a scale-free causal transform, never a level.** A PC1 taken
over raw levels is dominated by whichever column has the largest variance in its
own units, which on this block is the index itself — Ω would then be a
reparameterized S&P. So the columns are realized vol, vol of vol, ``|log
return|``, drawdown from a trailing running max, a differenced rolling
correlation against ``FRED:NASDAQ100``, an n-day change in ``FRED:DGS10``, the
``FRED:T10Y3M`` level, the ``FRED:NFCI`` level, and an n-day change in
``FRED:BAMLH0A0HYM2``. Windows are declared in months or years in
``omega.yaml`` and converted against each series' own ``periods_per_year``,
which is the convention ``features/kinematics.py`` uses so that one config
number means one span on a daily and on a monthly path.

**Availability is checked, never resolved by ``dropna``.** ``FRED:BAMLH0A0HYM2``
exists in the committed fixture from **2023-08-01** — 787 observations against
6,649 for the rates columns. The obvious implementation builds the frame, drops
incomplete rows and fits; that silently starts the research window in August
2023, produces an Ω with no 2008 and no 2020 in it, and reports nothing unusual.
Every number downstream would then be a statement about thirty-five months of a
bull market.

So each column is counted at the fit date, any column below
``min_column_observations`` knowable rows is **dropped and named** in
``OmegaSpec.dropped``, and the surviving set is logged at INFO. Dropped columns
are never imputed and never zero-filled: zero on a stress axis means "maximally
calm" and absence means nothing of the kind. This is the rule ``compute_rii``
already applies to its seven components — "six components, with the seventh
named" — reused rather than reinvented, because two implementations of "what to
do about a missing input" eventually disagree.

The window start is a config value. A column never decides it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - scaffold
    import pandas as pd

    from findynamics.core.contracts.pit import PITAccessor
    from findynamics.research.omega.config import OmegaConfig


def build_feature_frame(
    accessor: PITAccessor,
    config: OmegaConfig,
) -> pd.DataFrame:
    """The causal feature block ``Z_t``, one column per surviving transform.

    Read through the PIT accessor and nothing else: the accessor binds the
    information-set cutoff, so no transform here can widen it. Weekly inputs
    (``FRED:NFCI``) arrive forward-filled as of their **release** date, which
    the accessor already enforces — re-implementing a lag here would apply it
    twice.
    """
    raise NotImplementedError("KK1: implement Z_t (docs/research/kk-omega-design.md §3)")


def available_columns(
    frame: pd.DataFrame,
    min_observations: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(surviving, dropped)`` column names, both sorted.

    Sorted rather than in insertion order because the surviving set becomes
    ``OmegaSpec.columns`` and is dotted with a loading vector; a set or a dict
    iteration order would make the projection depend on how the frame happened
    to be assembled.
    """
    raise NotImplementedError("KK1: implement the availability check")


__all__ = ["available_columns", "build_feature_frame"]
