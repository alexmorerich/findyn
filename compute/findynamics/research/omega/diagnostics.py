"""The markdown panels the report is assembled from. **Scaffold (KK0).**

A ``render_report``-style renderer following ``engines/equity/diagnostics.py``:
it takes the artifacts the walk-forward wrote and produces markdown, so that
``docs/research/kk-omega-report.md`` is regenerable from the CSVs rather than
hand-assembled from a terminal.

Five panels, and one of them is the point of the track.

**Ω panel.** Explained-variance ratio per walk-forward window, the loading
table, the columns dropped and on which dates, the sign-reference column, and
the share of the path on which the sign reference changed. A loading history is
the only way to see that "Ω" meant a different linear combination in 2008 than
it did in 2020.

**Coupling panel.** For each of the three estimators: defined share, the ``β1``
path with Newey–West t-stats, ``R²``, and the winsorized share.

**Curvature panel.** Component correlation matrix, the weight table, and the
contribution decomposition on the five largest ``K`` dates.

**The redundancy panel — required, and the reason this module exists.**
Correlation and rank correlation of Ω, Ω̇, ``C`` and ``K`` against **each
existing published quantity**: ``velocity``, ``acceleration``, ``jerk_z``, the
RII, and each RII component from ``rii.py``. If ``K`` correlates above 0.9 with
the RII, the panel opens with the sentence *"K is a re-derivation of the
existing instability index."* That sentence is a legitimate result and the
renderer must not soften it — it is emitted by a threshold, not by a judgement
call, precisely so that nobody has to decide on the day whether to write it.

**Stationarity note.** An ADF or variance-ratio reading on Ω and ``C``, so the
KK3 regressions are not run on something with an obvious unit root without the
report saying so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - scaffold
    import pandas as pd


def render_report(artifacts: dict[str, pd.DataFrame]) -> str:
    """The full markdown report body, assembled from the stored CSVs."""
    raise NotImplementedError("KK2/KK5: implement the renderer")


def redundancy_panel(
    latent: dict[str, pd.Series],
    published: dict[str, pd.Series],
) -> str:
    """Ω, Ω̇, ``C``, ``K`` against every quantity the engine already publishes."""
    raise NotImplementedError("KK2: implement the redundancy panel")


__all__ = ["redundancy_panel", "render_report"]
