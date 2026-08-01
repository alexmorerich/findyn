"""Walk-forward evaluation of the gold regime model.

Every claim about which historical windows the model classifies correctly is
made through this module, and it is built so that the claim cannot be made any
other way. The one thing it will not do is fit on the window it is about to
grade.

What "walk-forward" means here, precisely
-----------------------------------------

For each reference window, the chain is fitted on **only the months that end
before the window starts**, and those parameters are then frozen. The posterior
inside the window is produced by *filtering* — running the Hamilton recursion
forward with the frozen parameters — so a month inside the window is conditioned
on data up to that month and on a model that never saw any of it.

That is the same arrangement production runs in: ``monthly_refit`` fits, every
daily ``predict`` filters against the frozen result. A backtest that instead
fitted once over the whole history and read the smoothed probabilities would
score a model nobody can run, and would score it generously — the fitted variance
of the violent state is estimated partly *from* the crash it is later credited
with recognising.

The reference windows
---------------------

Five, chosen because there is no serious disagreement about what they were, and
including two that the model must **not** call a crisis. A backtest with no
negative controls measures a model's enthusiasm, not its discrimination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from findynamics.engines.gold import regime as regime_mod
from findynamics.engines.gold.domain import GOLD_REGIMES

log = logging.getLogger("findynamics.engines.gold.backtest")


@dataclass(frozen=True)
class ReferenceWindow:
    """A stretch of history and what the vocabulary should call it."""

    name: str
    start: str
    end: str
    #: Names any of which counts as correct. Two are allowed for the crisis
    #: windows because ``hedge_bid`` and ``crisis_bid`` are both "gold is being
    #: bid for protection" — the distinction between them is how violently, and
    #: that is a matter of degree the vocabulary does not adjudicate.
    accepted: tuple[str, ...]
    note: str = ""


#: The windows P4 is graded on.
#:
#: 2013 is the **taper tantrum**, May to December, not the calendar year. The
#: rate shock started when Bernanke first said "taper" on 22 May 2013: the real
#: 10y yield was -0.64% in April and +0.80% by December. Grading January to April
#: as a rate headwind would be grading the model for not knowing what had not
#: happened yet, which is the no-lookahead law read backwards.
REFERENCE_WINDOWS: tuple[ReferenceWindow, ...] = (
    ReferenceWindow(
        name="2008H2",
        start="2008-07",
        end="2008-12",
        accepted=("crisis_bid", "hedge_bid"),
        note="Lehman. Gold -18% in October on forced liquidation, then +14% in November.",
    ),
    ReferenceWindow(
        name="2011",
        start="2011-01",
        end="2011-12",
        accepted=("crisis_bid", "hedge_bid"),
        note="Euro crisis and the US downgrade; gold to $1,900 in September.",
    ),
    ReferenceWindow(
        name="2013_taper",
        start="2013-05",
        end="2013-12",
        accepted=("carry_headwind",),
        note="Taper tantrum: real 10y from -0.64% to +0.80%, gold -27% on the year.",
    ),
    ReferenceWindow(
        name="2020_covid",
        start="2020-02",
        end="2020-05",
        accepted=("crisis_bid", "hedge_bid"),
        note="COVID. Gold -12% in the March dash for cash, then to a record by August.",
    ),
    ReferenceWindow(
        name="2022_tightening",
        start="2022-01",
        end="2022-12",
        accepted=("carry_headwind",),
        note="Fastest tightening cycle since 1981: real 10y +250bp, gold flat.",
    ),
)


@dataclass(frozen=True)
class WindowResult:
    """One window's verdict."""

    window: ReferenceWindow
    #: Newest month the chain was fitted on — strictly before the window.
    fitted_through: date
    months_in_fit: int
    #: Mean posterior over the window, one entry per regime.
    mean_posterior: dict[str, float]
    #: The name with the largest mean posterior.
    modal: str

    @property
    def ok(self) -> bool:
        return self.modal in self.window.accepted

    def __str__(self) -> str:
        shares = " ".join(
            f"{name}={self.mean_posterior.get(name, 0.0):.2f}" for name in GOLD_REGIMES
        )
        return (
            f"{self.window.name:16s} fit<={self.fitted_through} ({self.months_in_fit:3d}m) "
            f"-> {self.modal:15s} {'OK  ' if self.ok else 'FAIL'} [{shares}] "
            f"accepted={list(self.window.accepted)}"
        )


def _fit_cutoff(window: ReferenceWindow) -> pd.Timestamp:
    """The last instant the fit may see: the day before the window opens."""
    return pd.Timestamp(window.start) - pd.Timedelta(days=1)


def evaluate_window(
    monthly: pd.DataFrame,
    window: ReferenceWindow,
    rules: regime_mod.RegimeRules,
) -> WindowResult:
    """Fit before ``window``, freeze, filter forward, and read the posterior."""
    cutoff = _fit_cutoff(window)
    training = monthly.loc[:cutoff]

    model_fit = regime_mod.fit(training, rules)
    view = regime_mod.posterior(monthly, model_fit, rules)

    segment = view.posterior.loc[window.start : window.end]
    if segment.empty:
        raise regime_mod.RegimeUnavailable(
            f"{window.name}: the panel holds no months between {window.start} and {window.end}"
        )

    mean = {name: float(segment[name].mean()) for name in GOLD_REGIMES}
    return WindowResult(
        window=window,
        fitted_through=model_fit.fitted_through,
        months_in_fit=model_fit.n_observations,
        mean_posterior=mean,
        modal=max(mean, key=lambda name: mean[name]),
    )


def walk_forward(
    monthly: pd.DataFrame,
    rules: regime_mod.RegimeRules,
    windows: tuple[ReferenceWindow, ...] = REFERENCE_WINDOWS,
) -> list[WindowResult]:
    """Evaluate every reference window, each against its own fit."""
    results = []
    for window in windows:
        result = evaluate_window(monthly, window, rules)
        log.info("%s", result)
        results.append(result)
    return results


def summarize(results: list[WindowResult]) -> str:
    """A table, for a test failure message or a report."""
    passed = sum(1 for r in results if r.ok)
    lines = [f"walk-forward regime backtest: {passed}/{len(results)} windows"]
    lines.extend(f"  {r}" for r in results)
    return "\n".join(lines)


__all__ = [
    "REFERENCE_WINDOWS",
    "ReferenceWindow",
    "WindowResult",
    "evaluate_window",
    "summarize",
    "walk_forward",
]
