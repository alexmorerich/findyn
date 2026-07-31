"""FinDynamics compute job: regenerate the committed M4 diagnostics report.

Run deliberately, not on a schedule — same reasoning as `jobs.backtest`. The
report is an artifact of a model version, and a file that changed under a cron
would make every claim it contains un-anchored.

    python -m jobs.diagnostics --out docs/backtests/equity-p3c.md

Reads the committed price snapshot by default, so it is reproducible from a
clone with no API keys. The macro inputs the RII's credit and liquidity
components want are *not* in that snapshot, so those components are absent from
this report and the run says so — which is the behaviour under test as much as
the numbers are.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

from findynamics.core.artifacts import ArtifactStore
from findynamics.core.config import load_series_config
from findynamics.core.contracts.state import WorldState
from findynamics.data.accessor import PandasPITAccessor
from findynamics.engines.equity import diagnostics as diagnostics_mod
from findynamics.engines.equity.engine import ALL_ROLES, EquityEngine
from jobs._common import configure_logging
from jobs.backtest import DEFAULT_SNAPSHOT, load_snapshot

log = logging.getLogger("findynamics.jobs.diagnostics")


def run(*, snapshot: Path, out: Path, as_of: date, tmpdir: Path) -> int:
    observations = load_snapshot(snapshot)
    config = load_series_config()
    world = WorldState(as_of=as_of, factors={}, series=PandasPITAccessor(observations, as_of))

    # An in-memory store, then fit into it. The regime model is what the whole
    # instability layer is conditioned on, and without a fit `analyze` correctly
    # returns no instability view at all — so the report needs its own fit rather
    # than whatever happens to be in R2 for production. Same choice `jobs.backtest`
    # makes, and the reason both are reproducible from a clone.
    engine = EquityEngine(config, ArtifactStore(directory=tmpdir))
    engine.fit(world)
    engine._cache = None
    analysis = engine.analyze(world, roles=ALL_ROLES)

    # The engine's own view, not a reimplementation of it. A diagnostics report
    # that builds the tail fit differently from production is a report about a
    # model nobody is running.
    view = analysis.instability
    if view is None:
        log.error("the engine published no instability view; nothing to diagnose")
        return 1

    deep = analysis.features.get("deep_history")
    tail = None
    if view.tail is not None and deep is not None:
        tail = diagnostics_mod.tail_diagnostics(deep.log_price, fit=view.tail)
        log.info(
            "tail: %d raw exceedances declustered to %d episodes",
            tail.raw_exceedances,
            view.tail.exceedances,
        )

    # Diagnosed on the calibration path, not the published slice. The published
    # RII covers ten years — two of the four episodes and one of the four calm
    # windows — and grading an index designed to rank a century against that
    # would say more about FRED's ten-year cap than about §3.2.
    source = view.rii_source or view.rii
    rii = None
    try:
        rii = diagnostics_mod.rii_diagnostics(source.index, source.components)
        log.info("rii: episode/calm separation %+.1f points", rii.gap)
    except ValueError as err:
        log.warning("no RII diagnostics: %s", err)

    simulation = None
    if view.simulation is not None:
        # Against the calibration record too, and for the same reason: the
        # publication path is 2015+, which annualizes to ~13%/yr. Judging a
        # 12-year simulated median against the best decade in the sample would
        # make an honest simulation look pessimistic.
        reference = analysis.features.get("calibration") or analysis.features["publication"]
        simulation = diagnostics_mod.simulation_diagnostics(
            view.simulation,
            reference.log_price,
            periods_per_year=reference.series.periods_per_year,
        )
        log.info("simulation: realized drift %+.2f%%/yr", simulation.realized_drift * 100)

    if view.rii.missing:
        log.info("rii built without: %s", ", ".join(view.rii.missing))

    report = diagnostics_mod.render_report(
        tail=tail,
        rii=rii,
        simulation=simulation,
        calibration_series=analysis.roles.calibration.series_id,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d"),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    log.info("wrote %s (%d lines)", out, report.count("\n"))
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "backtests" / "equity-p3c.md",
    )
    parser.add_argument("--as-of", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    with tempfile.TemporaryDirectory() as tmp:
        return run(snapshot=args.snapshot, out=args.out, as_of=as_of, tmpdir=Path(tmp))


if __name__ == "__main__":
    sys.exit(main())
