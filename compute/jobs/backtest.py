"""FinDynamics compute job: regenerate the committed backtest report.

Run deliberately, not on a schedule. The report is an artifact of a model
version — the number that matters is "how did *this* model do", and a file that
silently changed under a cron would make every claim in the repository
un-anchored.

    python -m jobs.backtest --out docs/backtests/equity-p3b.md

Reads the committed price snapshot by default rather than the network, so the
report is reproducible from a clone with no API keys.
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from findynamics.core.artifacts import ArtifactStore
from findynamics.core.config import load_series_config
from findynamics.core.contracts.state import WorldState
from findynamics.data.accessor import PandasPITAccessor
from findynamics.engines.equity import backtest as backtest_mod
from findynamics.engines.equity.engine import ALL_ROLES, EquityEngine
from findynamics.engines.equity.regime.design import build_design
from findynamics.engines.equity.regime.hmm import RegimeModel, fit_hmm
from jobs._common import configure_logging

log = logging.getLogger("findynamics.jobs.backtest")

DEFAULT_SNAPSHOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "equity_prices.csv"


def load_snapshot(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("obs_date", "release_date", "revision_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def run(
    *,
    snapshot: Path,
    out: Path,
    as_of: date,
    refit_months: int,
    cross_check: bool = True,
) -> int:
    observations = load_snapshot(snapshot)
    config = load_series_config()
    world = WorldState(as_of=as_of, factors={}, series=PandasPITAccessor(observations, as_of))

    engine = EquityEngine(config, ArtifactStore(directory=None))
    analysis = engine.analyze(world, roles=ALL_ROLES)

    calibration = build_design(analysis.features["calibration"])
    is_proxy = analysis.roles.calibration_is_proxy

    log.info(
        "backtesting %s (%d rows) with %d-month refits",
        calibration.series.series_id,
        len(calibration),
        refit_months,
    )

    # Labels come from a single fit over the whole history: predictions must be
    # out of sample, but "what the regime turned out to be" is ex-post history.
    reference = RegimeModel(fit_hmm(calibration, is_proxy=is_proxy)).states(calibration)

    result = backtest_mod.run_backtest(
        calibration,
        analysis.features["calibration"].log_price,
        refit_months=refit_months,
        is_proxy=is_proxy,
        reference_regimes=reference,
    )

    if cross_check and "deep_history" in analysis.features:
        deep = build_design(analysis.features["deep_history"])
        log.info("cross-checking on %s (%d rows)", deep.series.series_id, len(deep))
        try:
            result.cross_check = backtest_mod.cross_check_deep_history(
                deep, analysis.features["deep_history"].log_price
            )
        except ValueError as err:
            log.warning("deep-history cross-check unavailable: %s", err)

    report = backtest_mod.render_report(
        result,
        calibration_series=calibration.series.series_id,
        is_proxy=is_proxy,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d"),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    log.info("wrote %s (%d lines)", out, report.count("\n"))
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__ or "backtest")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "backtests" / "equity-p3b.md",
    )
    parser.add_argument("--as-of", default="2026-07-30")
    parser.add_argument("--refit-months", type=int, default=backtest_mod.DEFAULT_REFIT_MONTHS)
    parser.add_argument("--no-cross-check", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    return run(
        snapshot=args.snapshot,
        out=args.out,
        as_of=datetime.strptime(args.as_of, "%Y-%m-%d").date(),
        refit_months=args.refit_months,
        cross_check=not args.no_cross_check,
    )


if __name__ == "__main__":
    sys.exit(main())
