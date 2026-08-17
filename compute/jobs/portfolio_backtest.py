"""FinDynamics compute job: the portfolio walk-forward backtest (task §4).

A thin CLI over :mod:`findynamics.backtest.portfolio`. Runs the balanced profile
against a static 60/30/10 benchmark, 2000-2024, monthly, point-in-time, and
writes the report artifact and a CSV of monthly returns.

    cd compute && python -m jobs.portfolio_backtest --out backtests/

The report is *robustness evidence*, not performance marketing — see the module
docstring of ``findynamics.backtest.portfolio`` for exactly what is replayed for
real and what is a documented reduced-form stand-in.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from findynamics.backtest.portfolio import render_report, returns_frame, run_backtest
from jobs._common import configure_logging

log = logging.getLogger("findynamics.jobs.portfolio_backtest")

#: The committed fixtures the backtest reads (S&P, gold, Treasury curve).
DEFAULT_FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "backtests"

REPORT_NAME = "portfolio_balanced_vs_60_30_10.md"
CSV_NAME = "portfolio_balanced_vs_60_30_10.csv"


def _parse_date(value: str | None, default: date) -> date:
    if not value:
        return default
    return datetime.strptime(value, "%Y-%m-%d").date()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "portfolio backtest")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--start", help="first rebalance month-end (YYYY-MM-DD)")
    parser.add_argument("--end", help="last rebalance month-end (YYYY-MM-DD)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    result = run_backtest(
        args.fixtures,
        start=_parse_date(args.start, date(2000, 1, 31)),
        end=_parse_date(args.end, date(2024, 12, 31)),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / REPORT_NAME
    csv_path = args.out / CSV_NAME
    report_path.write_text(render_report(result))
    returns_frame(result).to_csv(csv_path, index=False)

    metrics = result.metrics
    log.info(
        "balanced CAGR=%.2f%% maxDD=%.2f%% | benchmark CAGR=%.2f%% maxDD=%.2f%%",
        metrics["balanced"]["cagr"] * 100,
        metrics["balanced"]["max_drawdown"] * 100,
        metrics["benchmark_60_30_10"]["cagr"] * 100,
        metrics["benchmark_60_30_10"]["max_drawdown"] * 100,
    )
    log.info("wrote %s and %s", report_path, csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
