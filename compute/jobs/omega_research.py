"""KK-Ω research job. **EXPERIMENTAL — run deliberately, never on a schedule.**

    python -m jobs.omega_research --force-experimental --out backtests/omega

Runs the walk-forward comparison of arms A / A′ / B / C / D / E, answers Q1–Q5
against the pre-registered rules in ``docs/research/kk-omega-design.md`` §8, and
writes the artifacts KK5 assembles its report from.

Three properties, all deliberate:

**No network and no API keys.** The committed snapshot
``tests/fixtures/equity_prices.csv`` is the default input, so every number is
reproducible from a clone.

**No cron, no workflow, no console script.** This module is absent from
``[project.scripts]`` and from every file under ``.github/workflows/``. A
research job that can start itself is a research job that will eventually write
to something that matters — and the whole point of the ``Research is
quarantined`` contract is that nothing here can be reached by accident.

**It refuses to run unless told twice.** ``config/research/omega.yaml`` ships
``enabled: false``, so a plain invocation stops. ``--force-experimental`` is the
second word, and the banner prints either way: a reader who finds this output in
a directory six months from now needs to know what it is without going to look.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from findynamics.research.omega import load_omega_config
from findynamics.research.omega.backtest import walk_forward
from findynamics.research.omega.evaluate import evaluate
from findynamics.research.omega.lab import build_artifact, write_artifact
from findynamics.research.omega.report import load_artifacts, render_report, write_report
from jobs._common import configure_logging

log = logging.getLogger("findynamics.jobs.omega_research")

DEFAULT_SNAPSHOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "equity_prices.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "backtests" / "omega"
DEFAULT_REPORT = Path(__file__).resolve().parents[2] / "docs" / "research" / "kk-omega-report.md"

#: The Lab reads this file directly from the static site. It is a committed
#: research artifact and deliberately not an API response — see
#: ``findynamics/research/omega/lab.py`` for why that distinction is load-bearing.
#:
#: Outside ``dashboard/public/`` on purpose: Astro copies that directory into
#: every build, so an artifact kept there ships on production pages where the
#: Lab is compiled out. ``dashboard/scripts/copy-research-artifact.mjs`` puts
#: it into ``dist/`` only when ``PUBLIC_OMEGA_LAB`` is set.
DEFAULT_LAB_JSON = Path(__file__).resolve().parents[2] / "dashboard" / "research" / "omega.json"

BANNER = """
================================================================================
  KK-Ω RESEARCH RUN — EXPERIMENTAL, NOT A PRODUCTION SIGNAL
  A fifth-dimensional latent-state representation inspired by Kaluza-Klein
  geometry, tested as a quantitative modelling hypothesis.
  Outputs are states, scores and quantiles. Never a price target, never a trade.
  docs/research/kk-omega-design.md holds the pre-registered decision rules.
================================================================================
""".strip()


def load_snapshot(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("obs_date", "release_date", "revision_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def run(
    *,
    snapshot: Path,
    out: Path,
    shuffled: bool = False,
    lab_json: Path | None = None,
) -> int:
    """Walk forward, evaluate, and write the CSV artifacts."""
    observations = load_snapshot(snapshot)
    config = load_omega_config()

    result = walk_forward(observations, config, shuffle_targets=shuffled)
    evaluation = evaluate(result)

    out.mkdir(parents=True, exist_ok=True)
    prefix = "shuffled_" if shuffled else ""
    written: list[Path] = []

    for name, frame in (
        ("panel", result.panel),
        ("predictions", result.predictions),
        ("training_means", result.training_means),
        ("q1_forward_return", evaluation.q1),
        ("q2_forward_volatility", evaluation.q2),
        ("q3_transitions", evaluation.q3),
        ("q4_coupling", evaluation.q4),
        ("q5_performance", evaluation.q5),
        ("sub_periods", evaluation.sub_periods),
        ("regime_matrix", evaluation.regimes.matrix),
        ("regime_lead_lag", evaluation.regimes.lead_lag),
    ):
        path = out / f"{prefix}{name}.csv"
        # float_format so two runs on one machine produce identical bytes; the
        # default repr rounds differently across pandas versions.
        frame.to_csv(path, float_format="%.10g")
        written.append(path)

    windows = pd.DataFrame([record.as_dict() for record in result.windows])
    windows.to_csv(out / f"{prefix}windows.csv", index=False)
    written.append(out / f"{prefix}windows.csv")

    if lab_json is not None and not shuffled:
        written.append(write_artifact(build_artifact(result, evaluation), lab_json))

    log.info(
        "omega research: %d window(s), %d out-of-sample row(s), config %s, model %s; "
        "verdicts %s; wrote %d artifact(s) to %s",
        len(result.windows),
        len(result.panel),
        result.config_hash,
        result.model_version,
        ", ".join(f"{k}={v}" for k, v in evaluation.verdicts.items()),
        len(written),
        out,
    )
    return 0


def regenerate_report(*, artifacts: Path, out: Path) -> int:
    """Rewrite the KK5 report from the stored CSVs.

    Reads only committed artifacts, so it needs neither the fixture nor a
    two-minute walk-forward — which is what makes "regenerates byte-identically"
    a check anyone can run rather than a claim.
    """
    frames = load_artifacts(artifacts)
    windows = frames.get("windows")
    model_version = "unknown"
    if windows is not None and not windows.empty and "spec" in windows.columns:
        import ast

        spec = ast.literal_eval(str(windows["spec"].iloc[-1]))
        model_version = str(spec.get("model_version", "unknown"))

    body = render_report(
        frames,
        config_hash=_config_hash(),
        model_version=model_version,
    )
    path = write_report(body, out)
    log.info("omega research: regenerated %s from %s", path, artifacts)
    return 0


def _config_hash() -> str:
    from findynamics.research.omega.backtest import config_hash

    return config_hash(load_omega_config())


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="KK-Ω walk-forward research run")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--force-experimental",
        action="store_true",
        help="run even though config/research/omega.yaml ships enabled: false",
    )
    parser.add_argument(
        "--emit-lab-json",
        action="store_true",
        help=f"also write the decimated Lab artifact to {DEFAULT_LAB_JSON}",
    )
    parser.add_argument("--lab-json", type=Path, default=DEFAULT_LAB_JSON)
    parser.add_argument(
        "--shuffled",
        action="store_true",
        help="the acceptance-gate control: permute every target and re-run",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="regenerate docs/research/kk-omega-report.md from the stored CSVs and exit",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="where --report writes; defaults to docs/research/kk-omega-report.md",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)
    print(BANNER, file=sys.stderr)

    # --report reads committed artifacts and computes nothing, so it does not
    # need the experimental gate: regenerating a document from files already in
    # the repository is not running the research.
    if args.report:
        return regenerate_report(artifacts=args.out, out=args.report_out or DEFAULT_REPORT)

    config = load_omega_config()
    if not config.enabled and not args.force_experimental:
        log.error(
            "config/research/omega.yaml has enabled: false and --force-experimental was not "
            "passed. This module is quarantined; running it is a deliberate act and the flag "
            "is how you say so."
        )
        return 2

    return run(
        snapshot=args.snapshot,
        out=args.out,
        shuffled=args.shuffled,
        lab_json=args.lab_json if args.emit_lab_json else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
