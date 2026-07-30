"""FinDynamics compute job: daily run.

The orchestrator of ``01-target-architecture.md`` §8:

    config -> observations -> WorldState (factors) -> each enabled engine
    -> predict + outputs -> HMAC write-back

It names no engine and no series. Engines come from the registry filtered by
their config enable flags, series come from ``series.yaml``, so shipping a new
engine is a yaml edit and a package — never a change to this file.

A thin CLI: everything below the argument parsing lives in the package, because
logic that only exists in a job script cannot be tested or reused.
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from findynamics.core.config import SeriesConfig, load_series_config
from findynamics.core.contracts.state import AssetState, EngineOutput, FactorState, WorldState
from findynamics.core.registry import enabled_engines
from findynamics.data.accessor import PandasPITAccessor
from findynamics.data.store import load_observations, required_series_ids
from findynamics.engines import load_engines
from findynamics.factors.compute import compute_factors
from jobs._common import base_parser, configure_logging, write_back

log = logging.getLogger("findynamics.jobs.daily")


def parse_as_of(value: str | None) -> date:
    """The information-set cutoff. UTC, because the cron runs in UTC."""
    if not value:
        return datetime.now(UTC).date()
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_world(
    as_of: date,
    config: SeriesConfig,
    *,
    cache_dir: Path | None = None,
) -> tuple[WorldState, list[Any]]:
    """Load data, score the shared factors, and hand back the world and engines.

    The engines are constructed here and returned alongside the world because
    what data to load depends on which engines are enabled — asking each for its
    ``required_series`` is how a new engine's inputs get fetched without this
    file knowing what they are.
    """
    load_engines(config)
    engines = enabled_engines(config)

    engine_series = {series_id for engine in engines for series_id in engine.required_series()}
    wanted = required_series_ids(config, engine_series)
    log.info("loading %d series for %d engine(s)", len(wanted), len(engines))

    observations = load_observations(wanted, config=config, cache_dir=cache_dir)
    if observations.empty:
        raise RuntimeError("no observations could be loaded; refusing to publish an empty state")

    accessor = PandasPITAccessor(observations, as_of)
    factors = compute_factors(accessor, config)
    log.info("scored %d/%d factors", len(factors), len(config.factors))

    return WorldState(as_of=as_of, factors=factors, series=accessor), engines


def factor_payload(factors: dict[str, FactorState]) -> list[dict[str, Any]]:
    """Factor scores in the shape ``force_scores`` takes."""
    return [
        {
            "force": state.name,
            "as_of": state.as_of.isoformat(),
            "score": state.score,
            "components": state.components,
        }
        for state in factors.values()
    ]


def asset_state_payload(state: AssetState) -> dict[str, Any]:
    return {
        "asset": state.asset,
        "as_of": state.as_of.isoformat(),
        "model_version": state.model_version,
        "regime": state.regime,
        "expected_return": state.expected_return,
        "risk_score": state.risk_score,
        "confidence": state.confidence,
        "signals": [
            {
                "name": s.name,
                "value": s.value,
                "direction": s.direction,
                "note": s.note,
            }
            for s in state.signals
        ],
        "components": state.components,
    }


def engine_output_payload(output: EngineOutput) -> dict[str, Any]:
    return {
        "asset": output.asset,
        "metric": output.metric,
        "as_of": output.as_of.isoformat(),
        "value": output.value,
        "meta": output.meta,
    }


def run(
    as_of: date,
    *,
    config_path: Path | None = None,
    cache_dir: Path | None = None,
    dry_run: bool = False,
    out: Path | None = None,
) -> int:
    config = load_series_config(config_path)
    world, engines = build_world(as_of, config, cache_dir=cache_dir)

    if not engines:
        log.error("no engines are enabled; nothing to compute")
        return 2

    states: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    failed: list[str] = []

    for engine in engines:
        try:
            state = engine.predict(world)
            rows = engine.outputs(world)
        except Exception as err:  # one engine's bad day is not the run's
            failed.append(engine.name)
            log.exception("engine %s failed: %s", engine.name, err)
            continue

        states.append(asset_state_payload(state))
        outputs.extend(engine_output_payload(row) for row in rows)
        log.info(
            "%s: regime=%s risk=%.1f confidence=%.2f (+%d output rows)",
            engine.name,
            state.regime,
            state.risk_score,
            state.confidence,
            len(rows),
        )

    if not states:
        log.error("every engine failed (%s); withholding the write-back", ", ".join(failed))
        return 1

    payload = {
        "model_version": ",".join(sorted({s["model_version"] for s in states})),
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of": as_of.isoformat(),
        "factors": factor_payload(world.factors),
        "asset_state": states,
        "engine_output": outputs,
    }

    if out is not None:
        import json

        out.write_text(json.dumps(payload, indent=2))
        log.info("wrote payload to %s", out)

    write_back(payload, dry_run=dry_run)
    # A partial run still publishes; the caller learns from the exit code that
    # something is missing without losing the engines that did work.
    return 3 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "daily")
    parser.add_argument(
        "--config", type=Path, help="path to series.yaml (defaults to the shipped one)"
    )
    parser.add_argument("--cache-dir", type=Path, help="HTTP response cache directory")
    parser.add_argument("--out", type=Path, help="also write the payload to this file")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    return run(
        parse_as_of(args.as_of),
        config_path=args.config,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
        out=args.out,
    )


if __name__ == "__main__":
    sys.exit(main())
