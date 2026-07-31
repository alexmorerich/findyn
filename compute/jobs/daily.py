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

from findynamics.core.artifacts import build_artifact_store
from findynamics.core.config import SeriesConfig, load_series_config
from findynamics.core.contracts.state import (
    AssetState,
    DerivedFeature,
    EngineOutput,
    FactorState,
    ForecastQuantile,
    RegimeProbability,
    WorldState,
)
from findynamics.core.engine import StateUnavailable
from findynamics.core.registry import enabled_engines
from findynamics.data.accessor import PandasPITAccessor
from findynamics.data.store import load_observations, required_series_ids
from findynamics.engines import load_engines
from findynamics.factors.compute import compute_factors
from jobs._common import base_parser, chunk_on, configure_logging, write_back

log = logging.getLogger("findynamics.jobs.daily")

#: `engine_output` rows per write-back request.
#:
#: Each engine republishes a multi-year window of every metric it charts, so this
#: array grows with the number of *enabled engines*, not with the day's news: P1
#: alone sent ~7.5k rows, P2 sends ~20k, and P3-P6 will keep multiplying it.
#: Chunking here rather than trimming the window keeps the histories intact and
#: every request inside the Worker's CPU budget. Chunks are independently
#: idempotent, so a failure costs one chunk and a re-run repairs it.
ENGINE_OUTPUT_BATCH_SIZE = 5000


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
    # Artifact storage is chosen from the environment, not by the job: a run with
    # the serving plane configured reads its fitted models out of R2, and a
    # developer run with no secrets reads the local directory. Handing it in here
    # is what stops an engine defaulting to ephemeral local storage in CI.
    engines = enabled_engines(config, artifacts=build_artifact_store())

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


def regime_state_payload(row: RegimeProbability) -> dict[str, Any]:
    """One ``regime_state`` row — the posterior, not just the winning label."""
    return {
        "asset": row.asset,
        "as_of": row.as_of.isoformat(),
        "regime": row.regime,
        "probability": row.probability,
        "model_version": row.model_version,
    }


def forecast_payload(row: ForecastQuantile) -> dict[str, Any]:
    """One ``forecast_distribution`` row — a quantile, never a point forecast."""
    return {
        "asset": row.asset,
        "as_of": row.as_of.isoformat(),
        "horizon": row.horizon,
        "quantile": row.quantile,
        "value": row.value,
        "educational_only": row.educational_only,
        "model_version": row.model_version,
    }


def derived_feature_payload(feature: DerivedFeature) -> dict[str, Any]:
    """One ``derived_features`` row.

    ``model_version`` rides on the row rather than being read off the envelope,
    because it is part of this table's primary key: a run publishing two engines
    has two versions, and the envelope's joined string is a version no single
    model ever had.
    """
    return {
        "asset": feature.asset,
        "feature": feature.feature,
        "as_of": feature.as_of.isoformat(),
        "value": feature.value,
        "model_version": feature.model_version,
    }


def run(
    as_of: date,
    *,
    config_path: Path | None = None,
    cache_dir: Path | None = None,
    dry_run: bool = False,
    out: Path | None = None,
    full_history: bool = False,
) -> int:
    config = load_series_config(config_path)
    world, engines = build_world(as_of, config, cache_dir=cache_dir)

    if not engines:
        log.error("no engines are enabled; nothing to compute")
        return 2

    # Engine-agnostic: the job says "this is a backfill run" and each engine
    # decides what that means for its own windows.
    if full_history:
        log.info("full-history run: engines will republish from the start of their data")
        for engine in engines:
            engine.full_history = True

    states: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    regimes: list[dict[str, Any]] = []
    forecasts: list[dict[str, Any]] = []
    failed: list[str] = []
    silent: list[str] = []

    for engine in engines:
        # An engine that declines to publish a state is answering correctly, not
        # failing: its model may not be fitted yet, or the information set may be
        # too thin to describe. Its per-date outputs are still worth publishing —
        # that is the whole reason the two are separate methods — so the state is
        # skipped and the run carries on.
        state = None
        try:
            state = engine.predict(world)
        except StateUnavailable as err:
            silent.append(engine.name)
            log.info("engine %s published no state: %s", engine.name, err)
        except Exception as err:  # one engine's bad day is not the run's
            failed.append(engine.name)
            log.exception("engine %s failed: %s", engine.name, err)
            continue

        try:
            rows = engine.outputs(world)
            feature_rows = engine.derived_features(world)
            regime_rows = engine.regime_states(world)
            forecast_rows = engine.forecasts(world)
        except Exception as err:
            failed.append(engine.name)
            log.exception("engine %s failed publishing its outputs: %s", engine.name, err)
            continue

        if state is not None:
            states.append(asset_state_payload(state))
        outputs.extend(engine_output_payload(row) for row in rows)
        features.extend(derived_feature_payload(row) for row in feature_rows)
        regimes.extend(regime_state_payload(row) for row in regime_rows)
        forecasts.extend(forecast_payload(row) for row in forecast_rows)
        log.info(
            "%s: %s (+%d output rows, +%d features, +%d regime rows, +%d forecast rows)",
            engine.name,
            (
                "no state"
                if state is None
                else f"regime={state.regime} risk={state.risk_score:.1f} "
                f"confidence={state.confidence:.2f}"
            ),
            len(rows),
            len(feature_rows),
            len(regime_rows),
            len(forecast_rows),
        )

    if not states and not outputs and not features:
        log.error(
            "no engine produced anything (failed: %s; silent: %s); withholding the write-back",
            ", ".join(failed) or "none",
            ", ".join(silent) or "none",
        )
        return 1

    # The envelope's version is a run label for the factor rows, which belong to
    # no engine. Every per-engine row carries its own version in its own field.
    versions = sorted({s["model_version"] for s in states} | {f["model_version"] for f in features})

    payload = {
        "model_version": ",".join(versions) or "unversioned",
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of": as_of.isoformat(),
        "factors": factor_payload(world.factors),
        "asset_state": states,
        "engine_output": outputs,
        "derived_features": features,
        "regime_state": regimes,
        "forecast_distribution": forecasts,
    }

    if out is not None:
        import json

        out.write_text(json.dumps(payload, indent=2))
        log.info("wrote payload to %s", out)

    # Two tall arrays now, so the split is applied twice. Chaining works because
    # a later chunk of the outer split carries no `derived_features` key at all,
    # and chunk_on yields such a payload through untouched — so no row is sent
    # twice and no chunk exceeds the batch size on either array.
    chunks = [
        innermost
        for outer in chunk_on(payload, "engine_output", ENGINE_OUTPUT_BATCH_SIZE)
        for inner in chunk_on(outer, "derived_features", ENGINE_OUTPUT_BATCH_SIZE)
        for innermost in chunk_on(inner, "regime_state", ENGINE_OUTPUT_BATCH_SIZE)
    ]
    if len(chunks) > 1:
        log.info(
            "splitting %d output + %d feature + %d regime rows across %d requests",
            len(outputs),
            len(features),
            len(regimes),
            len(chunks),
        )
    for chunk in chunks:
        write_back(chunk, dry_run=dry_run)

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
    parser.add_argument(
        "--full-history",
        action="store_true",
        help=(
            "Republish every engine's per-date series from the start of its data. "
            "For a deliberate backfill — the nightly cron must not use this, or it "
            "writes the whole record every night to add one row."
        ),
    )
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    return run(
        parse_as_of(args.as_of),
        config_path=args.config,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
        out=args.out,
        full_history=args.full_history,
    )


if __name__ == "__main__":
    sys.exit(main())
