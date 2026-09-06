"""The decimated JSON artifact the Equity Dynamics Lab reads.

**A file, not an API.** ``/api/v1/assets/:asset/history`` reads the
``engine_output`` table and ``EngineOutput.asset`` is validated against
``ASSETS``, so publishing Ω through that path would write research rows into a
production table. This module writes a single static artifact to
``dashboard/research/omega.json`` instead: no migration, no route, no
``serving/`` change, and nothing the daily job can reach.

The artifact carries its own ``disclaimer`` field. A JSON file that outlives the
page it was written for should say what it is without the reader having to find
this docstring.

Decimation
----------

Capped at :data:`CHART_POINTS`, which is ``DEFAULT_CHART_POINTS`` in
``serving/src/api/assets.ts`` — matched rather than invented, so the page's
memory profile is the one it already has.

The **algorithm** is deliberately not the serving decimator. That one is
largest-triangle-three-buckets in TypeScript; reimplementing it in Python to be
byte-compatible would create a second implementation of one algorithm, and the
two would drift. This one buckets the series uniformly and keeps, per bucket,
the point furthest from that bucket's mean — which preserves single-day spikes,
the property LTTB is there for, without pretending to be it. The first and last
points are always kept so the artifact's span is the series' span.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from findynamics.research.omega.backtest import WalkForwardResult
from findynamics.research.omega.evaluate import Evaluation

log = logging.getLogger("findynamics.research.omega.lab")

#: ``DEFAULT_CHART_POINTS`` in ``serving/src/api/assets.ts``. One number, two
#: planes; a chart artifact that decimated to a different ceiling would change
#: the page's memory profile for no reason.
CHART_POINTS = 2000

#: The series the Lab panels draw, in the order they are stacked.
LAB_SERIES: tuple[str, ...] = (
    "omega",
    "omega_velocity",
    "omega_acceleration",
    "coupling",
    "curvature",
)

DISCLAIMER = (
    "Research artifact, not an API response. These series are an experimental "
    "hypothesis under test, are not part of any published engine state, regime or "
    "allocation, and no causal relationship is implied by their alignment on the "
    "time axis."
)


def decimate(series: pd.Series, points: int = CHART_POINTS) -> list[list[Any]]:
    """``[[iso, value], ...]``, capped at ``points``, spikes preserved.

    Non-finite values are dropped rather than emitted as ``null``: the Lab draws
    a gap wherever a date is absent, and a ``null`` in the array would have to
    be filtered on the other side anyway. Where the *absence itself* is the
    message — the coupling declining below its denominator floor — the gap is
    the message too, and the definitions block names it.
    """
    clean = series.dropna()
    clean = clean[np.isfinite(clean.to_numpy(dtype=float))]
    if clean.empty:
        return []
    if len(clean) <= points:
        return [[stamp.date().isoformat(), float(value)] for stamp, value in clean.items()]

    values = clean.to_numpy(dtype=float)
    edges = np.linspace(0, len(clean), points + 1, dtype=int)
    keep: list[int] = [0]
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        if end <= start:
            continue
        bucket = values[start:end]
        # Furthest from the bucket mean: a one-day crash survives, which is the
        # property the serving decimator exists to preserve.
        keep.append(start + int(np.argmax(np.abs(bucket - bucket.mean()))))
    keep.append(len(clean) - 1)

    ordered = sorted(set(keep))
    return [[clean.index[i].date().isoformat(), float(values[i])] for i in ordered]


def build_artifact(
    result: WalkForwardResult,
    evaluation: Evaluation,
    *,
    points: int = CHART_POINTS,
) -> dict[str, Any]:
    """The whole artifact, as a plain dict.

    Deterministic: no wall clock, no run id, no set iteration. ``generated`` is
    the **information set** the run described, not the moment the file was
    written — two runs on one snapshot must produce identical bytes, which is
    the same property ``OmegaSpec`` is built around.
    """
    spec = result.windows[-1].spec
    verdicts = evaluation.verdicts
    return {
        "disclaimer": DISCLAIMER,
        "generated": result.panel.index[-1].date().isoformat(),
        "model_version": result.model_version,
        "config_hash": result.config_hash,
        "spec": {
            "estimator": spec.estimator,
            "columns": list(spec.columns),
            "dropped": list(spec.dropped),
            "sign_reference": spec.sign_reference,
            "explained_variance_ratio": spec.explained_variance_ratio,
            "fit_start": spec.fit_start.isoformat(),
            "fit_end": spec.fit_end.isoformat(),
            "loadings": {column: spec.loading(column) for column in spec.columns},
        },
        "verdicts": {question: verdicts[question] for question in sorted(verdicts)},
        "series": {
            name: decimate(result.panel[name], points)
            for name in LAB_SERIES
            if name in result.panel.columns
        },
        "diagnostics": {
            "windows": float(len(result.windows)),
            "out_of_sample_rows": float(len(result.panel)),
            "curvature_rows": float(
                result.panel["curvature"].notna().sum()
                if "curvature" in result.panel.columns
                else 0
            ),
            "coupling_rows": float(
                result.panel["coupling"].notna().sum() if "coupling" in result.panel.columns else 0
            ),
            "arms_dropped": {
                arm: list(columns)
                for arm, columns in sorted(result.dropped_columns.items())
                if columns
            },
        },
    }


def write_artifact(artifact: dict[str, Any], path: Path) -> Path:
    """Write the artifact with sorted keys and a trailing newline.

    ``sort_keys`` and a fixed separator so two runs on one snapshot produce
    byte-identical files — the property the KK4 test asserts, and the reason the
    file is safe to commit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(artifact, sort_keys=True, indent=2, separators=(",", ": "))
    path.write_text(body + "\n")
    log.info(
        "omega lab: wrote %d series (%d points total) to %s",
        len(artifact["series"]),
        sum(len(points) for points in artifact["series"].values()),
        path,
    )
    return path


__all__ = [
    "CHART_POINTS",
    "DISCLAIMER",
    "LAB_SERIES",
    "build_artifact",
    "decimate",
    "write_artifact",
]
