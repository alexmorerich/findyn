"""One information set is one artifact — the property KK3 is built on.

``tests/engines/equity/test_reproducibility.py`` exists because "almost the same
model" cost this repository a monthly job that went red with a 409 nobody could
act on (issue #6): the HMM's k-means init is OpenMP-parallel, a threaded
floating-point reduction does not fix the order it sums in, and two fits at one
seed landed a relative ~1e-9 apart.

The same exposure exists here. PCA's SVD is BLAS-parallel for the same reason
k-means is, and KK3 stores one ``OmegaSpec`` per walk-forward window and compares
a truncated re-run against the full one bit-for-bit. So the estimator pins the
thread pool for the fit, takes ``svd_solver="full"`` rather than the randomized
solver, and projects with ``math.fsum`` instead of a dot product.

Across processes, not only within one: an in-process repeat shares a warmed BLAS
and a single thread-pool state, so it is the weaker of the two checks and would
pass on a build where the property does not hold.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd

from tests.research.omega.conftest import SNAPSHOT_AS_OF, accessor_at

COMPUTE_ROOT = Path(__file__).resolve().parents[3]


def test_two_fits_in_one_process_agree_on_every_bit(omega_engine, omega_observations):
    block = omega_engine.features(accessor_at(omega_observations))

    first = omega_engine.fit(block)
    second = omega_engine.fit(block)

    assert json.dumps(first.as_dict(), sort_keys=True) == json.dumps(
        second.as_dict(), sort_keys=True
    )


def test_the_spec_serializes_identically_in_a_separate_process(omega_engine, omega_observations):
    """The check that would catch a BLAS-dependent SVD.

    A subprocess gets a cold thread pool and re-reads the fixture from disk, so
    a fit that depended on either would disagree with the one computed here.
    """
    block = omega_engine.features(accessor_at(omega_observations))
    here = json.dumps(omega_engine.fit(block).as_dict(), sort_keys=True)

    program = textwrap.dedent(
        f"""
        import json
        import pandas as pd
        from datetime import date
        from findynamics.data.accessor import PandasPITAccessor
        from findynamics.research.omega import OmegaEngine

        frame = pd.read_csv("tests/fixtures/equity_prices.csv")
        for column in ("obs_date", "release_date", "revision_date"):
            frame[column] = pd.to_datetime(frame[column])

        engine = OmegaEngine.from_shipped_config()
        accessor = PandasPITAccessor(frame, date({SNAPSHOT_AS_OF.year},
                                                 {SNAPSHOT_AS_OF.month},
                                                 {SNAPSHOT_AS_OF.day}))
        spec = engine.fit(engine.features(accessor))
        print(json.dumps(spec.as_dict(), sort_keys=True))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=COMPUTE_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == here, result.stderr


def test_the_whole_path_is_byte_identical_across_two_runs(omega_engine, omega_observations):
    """Not only the spec: the serialized path is what an artifact would carry.

    ``test_reproducibility.py`` compares the stored document rather than a dict
    for the same reason — it is the serialization the storage layer sees, and a
    dict comparison can pass while the bytes differ.
    """
    accessor = accessor_at(omega_observations)

    first = omega_engine.fit_transform(accessor).frame().to_csv()
    second = omega_engine.fit_transform(accessor).frame().to_csv()

    assert first == second


def test_the_spec_round_trips_through_its_own_serialization(omega_engine, omega_observations):
    """KK3 stores one of these per window and reads them back to build the report."""
    block = omega_engine.features(accessor_at(omega_observations))
    spec = omega_engine.fit(block)

    restored = type(spec).from_dict(json.loads(json.dumps(spec.as_dict())))

    assert restored == spec
    pd.testing.assert_series_equal(
        omega_engine.transform(block, spec),
        omega_engine.transform(block, restored),
        check_exact=True,
    )


def test_no_wall_clock_reaches_the_spec(omega_engine, omega_observations):
    """The structural half of the determinism story (issue #6).

    ``fit_start`` / ``fit_end`` / ``n_observations`` describe the data. Nothing
    describes when the fit happened, which is what lets two runs a month apart
    produce the same bytes.
    """
    block = omega_engine.features(accessor_at(omega_observations))
    document = omega_engine.fit(block).as_dict()

    assert set(document) == {
        "estimator",
        "columns",
        "dropped",
        "mean",
        "scale",
        "loadings",
        "explained_variance_ratio",
        "sign_reference",
        "fit_start",
        "fit_end",
        "n_observations",
        "model_version",
    }
    for key in ("fitted_at", "created_at", "timestamp", "run_id"):
        assert key not in document
