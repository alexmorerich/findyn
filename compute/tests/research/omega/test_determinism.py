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
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from findynamics.research.omega import OmegaEngine
from findynamics.research.omega.diagnostics import equity_reference
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


@pytest.mark.slow
def test_the_coupling_and_curvature_paths_are_byte_identical_across_processes(
    omega_observations,
):
    """The KK2 half of the determinism story.

    ``C`` comes out of a closed-form solve over ``np.cumsum`` running totals and
    ``K`` out of a PCA in ``fitted`` mode, so both touch the reductions that put
    the ~1e-9 wobble into the HMM (issue #6). A subprocess gets a cold thread
    pool and re-reads the fixture from disk, so a path that depended on either
    would disagree with the one computed here.
    """
    engine = OmegaEngine.from_shipped_config()
    data = engine.analyze(accessor_at(omega_observations))
    here = _serialize_paths(data)

    program = textwrap.dedent(
        f"""
        import pandas as pd
        from datetime import date
        from findynamics.data.accessor import PandasPITAccessor
        from findynamics.research.omega import OmegaEngine
        from tests.research.omega.test_determinism import _serialize_paths

        frame = pd.read_csv("tests/fixtures/equity_prices.csv")
        for column in ("obs_date", "release_date", "revision_date"):
            frame[column] = pd.to_datetime(frame[column])

        engine = OmegaEngine.from_shipped_config()
        accessor = PandasPITAccessor(frame, date({SNAPSHOT_AS_OF.year},
                                                 {SNAPSHOT_AS_OF.month},
                                                 {SNAPSHOT_AS_OF.day}))
        print(_serialize_paths(engine.analyze(accessor)), end="")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=COMPUTE_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == here, result.stderr


def _serialize_paths(data) -> str:
    """Every published KK2 series as one CSV blob — what an artifact would hold."""
    frame = pd.DataFrame(
        {
            "omega": data.path.omega,
            "coupling_ratio": data.couplings["ratio"].coupling,
            "coupling_regression": data.couplings["regression"].coupling,
            "coupling_interaction": data.couplings["interaction"].coupling,
            "curvature": data.curvature.curvature,
            "curvature_percentile": data.curvature.percentile,
        }
    )
    return frame.to_csv(float_format="%.17g")


def test_the_curvature_weights_are_stable_across_two_fits(omega_observations, omega_config):
    """``fitted`` mode twice on one information set, to the last bit.

    The PCA behind the weighting is the same BLAS-parallel SVD the Ω estimator
    pins its thread pool for, and the weights travel in the artifact.
    """
    from tests.research.omega.conftest import with_params

    engine = OmegaEngine(config=with_params(omega_config, curvature={"weights": "fitted"}))
    accessor = accessor_at(omega_observations)
    reference = equity_reference(accessor)

    first = engine.analyze(accessor, reference=reference).curvature
    second = engine.analyze(accessor, reference=reference).curvature

    assert first.weights == second.weights
    assert (first.fit_start, first.fit_end) == (second.fit_start, second.fit_end)
    pd.testing.assert_series_equal(first.curvature, second.curvature, check_exact=True)


# ---------------------------------------------------------------------------
# KK6: the environment gate
# ---------------------------------------------------------------------------
#
# The tests above answer "does this fit twice the same way in one process". The
# three below answer the question that actually bit this repository in issue #6:
# does it fit the same way under a *different environment*. Each varies one
# thing that silently changes a floating-point reduction order or a container
# iteration order, and each would have caught a real defect:
#
#   PYTHONHASHSEED   set/dict iteration order. Ω's surviving columns come out of
#                    an availability check over a frame's columns; if that set
#                    were ever iterated instead of sorted, the loading vector
#                    would be permuted and Ω would be a different projection
#                    under a different seed, with no error anywhere.
#   OMP_NUM_THREADS  a threaded BLAS reduction sums its partials in whatever
#                    order the threads finish. `regime/hmm.py` documents the
#                    ~1e-9 this moved the HMM by, and the estimator pins the
#                    thread pool for the same reason. If one of these fails the
#                    fix is `threadpool_limits(1)` around the fit — never a
#                    tolerance on the assertion.


def _subprocess_spec(env_overrides: dict[str, str]) -> str:
    """Fit Ω in a clean interpreter under ``env_overrides`` and return its JSON."""
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
        block = engine.features(accessor)
        spec = engine.fit(block)
        omega = engine.transform(block, spec)
        print(json.dumps({{
            "spec": spec.as_dict(),
            "omega": [round(float(v), 15) for v in omega.dropna().to_numpy()[-500:]],
        }}, sort_keys=True))
        """
    )
    environment = {**os.environ, **env_overrides}
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=COMPUTE_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    return result.stdout


@pytest.mark.slow
def test_omega_is_identical_under_two_hash_seeds():
    """Catches any set or dict iteration order reaching the projection.

    ``PYTHONHASHSEED`` has to be set in the child's environment before the
    interpreter starts — setting it inside the process does nothing, which is
    why this is a subprocess test and not a monkeypatch.
    """
    first = _subprocess_spec({"PYTHONHASHSEED": "0"})
    second = _subprocess_spec({"PYTHONHASHSEED": "12345"})

    assert first == second


@pytest.mark.slow
def test_omega_is_identical_under_one_and_four_openmp_threads():
    """Catches an unpinned thread pool in the SVD.

    If this fails, the fix is `threadpool_limits(1)` around the fit, exactly as
    ``engines/equity/regime/hmm.py`` does it. It is **not** to loosen the
    assertion to a tolerance: artifacts here are compared by content, and
    "almost the same model" is what cost this repository a monthly job that went
    red with an unactionable 409.
    """
    single = _subprocess_spec({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    multiple = _subprocess_spec({"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})

    assert single == multiple


@pytest.mark.slow
def test_the_walk_forward_artifact_is_identical_across_processes(omega_observations):
    """The whole pipeline, not only Ω: `C`, `K` and every arm's prediction.

    A short window so the test is affordable; the point is the *environment*
    variation, and a leak of thread-order into the coupling's `np.cumsum` or the
    curvature's PCA would show up on 2,000 rows as readily as on 5,000.
    """
    program = textwrap.dedent(
        """
        import copy, json
        import pandas as pd
        from findynamics.research.omega import load_omega_config
        from findynamics.research.omega.config import OmegaConfig
        from findynamics.research.omega.backtest import walk_forward

        frame = pd.read_csv("tests/fixtures/equity_prices.csv")
        for column in ("obs_date", "release_date", "revision_date"):
            frame[column] = pd.to_datetime(frame[column])
        frame = frame[frame["obs_date"] <= pd.Timestamp(2012, 12, 31)]

        base = load_omega_config()
        params = copy.deepcopy(base.params)
        params["walk_forward"] = {**params.get("walk_forward", {}), "cadence_months": 6}
        config = OmegaConfig(enabled=base.enabled, experimental=base.experimental, params=params)

        result = walk_forward(frame, config)
        columns = [c for c in ("omega", "coupling", "curvature") if c in result.panel]
        print(result.panel[columns].to_csv(float_format="%.17g"), end="")
        print(result.predictions.to_csv(float_format="%.17g"), end="")
        """
    )

    def run(env_overrides: dict[str, str]) -> str:
        return subprocess.run(
            [sys.executable, "-c", program],
            cwd=COMPUTE_ROOT,
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, **env_overrides},
        ).stdout

    assert run({"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"}) == run(
        {"PYTHONHASHSEED": "999", "OMP_NUM_THREADS": "4"}
    )
