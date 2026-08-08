"""A refit of one information set is one artifact — issue #6.

Fitted artifacts are immutable and addressed by ``(model_version, fit date)``,
and the serving plane decides whether a re-write is a no-op or a conflict by
**comparing the bytes**. That contract only works if fitting the same data twice
produces the same bytes, which for a long time it did not:

* ``fitted_at`` stamped the wall clock into the document, so no two runs could
  ever agree, however identical the model;
* the HMM's k-means initialization is OpenMP-parallel, and a threaded
  floating-point reduction does not fix the order it sums in — two fits under
  the same seed landed a relative ~1e-9 apart;
* the transition classifiers are fitted on the HMM's posteriors, so that wobble
  moved a split threshold and re-serialized a 340 kB booster that differed from
  its predecessor.

The consequence was a monthly job that went red on its second run of the month
with a 409 nobody could act on. These tests are the standing check that it stays
fixed; ``jobs/_common.refit_cutoff`` is the other half, pinning *which* data the
month's refit sees.

Two neighbours cover the rest of the issue, and live in ``test_engine.py``
because they are about the document rather than about determinism:
``test_the_artifact_carries_no_wall_clock_timestamp`` checks structurally that
nothing clock-derived has crept back in, and
``test_two_fits_on_one_information_set_agree_on_what_they_publish`` follows two
separately stored artifacts through ``predict`` to the published state.
"""

from __future__ import annotations

import json

import pytest

from findynamics.engines.equity.engine import ALL_ROLES, ARTIFACT_NAME, EquityEngine
from findynamics.engines.equity.regime.design import build_design
from findynamics.engines.equity.regime.hmm import fit_hmm
from tests.engines.equity.conftest import world_from


@pytest.fixture(scope="module")
def calibration_design(equity_observations, config_module):
    """The design matrix the production refit fits its regime model on."""
    from findynamics.core.artifacts import ArtifactStore

    engine = EquityEngine(config_module, ArtifactStore(directory=None))
    analysis = engine.analyze(world_from(equity_observations), roles=ALL_ROLES)
    return build_design(analysis.features["calibration"])


@pytest.fixture(scope="module")
def config_module():
    from findynamics.core.config import load_series_config

    return load_series_config()


def test_the_hmm_lands_on_the_same_parameters_every_time(calibration_design):
    """The root cause, at the level it lives at.

    Two fits, one process, same data, same seed. Before the thread pool was
    pinned this failed reliably — not with a different model, but with the same
    model to nine significant figures and a different last bit, which is the
    hardest kind of difference to notice and the kind an equality test on bytes
    is unforgiving about.
    """
    first = fit_hmm(calibration_design)
    second = fit_hmm(calibration_design)

    assert first.as_dict() == second.as_dict()


def test_the_stored_artifact_is_byte_identical_across_refits(
    equity_engine, equity_observations, artifacts
):
    """The property the storage contract actually depends on.

    ``putArtifact`` treats a re-write of identical bytes as a no-op and different
    bytes as a 409. This is that comparison, run against the serialized document
    rather than against a dict, because it is the serialization that R2 sees.
    """
    world = world_from(equity_observations)

    equity_engine.fit(world)
    first = json.dumps(artifacts.load(ARTIFACT_NAME), sort_keys=True)

    equity_engine._cache = None  # defeat the per-run memoization
    equity_engine.fit(world)
    second = json.dumps(artifacts.load(ARTIFACT_NAME), sort_keys=True)

    assert first == second
