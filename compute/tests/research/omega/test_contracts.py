"""The invariants ``OmegaSpec`` and ``OmegaPath`` refuse to be built without.

Validation lives in ``__post_init__`` for the reason ``core/contracts/state.py``
gives: a nonsensical value is cheapest to catch where it is constructed. The
tests below are the standing check that each refusal still fires — a validator
nobody has watched reject anything is a validator that may have stopped.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from findynamics.research.omega.contracts import OmegaContractError, OmegaPath, OmegaSpec
from tests.research.omega.conftest import dummy_spec


def spec(**overrides) -> OmegaSpec:
    base = {
        "estimator": "pca",
        "columns": ("drawdown", "realized_vol"),
        "dropped": ("credit_velocity",),
        "mean": (0.1, 0.2),
        "scale": (1.0, 2.0),
        "loadings": (0.6, 0.8),
        "explained_variance_ratio": 0.55,
        "sign_reference": "realized_vol",
        "fit_start": date(2000, 1, 3),
        "fit_end": date(2020, 1, 3),
        "n_observations": 5_000,
        "model_version": "omega-0.1.0",
    }
    return OmegaSpec(**{**base, **overrides})


def test_a_valid_spec_round_trips():
    original = spec()

    assert OmegaSpec.from_dict(original.as_dict()) == original
    assert original.loading("realized_vol") == 0.8
    assert original.width == 2


def test_a_negative_loading_on_the_sign_reference_is_unconstructible():
    """The invariant that makes an unpinned Ω impossible rather than unlikely.

    An unpinned sign does not fail loudly — it produces a Ω̇ spike train aligned
    to the refit calendar that reads exactly like a signal. Refusing it in the
    contract means no estimator, present or future, can emit one.
    """
    with pytest.raises(OmegaContractError, match="spike train"):
        spec(loadings=(0.6, -0.8))


def test_a_sign_reference_outside_the_used_columns_is_refused():
    with pytest.raises(OmegaContractError, match="not among the used columns"):
        spec(sign_reference="credit_velocity")


def test_parameter_vectors_must_match_the_column_count():
    """A spec whose loadings and columns drifted apart projects onto wrong axes.

    The failure has no symptom: the result is still a plausible-looking float on
    every date, which is why it is a construction-time refusal rather than a
    review item.
    """
    with pytest.raises(OmegaContractError, match="expected 2 entries"):
        spec(loadings=(0.6, 0.8, 0.1))


def test_columns_must_be_sorted_and_unique():
    with pytest.raises(OmegaContractError, match="must be sorted"):
        spec(columns=("realized_vol", "drawdown"))
    with pytest.raises(OmegaContractError, match="duplicates"):
        spec(columns=("drawdown", "drawdown"))


def test_a_column_cannot_be_both_used_and_dropped():
    with pytest.raises(OmegaContractError, match="both used and dropped"):
        spec(dropped=("drawdown",))


def test_a_non_positive_scale_is_refused():
    """A constant column belongs in ``dropped``, not in a division."""
    with pytest.raises(OmegaContractError, match="carries no information"):
        spec(scale=(1.0, 0.0))


def test_a_non_finite_parameter_is_refused():
    with pytest.raises(OmegaContractError, match="must be finite"):
        spec(mean=(0.1, float("nan")))


def test_an_explained_variance_ratio_outside_zero_one_is_refused():
    with pytest.raises(OmegaContractError, match=r"within \[0, 1\]"):
        spec(explained_variance_ratio=1.4)


def test_a_datetime_is_not_a_date():
    """``datetime`` subclasses ``date`` and carries a time that would break t-1.

    The same refusal ``core/contracts/state.py::_check_date`` makes, for the
    same reason: a fit window whose bounds carry a clock is a fit window that
    cannot be compared against another run's.
    """
    with pytest.raises(OmegaContractError, match="expected datetime.date"):
        spec(fit_start=datetime(2000, 1, 3, 9, 30))


def test_an_inverted_fit_window_is_refused():
    with pytest.raises(OmegaContractError, match="is after fit_end"):
        spec(fit_start=date(2020, 1, 3), fit_end=date(2000, 1, 3))


def _path(**overrides) -> OmegaPath:
    index = pd.bdate_range(date(2020, 1, 1), periods=5)
    base = {
        "omega": pd.Series(np.arange(5.0), index=index),
        "omega_velocity": pd.Series(np.arange(5.0), index=index),
        "omega_acceleration": pd.Series(np.arange(5.0), index=index),
        "omega_volatility": pd.Series(np.arange(5.0), index=index),
        "omega_zscore": pd.Series(np.arange(5.0), index=index),
        "omega_regime": pd.Series(["latent_calm"] * 5, index=index),
        "spec": dummy_spec(),
        "diagnostics": {"observations": 5.0},
    }
    return OmegaPath(**{**base, **overrides})


def test_a_valid_path_frames_and_summarizes():
    path = _path()

    assert list(path.frame().columns) == [
        "omega",
        "omega_velocity",
        "omega_acceleration",
        "omega_volatility",
        "omega_zscore",
        "omega_regime",
    ]
    assert path.frame().index.name == "obs_date"
    assert path.latest()["omega"] == 4.0
    assert len(path) == 5


def test_every_series_on_a_path_must_share_one_index():
    """Two series on different calendars would align silently downstream."""
    shifted = pd.Series(np.arange(5.0), index=pd.bdate_range(date(2021, 1, 1), periods=5))

    with pytest.raises(OmegaContractError, match="different index"):
        _path(omega_velocity=shifted)


def test_a_label_outside_the_vocabulary_is_refused():
    index = pd.bdate_range(date(2020, 1, 1), periods=5)

    with pytest.raises(OmegaContractError, match="outside the vocabulary"):
        _path(omega_regime=pd.Series(["bear"] * 5, index=index))


def test_a_non_finite_diagnostic_is_refused():
    with pytest.raises(OmegaContractError, match="must be finite"):
        _path(diagnostics={"observations": float("inf")})


def test_a_path_must_carry_a_real_spec():
    with pytest.raises(OmegaContractError, match="must be an OmegaSpec"):
        _path(spec={"estimator": "pca"})


def test_replace_revalidates():
    """Frozen dataclasses are copied with ``replace``, which re-runs the checks."""
    path = _path()

    with pytest.raises(OmegaContractError, match="outside the vocabulary"):
        replace(path, omega_regime=path.omega_regime.replace("latent_calm", "calm"))
