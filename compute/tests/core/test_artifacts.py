"""Artifact storage: local, remote, and the lifecycle contract between them.

The failure this file exists to prevent is not a crash. It is a scheduled run
that quietly publishes nothing for months because the refit wrote a fitted model
into a container that no longer exists — which is precisely what
``compute/artifacts/`` does in GitHub Actions, and precisely what nobody would
notice until someone asked why ``/assets/equity/state`` had never returned a
number.
"""

from __future__ import annotations

import json

import httpx
import pytest

from findynamics.core.artifacts import (
    ADMIN_URL_ENV,
    SECRET_ENV,
    VERSION_ENV,
    ArtifactStore,
    RemoteArtifactStore,
    build_artifact_store,
)
from findynamics.core.signing import sign

SECRET = "test-secret"
BASE = "https://findyn.test/admin/v1"


# --- local -------------------------------------------------------------------


def test_local_round_trip(tmp_path):
    store = ArtifactStore(tmp_path)
    store.save("equity", {"model_version": "equity-1.0.0", "d": 0.35})
    assert store.load("equity")["d"] == 0.35


def test_a_missing_local_artifact_is_empty_not_an_error(tmp_path):
    """A daily run must not stop because the refit has not happened yet."""
    assert ArtifactStore(tmp_path).load("equity") == {}


def test_an_unreadable_local_artifact_is_empty_not_an_error(tmp_path):
    (tmp_path / "equity.json").write_text("{not json")
    assert ArtifactStore(tmp_path).load("equity") == {}


# --- the factory -------------------------------------------------------------


def test_the_factory_picks_local_when_the_serving_plane_is_not_configured(tmp_path):
    store = build_artifact_store(tmp_path, env={})
    assert isinstance(store, ArtifactStore)


def test_the_factory_picks_r2_when_both_credentials_are_present(tmp_path):
    store = build_artifact_store(tmp_path, env={ADMIN_URL_ENV: BASE, SECRET_ENV: SECRET})
    assert isinstance(store, RemoteArtifactStore)


def test_one_credential_alone_is_not_enough(tmp_path):
    """Half-configured is local, not a half-working remote store."""
    assert isinstance(build_artifact_store(tmp_path, env={ADMIN_URL_ENV: BASE}), ArtifactStore)
    assert isinstance(build_artifact_store(tmp_path, env={SECRET_ENV: SECRET}), ArtifactStore)


def test_the_write_back_endpoint_is_accepted_as_the_base(tmp_path):
    """Deployments already set FINDYN_ADMIN_URL to the /results endpoint; asking
    for a second, almost-identical variable is how they end up disagreeing."""
    store = build_artifact_store(
        tmp_path, env={ADMIN_URL_ENV: f"{BASE}/results", SECRET_ENV: SECRET}
    )
    assert isinstance(store, RemoteArtifactStore)
    assert store.directory == f"{BASE}/artifacts"


# --- remote ------------------------------------------------------------------


@pytest.fixture
def transport_calls():
    return []


def mock_transport(calls, handler):
    def dispatch(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    return httpx.MockTransport(dispatch)


def patch_httpx(monkeypatch, calls, handler):
    """Route httpx.get/put through a mock transport."""
    transport = mock_transport(calls, handler)

    def get(url, **kwargs):
        with httpx.Client(transport=transport) as client:
            return client.get(url, **kwargs)

    def put(url, **kwargs):
        with httpx.Client(transport=transport) as client:
            return client.put(url, **kwargs)

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "put", put)


def test_save_addresses_the_artifact_by_its_own_model_version(monkeypatch, transport_calls):
    store = RemoteArtifactStore(BASE, SECRET)
    patch_httpx(
        monkeypatch,
        transport_calls,
        lambda request: httpx.Response(201, json={"ok": True}),
    )

    version = store.save("equity", {"model_version": "equity-1.0.0+cal.yahoo_gspc", "d": 0.35})

    assert version == "equity-1.0.0+cal.yahoo_gspc"
    request = transport_calls[0]
    assert request.method == "PUT"
    # `+` must survive as a path segment, not decay into a space.
    assert "equity-1.0.0%2Bcal.yahoo_gspc" in str(request.url)


def test_save_signs_the_exact_bytes_it_sends(monkeypatch, transport_calls):
    store = RemoteArtifactStore(BASE, SECRET)
    patch_httpx(monkeypatch, transport_calls, lambda r: httpx.Response(201, json={"ok": True}))
    store.save("equity", {"model_version": "v1", "d": 0.35})

    request = transport_calls[0]
    body = request.content.decode()
    _, expected = sign(SECRET, body, int(request.headers["x-findyn-timestamp"]))
    assert request.headers["x-findyn-signature"] == expected


def test_save_refuses_a_payload_with_no_model_version(monkeypatch, transport_calls):
    """A fitted model that cannot be addressed by version cannot be published."""
    store = RemoteArtifactStore(BASE, SECRET)
    patch_httpx(monkeypatch, transport_calls, lambda r: httpx.Response(201))
    with pytest.raises(ValueError, match="model_version"):
        store.save("equity", {"d": 0.35})
    assert transport_calls == []


def test_a_conflicting_write_is_an_error_not_a_silent_overwrite(monkeypatch, transport_calls):
    """Immutability is the property that makes model_version mean anything."""
    store = RemoteArtifactStore(BASE, SECRET)
    patch_httpx(monkeypatch, transport_calls, lambda r: httpx.Response(409, json={"e": 1}))
    with pytest.raises(ValueError, match="immutable"):
        store.save("equity", {"model_version": "v1"})


def test_load_follows_latest_by_default(monkeypatch, transport_calls):
    store = RemoteArtifactStore(BASE, SECRET)
    patch_httpx(
        monkeypatch,
        transport_calls,
        lambda r: httpx.Response(
            200,
            json={"model_version": "v2", "d": 0.4},
            headers={"x-findyn-model-version": "v2"},
        ),
    )
    assert store.load("equity")["d"] == 0.4
    assert str(transport_calls[0].url).endswith("/artifacts/equity/latest")


def test_a_pinned_version_is_requested_exactly(monkeypatch, transport_calls):
    """A replay of a published state must load the model that produced it."""
    store = RemoteArtifactStore(BASE, SECRET, version="equity-1.0.0+cal.yahoo_gspc")
    patch_httpx(monkeypatch, transport_calls, lambda r: httpx.Response(200, json={"d": 1}))
    store.load("equity")
    assert "equity-1.0.0%2Bcal.yahoo_gspc" in str(transport_calls[0].url)


def test_the_pin_comes_from_the_environment(tmp_path):
    store = build_artifact_store(
        tmp_path,
        env={ADMIN_URL_ENV: BASE, SECRET_ENV: SECRET, VERSION_ENV: "equity-9.9.9"},
    )
    assert isinstance(store, RemoteArtifactStore)
    assert "equity-9.9.9" in repr(store)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404),
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=["a", "list"]),
    ],
)
def test_every_load_failure_degrades_to_empty(monkeypatch, transport_calls, response):
    """A network blip during a daily run must not abort the run.

    Every engine already handles "no fit yet" — that is the state of a fresh
    deployment — so returning ``{}`` puts the failure on a path that is already
    tested rather than inventing a second one.
    """
    store = RemoteArtifactStore(BASE, SECRET)
    patch_httpx(monkeypatch, transport_calls, lambda r: response)
    assert store.load("equity") == {}


def test_a_transport_error_degrades_to_empty(monkeypatch, transport_calls):
    def explode(request):
        raise httpx.ConnectError("no route to host")

    store = RemoteArtifactStore(BASE, SECRET)
    patch_httpx(monkeypatch, transport_calls, explode)
    assert store.load("equity") == {}


# --- the lifecycle contract --------------------------------------------------


def test_refit_then_predict_across_two_processes(monkeypatch, transport_calls, tmp_path):
    """The whole point, end to end.

    A refit stores a model; a *separate* store instance — standing in for the
    daily run's container, which shares no filesystem with the refit's — loads it
    back and gets the same parameters.
    """
    stored: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = str(request.url.path)
        if request.method == "PUT":
            stored[path] = request.content.decode()
            return httpx.Response(201, json={"ok": True})
        # `latest` resolves to the single stored version.
        if path.endswith("/latest"):
            body = next(iter(stored.values()), None)
            if body is None:
                return httpx.Response(404)
            version = json.loads(body)["model_version"]
            return httpx.Response(200, text=body, headers={"x-findyn-model-version": version})
        body = stored.get(path)
        return httpx.Response(200, text=body) if body else httpx.Response(404)

    patch_httpx(monkeypatch, transport_calls, handler)

    refit = RemoteArtifactStore(BASE, SECRET)
    refit.save("equity", {"model_version": "equity-1.0.0+cal.yahoo_gspc", "hmm": {"seed": 7}})

    daily = RemoteArtifactStore(BASE, SECRET)
    loaded = daily.load("equity")

    assert loaded["hmm"]["seed"] == 7
    assert loaded["model_version"] == "equity-1.0.0+cal.yahoo_gspc"


def test_local_storage_cannot_serve_a_second_process(tmp_path):
    """The bug the remote store exists for, stated as a test.

    Two 'containers' — two different directories — do not see each other's
    fitted models. On a laptop that is invisible because the directory persists;
    in CI it is the whole failure.
    """
    refit = ArtifactStore(tmp_path / "refit-container")
    refit.save("equity", {"model_version": "v1", "hmm": {"seed": 7}})

    daily = ArtifactStore(tmp_path / "daily-container")
    assert daily.load("equity") == {}
