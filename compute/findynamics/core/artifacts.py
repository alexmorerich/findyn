"""Fitted-parameter storage for engines.

``AssetEngine.fit`` runs monthly and ``predict`` runs daily, in separate
processes: whatever the refit chose has to survive between them.

Two implementations of the same tiny interface:

* :class:`ArtifactStore` — a JSON file per engine under ``compute/artifacts/``.
  Fine for a laptop, and **useless in production**: the refit and the daily run
  are different GitHub Actions containers, the directory is gitignored, and the
  runner is destroyed minutes later. The equity engine's correct response to a
  missing fit is to publish nothing, so a production cron on local storage would
  have published nothing forever.
* :class:`RemoteArtifactStore` — the same interface over R2, through the
  serving plane's HMAC-authenticated admin door (FINDYN_V1_SPEC.md §6).

:func:`build_artifact_store` picks between them from the environment, so no
engine and no job has to know which one it is talking to.

**Artifacts are immutable and addressed by (model_version, fit date).** A pair is
written once; re-writing the same bytes is a no-op and re-writing *different*
bytes is an error. That is the property that makes "which model produced this
state" an answerable question.

Both halves of the address are needed, and it took a production failure to see
why. ``model_version`` names the model *specification* — which features, which
estimator, which vocabulary. A monthly expanding-window refit re-estimates every
parameter, so the artifact's content legitimately changes each month while the
specification does not. Keyed on the version alone, immutability meant the second
refit of any engine conflicted with the first, forever; the first engine to run
twice did exactly that. "Version 1.1.0 is these bytes" was never a claim this
system could keep. "The state published on this date came from *this* fit of
1.1.0" is, and it is the one that matters for replay.

A missing or unreadable artifact is never fatal — ``load`` returns ``{}`` and the
engine falls back to its configured defaults or declines to publish. A daily run
must not stop because a refit has not happened yet.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from findynamics.core.signing import headers as sign_headers

log = logging.getLogger("findynamics.core.artifacts")

#: Overridable so a test or a CI run never writes into the working tree.
ENV_VAR = "FINDYN_ARTIFACT_DIR"

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def artifact_dir() -> Path:
    override = os.environ.get(ENV_VAR)
    return Path(override) if override else DEFAULT_DIR


class ArtifactStore:
    """Named JSON documents, one per engine."""

    def __init__(self, directory: Path | None = None) -> None:
        self._dir = directory or artifact_dir()

    @property
    def directory(self) -> Path:
        return self._dir

    def path_for(self, name: str) -> Path:
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"invalid artifact name: {name!r}")
        return self._dir / f"{name}.json"

    def load(self, name: str) -> dict[str, Any]:
        """Stored parameters, or ``{}`` when there are none to load."""
        path = self.path_for(name)
        if not path.exists():
            return {}
        try:
            body = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as err:
            log.warning("ignoring unreadable artifact %s: %s", path, err)
            return {}
        return body if isinstance(body, dict) else {}

    def save(self, name: str, payload: dict[str, Any]) -> Path:
        """Write parameters, replacing whatever was there.

        Written to a temporary file and renamed, so a crash mid-write leaves the
        previous parameters intact rather than a truncated file that every later
        run silently ignores.
        """
        path = self.path_for(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(path)
        log.info("wrote %s", path)
        return path


class ArtifactStorage(Protocol):
    """What an engine needs from artifact storage, and nothing more."""

    def load(self, name: str) -> dict[str, Any]: ...

    def save(self, name: str, payload: dict[str, Any]) -> Any: ...


#: Environment variable holding the serving plane's admin base URL. Accepts the
#: write-back endpoint too, so a deployment that already sets FINDYN_ADMIN_URL
#: needs no second variable.
ADMIN_URL_ENV = "FINDYN_ADMIN_URL"
SECRET_ENV = "ADMIN_HMAC_SECRET"

#: Set to pin a run to one exact model version instead of following `latest`.
#: A backtest or a replay of a published state should use this; a daily run
#: should not, or it would never pick up a refit.
#:
#: Accepts ``<version>`` — the newest fit of that specification — or
#: ``<version>@<YYYY-MM-DD>`` for one exact fit. Replaying a published state
#: wants the second form: the specification alone still moves under you every
#: month, which is the whole reason the fit date is part of the address.
VERSION_ENV = "FINDYN_MODEL_VERSION"

#: Field carrying the fit date in an artifact document. Engines spell their own
#: provenance differently (``as_of``, ``fitted_as_of``); :meth:`RemoteArtifactStore.save`
#: normalises to this one so the address does not depend on which engine wrote it.
FIT_FIELD = "fit_date"

#: Where the fit date is read from when an engine has not set `FIT_FIELD`.
FIT_SOURCES: tuple[str, ...] = (FIT_FIELD, "as_of", "fitted_as_of")


def _admin_base(url: str) -> str:
    """Normalise an admin URL to its base, accepting the write-back endpoint."""
    trimmed = url.rstrip("/")
    return trimmed[: -len("/results")] if trimmed.endswith("/results") else trimmed


def split_pin(pin: str) -> tuple[str, str | None]:
    """``"equity-1.1.0+cal.x@2026-07-31"`` -> ``("equity-1.1.0+cal.x", "2026-07-31")``.

    Split on the last ``@`` only: a model version may legitimately contain ``+``
    and ``.`` but never ``@``, so this is unambiguous, and splitting on the first
    one would break the day someone tags a version with an email-looking suffix.
    """
    version, sep, fit = pin.rpartition("@")
    return (version, fit) if sep else (pin, None)


def fit_date_of(payload: dict[str, Any]) -> str | None:
    """The date this artifact was fitted through, whatever the engine called it."""
    for field in FIT_SOURCES:
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()[:10]
    return None


class RemoteArtifactStore:
    """Fitted models in R2, via the serving plane's admin API.

    Reads and writes are HMAC-signed with the same secret and the same
    ``{timestamp}.{body}`` construction as the write-back, so there is one
    credential and one signature format for the whole compute → serving door.
    """

    def __init__(
        self,
        base_url: str,
        secret: str,
        *,
        version: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base = _admin_base(base_url)
        self._secret = secret
        self._version = version
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"RemoteArtifactStore(base={self._base!r}, version={self._version or 'latest'})"

    @property
    def directory(self) -> str:
        return f"{self._base}/artifacts"

    def _url(self, name: str, version: str, fit: str | None = None) -> str:
        url = f"{self._base}/artifacts/{quote(name, safe='')}/{quote(version, safe='')}"
        return f"{url}?fit={quote(fit, safe='')}" if fit else url

    def load(self, name: str) -> dict[str, Any]:
        """The pinned version if one is configured, else whatever is current.

        Every failure degrades to ``{}``: a network blip during a daily run
        should leave the engine publishing what it can rather than aborting, and
        every engine already handles "no fit yet".
        """
        import httpx

        version, fit = split_pin(self._version) if self._version else ("latest", None)
        url = self._url(name, version, fit)
        try:
            response = httpx.get(
                url,
                headers=sign_headers(self._secret, ""),
                timeout=self._timeout,
            )
        except httpx.HTTPError as err:
            log.warning("artifact %s unreachable (%s); continuing without it", url, err)
            return {}

        if response.status_code == 404:
            log.info("no stored artifact for %s@%s yet", name, version)
            return {}
        if response.status_code >= 400:
            log.warning(
                "artifact %s returned %d: %s", url, response.status_code, response.text[:200]
            )
            return {}

        try:
            body = response.json()
        except ValueError as err:
            log.warning("artifact %s is not JSON (%s)", url, err)
            return {}
        if not isinstance(body, dict):
            return {}

        resolved = response.headers.get("x-findyn-model-version", version)
        resolved_fit = response.headers.get("x-findyn-model-fit")
        log.info(
            "loaded artifact %s@%s from R2 (fit %s)",
            name,
            resolved,
            resolved_fit or "pre-dated, legacy key",
        )
        return body

    def save(self, name: str, payload: dict[str, Any]) -> str:
        """Store under the payload's own ``model_version`` and fit date. Write-once.

        Both halves of the address come from the payload rather than from
        arguments, so an artifact cannot be filed under a name or a date its own
        contents disagree with — the serving side rejects that mismatch too, on
        the same reasoning.

        The fit date is normalised into the document under `FIT_FIELD` before it
        is serialized, so the stored bytes say when they were fitted. An engine
        that spells it `as_of` or `fitted_as_of` needs no change; one that has no
        fit date at all is an error, because a fitted model that cannot be told
        apart from next month's refit of the same specification is exactly the
        thing this key exists to prevent.
        """
        import httpx

        version = str(payload.get("model_version") or "").strip()
        if not version:
            raise ValueError(
                f"artifact {name!r} has no model_version; a fitted model that "
                "cannot be addressed by version cannot be published"
            )

        fit = fit_date_of(payload)
        if not fit:
            raise ValueError(
                f"artifact {name!r}@{version} has no fit date; set one of "
                f"{', '.join(FIT_SOURCES)}. Without it, this month's fit and next "
                "month's are indistinguishable and neither can be replayed"
            )

        document = {**payload, FIT_FIELD: fit}
        body = json.dumps(document, indent=2, sort_keys=True)
        url = self._url(name, version)
        response = httpx.put(
            url,
            content=body,
            headers={
                "content-type": "application/json",
                **sign_headers(self._secret, body),
            },
            timeout=self._timeout,
        )
        if response.status_code == 409:
            raise ValueError(
                f"artifact {name}@{version} fitted {fit} already exists with different "
                "content; fitted models are immutable — a refit on a later date is a "
                "new fit, not an edit"
            )
        response.raise_for_status()
        log.info(
            "stored artifact %s@%s fitted %s in R2 (%s)", name, version, fit, response.status_code
        )
        return version


def build_artifact_store(
    directory: Path | None = None,
    env: dict[str, str] | None = None,
) -> ArtifactStorage:
    """Remote storage when the serving plane is configured, local otherwise.

    The environment decides, so a job never names a backend. A developer running
    the same command with no secrets gets the local directory and a production
    run gets R2 — and there is no flag anyone can forget to pass that would let
    an ephemeral local artifact reach production.
    """
    settings = os.environ if env is None else env
    url = settings.get(ADMIN_URL_ENV)
    secret = settings.get(SECRET_ENV)

    if url and secret:
        return RemoteArtifactStore(url, secret, version=settings.get(VERSION_ENV) or None)

    log.info(
        "%s/%s are not both set; using local artifact storage. This is correct for "
        "development and wrong for production — a scheduled run on local storage "
        "loses every refit when the container is destroyed.",
        ADMIN_URL_ENV,
        SECRET_ENV,
    )
    return ArtifactStore(directory)


__all__ = [
    "ADMIN_URL_ENV",
    "DEFAULT_DIR",
    "ENV_VAR",
    "SECRET_ENV",
    "VERSION_ENV",
    "ArtifactStorage",
    "ArtifactStore",
    "RemoteArtifactStore",
    "artifact_dir",
    "build_artifact_store",
]
