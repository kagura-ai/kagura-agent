"""Live memory-cloud adapter and credential-isolated command actor for #188."""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from kagura_agent.eval.bootstrap_ab import (
    Arm,
    ArmHandle,
    ArmPair,
    BootstrapEnvelope,
    BootstrapSnapshot,
    ExperimentInvariantError,
    ExperimentManifest,
    ObjectiveActor,
    OutcomeObservation,
    SnapshotMemory,
    TaskSpec,
    VerifiedSource,
    _memory_logical_id,
)

JsonRequest = Callable[[str, str, dict[str, Any] | None], Awaitable[dict[str, Any]]]
FeedbackMode = Literal["host", "public"]
#: A bearer source. Called with ``force=True`` after a 401 to demand a freshly
#: rotated token; ``force=False`` returns the currently valid token (which the
#: provider may still refresh on its own expiry clock).
TokenProvider = Callable[[bool], str]


class LiveEvalError(RuntimeError):
    """A live bootstrap experiment failed before it could produce valid evidence."""


class KaguraCliTokenProvider:  # pragma: no cover - shells out to the `kagura` CLI
    """A refreshing bearer sourced from the ``kagura`` CLI's OAuth session.

    The owner OAuth access token is the only credential that satisfies the
    operator-only host-feedback endpoint on a single-owner workspace, but it
    lives only ~1h — far short of a full 5-generation gate run. This provider
    rotates it via ``kagura auth refresh`` (proactively once per
    ``refresh_interval_s``, and on demand when the client reports a 401) and
    emits the current token via ``kagura auth token``, so the run inherits a
    continuously-valid owner bearer without the runner re-reading anything.

    Kept out of unit coverage precisely because it shells the external CLI; the
    recoverable-401 and refresh contract it feeds is fully tested at the
    :class:`BearerJsonClient` seam with an injected provider.
    """

    def __init__(
        self,
        *,
        kagura_bin: str = "kagura",
        refresh_interval_s: float = 2400.0,
        timeout: float = 60.0,
    ) -> None:
        self._bin = kagura_bin
        self._refresh_interval_s = refresh_interval_s
        self._timeout = timeout
        self._token: str | None = None
        self._last_refresh = 0.0

    def __call__(self, force: bool) -> str:
        import subprocess  # local: only the live path pays the import

        now = time.monotonic()
        stale = self._token is None or (now - self._last_refresh) >= self._refresh_interval_s
        if force or stale:
            subprocess.run(
                [self._bin, "auth", "refresh"],
                check=True,
                capture_output=True,
                timeout=self._timeout,
            )
            emitted = subprocess.run(
                [self._bin, "auth", "token"],
                check=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            token = emitted.stdout.strip()
            if not token:
                raise LiveEvalError("`kagura auth token` returned an empty access token")
            self._token = token
            self._last_refresh = now
        assert self._token is not None
        return self._token


class BearerJsonClient:
    """Minimal async JSON client that never stores the API key as a public attribute.

    Two robustness seams keep a **multi-hour** gate run (~3600 REST calls behind
    900 serial actor turns) from aborting on a recoverable fault:

    - ``token_provider`` — a refreshing bearer source. A live run outlives the
      ~1h OAuth access token, but the runner reads the bearer once, so a fixed
      key would 401 mid-run. When a provider is supplied the Authorization header
      is rebuilt from it per request, and a 401 triggers exactly one forced
      refresh + retry (the provider rotates the token). A fixed-key client (no
      provider) is unchanged: a 401 surfaces immediately.
    - ``retries`` — a bounded retry on *transient* faults (dropped connection /
      5xx) with linear backoff, so one blip over thousands of calls does not
      discard the run and its fresh 70-memory re-provision. A 4xx (other than the
      401-refresh case) is a request bug, not transient, and is never retried.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        token_provider: TokenProvider | None = None,
        retries: int = 0,
        retry_backoff_s: float = 0.5,
    ) -> None:
        if not api_key and token_provider is None:
            raise ValueError("KAGURA_API_KEY (or a token_provider) is required for a live eval")
        if timeout <= 0.0:
            raise ValueError("memory-cloud timeout must be positive")
        if retries < 0:
            raise ValueError("memory-cloud retries must be non-negative")
        stripped = base_url.rstrip("/")
        if not stripped.startswith("https://") and not (
            stripped.startswith("http://127.0.0.1") or stripped.startswith("http://localhost")
        ):
            raise ValueError("memory-cloud base URL must use HTTPS (loopback HTTP is allowed)")
        self.base_url = stripped
        self.timeout = timeout
        self._static_key = api_key
        self._token_provider = token_provider
        self._retries = retries
        self._retry_backoff_s = retry_backoff_s

    def _bearer(self, *, refresh: bool) -> str:
        """The current bearer — from the provider (optionally forced) or the fixed key."""
        if self._token_provider is not None:
            return self._token_provider(refresh)
        return self._static_key

    def _headers(self, *, refresh: bool) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._bearer(refresh=refresh)}",
            "Content-Type": "application/json",
            "User-Agent": "kagura-agent-bootstrap-eval/1",
        }

    async def request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._request_sync, method, path, body)

    def _request_sync(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        if not path.startswith("/"):
            raise LiveEvalError("memory-cloud request path must be absolute")
        data = None if body is None else json.dumps(body).encode()
        refreshed_after_401 = False
        # attempt 0..retries are the transient-retry budget; the 401 forced-refresh
        # retry is granted once ON TOP of that (an expiry is not a "transient" but a
        # recoverable credential fault), so it never consumes the transient budget.
        attempt = 0
        while True:
            request = urllib.request.Request(
                f"{self.base_url}{path}",
                data=data,
                headers=self._headers(refresh=refreshed_after_401),
                method=method,
            )
            try:
                with urllib.request.urlopen(  # noqa: S310
                    request, timeout=self.timeout
                ) as response:
                    raw_payload = response.read()
                break
            except urllib.error.HTTPError as exc:
                # A 401 with a refreshing provider means the bearer expired: force a
                # rotation and retry exactly once before giving up (fail closed on a
                # genuinely unauthorized credential, never loop).
                if exc.code == 401 and self._token_provider is not None and not refreshed_after_401:
                    refreshed_after_401 = True
                    continue
                # 5xx is transient (upstream hiccup); a 4xx is a request bug. Retry
                # only the former, within the bounded budget.
                if exc.code >= 500 and attempt < self._retries:
                    attempt += 1
                    time.sleep(self._retry_backoff_s * attempt)
                    continue
                # Never include response bodies: an upstream regression could reflect a bearer.
                raise LiveEvalError(
                    f"memory-cloud {method} {path} failed with HTTP {exc.code}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < self._retries:
                    attempt += 1
                    time.sleep(self._retry_backoff_s * attempt)
                    continue
                raise LiveEvalError(f"memory-cloud {method} {path} connection failed") from exc
        try:
            payload = raw_payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LiveEvalError(f"memory-cloud {method} {path} returned non-UTF-8") from exc
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise LiveEvalError(f"memory-cloud {method} {path} returned non-JSON") from exc
        if not isinstance(parsed, dict):
            raise LiveEvalError(f"memory-cloud {method} {path} returned a non-object")
        return parsed


_SEARCH_CONFIG_FIELDS = (
    "semantic_weight",
    "bm25_weight",
    "fetch_factor",
    "use_rerank",
    "reranker_provider",
    "reranker_model",
    "reinforce_enabled",
    "reinforce_max_boost",
    "reinforce_require_host_arbitration",
    "routing_mode",
)

_SEARCH_CONFIG_READ_ONLY_FIELDS = frozenset(
    {
        "context_id",
        "embedding_model",
        "embedding_dimensions",
        "created_at",
        "updated_at",
    }
)


def _search_config_payload(raw: Mapping[str, Any], *, enabled: bool) -> dict[str, Any]:
    missing = [field for field in _SEARCH_CONFIG_FIELDS if field not in raw]
    if missing:
        raise ExperimentInvariantError(f"search config lacks required fields: {missing}")
    # Preserve forward-compatible writable fields so teardown restores the exact
    # pre-run policy even when memory-cloud grows beyond this client's known set.
    payload = {
        field: value for field, value in raw.items() if field not in _SEARCH_CONFIG_READ_ONLY_FIELDS
    }
    payload["reinforce_enabled"] = enabled
    return payload


def _comparable_search_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: value
        for field, value in raw.items()
        if field not in _SEARCH_CONFIG_READ_ONLY_FIELDS and field != "reinforce_enabled"
    }


def _snapshot_from_export(raw: Mapping[str, Any], expected: BootstrapSnapshot) -> BootstrapSnapshot:
    records = raw.get("memories")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ExperimentInvariantError("context export has no memory list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        logical_id = _memory_logical_id(record)
        if logical_id in by_id:
            raise ExperimentInvariantError(f"context export repeats logical id {logical_id}")
        by_id[logical_id] = record
    if set(by_id) != set(expected.memory_ids):
        raise ExperimentInvariantError("context export does not contain the fixed snapshot exactly")
    memories: list[SnapshotMemory] = []
    for source in expected.memories:
        record = by_id[source.logical_id]
        tags = record.get("tags", [])
        memories.append(
            SnapshotMemory(
                logical_id=source.logical_id,
                summary=str(record.get("summary", "")),
                content=str(record.get("content", "")),
                tags=tuple(str(tag) for tag in tags) if isinstance(tags, list) else (),
                delivery_mode=str(record.get("delivery_mode", "on_recall")),
                # trust_tier is context-level and intentionally absent from the
                # portability export; bootstrap enforces trusted-only recall.
                trust_tier="trusted",
            )
        )
    return BootstrapSnapshot(name=expected.name, memories=tuple(memories))


@dataclass(frozen=True)
class LiveArmConfig:
    agent_id: str
    context_id: str
    feedback_journal: str


class RestBootstrapBackend:
    """Run #188 against two pre-provisioned, disposable memory-cloud contexts.

    ``prepare`` proves that both portability exports equal the committed logical
    snapshot, then pins the control context's reinforce actuator OFF and treatment
    ON while requiring every other search setting to match. Both arms receive the
    same independently-verified feedback events; only whether ranking consumes the
    journal differs.

    The contexts must be disposable because retrieval feedback is append-only. The
    adapter restores search settings in ``close`` but cannot erase feedback history.
    """

    def __init__(
        self,
        *,
        request: JsonRequest,
        control: LiveArmConfig,
        treatment: LiveArmConfig,
        feedback_mode: FeedbackMode,
        host_feedback_path: str | None = None,
    ) -> None:
        if feedback_mode == "host" and not host_feedback_path:
            raise ValueError("host feedback mode requires an operator-only endpoint path")
        if host_feedback_path is not None:
            if not host_feedback_path.startswith("/") or "{context_id}" not in host_feedback_path:
                raise ValueError("host feedback path must be absolute and contain {context_id}")
            try:
                host_feedback_path.format(agent_id="agent", context_id="context")
            except (KeyError, ValueError) as exc:
                raise ValueError("host feedback path contains an unsupported placeholder") from exc
        self._request = request
        self._configs = {Arm.CONTROL: control, Arm.TREATMENT: treatment}
        self.feedback_mode = feedback_mode
        self.host_feedback_path = host_feedback_path
        self._original_search_configs: dict[Arm, dict[str, Any]] = {}
        self._actual_to_logical: dict[Arm, dict[str, str]] = {}
        self._logical_to_actual: dict[Arm, dict[str, str]] = {}

    async def prepare(self, manifest: ExperimentManifest, snapshot: BootstrapSnapshot) -> ArmPair:
        expected_provenance = "host" if self.feedback_mode == "host" else "agent"
        if manifest.feedback_provenance != expected_provenance:
            raise ExperimentInvariantError(
                "manifest feedback provenance does not match the live transport"
            )
        pair = ArmPair(
            control=self._handle(Arm.CONTROL, manifest),
            treatment=self._handle(Arm.TREATMENT, manifest),
        )
        pair.validate(manifest)
        exports = await asyncio.gather(
            *(
                self._request("GET", f"/api/v1/contexts/{handle.context_id}/export", None)
                for handle in (pair.control, pair.treatment)
            )
        )
        for arm, exported in zip((Arm.CONTROL, Arm.TREATMENT), exports, strict=True):
            handle = pair.for_arm(arm)
            context = exported.get("context")
            if not isinstance(context, dict) or str(context.get("id")) != handle.context_id:
                raise ExperimentInvariantError(
                    f"{arm.value} export resolved outside its isolated context"
                )
            observed = _snapshot_from_export(exported, snapshot)
            if observed.fingerprint != manifest.snapshot_fingerprint:
                raise ExperimentInvariantError(
                    f"{arm.value} context export differs from the fixed snapshot"
                )
            records = exported["memories"]
            assert isinstance(records, list)
            actual_to_logical: dict[str, str] = {}
            logical_to_actual: dict[str, str] = {}
            for record in records:
                assert isinstance(record, dict)
                actual_id = record.get("id")
                if not isinstance(actual_id, str) or not actual_id:
                    raise ExperimentInvariantError(
                        f"{arm.value} context export has a memory without a stable id"
                    )
                logical_id = _memory_logical_id(record)
                if actual_id in actual_to_logical or logical_id in logical_to_actual:
                    raise ExperimentInvariantError(
                        f"{arm.value} context export has an ambiguous memory identity"
                    )
                actual_to_logical[actual_id] = logical_id
                logical_to_actual[logical_id] = actual_id
            self._actual_to_logical[arm] = actual_to_logical
            self._logical_to_actual[arm] = logical_to_actual
        paths = {
            arm: f"/api/v1/contexts/{pair.for_arm(arm).context_id}/search-config"
            for arm in (Arm.CONTROL, Arm.TREATMENT)
        }
        configs = await asyncio.gather(
            *(self._request("GET", paths[arm], None) for arm in (Arm.CONTROL, Arm.TREATMENT))
        )
        for arm, config in zip((Arm.CONTROL, Arm.TREATMENT), configs, strict=True):
            reported_context = config.get("context_id")
            if (
                reported_context is not None
                and str(reported_context) != pair.for_arm(arm).context_id
            ):
                raise ExperimentInvariantError(
                    f"{arm.value} search config resolved outside its isolated context"
                )
            _search_config_payload(config, enabled=arm is Arm.TREATMENT)
        if self.feedback_mode == "host" and any(
            not bool(config.get("reinforce_require_host_arbitration")) for config in configs
        ):
            raise ExperimentInvariantError(
                "host feedback experiment requires host arbitration in both contexts"
            )
        if _comparable_search_config(configs[0]) != _comparable_search_config(configs[1]):
            raise ExperimentInvariantError("A/B search configs differ outside reinforce_enabled")
        self._original_search_configs = {
            Arm.CONTROL: dict(configs[0]),
            Arm.TREATMENT: dict(configs[1]),
        }
        try:
            updated = await asyncio.gather(
                self._request(
                    "PUT",
                    paths[Arm.CONTROL],
                    _search_config_payload(configs[0], enabled=False),
                ),
                self._request(
                    "PUT",
                    paths[Arm.TREATMENT],
                    _search_config_payload(configs[1], enabled=True),
                ),
            )
            for arm, applied_config in zip((Arm.CONTROL, Arm.TREATMENT), updated, strict=True):
                expected = _search_config_payload(
                    configs[0 if arm is Arm.CONTROL else 1],
                    enabled=arm is Arm.TREATMENT,
                )
                if (
                    _comparable_search_config(applied_config) != _comparable_search_config(expected)
                    or applied_config.get("reinforce_enabled") is not expected["reinforce_enabled"]
                ):
                    raise ExperimentInvariantError(
                        f"{arm.value} context did not apply the registered ranking actuator"
                    )
        except BaseException:
            await self._restore_search_configs(pair)
            raise
        return pair

    def _handle(self, arm: Arm, manifest: ExperimentManifest) -> ArmHandle:
        config = self._configs[arm]
        return ArmHandle(
            arm=arm,
            agent_id=config.agent_id,
            context_id=config.context_id,
            snapshot_fingerprint=manifest.snapshot_fingerprint,
            feedback_journal=config.feedback_journal,
            feedback_influence=arm is Arm.TREATMENT,
        )

    async def bootstrap(
        self,
        handle: ArmHandle,
        task: TaskSpec,
        *,
        session_id: str,
        recall_k: int,
        evaluation_seed: int,
        exploration_floor: float,
        candidate_pool_k: int,
    ) -> BootstrapEnvelope:
        raw = await self._request(
            "POST",
            f"/api/v1/agents/{handle.agent_id}/bootstrap",
            {
                "context_id": handle.context_id,
                "session_id": session_id,
                "query": task.query,
                "recall_k": recall_k,
                "recall_evaluation": {
                    "seed": evaluation_seed,
                    "exploration_floor": exploration_floor,
                    "candidate_pool_k": candidate_pool_k,
                },
                "include": ["pinned", "recall", "upcoming", "state", "policy"],
            },
        )
        components = raw.get("components")
        if isinstance(components, dict):
            recall_component = components.get("recall")
            if isinstance(recall_component, dict):
                identity_map = self._actual_to_logical.get(handle.arm, {})
                records = recall_component.get("results")
                if not isinstance(records, list):
                    records = recall_component.get("memories")
                if isinstance(records, list):
                    for record in records:
                        if not isinstance(record, dict):
                            continue
                        actual_id = record.get("memory_id", record.get("id"))
                        logical_id = identity_map.get(str(actual_id))
                        if logical_id is None:
                            continue
                        details = record.get("details")
                        record["details"] = {
                            **(details if isinstance(details, dict) else {}),
                            "eval_id": logical_id,
                        }
                probabilities = recall_component.get("selection_probabilities")
                if isinstance(probabilities, dict):
                    recall_component["selection_probabilities"] = {
                        identity_map.get(str(memory_id), str(memory_id)): probability
                        for memory_id, probability in probabilities.items()
                    }
        return BootstrapEnvelope(raw)

    async def record_verified_feedback(
        self,
        handle: ArmHandle,
        *,
        logical_memory_id: str,
        query: str,
        helpful: bool,
        verdict_source: VerifiedSource,
        verdict_reference: str,
        experiment_id: str,
        note: str,
    ) -> None:
        memory_id = self._logical_to_actual.get(handle.arm, {}).get(logical_memory_id)
        if memory_id is None:
            raise ExperimentInvariantError(
                f"{handle.arm.value} feedback target is outside the registered snapshot"
            )
        body: dict[str, Any] = {
            "memory_id": memory_id,
            "helpful": helpful,
            "query": query,
            "note": note,
        }
        if self.feedback_mode == "host":
            assert self.host_feedback_path is not None
            path = self.host_feedback_path.format(
                agent_id=handle.agent_id, context_id=handle.context_id
            )
            body["verdict_source"] = verdict_source
            body["verdict_reference"] = verdict_reference
            body["experiment_id"] = experiment_id
        else:
            path = f"/api/v1/contexts/{handle.context_id}/feedback"
        await self._request("POST", path, body)

    async def close(self, pair: ArmPair) -> None:
        await self._restore_search_configs(pair)

    async def _restore_search_configs(self, pair: ArmPair) -> None:
        if not self._original_search_configs:
            return
        await asyncio.gather(
            *(
                self._request(
                    "PUT",
                    f"/api/v1/contexts/{pair.for_arm(arm).context_id}/search-config",
                    _search_config_payload(
                        self._original_search_configs[arm],
                        enabled=bool(self._original_search_configs[arm]["reinforce_enabled"]),
                    ),
                )
                for arm in (Arm.CONTROL, Arm.TREATMENT)
            )
        )
        self._original_search_configs.clear()


_DEFAULT_STRIPPED_ENV = frozenset(
    {
        "KAGURA_API_KEY",
        "KAGURA_MCP_URL",
        "KAGURA_AGENT_ID",
        "KAGURA_AGENT_MEMORY_MCP_CONTEXT",
        "KAGURA_AGENT_MEMORY_MCP_SERVER",
    }
)


class CommandObjectiveActor(ObjectiveActor):
    """Run an actor command with bootstrap data on stdin and score it host-side."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout: float = 300.0,
        env: Mapping[str, str] | None = None,
        retries: int = 0,
        retry_backoff_s: float = 2.0,
    ) -> None:
        if not command or any(not item for item in command):
            raise ValueError("actor command must be a non-empty argv sequence")
        if timeout <= 0.0:
            raise ValueError("actor timeout must be positive")
        if retries < 0:
            raise ValueError("actor retries must be non-negative")
        self.command = tuple(command)
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s
        source_env = os.environ if env is None else env
        self.env = {
            key: value
            for key, value in source_env.items()
            if key.upper() not in _DEFAULT_STRIPPED_ENV
        }

    async def run(
        self, task: TaskSpec, bootstrap: BootstrapEnvelope, *, seed: int
    ) -> OutcomeObservation:
        payload = {
            "task": task.actor_payload(),
            "bootstrap_context": bootstrap.model_context(),
            "seed": seed,
        }
        # The 900 serial codex turns are the run's most failure-prone surface: a
        # transient (network blip / momentary rate throttle) fails one trial, which
        # would otherwise abort the whole multi-hour run. Retry the invocation a
        # bounded number of times with linear backoff before surfacing the error.
        # The actor is a pure, idempotent one-shot Q&A, so a retry is safe.
        attempt = 0
        while True:
            try:
                return await self._invoke_once(task, payload)
            except LiveEvalError:
                if attempt >= self.retries:
                    raise
                attempt += 1
                await asyncio.sleep(self.retry_backoff_s * attempt)

    async def _invoke_once(self, task: TaskSpec, payload: dict[str, Any]) -> OutcomeObservation:
        process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
        )
        try:
            stdout, _stderr = await asyncio.wait_for(
                process.communicate(json.dumps(payload, ensure_ascii=False).encode()),
                timeout=self.timeout,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise LiveEvalError(f"actor command timed out after {self.timeout:g}s") from None
        if process.returncode != 0:
            raise LiveEvalError(f"actor command exited {process.returncode}")
        try:
            output = stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LiveEvalError("actor command output is not UTF-8") from exc
        score = task.check.score(output)
        return OutcomeObservation(
            score=score,
            passed=score == 1.0,
            source="objective_check",
            output=output,
        )
