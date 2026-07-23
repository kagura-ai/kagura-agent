"""CLI for the live bootstrap A/B gate in :mod:`kagura_agent.eval.bootstrap_ab`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kagura_agent.eval.bootstrap_ab import (
    CorpusError,
    ExperimentInvariantError,
    ExperimentManifest,
    GateThresholds,
    VersionStamp,
    load_default_snapshot,
    load_default_tasks,
    run_experiment,
)
from kagura_agent.eval.bootstrap_live import (
    BearerJsonClient,
    CommandObjectiveActor,
    KaguraCliTokenProvider,
    LiveArmConfig,
    LiveEvalError,
    RestBootstrapBackend,
    TokenProvider,
)


def _bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _object(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _thresholds(raw: Mapping[str, Any]) -> GateThresholds:
    defaults = GateThresholds()
    return GateThresholds(
        delta_min=_number(raw.get("delta_min", defaults.delta_min), field="delta_min"),
        confidence=_number(raw.get("confidence", defaults.confidence), field="confidence"),
        bootstrap_resamples=_integer(
            raw.get("bootstrap_resamples", defaults.bootstrap_resamples),
            field="bootstrap_resamples",
        ),
        max_degraded_rate=_number(
            raw.get("max_degraded_rate", defaults.max_degraded_rate),
            field="max_degraded_rate",
        ),
        max_tail_regression=_number(
            raw.get("max_tail_regression", defaults.max_tail_regression),
            field="max_tail_regression",
        ),
        max_entropy_drop=_number(
            raw.get("max_entropy_drop", defaults.max_entropy_drop),
            field="max_entropy_drop",
        ),
        utility_flat_tolerance=_number(
            raw.get("utility_flat_tolerance", defaults.utility_flat_tolerance),
            field="utility_flat_tolerance",
        ),
        max_goodhart_decline=_number(
            raw.get("max_goodhart_decline", defaults.max_goodhart_decline),
            field="max_goodhart_decline",
        ),
    )


def _version_stamp(raw: Mapping[str, Any]) -> VersionStamp:
    judge = raw.get("judge")
    if judge is not None and not isinstance(judge, str):
        raise ValueError("versions.judge must be a string or null")
    return VersionStamp(
        code=_string(raw.get("code"), field="versions.code"),
        actor_model=_string(raw.get("actor_model"), field="versions.actor_model"),
        actor=_string(raw.get("actor"), field="versions.actor"),
        bootstrap_api=_string(raw.get("bootstrap_api"), field="versions.bootstrap_api"),
        ranking_policy=_string(raw.get("ranking_policy"), field="versions.ranking_policy"),
        judge=judge,
    )


def _arm(raw: Mapping[str, Any], *, field: str) -> LiveArmConfig:
    return LiveArmConfig(
        agent_id=_string(raw.get("agent_id"), field=f"{field}.agent_id"),
        context_id=_string(raw.get("context_id"), field=f"{field}.context_id"),
        feedback_journal=_string(raw.get("feedback_journal"), field=f"{field}.feedback_journal"),
    )


def _derive_base_url(raw: Mapping[str, Any]) -> str:
    explicit = raw.get("base_url")
    if explicit is not None:
        return _string(explicit, field="memory_cloud.base_url")
    mcp_url = os.environ.get("KAGURA_MCP_URL", "https://memory.kagura-ai.com/mcp")
    marker = mcp_url.find("/mcp")
    return mcp_url[:marker] if marker >= 0 else mcp_url.rstrip("/")


async def _run(config: Mapping[str, Any]) -> tuple[str, bool]:
    snapshot = load_default_snapshot()
    tasks = load_default_tasks()
    versions = _object(config.get("versions"), field="versions")
    thresholds = _object(config.get("thresholds", {}), field="thresholds")
    configured_fingerprint = _string(
        config.get("snapshot_fingerprint"), field="snapshot_fingerprint"
    )
    if configured_fingerprint != snapshot.fingerprint:
        raise ValueError(
            "snapshot_fingerprint does not match the committed bootstrap_snapshot.json"
        )
    feedback_provenance = config.get("feedback_provenance", "host")
    if feedback_provenance not in ("host", "agent"):
        raise ValueError("feedback_provenance must be host or agent")
    manifest = ExperimentManifest(
        experiment_id=_string(config.get("experiment_id"), field="experiment_id"),
        snapshot_fingerprint=configured_fingerprint,
        versions=_version_stamp(versions),
        thresholds=_thresholds(thresholds),
        generations=_integer(config.get("generations", 5), field="generations"),
        repetitions=_integer(config.get("repetitions", 3), field="repetitions"),
        recall_k=_integer(config.get("recall_k", 5), field="recall_k"),
        exploration_floor=_number(config.get("exploration_floor", 0.01), field="exploration_floor"),
        candidate_pool_k=_integer(config.get("candidate_pool_k", 100), field="candidate_pool_k"),
        seed=_integer(config.get("seed", 188), field="seed"),
        primary_generation=_integer(
            config.get("primary_generation", -1), field="primary_generation"
        ),
        feedback_provenance=feedback_provenance,
    )
    cloud = _object(config.get("memory_cloud"), field="memory_cloud")
    control = _arm(_object(cloud.get("control"), field="memory_cloud.control"), field="control")
    treatment = _arm(
        _object(cloud.get("treatment"), field="memory_cloud.treatment"), field="treatment"
    )
    feedback_mode = cloud.get("feedback_mode", "host")
    if feedback_mode not in ("host", "public"):
        raise ValueError("memory_cloud.feedback_mode must be host or public")
    host_feedback_path = cloud.get("host_feedback_path")
    if host_feedback_path is not None and not isinstance(host_feedback_path, str):
        raise ValueError("memory_cloud.host_feedback_path must be a string or null")
    # Robustness knobs for a multi-hour run (default off/zero → behavior unchanged):
    #  - token_refresh: source the owner bearer from a refreshing `kagura` CLI
    #    provider (the static ~1h OAuth token would 401 mid-run).
    #  - retries/retry_backoff_s: bounded retry on transient REST faults.
    token_refresh = _bool(cloud.get("token_refresh", False), field="memory_cloud.token_refresh")
    token_provider: TokenProvider | None = KaguraCliTokenProvider() if token_refresh else None
    client = BearerJsonClient(
        _derive_base_url(cloud),
        os.environ.get("KAGURA_API_KEY", ""),
        timeout=_number(cloud.get("timeout", 30.0), field="memory_cloud.timeout"),
        token_provider=token_provider,
        retries=_integer(cloud.get("retries", 0), field="memory_cloud.retries"),
        retry_backoff_s=_number(
            cloud.get("retry_backoff_s", 0.5), field="memory_cloud.retry_backoff_s"
        ),
    )
    backend = RestBootstrapBackend(
        request=client.request,
        control=control,
        treatment=treatment,
        feedback_mode=feedback_mode,
        host_feedback_path=host_feedback_path,
    )
    actor_config = _object(config.get("actor"), field="actor")
    command = actor_config.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("actor.command must be an argv string list")
    actor = CommandObjectiveActor(
        command,
        timeout=_number(actor_config.get("timeout", 300.0), field="actor.timeout"),
        retries=_integer(actor_config.get("retries", 0), field="actor.retries"),
        retry_backoff_s=_number(
            actor_config.get("retry_backoff_s", 2.0), field="actor.retry_backoff_s"
        ),
    )
    result = await run_experiment(manifest, snapshot, tasks, backend, actor)
    return result.to_json(), result.gate.default_on_allowed


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kagura-agent-bootstrap-eval",
        description="Run the #188 outcome-level A/B gate on disposable bootstrap contexts.",
    )
    parser.add_argument("--config", required=True, help="pre-registered experiment JSON")
    parser.add_argument("--output", required=True, help="result JSON path")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        raw = json.loads(Path(ns.config).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("config root must be an object")
        result_json, passed = asyncio.run(_run(raw))
        Path(ns.output).write_text(result_json, encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError, CorpusError) as exc:
        print(f"bootstrap eval config error: {exc}", file=sys.stderr)
        return 2
    except (ExperimentInvariantError, LiveEvalError) as exc:
        print(f"bootstrap eval invalid: {exc}", file=sys.stderr)
        return 3
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
