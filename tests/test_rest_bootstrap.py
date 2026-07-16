"""#187: trusted-host REST bootstrap composed with the existing memory transport."""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import pytest

from kagura_agent.mcp.memory_cloud import (
    ALWAYS_DELIVERY,
    TRUSTED_TIER,
    LocalMemoryClient,
    MemoryClient,
    MemoryUnreachableError,
)
from kagura_agent.mcp.rest_bootstrap import (
    BootstrapContractError,
    RestBootstrapMemoryClient,
)

_AGENT = "550e8400-e29b-41d4-a716-446655440001"
_CTX = "550e8400-e29b-41d4-a716-446655440000"


def _bootstrap_envelope() -> dict[str, Any]:
    return {
        "status": "success",
        "degraded": False,
        "agent": {
            "agent_id": _AGENT,
            "binding": {"context_id": _CTX, "is_default": True},
        },
        "context": {"id": _CTX, "usage_guide": "Use project facts."},
        "instructions": "Use project facts.\n\nNever store credentials.",
        "components": {
            "pinned": {
                "status": "ok",
                "memories": [
                    {
                        "memory_id": "pin-1",
                        "summary": "Never deploy without approval",
                        "tags": ["guardrail"],
                        "delivery_mode": "always",
                    }
                ],
            },
            "recall": {
                "status": "ok",
                "trust_filter": "trusted",
                "results": [{"memory_id": "recall-1", "summary": "Use decorrelated jitter"}],
            },
            "upcoming": {
                "status": "ok",
                "results": [{"memory_id": "time-1", "summary": "Rotate the key tomorrow"}],
            },
            "state": {"status": "ok", "states": {"phase": {"value": "verify"}}},
            "policy": {"status": "skipped", "reason": "no_policy_bundle"},
        },
    }


class _FakeBootstrap:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _SdkModel:
    """Small stand-in for AgentBootstrapResponse.model_dump(mode='json')."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return self.raw


def _client(result: Any) -> tuple[RestBootstrapMemoryClient, _FakeBootstrap]:
    fake = _FakeBootstrap(result)
    return (
        RestBootstrapMemoryClient(
            LocalMemoryClient(),
            fake,
            agent_id=_AGENT,
            context_id=_CTX,
        ),
        fake,
    )


async def test_rest_bootstrap_is_one_call_and_normalizes_trusted_lanes() -> None:
    client, fake = _client(_SdkModel(_bootstrap_envelope()))

    out = await client.get_agent_bootstrap(
        session_id="session-1", query="how should retries work?", recall_k=5
    )

    assert isinstance(client, MemoryClient)
    assert fake.calls == [
        (
            (_AGENT,),
            {
                "context_id": _CTX,
                "session_id": "session-1",
                "query": "how should retries work?",
                "recall_k": 5,
                "include": ["pinned", "recall", "upcoming", "state", "policy"],
            },
        )
    ]
    assert out.agent_id == _AGENT and out.context_id == _CTX
    assert [memory.id for memory in out.pinned] == ["pin-1"]
    assert [memory.id for memory in out.recall] == ["recall-1"]
    assert [memory.id for memory in out.upcoming] == ["time-1"]
    assert all(
        memory.trust_tier == TRUSTED_TIER for memory in (*out.pinned, *out.recall, *out.upcoming)
    )
    assert out.pinned[0].delivery_mode == ALWAYS_DELIVERY
    assert out.pinned[0].tags == ("guardrail",)
    assert out.state == {"phase": {"value": "verify"}}
    assert out.degraded is False and out.component_failures == ()


async def test_rest_bootstrap_preserves_fail_soft_component_errors_and_policy() -> None:
    envelope = _bootstrap_envelope()
    envelope["degraded"] = True
    envelope["instructions"] = None
    envelope["components"]["recall"] = {"status": "error", "error": "rate_limited"}
    envelope["components"]["policy"] = {"status": "ok", "mode": "enforce"}
    client, _ = _client(envelope)

    out = await client.get_agent_bootstrap(session_id="s", query="q")

    assert out.degraded is True
    assert out.component_failures == ("recall",)
    assert out.recall == ()
    assert out.pinned and out.upcoming
    assert out.instructions == ""
    assert out.policy == {"mode": "enforce"}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update(status="error"), "top-level status"),
        (lambda raw: raw.update(degraded="false"), "degraded flag"),
        (lambda raw: raw.update(agent={}), "agent identity"),
        (lambda raw: raw.update(context={}), "context identity"),
        (lambda raw: raw.update(instructions=42), "instruction block"),
        (lambda raw: raw.update(components=[]), "component map"),
        (lambda raw: raw["components"].pop("state"), "state component"),
        (lambda raw: raw["components"]["state"].update(status="bogus"), "status is invalid"),
        (lambda raw: raw["components"]["state"].update(states=[]), "state component"),
        (
            lambda raw: raw["components"]["recall"].update(trust_filter="external"),
            "trusted-only",
        ),
        (lambda raw: raw["components"]["recall"].update(status="skipped"), "skipped"),
        (lambda raw: raw.update(degraded=True), "disagrees"),
        (
            lambda raw: raw["components"]["pinned"].update(memories=["not-an-object"]),
            "non-object",
        ),
        (
            lambda raw: raw["components"]["upcoming"].update(
                results=[{"memory_id": "missing-text"}]
            ),
            "stable id/text",
        ),
        (
            lambda raw: raw["components"]["pinned"]["memories"][0].update(
                memory_id={"unexpected": "object"}
            ),
            "memory id is not a non-empty string",
        ),
        (
            lambda raw: raw["components"]["recall"]["results"][0].update(
                summary=["unexpected", "list"]
            ),
            "memory text is not a non-empty string",
        ),
    ],
)
async def test_rest_bootstrap_rejects_inconsistent_contract(mutate, message) -> None:
    envelope = copy.deepcopy(_bootstrap_envelope())
    mutate(envelope)
    client, _ = _client(envelope)

    with pytest.raises(BootstrapContractError, match=message):
        await client.get_agent_bootstrap(session_id="s", query="q")


async def test_rest_bootstrap_rejects_identity_mismatch_and_invalid_k() -> None:
    internal = _bootstrap_envelope()
    internal["agent"]["binding"]["context_id"] = "other"
    client, _ = _client(internal)
    with pytest.raises(BootstrapContractError, match="binding disagrees"):
        await client.get_agent_bootstrap(session_id="s", query="q")

    wrong_agent = _bootstrap_envelope()
    wrong_agent["agent"]["agent_id"] = "other"
    client, _ = _client(wrong_agent)
    with pytest.raises(BootstrapContractError, match="configured agent"):
        await client.get_agent_bootstrap(session_id="s", query="q")

    wrong_context = _bootstrap_envelope()
    wrong_context["context"]["id"] = "other"
    wrong_context["agent"]["binding"]["context_id"] = "other"
    client, _ = _client(wrong_context)
    with pytest.raises(BootstrapContractError, match="configured context"):
        await client.get_agent_bootstrap(session_id="s", query="q")

    client, fake = _client(_bootstrap_envelope())
    with pytest.raises(ValueError, match="recall_k"):
        await client.get_agent_bootstrap(session_id="s", query="q", recall_k=0)
    assert fake.calls == []


async def test_rest_bootstrap_wraps_transport_failure_as_memory_unreachable() -> None:
    client, _ = _client(OSError("network down: Authorization=Bearer secret-key"))
    with pytest.raises(MemoryUnreachableError, match="REST request failed") as exc:
        await client.get_agent_bootstrap(session_id="s", query="q")
    assert "secret-key" not in str(exc.value)
    assert isinstance(exc.value.__cause__, OSError)


async def test_rest_bootstrap_propagates_cancellation() -> None:
    client, _ = _client(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await client.get_agent_bootstrap(session_id="s", query="q")


async def test_rest_wrapper_delegates_non_bootstrap_memory_operations() -> None:
    inner = LocalMemoryClient()
    client = RestBootstrapMemoryClient(
        inner,
        _FakeBootstrap(_bootstrap_envelope()),
        agent_id=_AGENT,
        context_id=_CTX,
    )
    memory_id = await client.remember("alpha", tags=("tag",))
    assert [memory.id for memory in await client.recall("alpha", tags=("tag",))] == [memory_id]
    assert await client.load_pinned() == []
    await client.create_edge(memory_id, "target", type="prevents")
    assert inner.edges_of(memory_id) == [("target", "prevents")]
