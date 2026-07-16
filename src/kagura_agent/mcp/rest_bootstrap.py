"""Trusted-host REST adapter for one-call agent bootstrap (#187).

The existing :class:`McpMemoryClient` remains the memory read/write transport.
Session-start grounding is different: it is host orchestration, uses an
agent-bound member key, and must never become an MCP tool exposed to the brain.
This module composes those two surfaces without widening ``MemoryClient`` with
credentials or raw SDK response objects.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from kagura_agent.mcp.memory_cloud import (
    ALWAYS_DELIVERY,
    ON_RECALL_DELIVERY,
    TRUSTED_TIER,
    AgentBootstrap,
    Memory,
    MemoryUnreachableError,
)

_BOOTSTRAP_COMPONENTS = ("pinned", "recall", "upcoming", "state", "policy")

BootstrapCall = Callable[..., Awaitable[Any]]


class MemoryTransport(Protocol):
    """The non-bootstrap memory operations delegated to the existing transport."""

    async def remember(
        self,
        text: str,
        *,
        tags: tuple[str, ...] = (),
        trust_tier: str = TRUSTED_TIER,
        delivery_mode: str = ON_RECALL_DELIVERY,
    ) -> str: ...

    async def recall(
        self,
        query: str,
        *,
        trusted_only: bool = False,
        tags: tuple[str, ...] = (),
    ) -> list[Memory]: ...

    async def load_pinned(self) -> list[Memory]: ...

    async def create_edge(self, src_id: str, dst_id: str, *, type: str) -> None: ...


class BootstrapContractError(MemoryUnreachableError):
    """The server returned an unsafe or internally inconsistent bootstrap."""


def _as_mapping(result: Any) -> Mapping[str, Any]:
    """Accept an SDK Pydantic model or a raw mapping at the injected test seam."""
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        result = model_dump(mode="json")
    if not isinstance(result, Mapping):
        raise BootstrapContractError("bootstrap response is not an object")
    return result


def _trusted_component_memories(
    component: Mapping[str, Any], *record_keys: str
) -> tuple[Memory, ...]:
    """Parse a server-trusted compact lane into canonical runtime memories."""
    records: list[Mapping[str, Any]] = []
    for key in record_keys:
        value = component.get(key)
        if isinstance(value, list):
            if not all(isinstance(record, Mapping) for record in value):
                raise BootstrapContractError(f"bootstrap {key} contains a non-object")
            records = value
            break

    memories: list[Memory] = []
    for record in records:
        raw_tags = record.get("tags")
        tags = tuple(str(tag) for tag in raw_tags) if isinstance(raw_tags, (list, tuple)) else ()
        memory = Memory(
            id=str(record.get("memory_id") or record.get("id") or ""),
            text=str(
                record.get("summary")
                or record.get("text")
                or record.get("content")
                or record.get("details")
                or ""
            ),
            tags=tags,
            # These compact rows intentionally omit trust_tier. Trust is asserted
            # by the server bootstrap lane, with recall additionally proven below.
            trust_tier=TRUSTED_TIER,
            delivery_mode=str(record.get("delivery_mode") or ON_RECALL_DELIVERY),
        )
        if not memory.id or not memory.text:
            raise BootstrapContractError("bootstrap memory has no stable id/text")
        memories.append(memory)
    return tuple(memories)


def _parse_agent_bootstrap(result: Any) -> AgentBootstrap:
    """Validate and normalize one production ``AgentsClient.bootstrap`` envelope."""
    raw = _as_mapping(result)
    if raw.get("status") != "success":
        raise BootstrapContractError("bootstrap top-level status is not success")
    degraded = raw.get("degraded")
    agent = raw.get("agent")
    context = raw.get("context")
    instructions = raw.get("instructions")
    raw_components = raw.get("components")
    if not isinstance(degraded, bool):
        raise BootstrapContractError("bootstrap degraded flag is not boolean")
    if (
        not isinstance(agent, Mapping)
        or not isinstance(agent.get("agent_id"), str)
        or not agent["agent_id"]
    ):
        raise BootstrapContractError("bootstrap has no agent identity")
    if (
        not isinstance(context, Mapping)
        or not isinstance(context.get("id"), str)
        or not context["id"]
    ):
        raise BootstrapContractError("bootstrap has no context identity")
    if instructions is not None and not isinstance(instructions, str):
        raise BootstrapContractError("bootstrap instruction block is malformed")
    if not isinstance(raw_components, Mapping):
        raise BootstrapContractError("bootstrap has no component map")

    components: dict[str, Mapping[str, Any]] = {}
    statuses: list[tuple[str, str]] = []
    for name in _BOOTSTRAP_COMPONENTS:
        component = raw_components.get(name)
        if not isinstance(component, Mapping):
            raise BootstrapContractError(f"bootstrap has no {name} component provenance")
        status = component.get("status")
        if status not in ("ok", "error", "skipped"):
            raise BootstrapContractError(f"bootstrap {name} component status is invalid")
        components[name] = component
        statuses.append((name, str(status)))

    failures = tuple(name for name, status in statuses if status == "error")
    if degraded != bool(failures):
        raise BootstrapContractError("bootstrap degraded flag disagrees with component failures")
    recall_component = components["recall"]
    if recall_component.get("status") == "skipped":
        raise BootstrapContractError("bootstrap recall skipped despite a query")
    if (
        recall_component.get("status") == "ok"
        and recall_component.get("trust_filter") != TRUSTED_TIER
    ):
        raise BootstrapContractError("bootstrap recall is not proven trusted-only")

    pinned = (
        _trusted_component_memories(components["pinned"], "memories")
        if components["pinned"].get("status") == "ok"
        else ()
    )
    pinned = tuple(
        memory
        if memory.delivery_mode == ALWAYS_DELIVERY
        else Memory(
            id=memory.id,
            text=memory.text,
            tags=memory.tags,
            trust_tier=memory.trust_tier,
            delivery_mode=ALWAYS_DELIVERY,
        )
        for memory in pinned
    )
    recalled = (
        _trusted_component_memories(recall_component, "results", "memories")
        if recall_component.get("status") == "ok"
        else ()
    )
    upcoming = (
        _trusted_component_memories(components["upcoming"], "results", "memories")
        if components["upcoming"].get("status") == "ok"
        else ()
    )

    state_component = components["state"]
    raw_state = state_component.get("states", {})
    if state_component.get("status") == "ok" and not isinstance(raw_state, Mapping):
        raise BootstrapContractError("bootstrap state component is malformed")
    state = dict(raw_state) if isinstance(raw_state, Mapping) else {}
    policy_component = components["policy"]
    policy = (
        {key: value for key, value in policy_component.items() if key != "status"}
        if policy_component.get("status") == "ok"
        else None
    )

    context_id = str(context["id"])
    binding = agent.get("binding")
    binding_context = binding.get("context_id") if isinstance(binding, Mapping) else None
    if binding_context is not None and str(binding_context) != context_id:
        raise BootstrapContractError("bootstrap agent binding disagrees with context identity")

    return AgentBootstrap(
        agent_id=str(agent["agent_id"]),
        context_id=context_id,
        instructions=instructions or "",
        pinned=pinned,
        recall=recalled,
        upcoming=upcoming,
        state=state,
        policy=policy,
        degraded=degraded,
        component_failures=failures,
        component_statuses=tuple(statuses),
    )


class RestBootstrapMemoryClient:
    """Delegate memory I/O to ``inner`` and bootstrap through host-side REST."""

    def __init__(
        self,
        inner: MemoryTransport,
        bootstrap_call: BootstrapCall,
        *,
        agent_id: str,
        context_id: str,
    ) -> None:
        self._inner = inner
        self._bootstrap_call = bootstrap_call
        self._agent_id = agent_id
        self._context_id = context_id

    async def remember(
        self,
        text: str,
        *,
        tags: tuple[str, ...] = (),
        trust_tier: str = TRUSTED_TIER,
        delivery_mode: str = ON_RECALL_DELIVERY,
    ) -> str:
        return await self._inner.remember(
            text,
            tags=tags,
            trust_tier=trust_tier,
            delivery_mode=delivery_mode,
        )

    async def recall(
        self,
        query: str,
        *,
        tags: tuple[str, ...] = (),
        trusted_only: bool = False,
    ) -> list[Memory]:
        return await self._inner.recall(query, tags=tags, trusted_only=trusted_only)

    async def load_pinned(self) -> list[Memory]:
        return await self._inner.load_pinned()

    async def get_agent_bootstrap(
        self,
        *,
        session_id: str,
        query: str,
        recall_k: int = 5,
    ) -> AgentBootstrap:
        if not 1 <= recall_k <= 100:
            raise ValueError("recall_k must be in [1, 100]")
        try:
            result = await self._bootstrap_call(
                self._agent_id,
                context_id=self._context_id,
                session_id=session_id,
                query=query,
                recall_k=recall_k,
                include=list(_BOOTSTRAP_COMPONENTS),
            )
        except Exception as exc:
            raise MemoryUnreachableError(f"agent bootstrap REST request failed: {exc}") from exc
        bootstrap = _parse_agent_bootstrap(result)
        if bootstrap.agent_id != self._agent_id:
            raise BootstrapContractError("bootstrap resolved outside the configured agent")
        if bootstrap.context_id != self._context_id:
            raise BootstrapContractError("bootstrap resolved outside the configured context")
        return bootstrap

    async def create_edge(self, src_id: str, dst_id: str, *, type: str) -> None:
        await self._inner.create_edge(src_id, dst_id, type=type)


def build_agents_bootstrap_call(
    *, api_key: str, mcp_url: str | None = None
) -> BootstrapCall:  # pragma: no cover - deployment edge (SDK + live REST service)
    """Build a short-lived SDK REST call using an agent-bound member key."""

    async def call(agent_id: str, **kwargs: Any) -> Any:
        try:
            from kagura_memory import AgentsClient
        except ImportError as exc:
            raise MemoryUnreachableError(
                "cloud bootstrap requires the optional memory SDK "
                "(install: pip install 'kagura-agent[memory]')"
            ) from exc
        async with AgentsClient.from_mcp_url(api_key=api_key, mcp_url=mcp_url) as client:
            return await client.bootstrap(agent_id, **kwargs)

    return call
