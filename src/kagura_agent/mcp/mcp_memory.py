"""#111: the trust-aware MCP-server-backed MemoryClient — #107's headline tier.

`LocalMemoryClient` is in-process and `SqliteMemoryClient` (#110) is durable but
single-host. `McpMemoryClient` is the production backbone: it speaks to the
kagura-memory **MCP server**, so memory persists and is shared across hosts and
its trust-aware surface (`recall(trusted_only=True)`, `load_pinned`, `create_edge`,
`feedback`) is the real one.

**Pure translation behind an injected transport.** The adapter never imports the
`mcp` SDK; it depends only on an injected ``call_tool(name, args) -> result``
coroutine. So the translation it owns — building each tool's args, enforcing
`trusted_only`, parsing results into :class:`Memory` — is fully unit-testable with
a fake transport. The real MCP `ClientSession` connection + auth lives behind
:func:`build_mcp_call_tool`, the ``# pragma: no cover`` deployment edge (it needs
the SDK + a live server, so its end-to-end behaviour is verified at deployment).

**Provenance (the membrane rule).** `recall(trusted_only=True)` both sends the
server's `trust_tier='trusted'` filter AND re-filters client-side, so a quarantined
/ externally-ingested memory is *never* fed back as behaviour-influencing context
even if the server filter regressed (OWASP LLM01/LLM03). `remember`'s `trust_tier`
is accepted for protocol parity but **not forwarded** — the server assigns the tier
by the connection's identity, so the agent can never self-assert trust (the
`QuarantinedMemoryClient` posture).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from kagura_agent.mcp.memory_cloud import (
    _VALID_DELIVERY,
    ON_RECALL_DELIVERY,
    QUARANTINE_TIER,
    TRUSTED_TIER,
    Memory,
)

#: The injected MCP transport: call tool ``name`` with ``args``, return its result.
McpCall = Callable[[str, dict[str, Any]], Awaitable[Any]]

#: The kagura-memory `remember` tool requires a non-empty `summary` (10-500 chars,
#: the searchable conclusion). The MemoryClient surface has only `text`, so the
#: summary is derived from it.
_SUMMARY_MAX = 500


def _records_of(result: Any) -> list[dict[str, Any]]:
    """Extract the list of memory records from a recall / load_pinned result,
    tolerant of the server returning a bare list or wrapping it under a key."""
    if isinstance(result, list):
        return [r for r in result if isinstance(r, dict)]
    if isinstance(result, dict):
        for key in ("results", "memories", "items"):
            value = result.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
    return []


def _is_trusted(record: Mapping[str, Any]) -> bool:
    """Whether a raw record carries an EXPLICIT trusted tier — case/space tolerant,
    fail-closed (a missing/null/unrecognised tier is NOT trusted). The single
    provenance gate; never default a tier to trusted."""
    return str(record.get("trust_tier") or "").strip().lower() == TRUSTED_TIER


def _to_memory(record: Mapping[str, Any]) -> Memory:
    """Map one MCP memory record onto the runtime :class:`Memory`.

    A missing/blank ``trust_tier`` defaults to **QUARANTINE** (fail-closed): an
    unprovable tier must never present as trusted to a downstream consumer (e.g.
    continuity's ``load_pinned`` trust gate), so the only path that asserts trust
    is :func:`_is_trusted`."""
    # Only a list/tuple is a tag list. A bare string would otherwise iterate into
    # per-character tags ("goal" → 'g','o','a','l'); anything non-list is dropped.
    raw_tags = record.get("tags")
    tags = tuple(str(t) for t in raw_tags) if isinstance(raw_tags, (list, tuple)) else ()
    # Canonicalize the tier (strip + lower) so the SURFACED label matches what every
    # downstream gate checks by exact equality — notably continuity.load_guardrails,
    # which keeps only `m.trust_tier == TRUSTED_TIER`. Without this, a server-returned
    # trusted pinned guardrail labelled "Trusted" / " trusted " is silently dropped
    # from the prompt every turn (the pinned lane has no _is_trusted pre-filter, only
    # recall does). A missing/blank tier defaults to QUARANTINE (fail-closed): an
    # unprovable tier must never present as trusted.
    trust_tier = str(record.get("trust_tier") or "").strip().lower() or QUARANTINE_TIER
    return Memory(
        id=str(record.get("memory_id") or record.get("id") or ""),
        text=str(record.get("summary") or record.get("text") or record.get("content") or ""),
        tags=tags,
        trust_tier=trust_tier,
        delivery_mode=str(record.get("delivery_mode") or ON_RECALL_DELIVERY),
    )


def _memory_id_of(result: Any) -> str:
    """Extract the new memory id from a `remember` result (bare id or wrapped)."""
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        return str(result.get("memory_id") or result.get("id") or "")
    return ""


class McpMemoryClient:
    """A :class:`~kagura_agent.mcp.memory_cloud.MemoryClient` backed by the
    kagura-memory MCP server, via an injected ``call_tool`` transport."""

    def __init__(
        self, call_tool: McpCall, *, context_id: str, default_type: str = "note"
    ) -> None:
        self._call = call_tool
        self._context_id = context_id
        self._default_type = default_type

    async def remember(
        self,
        text: str,
        *,
        tags: tuple[str, ...] = (),
        trust_tier: str = TRUSTED_TIER,
        delivery_mode: str = ON_RECALL_DELIVERY,
    ) -> str:
        # Fail-CLOSED at the write boundary, like LocalMemoryClient: a typo'd
        # delivery mode must never be stored verbatim and silently never pin (the
        # #88 standing-guardrail hazard).
        if delivery_mode not in _VALID_DELIVERY:
            raise ValueError(
                f"unknown delivery_mode {delivery_mode!r} (expected one of {_VALID_DELIVERY})"
            )
        # trust_tier is intentionally NOT forwarded: the server assigns the tier by
        # the connection identity, so the agent cannot self-assert trust.
        result = await self._call(
            "remember",
            {
                "context_id": self._context_id,
                "summary": text[:_SUMMARY_MAX],
                "content": text,
                "type": self._default_type,
                "tags": list(tags),
                "delivery_mode": delivery_mode,
            },
        )
        mid = _memory_id_of(result)
        if not mid:
            # A write that returns no id did not persist — fail closed rather than
            # hand back "" that would become an empty src_id in a later create_edge.
            raise RuntimeError("remember: the MCP server returned no memory id")
        return mid

    async def recall(
        self,
        query: str,
        *,
        trusted_only: bool = False,
        tags: tuple[str, ...] = (),
    ) -> list[Memory]:
        filters: dict[str, Any] = {}
        if trusted_only:
            filters["trust_tier"] = "trusted"  # server-side exclusion of external memories
        if tags:
            filters["tags"] = list(tags)
        args: dict[str, Any] = {"context_id": self._context_id, "query": query}
        if filters:
            args["filters"] = filters
        records = _records_of(await self._call("recall", args))
        if trusted_only:
            # Defense-in-depth: never feed a non-trusted memory as behaviour context,
            # even if the server's trust filter regressed (the membrane rule). Keep
            # only records with an EXPLICIT trusted tier (fail-closed).
            records = [r for r in records if _is_trusted(r)]
        return [_to_memory(r) for r in records]

    async def load_pinned(self) -> list[Memory]:
        result = await self._call("load_pinned", {"context_id": self._context_id})
        return [_to_memory(r) for r in _records_of(result)]

    async def create_edge(self, src_id: str, dst_id: str, *, type: str) -> None:
        # ``type`` is passed through as the server's ``edge_type``. The kagura-memory
        # server validates it against its own edge vocabulary, so a caller type the
        # server does not accept (the codebase's "prevents" is not in the default
        # enum) surfaces as a server error here rather than being silently rewritten
        # — mapping the edge vocabulary to the server's is a deployment concern.
        await self._call(
            "create_edge",
            {
                "context_id": self._context_id,
                "source_id": src_id,
                "target_id": dst_id,
                "edge_type": type,
            },
        )

    async def record_feedback(self, memory_id: str, query: str, *, helpful: bool) -> None:
        """Host-side retrieval-quality signal (#90): was ``memory_id`` useful for
        ``query``? Off the agent protocol — an independent verdict, never the
        agent's self-report. Append-only on the server, excluded from recall."""
        await self._call(
            "feedback",
            {
                "context_id": self._context_id,
                "memory_id": memory_id,
                "helpful": helpful,
                "query": query,
            },
        )


def build_mcp_call_tool(  # pragma: no cover - deployment edge (needs the mcp SDK + a live server)
    env: Mapping[str, str],
) -> McpCall:
    """Build the real MCP transport coroutine for :class:`McpMemoryClient`.

    Lazy: it captures the connection config now (so ``make_memory_client`` stays
    synchronous and dependency-free) and establishes the `ClientSession` to the
    kagura-memory MCP server on first use, importing the optional ``mcp`` SDK then.
    Fail-closed: a missing SDK or unreachable server raises on first call, so the
    run fails rather than silently degrading to memory-less. The server command /
    URL + auth are read from the host environment, never baked.
    """
    import shlex

    # Split into program + args so a server command WITH arguments (the natural
    # "kagura-memory-mcp serve --stdio") execs correctly, not as one literal name.
    # NOTE: re-establishes the session per call (a fresh server process each memory
    # op) — correct but spawn-heavy; reusing one lazy session is a follow-up.
    argv = shlex.split(env.get("KAGURA_AGENT_MEMORY_MCP_SERVER", ""))

    async def call_tool(name: str, args: dict[str, Any]) -> Any:
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
        except ImportError as exc:
            raise RuntimeError(
                "the MCP cloud memory backend needs the 'mcp' SDK "
                "(install: pip install mcp) — or unset KAGURA_AGENT_MEMORY_MCP_CONTEXT"
            ) from exc
        params = StdioServerParameters(command=argv[0], args=argv[1:])
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, args)
            # Prefer the structured payload — but only when truly absent (None) fall
            # back, so a VALID-but-empty result ([]/{}) is not discarded by `or`.
            structured = getattr(result, "structuredContent", None)
            return structured if structured is not None else result

    return call_tool
