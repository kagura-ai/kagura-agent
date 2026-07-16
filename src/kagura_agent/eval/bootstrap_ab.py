"""Outcome-level A/B gate for bootstrap-grounded reinforcement ranking (#188).

The harness deliberately has no legacy ``recall()`` arm. Every trial must arrive
through the server-shaped ``get_agent_bootstrap`` envelope introduced by #187.
Control and treatment start from byte-equivalent logical snapshots in isolated
agent/context/feedback lanes; the only configured difference is whether verified
feedback may influence ranking.

The module keeps orchestration and decision math dependency-free so it remains
unit-testable in the core package. A live adapter may use memory-cloud REST, MCP,
or an in-process service, but it must implement :class:`BootstrapExperimentBackend`
and satisfy the same isolation invariants.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from importlib.resources import files
from typing import Any, Literal, Protocol, runtime_checkable


class ExperimentInvariantError(RuntimeError):
    """The experiment cannot identify the ranking actuator cleanly."""


class CorpusError(ValueError):
    """The fixed task/snapshot corpus is malformed or leaks its gold answer."""


class Arm(StrEnum):
    CONTROL = "control"
    TREATMENT = "treatment"


CheckKind = Literal["contains_all", "exact", "regex"]
VerifiedSource = Literal["objective_check", "host_check", "hitl_approval"]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ObjectiveCheck:
    """A hidden, host-side deterministic task-success check.

    The actor receives the task prompt and bootstrap context, never this object.
    This keeps the verdict independent of the agent's self-report while avoiding
    an LLM judge for the committed fixed corpus.
    """

    kind: CheckKind
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in ("contains_all", "exact", "regex"):
            raise CorpusError(f"unsupported objective check: {self.kind!r}")
        if not self.values or any(not value for value in self.values):
            raise CorpusError("objective check values must be non-empty")
        if self.kind == "exact" and len(self.values) != 1:
            raise CorpusError("an exact objective check has exactly one value")
        if self.kind == "regex":
            for pattern in self.values:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise CorpusError(f"invalid objective regex {pattern!r}: {exc}") from exc

    def score(self, text: str) -> float:
        normalized = " ".join(text.casefold().split())
        if self.kind == "exact":
            return float(normalized == " ".join(self.values[0].casefold().split()))
        if self.kind == "contains_all":
            hits = sum(value.casefold() in normalized for value in self.values)
        else:
            hits = sum(
                re.search(pattern, text, flags=re.IGNORECASE) is not None for pattern in self.values
            )
        return hits / len(self.values)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> ObjectiveCheck:
        kind = raw.get("kind")
        values = raw.get("values")
        if kind not in ("contains_all", "exact", "regex"):
            raise CorpusError(f"unsupported objective check: {kind!r}")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise CorpusError("objective check values must be a string list")
        return cls(kind=kind, values=tuple(values))


@dataclass(frozen=True)
class TaskSpec:
    id: str
    category: str
    query: str
    prompt: str
    check: ObjectiveCheck
    gold_memory_id: str
    candidate_memory_ids: tuple[str, ...]
    tail: bool = False
    held_out: bool = False

    def actor_payload(self) -> dict[str, object]:
        """Return the task data visible to an actor (gold/check intentionally absent)."""
        return {
            "id": self.id,
            "category": self.category,
            "query": self.query,
            "prompt": self.prompt,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> TaskSpec:
        check = raw.get("check")
        candidates = raw.get("candidate_memory_ids")
        if not isinstance(check, dict):
            raise CorpusError("task.check must be an object")
        if not isinstance(candidates, list) or not all(
            isinstance(candidate, str) and candidate for candidate in candidates
        ):
            raise CorpusError("task.candidate_memory_ids must be a non-empty string list")
        task = cls(
            id=_require_string(raw.get("id"), field="task.id"),
            category=_require_string(raw.get("category"), field="task.category"),
            query=_require_string(raw.get("query"), field="task.query"),
            prompt=_require_string(raw.get("prompt"), field="task.prompt"),
            check=ObjectiveCheck.from_dict(check),
            gold_memory_id=_require_string(raw.get("gold_memory_id"), field="task.gold_memory_id"),
            candidate_memory_ids=tuple(candidates),
            tail=bool(raw.get("tail", False)),
            held_out=bool(raw.get("held_out", False)),
        )
        if task.gold_memory_id not in task.candidate_memory_ids:
            raise CorpusError(f"{task.id}: gold memory must be in candidate_memory_ids")
        prompt_folded = f"{task.query}\n{task.prompt}".casefold()
        leaked = [value for value in task.check.values if value.casefold() in prompt_folded]
        if leaked:
            raise CorpusError(f"{task.id}: objective answer leaks into task text: {leaked!r}")
        return task


@dataclass(frozen=True)
class SnapshotMemory:
    logical_id: str
    summary: str
    content: str
    tags: tuple[str, ...]
    delivery_mode: str = "on_recall"
    trust_tier: str = "trusted"

    def canonical(self) -> dict[str, object]:
        return {
            "logical_id": self.logical_id,
            "summary": self.summary,
            "content": self.content,
            "tags": list(self.tags),
            "delivery_mode": self.delivery_mode,
            "trust_tier": self.trust_tier,
        }


@dataclass(frozen=True)
class BootstrapSnapshot:
    name: str
    memories: tuple[SnapshotMemory, ...]

    @property
    def fingerprint(self) -> str:
        return _sha256([memory.canonical() for memory in self.memories])

    @property
    def memory_ids(self) -> tuple[str, ...]:
        return tuple(memory.logical_id for memory in self.memories)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> BootstrapSnapshot:
        values = raw.get("memories")
        if not isinstance(values, list) or not values:
            raise CorpusError("snapshot.memories must be a non-empty list")
        memories: list[SnapshotMemory] = []
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                raise CorpusError(f"snapshot.memories[{index}] must be an object")
            tags = value.get("tags", [])
            if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                raise CorpusError(f"snapshot.memories[{index}].tags must be a string list")
            memories.append(
                SnapshotMemory(
                    logical_id=_require_string(
                        value.get("logical_id"), field=f"snapshot.memories[{index}].logical_id"
                    ),
                    summary=_require_string(
                        value.get("summary"), field=f"snapshot.memories[{index}].summary"
                    ),
                    content=_require_string(
                        value.get("content"), field=f"snapshot.memories[{index}].content"
                    ),
                    tags=tuple(tags),
                    delivery_mode=str(value.get("delivery_mode", "on_recall")),
                    trust_tier=str(value.get("trust_tier", "trusted")),
                )
            )
        ids = [memory.logical_id for memory in memories]
        if len(ids) != len(set(ids)):
            raise CorpusError("snapshot logical ids must be unique")
        return cls(
            name=_require_string(raw.get("name"), field="snapshot.name"),
            memories=tuple(memories),
        )


@dataclass(frozen=True)
class GateThresholds:
    delta_min: float = 0.02
    confidence: float = 0.95
    bootstrap_resamples: int = 5000
    max_degraded_rate: float = 0.0
    max_tail_regression: float = 0.02
    max_entropy_drop: float = 0.10
    utility_flat_tolerance: float = 0.01
    max_goodhart_decline: float = 0.02

    def __post_init__(self) -> None:
        if not -1.0 <= self.delta_min <= 1.0:
            raise ValueError("delta_min must be in [-1, 1]")
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0.5, 1)")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")
        for name in (
            "max_degraded_rate",
            "max_tail_regression",
            "max_entropy_drop",
            "utility_flat_tolerance",
            "max_goodhart_decline",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class VersionStamp:
    code: str
    actor_model: str
    actor: str
    bootstrap_api: str
    ranking_policy: str
    judge: str | None = None

    def __post_init__(self) -> None:
        for name in ("code", "actor_model", "actor", "bootstrap_api", "ranking_policy"):
            if not getattr(self, name).strip():
                raise ValueError(f"version stamp {name} must not be blank")


@dataclass(frozen=True)
class ExperimentManifest:
    experiment_id: str
    snapshot_fingerprint: str
    versions: VersionStamp
    thresholds: GateThresholds = GateThresholds()
    generations: int = 5
    repetitions: int = 3
    recall_k: int = 5
    seed: int = 188
    primary_generation: int = -1
    feedback_provenance: Literal["host", "agent"] = "host"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", self.experiment_id):
            raise ValueError("experiment_id must match [A-Za-z0-9._-]{1,64}")
        if not self.snapshot_fingerprint.startswith("sha256:"):
            raise ValueError("snapshot_fingerprint must be a sha256 fingerprint")
        if self.generations < 2:
            raise ValueError("long-horizon evaluation requires at least 2 generations")
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if not 1 <= self.recall_k <= 100:
            raise ValueError("recall_k must be in [1, 100]")
        if self.primary_generation != -1 and not 0 <= self.primary_generation < self.generations:
            raise ValueError("primary_generation must be -1 (last) or a valid generation")
        if self.feedback_provenance not in ("host", "agent"):
            raise ValueError("feedback_provenance must be host or agent")

    @property
    def resolved_primary_generation(self) -> int:
        return self.generations - 1 if self.primary_generation == -1 else self.primary_generation


@dataclass(frozen=True)
class ArmHandle:
    arm: Arm
    agent_id: str
    context_id: str
    snapshot_fingerprint: str
    feedback_journal: str
    feedback_influence: bool


@dataclass(frozen=True)
class ArmPair:
    control: ArmHandle
    treatment: ArmHandle

    def validate(self, manifest: ExperimentManifest) -> None:
        if self.control.arm is not Arm.CONTROL or self.treatment.arm is not Arm.TREATMENT:
            raise ExperimentInvariantError("backend returned handles under the wrong arm")
        if self.control.feedback_influence:
            raise ExperimentInvariantError("control must freeze feedback influence")
        if not self.treatment.feedback_influence:
            raise ExperimentInvariantError("treatment must enable feedback influence")
        for handle in (self.control, self.treatment):
            if handle.snapshot_fingerprint != manifest.snapshot_fingerprint:
                raise ExperimentInvariantError(
                    f"{handle.arm.value} snapshot does not match the pre-registered snapshot"
                )
        if self.control.context_id == self.treatment.context_id:
            raise ExperimentInvariantError("A/B arms must use distinct contexts")
        if self.control.agent_id == self.treatment.agent_id:
            raise ExperimentInvariantError("A/B arms must use distinct agent identities")
        if self.control.feedback_journal == self.treatment.feedback_journal:
            raise ExperimentInvariantError("A/B arms must use distinct feedback journals")

    def for_arm(self, arm: Arm) -> ArmHandle:
        return self.control if arm is Arm.CONTROL else self.treatment


def _memory_logical_id(record: Mapping[str, Any]) -> str:
    for key in ("logical_id", "external_id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    for container_name in ("details", "context", "metadata"):
        container = record.get(container_name)
        if isinstance(container, dict):
            for key in ("eval_id", "logical_id", "external_id"):
                value = container.get(key)
                if isinstance(value, str) and value:
                    return value
    value = record.get("id", record.get("memory_id"))
    if isinstance(value, str) and value:
        return value
    raise ExperimentInvariantError("bootstrap memory record has no stable identifier")


def _memory_actual_id(record: Mapping[str, Any]) -> str:
    value = record.get("id", record.get("memory_id"))
    if not isinstance(value, str) or not value:
        raise ExperimentInvariantError("bootstrap recall record has no memory UUID")
    return value


def _records(component: Mapping[str, Any], *keys: str) -> tuple[dict[str, Any], ...]:
    for key in keys:
        value = component.get(key)
        if isinstance(value, list):
            if not all(isinstance(record, dict) for record in value):
                raise ExperimentInvariantError(f"bootstrap component {key} contains a non-object")
            return tuple(value)
    return ()


def _normalized_memory(record: Mapping[str, Any]) -> dict[str, object]:
    """Logical clone comparison that deliberately ignores row ids and timestamps."""
    normalized: dict[str, object] = {"logical_id": _memory_logical_id(record)}
    for key in (
        "summary",
        "content",
        "context_summary",
        "type",
        "importance",
        "tags",
        "trust_tier",
        "delivery_mode",
    ):
        value = record.get(key)
        if value is not None:
            normalized[key] = value
    return normalized


@dataclass(frozen=True)
class BootstrapEnvelope:
    """Validated server-shaped ``get_agent_bootstrap`` response."""

    raw: dict[str, Any]

    def __post_init__(self) -> None:
        if self.raw.get("status") != "success":
            raise ExperimentInvariantError("bootstrap top-level status is not success")
        agent = self.raw.get("agent")
        context = self.raw.get("context")
        instructions = self.raw.get("instructions")
        components = self.raw.get("components")
        if not isinstance(agent, dict) or not isinstance(agent.get("agent_id"), str):
            raise ExperimentInvariantError("bootstrap envelope has no agent identity")
        if not isinstance(context, dict):
            raise ExperimentInvariantError("bootstrap envelope has no context metadata")
        if not isinstance(instructions, str):
            raise ExperimentInvariantError("bootstrap envelope has no instruction block")
        if not isinstance(components, dict):
            raise ExperimentInvariantError("bootstrap envelope has no component map")
        required = ("pinned", "recall", "upcoming", "state", "policy")
        for name in required:
            component = components.get(name)
            if not isinstance(component, dict):
                raise ExperimentInvariantError(
                    f"bootstrap envelope has no {name} component provenance"
                )
            if component.get("status") not in ("ok", "error", "skipped"):
                raise ExperimentInvariantError(f"bootstrap {name} component has an invalid status")
        recall = components["recall"]
        assert isinstance(recall, dict)
        if recall.get("status") not in ("ok", "error"):
            raise ExperimentInvariantError(
                "bootstrap recall was skipped; every eval task needs a query"
            )
        declared_degraded = self.raw.get("degraded")
        if not isinstance(declared_degraded, bool):
            raise ExperimentInvariantError("bootstrap envelope has no boolean degraded status")
        observed_degraded = any(
            isinstance(components[name], dict) and components[name].get("status") == "error"
            for name in required
        )
        if declared_degraded != observed_degraded:
            raise ExperimentInvariantError(
                "bootstrap degraded status disagrees with component failures"
            )

    @property
    def degraded(self) -> bool:
        return bool(self.raw.get("degraded", False))

    @property
    def agent_id(self) -> str:
        agent = self.raw["agent"]
        assert isinstance(agent, dict)
        return str(agent["agent_id"])

    @property
    def context_id(self) -> str:
        agent = self.raw["agent"]
        assert isinstance(agent, dict)
        binding = agent.get("binding")
        if isinstance(binding, dict) and isinstance(binding.get("context_id"), str):
            return str(binding["context_id"])
        context = self.raw.get("context")
        if isinstance(context, dict) and isinstance(context.get("id"), str):
            return str(context["id"])
        raise ExperimentInvariantError("bootstrap envelope has no resolved context id")

    @property
    def components(self) -> dict[str, dict[str, Any]]:
        raw_components = self.raw["components"]
        assert isinstance(raw_components, dict)
        return {
            str(name): component
            for name, component in raw_components.items()
            if isinstance(component, dict)
        }

    def component_failures(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                name
                for name, component in self.components.items()
                if component.get("status") == "error"
            )
        )

    def recall_records(self) -> tuple[dict[str, Any], ...]:
        return _records(self.components["recall"], "results", "memories")

    def selected_logical_ids(self) -> tuple[str, ...]:
        return tuple(_memory_logical_id(record) for record in self.recall_records())

    def feedback_memory_ids(self) -> tuple[str, ...]:
        return tuple(_memory_actual_id(record) for record in self.recall_records())

    def selection_probabilities(self) -> dict[str, float] | None:
        component = self.components["recall"]
        declared = component.get("selection_probabilities")
        actual_to_logical = {
            _memory_actual_id(record): _memory_logical_id(record)
            for record in self.recall_records()
        }
        if isinstance(declared, dict):
            out: dict[str, float] = {}
            for key, value in declared.items():
                if (
                    not isinstance(key, str)
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                ):
                    return None
                probability = float(value)
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    return None
                out[actual_to_logical.get(key, key)] = probability
            return out
        out = {}
        for record in self.recall_records():
            value = record.get("selection_probability")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            probability = float(value)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                return None
            out[_memory_logical_id(record)] = probability
        return out

    def non_ranking_payload(self) -> dict[str, object]:
        """Canonical bootstrap data that must be identical between cloned arms."""
        context = self.raw.get("context")
        context_payload: dict[str, object] = {}
        if isinstance(context, dict):
            for key in ("description", "summary", "usage_guide", "is_public", "is_locked"):
                if key in context:
                    context_payload[key] = context[key]
        payload: dict[str, object] = {
            "instructions": self.raw.get("instructions"),
            "context": context_payload,
        }
        for name in ("pinned", "upcoming"):
            component = self.components.get(name)
            if component is None:
                payload[name] = None
                continue
            rows = _records(component, "memories", "results")
            payload[name] = {
                "status": component.get("status"),
                "records": sorted(
                    (_normalized_memory(record) for record in rows),
                    key=lambda record: str(record["logical_id"]),
                ),
            }
        for name in ("state", "policy"):
            component = self.components.get(name)
            if component is None:
                payload[name] = None
                continue
            payload[name] = {
                key: value
                for key, value in component.items()
                if key not in {"context_id", "agent_id", "generated_at", "from", "until"}
            }
        return payload

    def non_ranking_fingerprint(self) -> str:
        return _sha256(self.non_ranking_payload())

    def model_context(self) -> str:
        """Render model-visible bootstrap data; health/correlation metadata stays host-side."""
        blocks: list[str] = []
        instructions = self.raw.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            blocks.append(f"Instructions:\n{instructions.strip()}")
        for name, keys in (
            ("pinned", ("memories",)),
            ("recall", ("results", "memories")),
            ("upcoming", ("results",)),
        ):
            component = self.components.get(name, {})
            rows = _records(component, *keys)
            if rows:
                rendered = []
                for record in rows:
                    text = record.get("content") or record.get("summary") or ""
                    rendered.append(f"- [{_memory_logical_id(record)}] {text}")
                blocks.append(f"{name.title()}:\n" + "\n".join(rendered))
        state = self.components.get("state", {}).get("states")
        if isinstance(state, dict) and state:
            blocks.append("Agent state:\n" + _canonical_json(state))
        return "\n\n".join(blocks)


@dataclass(frozen=True)
class OutcomeObservation:
    score: float
    passed: bool
    source: VerifiedSource
    output: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("outcome score must be in [0, 1]")
        if self.source not in ("objective_check", "host_check", "hitl_approval"):
            raise ValueError("feedback requires an independent host-arbitrated source")


@runtime_checkable
class ObjectiveActor(Protocol):
    async def run(
        self,
        task: TaskSpec,
        bootstrap: BootstrapEnvelope,
        *,
        seed: int,
    ) -> OutcomeObservation: ...


@runtime_checkable
class BootstrapExperimentBackend(Protocol):
    """Live/fake adapter contract; every read is ``get_agent_bootstrap`` shaped."""

    async def prepare(
        self, manifest: ExperimentManifest, snapshot: BootstrapSnapshot
    ) -> ArmPair: ...

    async def bootstrap(
        self,
        handle: ArmHandle,
        task: TaskSpec,
        *,
        session_id: str,
        recall_k: int,
    ) -> BootstrapEnvelope: ...

    async def record_verified_feedback(
        self,
        handle: ArmHandle,
        *,
        memory_id: str,
        query: str,
        helpful: bool,
        verdict_source: VerifiedSource,
        note: str,
    ) -> None: ...

    async def close(self, pair: ArmPair) -> None: ...


@dataclass(frozen=True)
class TrialRecord:
    task_id: str
    category: str
    arm: str
    generation: int
    repetition: int
    seed: int
    session_id: str
    score: float
    passed: bool
    verdict_source: str
    degraded: bool
    component_failures: tuple[str, ...]
    selected_memory_ids: tuple[str, ...]
    declared_probabilities: dict[str, float] | None
    feedback_writes: int


@dataclass(frozen=True)
class GenerationMetrics:
    arm: str
    generation: int
    task_success: float
    held_out_gold_success: float | None
    tail_success: float | None
    grounded_diversity: float
    selection_entropy: float
    selection_gini: float
    minimum_declared_probability: float | None
    degraded_rate: float
    component_failures: dict[str, int]
    selection_counts: dict[str, int]


@dataclass(frozen=True)
class PairedEffect:
    generation: int
    task_count: int
    mean_lift: float
    confidence: float
    ci_lower: float
    ci_upper: float
    task_differences: dict[str, float]


@dataclass(frozen=True)
class GateVerdict:
    default_on_allowed: bool
    reasons: tuple[str, ...]
    collapse_detected: bool
    goodhart_detected: bool
    max_tail_regression: float
    positivity_proven: bool


@dataclass(frozen=True)
class ExperimentResult:
    schema_version: int
    experiment_id: str
    snapshot_fingerprint: str
    versions: VersionStamp
    thresholds: GateThresholds
    manifest: ExperimentManifest
    arms: ArmPair
    paired_effect: PairedEffect
    generations: tuple[GenerationMetrics, ...]
    gate: GateVerdict
    trials: tuple[TrialRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _trial_seed(base: int, task_id: str, generation: int, repetition: int) -> int:
    value = f"{base}:{task_id}:{generation}:{repetition}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires values")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _paired_effect(
    records: Sequence[TrialRecord],
    thresholds: GateThresholds,
    *,
    generation: int,
    seed: int,
) -> PairedEffect:
    by_task_arm: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        if record.generation == generation:
            by_task_arm[(record.task_id, record.arm)].append(record.score)
    differences: dict[str, float] = {}
    task_ids = sorted({task_id for task_id, _arm in by_task_arm})
    for task_id in task_ids:
        control = by_task_arm.get((task_id, Arm.CONTROL.value), [])
        treatment = by_task_arm.get((task_id, Arm.TREATMENT.value), [])
        if not control or not treatment:
            raise ExperimentInvariantError(f"primary endpoint is not paired for task {task_id}")
        differences[task_id] = _mean(treatment) - _mean(control)
    if len(differences) < 2:
        raise ExperimentInvariantError("paired confidence interval requires at least two tasks")
    values = list(differences.values())
    rng = random.Random(seed)
    samples = sorted(
        _mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(thresholds.bootstrap_resamples)
    )
    alpha = (1.0 - thresholds.confidence) / 2.0
    return PairedEffect(
        generation=generation,
        task_count=len(values),
        mean_lift=_mean(values),
        confidence=thresholds.confidence,
        ci_lower=_quantile(samples, alpha),
        ci_upper=_quantile(samples, 1.0 - alpha),
        task_differences=differences,
    )


def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
        return 0.0
    value = -sum((count / total) * math.log(count / total) for count in counts if count)
    return value / math.log(len(counts))


def _gini(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
        return 0.0
    ordered = sorted(counts)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2.0 * weighted) / (len(ordered) * total) - (len(ordered) + 1) / len(ordered)


def _generation_metrics(
    records: Sequence[TrialRecord], tasks: Mapping[str, TaskSpec], snapshot: BootstrapSnapshot
) -> tuple[GenerationMetrics, ...]:
    groups: dict[tuple[str, int], list[TrialRecord]] = defaultdict(list)
    for record in records:
        groups[(record.arm, record.generation)].append(record)
    universe = snapshot.memory_ids
    out: list[GenerationMetrics] = []
    for (arm, generation), group in sorted(
        groups.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        counts: Counter[str] = Counter()
        failures: Counter[str] = Counter()
        for record in group:
            counts.update(record.selected_memory_ids)
            failures.update(record.component_failures)
        held_out = [record.score for record in group if tasks[record.task_id].held_out]
        tail = [record.score for record in group if tasks[record.task_id].tail]
        declared_mins: list[float] = []
        probabilities_complete = True
        for record in group:
            probabilities = record.declared_probabilities
            candidate_ids = tasks[record.task_id].candidate_memory_ids
            if probabilities is None or any(
                candidate not in probabilities for candidate in candidate_ids
            ):
                probabilities_complete = False
                break
            declared_mins.append(min(probabilities[candidate] for candidate in candidate_ids))
        vector = [counts.get(memory_id, 0) for memory_id in universe]
        out.append(
            GenerationMetrics(
                arm=arm,
                generation=generation,
                task_success=_mean([record.score for record in group]),
                held_out_gold_success=_mean(held_out) if held_out else None,
                tail_success=_mean(tail) if tail else None,
                grounded_diversity=sum(value > 0 for value in vector) / len(vector),
                selection_entropy=_entropy(vector),
                selection_gini=_gini(vector),
                minimum_declared_probability=(
                    min(declared_mins) if probabilities_complete and declared_mins else None
                ),
                degraded_rate=sum(record.degraded for record in group) / len(group),
                component_failures=dict(sorted(failures.items())),
                selection_counts={key: counts[key] for key in sorted(counts)},
            )
        )
    return tuple(out)


def _gate(
    effect: PairedEffect,
    metrics: Sequence[GenerationMetrics],
    thresholds: GateThresholds,
    *,
    feedback_provenance: str,
) -> GateVerdict:
    reasons: list[str] = []
    if effect.ci_lower <= thresholds.delta_min:
        reasons.append(
            f"primary CI lower bound {effect.ci_lower:.6f} does not clear delta_min "
            f"{thresholds.delta_min:.6f}"
        )
    if feedback_provenance != "host":
        reasons.append("feedback was not server-stamped as host-arbitrated")
    if any(metric.degraded_rate > thresholds.max_degraded_rate for metric in metrics):
        reasons.append("bootstrap degraded rate exceeds the pre-registered maximum")
    by_arm_generation = {(metric.arm, metric.generation): metric for metric in metrics}
    generations = sorted({metric.generation for metric in metrics})
    max_tail_regression = 0.0
    for generation in generations:
        control = by_arm_generation[(Arm.CONTROL.value, generation)]
        treatment = by_arm_generation[(Arm.TREATMENT.value, generation)]
        if control.tail_success is not None and treatment.tail_success is not None:
            max_tail_regression = max(
                max_tail_regression, control.tail_success - treatment.tail_success
            )
    if max_tail_regression > thresholds.max_tail_regression:
        reasons.append("rare-but-correct tail regression exceeds the pre-registered maximum")

    treatment_metrics = [
        by_arm_generation[(Arm.TREATMENT.value, generation)] for generation in generations
    ]
    baseline = treatment_metrics[0]
    collapse = any(
        baseline.selection_entropy - metric.selection_entropy > thresholds.max_entropy_drop
        and metric.task_success <= baseline.task_success + thresholds.utility_flat_tolerance
        for metric in treatment_metrics[1:]
    )
    if collapse:
        reasons.append("grounded-set entropy collapses while utility is flat or falling")

    held_out = [
        metric.held_out_gold_success
        for metric in treatment_metrics
        if metric.held_out_gold_success is not None
    ]
    goodhart = False
    peak = -1.0
    for value in held_out:
        assert value is not None
        peak = max(peak, value)
        if peak - value > thresholds.max_goodhart_decline:
            goodhart = True
    if goodhart:
        reasons.append("held-out gold metric shows a rise-then-fall Goodhart signature")

    positivity = all(
        metric.minimum_declared_probability is not None
        and metric.minimum_declared_probability > 0.0
        for metric in treatment_metrics
    )
    if not positivity:
        reasons.append(
            "strictly-positive per-candidate selection propensity is missing or violated"
        )
    return GateVerdict(
        default_on_allowed=not reasons,
        reasons=tuple(reasons),
        collapse_detected=collapse,
        goodhart_detected=goodhart,
        max_tail_regression=max_tail_regression,
        positivity_proven=positivity,
    )


async def run_experiment(
    manifest: ExperimentManifest,
    snapshot: BootstrapSnapshot,
    tasks: Sequence[TaskSpec],
    backend: BootstrapExperimentBackend,
    actor: ObjectiveActor,
) -> ExperimentResult:
    """Run paired bootstrap arms over successive feedback generations."""
    if manifest.snapshot_fingerprint != snapshot.fingerprint:
        raise ExperimentInvariantError("manifest snapshot fingerprint does not match the corpus")
    validate_corpus(snapshot, tasks)
    pair = await backend.prepare(manifest, snapshot)
    task_map = {task.id: task for task in tasks}
    records: list[TrialRecord] = []
    try:
        pair.validate(manifest)
        for generation in range(manifest.generations):
            generation_records: list[TrialRecord] = []
            pending_feedback: list[
                tuple[
                    int,
                    ArmHandle,
                    BootstrapEnvelope,
                    str,
                    bool,
                    VerifiedSource,
                    str,
                ]
            ] = []
            ordered_tasks = list(tasks)
            random.Random(manifest.seed + generation).shuffle(ordered_tasks)
            for task in ordered_tasks:
                for repetition in range(manifest.repetitions):
                    seed = _trial_seed(manifest.seed, task.id, generation, repetition)
                    envelopes: dict[Arm, BootstrapEnvelope] = {}
                    for arm in (Arm.CONTROL, Arm.TREATMENT):
                        handle = pair.for_arm(arm)
                        session_id = (
                            f"{manifest.experiment_id}.{task.id}.g{generation}.r{repetition}."
                            f"{arm.value[0]}"
                        )
                        envelope = await backend.bootstrap(
                            handle,
                            task,
                            session_id=session_id,
                            recall_k=manifest.recall_k,
                        )
                        if (
                            envelope.agent_id != handle.agent_id
                            or envelope.context_id != handle.context_id
                        ):
                            raise ExperimentInvariantError(
                                f"{arm.value} bootstrap resolved outside its isolated agent/context"
                            )
                        envelopes[arm] = envelope
                    if (
                        envelopes[Arm.CONTROL].non_ranking_fingerprint()
                        != envelopes[Arm.TREATMENT].non_ranking_fingerprint()
                    ):
                        raise ExperimentInvariantError(
                            "bootstrap arms differ outside recall ranking "
                            "(guide/pinned/upcoming/state/policy)"
                        )
                    outcomes: dict[Arm, OutcomeObservation] = {}
                    actor_order = [Arm.CONTROL, Arm.TREATMENT]
                    random.Random(seed).shuffle(actor_order)
                    for arm in actor_order:
                        outcomes[arm] = await actor.run(task, envelopes[arm], seed=seed)
                    for arm in (Arm.CONTROL, Arm.TREATMENT):
                        envelope = envelopes[arm]
                        outcome = outcomes[arm]
                        handle = pair.for_arm(arm)
                        if not envelope.degraded:
                            pending_feedback.append(
                                (
                                    len(generation_records),
                                    handle,
                                    envelope,
                                    task.query,
                                    outcome.passed,
                                    outcome.source,
                                    (
                                        f"{manifest.experiment_id}:{task.id}:g{generation}:"
                                        f"r{repetition}:{outcome.source}"
                                    ),
                                )
                            )
                        generation_records.append(
                            TrialRecord(
                                task_id=task.id,
                                category=task.category,
                                arm=arm.value,
                                generation=generation,
                                repetition=repetition,
                                seed=seed,
                                session_id=(
                                    f"{manifest.experiment_id}.{task.id}.g{generation}."
                                    f"r{repetition}.{arm.value[0]}"
                                ),
                                score=outcome.score,
                                passed=outcome.passed,
                                verdict_source=outcome.source,
                                degraded=envelope.degraded,
                                component_failures=envelope.component_failures(),
                                selected_memory_ids=envelope.selected_logical_ids(),
                                declared_probabilities=envelope.selection_probabilities(),
                                feedback_writes=0,
                            )
                        )
            # A generation is a clean policy snapshot: all tasks are scored before
            # any verified signal can alter the next generation's ranking.
            for (
                record_index,
                handle,
                envelope,
                query,
                helpful,
                verdict_source,
                note,
            ) in pending_feedback:
                writes = 0
                for memory_id in envelope.feedback_memory_ids():
                    await backend.record_verified_feedback(
                        handle,
                        memory_id=memory_id,
                        query=query,
                        helpful=helpful,
                        verdict_source=verdict_source,
                        note=note,
                    )
                    writes += 1
                generation_records[record_index] = replace(
                    generation_records[record_index], feedback_writes=writes
                )
            records.extend(generation_records)
    finally:
        await backend.close(pair)
    effect = _paired_effect(
        records,
        manifest.thresholds,
        generation=manifest.resolved_primary_generation,
        seed=manifest.seed,
    )
    metrics = _generation_metrics(records, task_map, snapshot)
    verdict = _gate(
        effect,
        metrics,
        manifest.thresholds,
        feedback_provenance=manifest.feedback_provenance,
    )
    return ExperimentResult(
        schema_version=2,
        experiment_id=manifest.experiment_id,
        snapshot_fingerprint=snapshot.fingerprint,
        versions=manifest.versions,
        thresholds=manifest.thresholds,
        manifest=manifest,
        arms=pair,
        paired_effect=effect,
        generations=metrics,
        gate=verdict,
        trials=tuple(records),
    )


def validate_corpus(snapshot: BootstrapSnapshot, tasks: Sequence[TaskSpec]) -> None:
    if len(tasks) < 20:
        raise CorpusError("outcome gate requires at least 20 fixed tasks")
    task_ids = [task.id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise CorpusError("task ids must be unique")
    categories = Counter(task.category for task in tasks)
    if len(categories) < 3 or min(categories.values()) < 3:
        raise CorpusError("task set must be stratified across at least three categories")
    if not any(task.tail for task in tasks):
        raise CorpusError("task set needs rare-but-correct tail tasks")
    if not any(task.held_out for task in tasks):
        raise CorpusError("task set needs held-out gold tasks")
    by_id = {memory.logical_id: memory for memory in snapshot.memories}
    gold_ids: set[str] = set()
    for task in tasks:
        missing = set(task.candidate_memory_ids) - by_id.keys()
        if missing:
            raise CorpusError(f"{task.id}: unknown candidate memories: {sorted(missing)}")
        if task.gold_memory_id in gold_ids:
            raise CorpusError(f"gold memory reused across tasks: {task.gold_memory_id}")
        gold_ids.add(task.gold_memory_id)
        gold_memory = by_id[task.gold_memory_id]
        gold_text = f"{gold_memory.summary}\n{gold_memory.content}".casefold()
        absent = [
            value
            for value in task.check.values
            if task.check.kind != "regex" and value.casefold() not in gold_text
        ]
        if absent:
            raise CorpusError(
                f"{task.id}: gold memory does not contain objective evidence: {absent}"
            )


def _load_json_resource(name: str) -> Any:
    resource = files("kagura_agent.eval").joinpath("fixtures", name)
    return json.loads(resource.read_text(encoding="utf-8"))


def load_default_snapshot() -> BootstrapSnapshot:
    raw = _load_json_resource("bootstrap_snapshot.json")
    if not isinstance(raw, dict):
        raise CorpusError("default snapshot root must be an object")
    return BootstrapSnapshot.from_dict(raw)


def load_default_tasks() -> tuple[TaskSpec, ...]:
    raw = _load_json_resource("bootstrap_tasks.json")
    if not isinstance(raw, list) or not all(isinstance(task, dict) for task in raw):
        raise CorpusError("default tasks root must be an object list")
    return tuple(TaskSpec.from_dict(task) for task in raw)
