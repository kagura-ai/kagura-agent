"""Capability graduation — memory-driven capability unlock.

The Phase-1 In/Out list is a *curve*, not a fixed table. Dangerous categories
(DNS writes, `apt install`, …) start locked. As a category accrues verified
successes across distinct tasks with zero failures, the engine surfaces an HITL
*proposal* to unlock it — the human grant is always the final gate, so the
thresholds are tuned low enough that even a low-volume self-host sees proposals.

Safety rails:
- fail-closed, per-category: one failure resets the counters and demotes.
- input-trust gate: a run drawing on untrusted (externally ingested) memory
  never qualifies (CSO C1 memory provenance).
- verified successes only: the signal is exit code / test / approval, never the
  agent's self-report (CSO M1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GraduationPolicy:
    min_successes: int = 5
    min_distinct_tasks: int = 3
    cooldown_seconds: int = 7 * 86400


@dataclass
class _Stats:
    successes: int = 0
    distinct_tasks: set[str] = field(default_factory=set)
    failures: int = 0  # lifetime counter (telemetry / trust score)
    last_proposed_at: float | None = None
    demoted_until: float | None = None  # gate: blocked until this time

    def reset(self) -> None:
        self.successes = 0
        self.distinct_tasks = set()


class GraduationEngine:
    def __init__(self, policy: GraduationPolicy, *, clock: Callable[[], float]) -> None:
        self._policy = policy
        self._clock = clock
        self._by_category: dict[str, _Stats] = {}

    def _stats(self, category: str) -> _Stats:
        return self._by_category.setdefault(category, _Stats())

    def record_success(self, category: str, *, task_id: str, verified: bool) -> None:
        if not verified:
            return  # self-reported successes never count
        stats = self._stats(category)
        stats.successes += 1
        stats.distinct_tasks.add(task_id)

    def record_failure(self, category: str) -> None:
        # fail-closed, per-category: wipe progress AND demote for a cooldown window.
        # Demotion is recoverable (README: "resets the counter") — after the window
        # elapses, re-accruing verified successes re-qualifies the category.
        stats = self._stats(category)
        stats.failures += 1
        stats.reset()
        stats.demoted_until = self._clock() + self._policy.cooldown_seconds

    def mark_proposed(self, category: str) -> None:
        self._stats(category).last_proposed_at = self._clock()

    def should_propose(self, category: str, *, input_trust: str) -> bool:
        if input_trust != "trusted":
            return False  # input-trust gate (CSO C1)
        stats = self._stats(category)
        if stats.demoted_until is not None and self._clock() < stats.demoted_until:
            return False  # fail-closed: within the post-failure demotion window
        if stats.successes < self._policy.min_successes:
            return False
        if len(stats.distinct_tasks) < self._policy.min_distinct_tasks:
            return False
        if stats.last_proposed_at is not None:
            if self._clock() - stats.last_proposed_at < self._policy.cooldown_seconds:
                return False  # within cooldown
        return True
