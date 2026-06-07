"""thread ⇄ session table.

v0.1 only needed "which threads have a session". v0.3 stores a full record per
thread (container id, image, granted capabilities, status) and reconciles
against the live containers after a cockpit restart — a session whose container
vanished is marked dead so it is not mistaken for resumable.
"""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass, field


@dataclass
class SessionRecord:
    thread_id: str
    container_id: str | None = None
    image: str | None = None
    granted_caps: frozenset[str] = field(default_factory=frozenset)
    status: str = "running"  # running | dead | closed


class SessionRegistry:
    def __init__(self) -> None:
        self._records: dict[str, SessionRecord] = {}

    def add(
        self,
        thread_id: str,
        *,
        container_id: str | None = None,
        image: str | None = None,
        granted_caps: frozenset[str] = frozenset(),
    ) -> None:
        self._records[thread_id] = SessionRecord(
            thread_id=thread_id,
            container_id=container_id,
            image=image,
            granted_caps=granted_caps,
        )

    def has(self, thread_id: str) -> bool:
        return thread_id in self._records

    def get(self, thread_id: str) -> SessionRecord | None:
        return self._records.get(thread_id)

    def sessions(self) -> Set[str]:
        return frozenset(self._records)

    def close(self, thread_id: str) -> None:
        rec = self._records.get(thread_id)
        if rec is not None:
            rec.status = "closed"

    def reconcile(self, live_container_ids: Set[str]) -> None:
        for rec in self._records.values():
            if rec.container_id is not None and rec.container_id not in live_container_ids:
                rec.status = "dead"
