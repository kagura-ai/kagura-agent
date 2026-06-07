"""Local CLI adapter — the v1 cut. Slack/Discord come later, same protocol.

Driven by an injected `inbox` of events (so it is testable and scriptable);
captures replies in `sent`; answers HITL `ask`s from a preset `answers` queue
(falling back to interactive stdin when none remain).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from kagura_agent.cockpit.transports.base import Event


class CliTransport:
    def __init__(
        self,
        inbox: list[Event] | None = None,
        answers: list[str] | None = None,
    ) -> None:
        self._inbox = list(inbox or [])
        self._answers = list(answers or [])
        self.sent: list[tuple[str, str]] = []

    async def listen(self) -> AsyncIterator[Event]:
        for event in self._inbox:
            yield event

    async def send(self, thread_id: str, text: str) -> None:
        self.sent.append((thread_id, text))

    async def ask(self, thread_id: str, question: str, options: list[str]) -> str:
        if self._answers:
            return self._answers.pop(0)
        # interactive fallback for real local use
        prompt = f"[{thread_id}] {question} {options}: "  # pragma: no cover
        return input(prompt).strip()  # pragma: no cover
