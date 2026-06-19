"""#102 (PR3): the in-container brain entrypoint.

This runs INSIDE the hardened agent container (the image's ENTRYPOINT). It reads
the encoded run input (Task + optional resume) from stdin, builds the brain from
the container's environment, and streams the brain's events as JSON lines on
stdout — the container half of PR1's wire protocol.

Auth (#113): the brain authenticates with BYOK — ``make_brain`` resolves the
``ANTHROPIC_API_KEY`` the membrane injected into the (egress-sealed) container.
The entrypoint itself is auth-agnostic: it just runs whatever ``make_brain(env)``
yields, so the auth model is entirely a property of the injected environment.

``run_brain_entrypoint`` is the pure, testable core; ``main`` is the thin
deployment glue (``# pragma: no cover`` — it touches real stdio + the real SDK).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from kagura_agent.core.brain.base import BrainProvider
from kagura_agent.core.brain.container import stream_brain_events


async def run_brain_entrypoint(
    stdin_bytes: bytes,
    *,
    make_brain: Callable[[Mapping[str, str]], BrainProvider],
    env: Mapping[str, str],
    emit: Callable[[str], None],
) -> None:
    """Build the brain from ``env`` and stream its events for the stdin run input.

    Pure + injectable: ``make_brain`` / ``env`` / ``emit`` are seams so the
    entrypoint is unit-tested without the real SDK or container stdio."""
    brain = make_brain(env)
    await stream_brain_events(stdin_bytes, brain, emit)


def main() -> int:  # pragma: no cover - container entrypoint glue (real stdio + SDK)
    """Container ENTRYPOINT: stdin → brain → stdout JSON lines.

    Wires the real seams — the run input from ``sys.stdin``, the brain from
    ``make_brain(os.environ)`` (which resolves the injected BYOK key), and ``emit``
    to a flushed ``sys.stdout`` (the pure event channel; logs go to stderr)."""
    import asyncio
    import os
    import sys

    from kagura_agent.core.brain.select import make_brain

    payload = sys.stdin.buffer.read()

    # Protect the event channel: events go to the REAL stdout, but bind everything
    # else's stdout (a stray SDK/library banner or print) to stderr, so it can't
    # inject a non-JSON line into the stream the host decodes (which would abort an
    # otherwise-healthy run). Logs already belong on stderr by protocol.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    def emit(line: str) -> None:
        real_stdout.write(line + "\n")
        real_stdout.flush()

    asyncio.run(run_brain_entrypoint(payload, make_brain=make_brain, env=os.environ, emit=emit))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
