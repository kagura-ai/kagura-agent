"""v0.4: failure learning wires memory's prevents-edges to graduation.

A failure both records a `prevents` edge in memory (so a future run recalls "this
broke things") and demotes the category in the graduation engine — the edge
accumulation *is* the trust signal. A verified success advances the curve.
"""

from kagura_agent.mcp.memory_cloud import LocalMemoryClient
from kagura_agent.membrane.graduation import GraduationEngine, GraduationPolicy
from kagura_agent.patterns.failure_learning import FailureLearner


class _Clock:
    def __call__(self) -> float:
        return 1000.0


async def test_failure_records_prevents_edge_and_demotes() -> None:
    memory = LocalMemoryClient()
    engine = GraduationEngine(GraduationPolicy(), clock=_Clock())
    learner = FailureLearner(memory=memory, graduation=engine)

    action_mid = await memory.remember("ran apt install foo", tags=("apt",))
    await learner.failed(
        "apt", action_mid=action_mid, description="apt install foo corrupted the container"
    )

    # prevents edge recorded
    fail_hits = await memory.recall("corrupted", tags=("failure",))
    assert fail_hits
    assert memory.edges_of(fail_hits[0].id) == [(action_mid, "prevents")]
    # and the category is demoted (fail-closed)
    assert engine.should_propose("apt", input_trust="trusted") is False


async def test_verified_success_advances_curve() -> None:
    memory = LocalMemoryClient()
    engine = GraduationEngine(GraduationPolicy(), clock=_Clock())
    learner = FailureLearner(memory=memory, graduation=engine)

    for i in range(5):
        await learner.succeeded("apt", task_id=f"task{i % 3}", verified=True)

    assert engine.should_propose("apt", input_trust="trusted") is True
