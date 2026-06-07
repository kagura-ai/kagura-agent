"""v0.4: capability graduation — a curve, not a fixed allow-list.

A category unlocks only after enough *verified* successes across distinct tasks,
zero failures since the last reset, and outside the cooldown. Crossing the bar
merely surfaces an HITL *proposal* — the human is still the final gate (no auto
grant). Two hard safety rails:
- **fail-closed**: one failure resets counters and demotes the category.
- **input-trust gate**: a run whose inputs include untrusted (externally
  ingested) memory never qualifies, regardless of stats (CSO C1).
- success signals must be **verified** (exit code / test / approval), never
  self-reported (CSO M1).
"""

from kagura_agent.membrane.graduation import GraduationEngine, GraduationPolicy

DAY = 86400


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _qualify(engine: GraduationEngine, category: str) -> None:
    # 5 verified successes across 3 distinct tasks
    for i in range(5):
        engine.record_success(category, task_id=f"task{i % 3}", verified=True)


def test_below_threshold_does_not_propose() -> None:
    engine = GraduationEngine(GraduationPolicy(), clock=Clock())
    engine.record_success("apt", task_id="t0", verified=True)
    assert engine.should_propose("apt", input_trust="trusted") is False


def test_meeting_threshold_proposes() -> None:
    engine = GraduationEngine(GraduationPolicy(), clock=Clock())
    _qualify(engine, "apt")
    assert engine.should_propose("apt", input_trust="trusted") is True


def test_unverified_successes_do_not_count() -> None:
    engine = GraduationEngine(GraduationPolicy(), clock=Clock())
    for i in range(5):
        engine.record_success("apt", task_id=f"task{i}", verified=False)
    assert engine.should_propose("apt", input_trust="trusted") is False


def test_single_failure_resets_and_blocks() -> None:
    engine = GraduationEngine(GraduationPolicy(), clock=Clock())
    _qualify(engine, "apt")
    engine.record_failure("apt")
    assert engine.should_propose("apt", input_trust="trusted") is False


def test_failure_is_per_category() -> None:
    engine = GraduationEngine(GraduationPolicy(), clock=Clock())
    _qualify(engine, "apt")
    _qualify(engine, "dns")
    engine.record_failure("dns")
    assert engine.should_propose("apt", input_trust="trusted") is True
    assert engine.should_propose("dns", input_trust="trusted") is False


def test_untrusted_input_never_proposes() -> None:
    engine = GraduationEngine(GraduationPolicy(), clock=Clock())
    _qualify(engine, "apt")
    assert engine.should_propose("apt", input_trust="external") is False


def test_cooldown_blocks_reproposal_then_clears() -> None:
    clock = Clock()
    engine = GraduationEngine(GraduationPolicy(cooldown_seconds=7 * DAY), clock=clock)
    _qualify(engine, "apt")

    assert engine.should_propose("apt", input_trust="trusted") is True
    engine.mark_proposed("apt")

    # within cooldown: blocked
    assert engine.should_propose("apt", input_trust="trusted") is False
    # after cooldown: clears
    clock.t += 7 * DAY + 1
    assert engine.should_propose("apt", input_trust="trusted") is True
