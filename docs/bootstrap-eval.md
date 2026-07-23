# Bootstrap outcome A/B gate

This is the executable gate for [issue #188](https://github.com/kagura-ai/kagura-agent/issues/188).
It evaluates feedback-influenced ranking on the production `get_agent_bootstrap`
path. There is deliberately no legacy `recall()` arm.

## What the runner proves

- Control and treatment use distinct agents, contexts, and feedback journals.
- Both context exports exactly match the committed logical snapshot.
- Every search-config field matches except `reinforce_enabled`.
- Context guide, pinned, upcoming, state, and policy components match for every
  paired trial; only recall ordering may differ.
- Both arms receive the same independently checked candidate labels. For each
  paired trial, the host takes the union of registered task candidates recalled
  by either arm, labels the fixed gold candidate helpful and the fixed decoy
  unhelpful, then writes that identical stream to both journals. Arm-specific
  task success never becomes a training-signal difference, unrelated recalled
  memories are never rated, and a degraded pair is never reinforced.
- Every generation is evaluated as a frozen batch; verified feedback is applied
  only after all tasks in that generation finish, so task order cannot change
  the measured policy within a generation.
- The primary result is a task-level paired effect and confidence interval at a
  pre-registered generation, not a run-level pseudo-replicated estimate.
- Every generation reports task success, held-out success, rare-tail success,
  grounded diversity, normalized entropy, Gini, declared minimum selection
  propensity, degraded rate, and component failures.

The default-ON gate fails when the confidence-interval lower bound does not clear
`delta_min`, feedback is not server-stamped `host`, a partial bootstrap exceeds
the degradation allowance, tail success regresses, diversity collapses without
utility, held-out success has a rise-then-fall signature, or strictly positive
candidate propensity is absent/unproven.

## Fixed corpus

The package contains 30 tasks stratified across code, operations, security,
product, and research, plus a 35-memory snapshot. Ten tasks form the rare tail
and five are held out for the long-horizon gold monitor. Objective answers are
stored only in host-side checks and the snapshot; they are not sent in the actor
payload. Corpus validation rejects prompt leakage, missing gold memories,
duplicate gold use, missing strata, and an absent tail/held-out population.
The fixed memories use answer-bearing L1 summaries because the production
bootstrap exposes summary/context-summary grounding, not L3 memory content.

When provisioning each disposable context, copy every memory from
`src/kagura_agent/eval/fixtures/bootstrap_snapshot.json` and preserve its
`logical_id` as `context.eval_id` (or `details.eval_id`). The export endpoint does
not preserve an `external_id`, so this logical marker is how the runner compares
cloned rows whose database UUIDs differ.

## Live prerequisites

1. Create two fresh, disposable contexts from the same snapshot. Retrieval
   feedback is append-only, so do not reuse contexts from an earlier run.
2. Register and bind a distinct agent to each context.
3. Give both contexts identical search settings and host arbitration posture.
   The runner temporarily sets `reinforce_enabled=false` for control and `true`
   for treatment, then restores both settings in `finally`.
4. Provide the operator-only host-feedback endpoint in `host_feedback_path`
   (`/api/v1/contexts/{context_id}/host-feedback` for memory-cloud #1305).
   The current public memory-cloud feedback route stamps provenance as `agent`;
   it may be used with `feedback_mode=public` and
   `feedback_provenance=agent` for diagnostics, but such a run is structurally
   blocked from authorizing default-ON. The upstream transport dependency is
   [memory-cloud#1305](https://github.com/kagura-ai/memory-cloud/issues/1305).
   Truthful per-candidate propensity/exploration-floor evidence is provided by
   [memory-cloud#1306](https://github.com/kagura-ai/memory-cloud/issues/1306);
   missing or incomplete evidence blocks the gate. The registered
   `exploration_floor` and `candidate_pool_k` are sent with the same per-trial
   deterministic seed to both arms; the runner verifies the response policy
   stamps and permits only `reinforce_enabled` to differ.
5. Use an actor adapter that reads one JSON object from stdin and writes only the
   answer text to stdout. Its input contains `task`, `bootstrap_context`, and a
   deterministic `seed`; it never contains the gold check. The runner strips
   `KAGURA_API_KEY`, `KAGURA_MCP_URL`, agent identity, and memory-context variables
   from the child environment. A committed reference actor drives OpenAI's Codex
   CLI on ChatGPT-subscription auth (`codex login`; needs the `brain` extra):
   set `actor.command` to `["<venv>/bin/python", "-m", "kagura_agent.eval.actor_codex"]`,
   optionally pinning `KAGURA_EVAL_ACTOR_MODEL` / `KAGURA_EVAL_ACTOR_TIMEOUT` in the
   host env — stamp the pinned model in `versions.actor_model`.

### Surviving a multi-hour run

A full 5-generation × 3-repetition gate is ~900 serial actor turns behind ~3600
REST calls — hours of wall-clock. Two `memory_cloud` knobs keep a recoverable
fault from discarding the whole run (both default off/zero, so a short run is
unchanged):

- `"token_refresh": true` — source the owner bearer from the `kagura` CLI's
  OAuth session instead of a fixed `KAGURA_API_KEY`. The static access token
  lives only ~1h and the runner reads it once, so a long run would 401 mid-flight;
  with this set, the runner rotates the token (`kagura auth refresh`) proactively
  and again on any 401, then retries that request once. Use this on a single-owner
  workspace where the owner OAuth token is the only credential the operator-only
  host-feedback endpoint accepts. `KAGURA_API_KEY` may be left unset.
- `"retries": N` (+ optional `"retry_backoff_s"`) — bounded retry on *transient*
  REST faults (dropped connection / 5xx). A 4xx is a request bug and is never
  retried. `"actor": {"retries": N}` does the same for the actor subprocess (the
  most failure-prone surface); the codex actor is an idempotent one-shot, so a
  retry is safe.

Copy [bootstrap-eval.example.json](bootstrap-eval.example.json), freeze every
version and threshold, then run. On a single-owner workspace prefer the refreshing
provider (no `KAGURA_API_KEY` needed); otherwise export the API key only in the
trusted host process:

```bash
# refreshing owner bearer (token_refresh: true in the config) — nothing to export
kagura-agent-bootstrap-eval \
  --config bootstrap-eval.json \
  --output bootstrap-eval-result.json

# …or a fixed key for a short run / a long-lived member key:
export KAGURA_API_KEY='...'
kagura-agent-bootstrap-eval --config bootstrap-eval.json --output result.json
```

Exit status is `0` only when default-ON is allowed, `1` for a scientifically
valid FAIL, `2` for bad configuration, and `3` for invalid/incomplete evidence.
Result JSON includes every trial, the complete registered manifest, and both
agent/context bindings so a proposal can audit and reproduce the aggregate.
