# Changelog

All notable changes to **kagura-agent** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The per-milestone module map is
the [README implementation-status table](README.md#implementation-status-v01v07-skeleton).

## [Unreleased]

## [0.6.0] - 2026-06-21

The forge-resistant self-improve loop (milestone v0.8). A host-arbitrated verified outcome
reinforces the recall ranking of the trusted memories that grounded a run. Every piece ships
**default-OFF**, so existing runs are byte-for-byte unchanged; flipping the re-rank default-ON
is gated on the [#166](https://github.com/kagura-ai/kagura-agent/issues/166) outcome eval. The
loop only persists across runs with a persistent backend (`KAGURA_AGENT_MEMORY_DB`).

### Added

- **Forge-resistant verified outcome** — `VerifiedOutcome`, a frozen value object whose
  `verified=True` is reachable only from an independent host signal (a check's exit code or
  HITL approval), never the agent's self-report (#170).
- **Grounding provenance with trust tiers** — `ProvenanceLog` captures the trust tier of each
  memory used to ground a run, so the input-trust gate derives from real tiers, not an
  ids-only vacuous check (#171).
- **Host-side MEASURE producer** — `measure_outcome(...)` turns a run plus an independent
  verdict (exit code / approval) into a `VerifiedOutcome` (#172).
- **`OutcomeReinforcer`** — verified outcomes reinforce the recall ranking of the trusted
  memories that grounded them; unverified runs reinforce nothing (#174, #175).
- **`FeedbackSink` protocol + `verify_and_reinforce`** — the config-armed loop arm reading
  `KAGURA_AGENT_VERIFY_CHECK` / `KAGURA_AGENT_VERIFY_CATEGORY` (#176), wired into
  `kagura-agent run` (#177).
- **Bounded recall re-rank (default-OFF)** — `KAGURA_AGENT_RECALL_RERANK` surfaces
  verified-useful trusted memories first, clamped to ±`RERANK_BOUND` with a stable cold-start
  floor; `LocalMemoryClient` and `SqliteMemoryClient` re-rank identically (#178, #179).
- **Exploration floor** — `KAGURA_AGENT_RECALL_EXPLORE` adds a strictly-nonzero, seeded
  exploration probability so feedback never permanently buries a memory (the Δ4 positivity
  floor); non-finite values are rejected to off (#180).

### Changed

- **Fail-closed operator gate** — cockpit gate callers accept `require_operator`; when set,
  only the originating operator's click authorizes, and an absent operator id denies (#173).
- README storefront: removed stale "private repository" claims now that the repo is public on
  PyPI, and aligned `keywords` with the repo topics (#181). Added a GitHub social-preview card
  (#182).

### Security

- The self-improve loop cannot be steered by a hijacked agent: verdicts come only from an
  independent host signal, and `record_feedback` / promotion live off the agent's memory
  surface — a forged "done" cannot reinforce anything (#170, #173, #174).

## [0.5.1] - 2026-06-21

### Fixed

- **Headless `run` / `repl` can write files.** The Agent SDK permission mode is now
  configurable via `KAGURA_AGENT_PERMISSION_MODE`; operator-typed `run` / `repl` default to
  `acceptEdits` — 0.5.0 used `default`, which dead-ended every mutating tool in a headless
  run — while `serve` keeps the safe `default` (#162).

### Changed

- README installs from PyPI (`pip install 'kagura-agent[claude]'`) now that the package is
  published (#163).
- CI: bump `actions/upload-artifact` and `actions/download-artifact` (#160, #161).

## [0.5.0] - 2026-06-20

First public release — Apache-2.0, on PyPI. The implemented skeleton of milestones v0.1–v0.7.

### Added

- **v0.1 walking skeleton** — the brain seam (`BrainProvider`), `ClaudeBrain`, the
  memory-reachability startup gate (CLI-primary, brain-independent), per-provider auth, the
  CLI transport, a structural intent router, session + checkpoint, and cockpit wiring.
- **v0.2 security membrane** — mount guards (no `docker.sock` / host FS), baked container
  hardening, default-deny egress, `CredentialBroker`/`Lease`, and the lease ledger + sweeper.
- **v0.3 cockpit + HITL** — fail-closed HITL approval with a graduation trail, the session
  registry with restart reconcile, and the status / kill intents.
- **v0.4 capability graduation** — the per-category trust curve (verified successes,
  fail-closed, cooldown), the input-trust gate, and `prevents`-edge failure learning.
- **v0.5 transports** — Slack (Bolt, Socket Mode) and Discord normalizers as pure additions
  behind the shared `Transport` protocol.
- **v0.6 credential config** — secret references (env / OS-keychain `*_keyring`), the
  provider registry + validator, and the `GrantedBroker` default-deny chokepoint.
- **v0.7 run path + doctor** — grants enforced end-to-end on `run`, suffix-agnostic secret
  resolution, doctor secret-backend awareness, and the `serve` cockpit loop.
- **Brain-in-container** (#102) — run the brain inside the hardened, egress-sealed container
  over JSON-lines IPC, with a BYOK launch spec and `serve --container`.
- **kagura-brain backend** (#134) — a second brain behind the same protocol, selected by
  `KAGURA_AGENT_BRAIN=kagura-brain` (claude / codex, local + cloud ollama).
- **CLI** — `run --prompt-file PATH` and `run -` to read the task body from a file or stdin
  (#142); `serve` now fails closed with a clean install hint when a transport extra is
  missing, instead of a raw traceback (#146).
- **Docs & OSS** — a top-of-file Quickstart (#144, #145); Apache-2.0 relicensing with badges
  and `NOTICE` (#95, #150); and the community-health files — `CONTRIBUTING.md` (DCO),
  `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue/PR templates, and Dependabot (#97).

[Unreleased]: https://github.com/kagura-ai/kagura-agent/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/kagura-ai/kagura-agent/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/kagura-ai/kagura-agent/releases/tag/v0.5.0
