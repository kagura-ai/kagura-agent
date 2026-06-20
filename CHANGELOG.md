# Changelog

All notable changes to **kagura-agent** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project will follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) from its first tagged release.

This is a pre-1.0 implemented skeleton: milestones v0.1–v0.7 are built and tested, but no
versioned release has been tagged yet, so everything to date lives under _Unreleased_. The
per-milestone module map is the
[README implementation-status table](README.md#implementation-status-v01v07-skeleton).

## [Unreleased]

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

[Unreleased]: https://github.com/kagura-ai/kagura-agent/commits/main
