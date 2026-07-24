# Documentation

Use this page as the entry point for kagura-agent documentation. The repository
root [README](../README.md) stays focused on orientation and the shortest path to
a working command.

## Start here

- [Getting started](getting-started.md) — prerequisites, installation,
  authentication, CLI usage, cloud bootstrap, and troubleshooting.
- [Design](design.md) — project boundaries, architecture, provider seams,
  capabilities, security membrane, and cockpit internals.
- [Project status and scope](project-status.md) — current implementation status,
  remaining production work, repository layout, and milestone coverage.

## Build and operate

- [Extending the agent](extending.md) — add custom MCP tools, egress rules, and
  leased cloud credentials without baking secrets into images.
- [Operations](operations.md) — incident response, hijack containment, and
  credential rotation.
- [Bootstrap ranking evaluation](bootstrap-eval.md) — fixed-corpus evaluation,
  confidence gates, and rollout criteria. The example configuration is
  [`bootstrap-eval.example.json`](bootstrap-eval.example.json).

## Policy and project information

- [Legal notes](legal.md) — open terms-of-service and operator-responsibility
  questions.
- [Contributing](../CONTRIBUTING.md) — local development and contribution process.
- [Security policy](../SECURITY.md) — vulnerability reporting and supported
  versions.
- [Changelog](../CHANGELOG.md) — release history.
