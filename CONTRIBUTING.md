# Contributing to kagura-agent

Thanks for your interest in contributing! This document covers the dev setup, the
quality gate every change must pass, and the sign-off we require on commits.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Dev setup

Requires **Python ≥ 3.11**. The package is install-from-clone (see the [README](README.md#quickstart)).

```bash
git clone https://github.com/kagura-ai/kagura-agent && cd kagura-agent
python -m venv .venv && source .venv/bin/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e '.[dev]'                              # ruff + mypy + pytest + coverage tooling
```

To actually *run* the agent (not just the tests) you also need a brain extra
(`.[claude]` or `.[brain]`) and the logins described in the README Quickstart — but the
test suite needs only `.[dev]`.

## The quality gate

CI runs three checks (`.github/workflows/ci.yml`); run them locally before opening a PR:

```bash
ruff check src/ tests/        # lint  — NOTE: this project does NOT use `ruff format`
mypy                          # strict types (config in pyproject)
pytest --cov=src/kagura_agent --cov-report=term   # tests + coverage gate (fail_under = 95)
```

All three must pass, and coverage must stay **≥ 95%**. New behaviour is **test-driven**:
write a failing test first, watch it fail, then make it pass. Every bug fix ships with a
regression test.

To refresh the README coverage badge after a coverage change:

```bash
pytest --cov=src/kagura_agent --cov-report=xml
genbadge coverage -i coverage.xml -o docs/coverage.svg
```

## Architecture guardrail

`core/session.py` must never import a brain SDK directly — it depends on the
`BrainProvider` protocol. `test_seam` enforces this; do not work around it. New brains
and transports are **pure additions behind their protocol**, not core changes.

## Pull requests

- Branch off `main`; keep PRs focused and reviewable.
- Reference the issue you're addressing.
- Make sure the gate is green and CI passes.
- A maintainer review (and, for non-trivial changes, an adversarial code review) is expected before merge.

## Developer Certificate of Origin (DCO)

We require every commit to be **signed off** under the
[Developer Certificate of Origin](https://developercertificate.org/) (DCO). This is a
lightweight statement that you wrote the contribution or otherwise have the right to
submit it under the project's Apache-2.0 license — no CLA, no paperwork.

Add a `Signed-off-by` line to each commit by committing with `-s`:

```bash
git commit -s -m "fix: ..."
```

which appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

(The name/email must match your `git config user.name` / `user.email`.) Forgot to sign
off? Amend the last commit with `git commit --amend -s --no-edit`, or for a series use
`git rebase --signoff`.

### AI-assisted contributions

AI-assisted or AI-generated contributions are welcome, but the **same bar applies**: you
must understand the change, it must pass the quality gate, and your DCO sign-off certifies
that you have the right to submit it under Apache-2.0 — you are responsible for the code you
submit, however it was produced. Disclose substantial AI assistance in the PR description
where it helps reviewers.

### The DCO

```
Developer Certificate of Origin
Version 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## License

By contributing, you agree that your contributions are licensed under the project's
[Apache License 2.0](LICENSE).
