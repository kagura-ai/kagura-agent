# Legal posture (self-host v1)

> Legal companion to the canonical design doc (`../README.md`).
>
> ⚠️ This document **flags questions for qualified legal review** — it is not
> legal advice and reaches no binding conclusion. Verify against the _current_
> Anthropic terms before any commercial or public launch.

Two distinct questions, often conflated:

## 1. Subscription-via-CLI automation

Does running a Pro/Max **subscription** through a subprocess-wrapped Claude Code
CLI inside an autonomous self-hosted agent fall within Anthropic's Consumer
Terms / Usage Policy / any Claude Code terms governing automated or programmatic
use? This is **open and must be verified**, not assumed.

**The commercial context sharpens the risk.** The README states this is "part of
the Kagura Memory Cloud commercial offering." Pro/Max are **consumer** products.
Powering a commercial offering with a consumer subscription — or shipping a
product that requires the *customer* to plug in their own Pro/Max — risks
(a) the customer breaching Consumer Terms, and (b) Kagura **inducing** that
breach. Verify specifically: personal/non-commercial-use clauses, any
competing-product restriction, and whether Claude Code permits headless /
programmatic invocation.

**Conservative stance (a legal framing to fix in the design, not just billing):**

- The **subscription path is the operator's own personal use, at their own
  risk — not a capability Kagura provides or supports commercially.**
- The **commercial lane Kagura sells is BYOK API keys only** (Commercial Terms;
  already the design — see README "Auth model"). This distinction matters for
  liability and inducement, not just cost.
- single subscriber = single operator; **never share one subscription across
  users** (the clearest line not to cross).

## 1b. Third-party CLI redistribution (image distribution)

Bundling `gh` (MIT), `awscli` v2, and `gcloud` SDK into a **distributed** Docker
image is redistribution. The CLI sources are permissive (Apache-2.0), but the
v2 / SDK *distributions* carry additional terms that may restrict redistribution,
and the base image (Debian/Ubuntu) bundles GPL components requiring
attribution/source.

→ **Mitigation = ship Dockerfiles, not prebuilt images** (README image section).
The operator builds locally, pulling upstream directly, which shifts
redistribution exposure to operator/upstream and matches the self-host model.
Carry `NOTICE`/attribution for whatever *is* baked.

## 2. Operator self-responsibility & liability

The membrane's threat model puts hijack risk on the operator. That needs an
**operator-facing terms / disclaimer document** stating the accepted-risk scope
**explicitly includes "hijacked → credentials exfiltrated"**, not merely "I
broke my own files." Note disclaimers have limits (gross negligence /
consumer-protection cannot always be waived) and must be **affirmatively
accepted**, not buried, to be enforceable.

## 3. External-chat PII flowing through the agent

`kagura-memory-ai-worker` ingests Slack/Teams chat into memory-cloud, which the
agent reads (`recall`) and may act on, transmit, or derive new memories from.
That chat contains **third parties' personal data** whose authors never consented
to Kagura — so the operator/Kagura takes on processor/controller obligations
(GDPR/CCPA), and the agent **inherits** them by processing and potentially
exfiltrating that data (the legal face of CSO finding C1 / README "Memory
provenance").

- **Erasure must cascade.** A `forget` has to reach derived artifacts —
  embeddings, edges, checkpoints — not just the primary memory.
- **Don't hold a standalone DPA / data-flow doc** — reference memory-cloud's and
  state the inheritance explicitly.

## Action items (pre-launch, held for CLO review)

- [ ] Read current Anthropic Consumer Terms + Usage Policy + Claude Code terms
      re: automation / programmatic use / subscription sharing / commercial use.
- [ ] Fix the **subscription = personal/self-responsibility vs commercial = BYOK**
      framing in product copy and terms (inducement risk).
- [ ] Decide image **distribution model** (Dockerfile vs prebuilt) on the legal
      analysis above; carry `NOTICE` for baked CLIs.
- [ ] Draft the operator self-responsibility terms (affirmatively accepted;
      liability disclaimer incl. the hijack scope above).
- [ ] Specify `forget` **cascade scope** (embeddings/edges/checkpoints) and
      reference memory-cloud's DPA rather than duplicating it.
