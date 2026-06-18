# Egress proxy — the auditable default-deny + allowlist sidecar (#94).
#
# Distributed as a recipe (this Dockerfile + proxy.py), NOT an opaque prebuilt
# image: the previous `ghcr.io/kagura-ai/egress-proxy:pinned-by-digest` reference
# was unverifiable from this repo. The enforcement logic is `proxy.py`, which
# delegates every decision to the unit-tested
# `kagura_agent.membrane.egress_proxy` (a thin layer over the same EgressPolicy
# the membrane validates launch specs with) — so the proxy provably enforces the
# same policy the launcher derives.
#
# Pin the digest below for reproducibility before deployment; `doctor` flags the
# all-zero placeholder (membrane: egress-proxy not pinned). Same pin discipline as
# Dockerfile.base / Dockerfile.python.
FROM python:3.11-slim@sha256:0000000000000000000000000000000000000000000000000000000000000000

# Install the audited decision core. Pin kagura-agent to the deployed version so
# the proxy and the membrane share the exact EgressPolicy semantics.
RUN pip install --no-cache-dir "kagura-agent"

COPY proxy.py /app/proxy.py
WORKDIR /app

# Non-root: the proxy needs no privileges beyond opening its listen socket and
# dialling allowed upstreams.
RUN useradd --create-home --uid 1000 proxy
USER proxy

EXPOSE 3128
ENTRYPOINT ["python", "/app/proxy.py"]
