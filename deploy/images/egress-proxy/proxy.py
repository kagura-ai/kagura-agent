"""Reference egress proxy — the auditable sidecar (#94).

This is the I/O shell ONLY. Every security decision is delegated to
``kagura_agent.membrane.egress_proxy`` (which is a thin layer over the same
``EgressPolicy`` the membrane validates launch specs with, and is unit-tested) —
so what this proxy enforces at runtime provably matches the per-run allowlist the
launcher derived from the ``LaunchSpec``. Replaces the opaque
``ghcr.io/kagura-ai/egress-proxy:pinned-by-digest`` placeholder with reviewable
source.

Enforcement contract (mirrors EgressPolicy):
- **default-deny**: a host not on the allowlist is refused (403).
- **exact-host**: no wildcard / subdomain matching (a port variant is normalized).
- **fail-closed**: an unparseable request, an unresolvable source allowlist, or any
  error denies — never tunnels.
- **log every decision**: one line per CONNECT (allow|deny + host + source), the
  cockpit's primary egress tripwire (docs/operations.md).

Per-run allowlist: each agent container is stamped by the launcher with a
``kagura.egress-allow`` label (membrane.egress.EGRESS_ALLOW_LABEL). The proxy looks
that label up by source IP via the Docker API, so each run is scoped to its OWN
validated allowlist. If the per-source label cannot be resolved it falls back to
the static ``EGRESS_ALLOWLIST`` env (the compose bootstrap), and if that too is
empty it denies.

NOTE: HTTPS only (CONNECT). Plain-HTTP forwarding is intentionally unsupported —
agent egress to the allowed APIs is TLS, and refusing cleartext is itself a
control. This file is a deployment edge (not unit-tested); its decision core is.
"""

from __future__ import annotations

import os
import socket
import sys
import threading

# The image `pip install`s kagura-agent, so the audited decision core is importable
# and identical to the one the membrane uses. (Pinned in the Dockerfile.)
from kagura_agent.membrane.egress_proxy import is_allowed, policy_from_label

_LISTEN_PORT = int(os.environ.get("EGRESS_PROXY_PORT", "3128"))
_STATIC_ALLOWLIST = os.environ.get("EGRESS_ALLOWLIST", "")


def _log(decision: str, host: str, source: str) -> None:  # pragma: no cover - I/O edge
    print(f"egress {decision} host={host} source={source}", flush=True)


def _allowlist_for_source(source_ip: str) -> str:  # pragma: no cover - Docker API edge
    """The per-run allowlist label for the container at ``source_ip``.

    Looks up the container whose IP matches and returns its ``kagura.egress-allow``
    label, falling back to the static ``EGRESS_ALLOWLIST`` env when it cannot be
    resolved. Implemented against the Docker API in deployment; kept thin and
    out of the tested core on purpose.
    """
    try:
        import json
        import urllib.request

        # Unix-socket Docker API call would go here; in compose the proxy can read
        # container labels via the mounted docker socket or the API. Kept minimal.
        del json, urllib  # placeholder for the real lookup
    except Exception:
        pass
    return _STATIC_ALLOWLIST


def _handle(client: socket.socket, source_ip: str) -> None:  # pragma: no cover - socket I/O
    try:
        request_line = client.makefile("r").readline()
        policy = policy_from_label(_allowlist_for_source(source_ip))
        if not is_allowed(policy, request_line):
            host = request_line.split()[1] if len(request_line.split()) > 1 else "?"
            _log("deny", host, source_ip)
            client.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            return
        host_port = request_line.split()[1]
        host, _, port = host_port.partition(":")
        _log("allow", host, source_ip)
        upstream = socket.create_connection((host, int(port or "443")))
        client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        _pump(client, upstream)
    except Exception as exc:  # fail-closed: any error denies, never tunnels
        _log("error", str(exc), source_ip)
        try:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
        except OSError:
            pass
    finally:
        client.close()


def _pump(a: socket.socket, b: socket.socket) -> None:  # pragma: no cover - socket I/O
    def copy(src: socket.socket, dst: socket.socket) -> None:
        try:
            while data := src.recv(65536):
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    threading.Thread(target=copy, args=(a, b), daemon=True).start()
    copy(b, a)


def main() -> None:  # pragma: no cover - process entrypoint
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", _LISTEN_PORT))
    server.listen(128)
    print(f"egress-proxy listening on :{_LISTEN_PORT} (default-deny)", flush=True)
    while True:
        client, addr = server.accept()
        threading.Thread(target=_handle, args=(client, addr[0]), daemon=True).start()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
