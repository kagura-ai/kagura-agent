"""Reference egress proxy — the auditable sidecar (#94).

This is the I/O shell ONLY. Every security DECISION is delegated to
``kagura_agent.membrane.egress_proxy`` (a thin layer over the same
``EgressPolicy`` the membrane validates launch specs with, and unit-tested) — so
what this proxy enforces provably matches the membrane's policy semantics:
default-deny, exact-host, fail-closed, log-every-decision. This replaces the
opaque ``ghcr.io/kagura-ai/egress-proxy:pinned-by-digest`` placeholder with
reviewable source.

Allowlist sourcing — be precise about what is and is not wired here:
- The decision core (`policy_from_label`) ALREADY consumes a per-run
  ``kagura.egress-allow`` label (the launcher stamps it, #92) and is tested.
- This reference shell resolves the allowlist from the static ``EGRESS_ALLOWLIST``
  env (the compose bootstrap). Mapping a *source container* to its per-run label
  needs a Docker-API lookup (by source IP), which requires the proxy to reach the
  Docker API — a deployment integration that is intentionally NOT enabled by
  default. ``_allowlist_for_source`` is that documented seam: drop the per-source
  lookup in there and the per-run least-privilege scoping is live, with zero change
  to the audited decision core. Until then the proxy enforces the static allowlist.

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
_MAX_HEAD = 8192  # a CONNECT request head is tiny; bound the read to refuse a flood


def _log(decision: str, host: str, source: str) -> None:  # pragma: no cover - I/O edge
    print(f"egress {decision} host={host} source={source}", flush=True)


def _allowlist_for_source(source_ip: str) -> str:  # pragma: no cover - integration seam
    """The allowlist label for the container at ``source_ip``.

    Integration seam for per-run scoping: resolve the source container's
    ``kagura.egress-allow`` label (membrane.egress.EGRESS_ALLOW_LABEL) via the
    Docker API here, and the per-run least-privilege the launcher stamps becomes
    live — ``policy_from_label`` already handles it. Not wired by default (it needs
    the proxy to reach the Docker API), so this reference returns the static
    ``EGRESS_ALLOWLIST`` env. ``source_ip`` is the lookup key once wired.
    """
    return _STATIC_ALLOWLIST


def _read_request_head(sock: socket.socket) -> str:  # pragma: no cover - socket I/O
    """Read the HTTP request head (through the blank line) byte-accurately.

    Deliberately NOT ``makefile().readline()``: a buffered reader over-reads past
    the request line into its own buffer, and those bytes would be lost when the
    raw socket is later tunnelled (a real data-framing bug for a TLS MITM proxy).
    Reading exactly up to ``\\r\\n\\r\\n`` leaves the socket buffer intact for the
    tunnel. Bounded so a client cannot stream forever before the blank line."""
    data = b""
    while b"\r\n\r\n" not in data and len(data) < _MAX_HEAD:
        chunk = sock.recv(1)
        if not chunk:
            break
        data += chunk
    return data.decode("latin-1", "replace")


def _split_authority(authority: str) -> tuple[str, int]:  # pragma: no cover - parse
    """Split a CONNECT authority into (host, port), IPv6-bracket aware.

    ``[2001:db8::1]:443`` → ``("2001:db8::1", 443)`` (a plain ``partition(':')``
    would mis-split the address). ``host:443`` → ``("host", 443)``; a missing port
    defaults to 443. Mirrors EgressPolicy._normalize_host's bracket handling so an
    allowed IPv6 host is actually dialable (not just allow-decided then 502'd)."""
    if authority.startswith("["):
        host, _, rest = authority[1:].partition("]")
        return host, int(rest.lstrip(":") or "443")
    host, _, port = authority.partition(":")  # no colon → port "" → default 443
    return host, int(port or "443")


def _handle(client: socket.socket, source_ip: str) -> None:  # pragma: no cover - socket I/O
    try:
        head = _read_request_head(client)
        request_line = head.split("\r\n", 1)[0]
        policy = policy_from_label(_allowlist_for_source(source_ip))
        if not is_allowed(policy, request_line):
            parts = request_line.split()
            _log("deny", parts[1] if len(parts) > 1 else "?", source_ip)
            client.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            return
        host, port = _split_authority(request_line.split()[1])
        _log("allow", host, source_ip)
        upstream = socket.create_connection((host, port))
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
