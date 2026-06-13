"""v0.2 (A5): container-image invariants — guard tests for the membrane images.

The image philosophy is "tools are baked; credentials and first-party code are
injected at run time". These tests pin that contract so a future edit cannot
quietly bake a secret, COPY app code in, or break the base -> python inheritance
(the same invariants the images CI workflow enforces on every push).
"""

import re
from pathlib import Path

IMAGES = Path(__file__).resolve().parent.parent / "deploy" / "images"

_SECRET_MARKERS = (
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN",
    "CLOUDFLARE_API_TOKEN",
    "BEGIN PRIVATE KEY",
    "PASSWORD",
)


def test_python_image_inherits_from_base() -> None:
    text = (IMAGES / "Dockerfile.python").read_text(encoding="utf-8")
    assert "FROM kagura-agent:base" in text


def test_python_image_bakes_the_kagura_memory_cli() -> None:
    # A6: the agent's memory entry point ships by default (recall/remember/ingest).
    # The CLI is a baked tool; the membrane injects a short-lived, read-scoped
    # access token at run time (the refresh token never enters the container).
    text = (IMAGES / "Dockerfile.python").read_text(encoding="utf-8")
    assert "kagura-memory" in text


def test_base_pins_the_upstream_by_digest() -> None:
    # Reproducibility: base must pin debian by @sha256 digest, not a floating tag.
    # A prefix-only assertion let an all-zeros placeholder digest pass CI while the
    # image was unbuildable (`manifest not found`), so also require a full 64-hex
    # digest that is NOT the all-zeros placeholder.
    text = (IMAGES / "Dockerfile.base").read_text(encoding="utf-8")
    m = re.search(r"FROM debian:bookworm-slim@sha256:([0-9a-f]{64})\b", text)
    assert m, "base must pin debian by a full 64-hex sha256 digest"
    assert m.group(1) != "0" * 64, "digest must be a real digest, not the all-zeros placeholder"


def test_images_bake_no_secrets() -> None:
    for name in ("Dockerfile.base", "Dockerfile.python"):
        upper = (IMAGES / name).read_text(encoding="utf-8").upper()
        for marker in _SECRET_MARKERS:
            assert marker not in upper, f"{name} must not bake {marker}"


def test_images_inject_code_rather_than_copy_it_in() -> None:
    # First-party code is mounted by the membrane at run time, never COPY/ADD'd
    # into the image (a baked-in source tree would escape the per-run mount gate).
    for name in ("Dockerfile.base", "Dockerfile.python"):
        for line in (IMAGES / name).read_text(encoding="utf-8").splitlines():
            instr = line.strip().upper()
            if instr.startswith("#"):
                continue
            assert not instr.startswith("COPY "), f"{name} must not COPY code in"
            assert not instr.startswith("ADD "), f"{name} must not ADD code in"


def test_images_declare_non_root() -> None:
    # Defence in depth alongside the launcher's --user: the base must drop to a
    # non-root user so the image is non-root even if a caller bypasses the flag.
    text = (IMAGES / "Dockerfile.base").read_text(encoding="utf-8")
    assert "USER agent" in text
