"""v0.2 (A5): container-image invariants — guard tests for the membrane images.

The image philosophy is "tools are baked; credentials and first-party code are
injected at run time". These tests pin that contract so a future edit cannot
quietly bake a secret, COPY app code in, or break the base -> python inheritance
(the same invariants the images CI workflow enforces on every push).
"""

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
    text = (IMAGES / "Dockerfile.python").read_text()
    assert "FROM kagura-agent:base" in text


def test_base_pins_the_upstream_by_digest() -> None:
    # Reproducibility: base must pin debian by @sha256 digest, not a floating tag.
    text = (IMAGES / "Dockerfile.base").read_text()
    assert "FROM debian:bookworm-slim@sha256:" in text


def test_images_bake_no_secrets() -> None:
    for name in ("Dockerfile.base", "Dockerfile.python"):
        upper = (IMAGES / name).read_text().upper()
        for marker in _SECRET_MARKERS:
            assert marker not in upper, f"{name} must not bake {marker}"


def test_images_inject_code_rather_than_copy_it_in() -> None:
    # First-party code is mounted by the membrane at run time, never COPY/ADD'd
    # into the image (a baked-in source tree would escape the per-run mount gate).
    for name in ("Dockerfile.base", "Dockerfile.python"):
        for line in (IMAGES / name).read_text().splitlines():
            instr = line.strip().upper()
            if instr.startswith("#"):
                continue
            assert not instr.startswith("COPY "), f"{name} must not COPY code in"
            assert not instr.startswith("ADD "), f"{name} must not ADD code in"


def test_images_declare_non_root() -> None:
    # Defence in depth alongside the launcher's --user: the base must drop to a
    # non-root user so the image is non-root even if a caller bypasses the flag.
    text = (IMAGES / "Dockerfile.base").read_text()
    assert "USER agent" in text
