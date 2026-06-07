"""The seam's load-bearing invariant, enforced as a test.

`core/session.py` must never import the Claude Agent SDK (or the Claude brain
module) directly. If this test fails, the brain seam has leaked and a future
Codex brain is no longer a pure addition.
"""

import ast
from pathlib import Path

import kagura_agent.core.session as session_mod

FORBIDDEN_ROOTS = {"anthropic", "claude_agent_sdk"}
FORBIDDEN_MODULES = {"kagura_agent.core.brain.claude", "kagura_agent.core.brain.sdk_engine"}


def _imported_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_session_does_not_import_the_sdk_or_claude_brain() -> None:
    source = Path(session_mod.__file__).read_text()
    imports = _imported_names(source)

    leaked_roots = {name for name in imports if name.split(".")[0] in FORBIDDEN_ROOTS}
    leaked_modules = imports & FORBIDDEN_MODULES

    assert not leaked_roots, f"session.py leaks SDK import(s): {leaked_roots}"
    assert not leaked_modules, f"session.py leaks brain impl import(s): {leaked_modules}"
