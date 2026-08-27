#!/usr/bin/env python3
"""Repoint the ``tmlt.core`` uv source from the dev-time path to the fork wheel.

``pyproject.toml`` on this fork resolves ``tmlt.core`` from a sibling checkout of
The-Everyone-Project/tumult-core so that the two forks can be co-developed. That
path does not exist in CI (or in any consumer's environment), so this script
rewrites the ``[tool.uv.sources]`` entry in place to the published fork wheel and
re-locks.

In CI this must run *before* the runner-setup action, whose own ``uv sync`` would
otherwise be the first thing to resolve the path source and fail on it. That is
earlier than ``uv`` exists on the runner, so ``--no-lock`` skips the re-lock and
leaves it to the ``uv sync`` that follows; the rewrite itself is plain text
editing and needs nothing but the interpreter.

It is deliberately a separate, idempotent script rather than an inline ``run:``
block so that the same command can be used to reproduce a CI environment
locally. Run it from the repository root; it edits ``pyproject.toml`` and
``uv.lock`` in the working tree and is not meant to be committed back.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

WHEEL_BASE = (
    "https://github.com/The-Everyone-Project/tumult-core/releases/download/"
    "0.19.1-ep-backend-3/tmlt_core-0.19.1+ep.backend.3-py3-none-"
)

# The fork wheels are tagged py3-none-<platform>: pure Python, but with
# platform-specific vendored Arb/FLINT/GMP/MPFR libraries. One URL per platform
# covers every supported Python version.
WHEEL_SOURCES = f"""[
  {{ url = "{WHEEL_BASE}macosx_11_0_arm64.whl", marker = "sys_platform == 'darwin' and platform_machine == 'arm64'" }},
  {{ url = "{WHEEL_BASE}macosx_11_0_x86_64.whl", marker = "sys_platform == 'darwin' and platform_machine == 'x86_64'" }},
  {{ url = "{WHEEL_BASE}manylinux_2_17_x86_64.manylinux2014_x86_64.whl", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" }},
]"""

PATH_SOURCE = re.compile(
    r'^"tmlt\.core" = \{ path = "\.\./tumult-core", editable = true \}$', re.MULTILINE
)


def main() -> int:
    """Rewrite the source and, unless ``--no-lock`` is given, re-lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="rewrite pyproject.toml only; do not invoke `uv lock`. For use "
        "before uv is installed -- the next `uv sync` re-resolves anyway.",
    )
    args = parser.parse_args()

    pyproject = Path("pyproject.toml")
    if not pyproject.is_file():
        print("error: run this from the repository root", file=sys.stderr)
        return 1

    text = pyproject.read_text()
    text, count = PATH_SOURCE.subn(f'"tmlt.core" = {WHEEL_SOURCES}', text)
    if count != 1:
        # Idempotence is decided by the rewritten source block, not by the URL:
        # the comment above the dev-time source quotes one of these wheel URLs
        # as its example, so a `WHEEL_BASE in text` check matches a file that
        # has never been rewritten and silently leaves the path source in place.
        if count == 0 and WHEEL_SOURCES in text:
            print("tmlt.core already points at the fork wheel; nothing to do")
            return 0
        print(
            "error: expected exactly one editable-path tmlt.core source in "
            f"pyproject.toml, found {count}",
            file=sys.stderr,
        )
        return 1

    pyproject.write_text(text)
    # The committed lock refers to the path source, so it must be regenerated.
    # `uv lock --check` is intentionally *not* run after this point.
    if args.no_lock:
        print("repointed tmlt.core at the fork wheel; leaving the lock stale")
        return 0
    print("repointed tmlt.core at the fork wheel; re-locking")
    return subprocess.call(["uv", "lock"])


if __name__ == "__main__":
    raise SystemExit(main())
