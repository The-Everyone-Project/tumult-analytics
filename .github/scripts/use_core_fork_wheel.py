#!/usr/bin/env python3
"""Repoint the ``tmlt.core`` uv source from the dev-time path to the fork wheel.

``pyproject.toml`` on this fork resolves ``tmlt.core`` from a sibling checkout of
The-Everyone-Project/tumult-core so that the two forks can be co-developed. That
path does not exist in CI (or in any consumer's environment), so this script
rewrites the ``[tool.uv.sources]`` entry in place to the published fork wheel and
re-locks.

It is deliberately a separate, idempotent script rather than an inline ``run:``
block so that the same command can be used to reproduce a CI environment
locally. Run it from the repository root; it edits ``pyproject.toml`` and
``uv.lock`` in the working tree and is not meant to be committed back.
"""

import re
import subprocess
import sys
from pathlib import Path

WHEEL_BASE = (
    "https://github.com/The-Everyone-Project/tumult-core/releases/download/"
    "0.19.1-ep-pandas-1/tmlt_core-0.19.1+ep.pandas.1-py3-none-"
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
    """Rewrite the source and re-lock."""
    pyproject = Path("pyproject.toml")
    if not pyproject.is_file():
        print("error: run this from the repository root", file=sys.stderr)
        return 1

    text = pyproject.read_text()
    if WHEEL_BASE in text:
        print("tmlt.core already points at the fork wheel; nothing to do")
        return 0

    text, count = PATH_SOURCE.subn(f'"tmlt.core" = {WHEEL_SOURCES}', text)
    if count != 1:
        print(
            "error: expected exactly one editable-path tmlt.core source in "
            f"pyproject.toml, found {count}",
            file=sys.stderr,
        )
        return 1

    pyproject.write_text(text)
    print("repointed tmlt.core at the fork wheel; re-locking")
    # The committed lock refers to the path source, so it must be regenerated.
    # `uv lock --check` is intentionally *not* run after this point.
    return subprocess.call(["uv", "lock"])


if __name__ == "__main__":
    raise SystemExit(main())
