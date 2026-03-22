from __future__ import annotations

from typing import Any

try:
    import cutlass.cute as cute  # type: ignore

    HAVE_CUTE = True
    CUTE_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - depends on local install
    cute = None  # type: ignore
    HAVE_CUTE = False
    CUTE_IMPORT_ERROR = str(exc)


def capability_summary() -> dict[str, Any]:
    return {
        "available": HAVE_CUTE,
        "import_error": CUTE_IMPORT_ERROR,
        "module": "cutlass.cute",
    }
