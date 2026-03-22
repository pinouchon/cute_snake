from __future__ import annotations

from typing import Any

from snake.implementations import implementation1, implementation2, implementation3, implementation4

IMPLEMENTATIONS = {
    "implementation1": implementation1,
    "implementation2": implementation2,
    "implementation3": implementation3,
    "implementation4": implementation4,
}


def get_implementation_module(config: dict[str, Any]):
    name = str(config.get("implementation", "implementation1"))
    try:
        return IMPLEMENTATIONS[name]
    except KeyError as exc:
        available = ", ".join(sorted(IMPLEMENTATIONS))
        raise ValueError(f"Unknown implementation {name!r}. Available: {available}") from exc
