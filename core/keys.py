"""Random multi-key pool — shuffle + auth-fail drop."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def _is_auth_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "invalid credentials",
        "invalid_auth",
        "unauthorized",
        "401",
        "invalid api key",
        "authentication",
    )
    status = getattr(exc, "status_code", None)
    return status == 401 or any(m in text for m in markers)


class KeyPool:
    """Holds many API keys; picks randomly and can drop bad ones."""

    def __init__(self, keys: Sequence[str]) -> None:
        cleaned = [k.strip() for k in keys if k and k.strip()]
        if not cleaned:
            raise ValueError("KeyPool needs at least one key")
        self._keys = list(dict.fromkeys(cleaned))  # unique, stable

    def pick(self) -> str:
        return random.choice(self._keys)

    def shuffled(self) -> list[str]:
        keys = list(self._keys)
        random.shuffle(keys)
        return keys

    def drop(self, key: str) -> None:
        remaining = [k for k in self._keys if k != key]
        if remaining:
            self._keys = remaining

    def run(self, fn: Callable[[str], T]) -> T:
        """Try keys in random order; drop auth-failed keys and retry."""
        errors: list[BaseException] = []
        for key in self.shuffled():
            try:
                return fn(key)
            except Exception as exc:  # noqa: BLE001 — key rotation boundary
                if _is_auth_error(exc) and len(self._keys) > 1:
                    self.drop(key)
                    errors.append(exc)
                    continue
                raise
        if errors:
            raise RuntimeError(
                f"All API keys failed auth ({len(errors)} attempts)"
            ) from errors[-1]
        raise RuntimeError("KeyPool is empty")

    def __len__(self) -> int:
        return len(self._keys)
