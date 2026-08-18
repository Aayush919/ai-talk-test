"""Merge Deepgram finals in one breath. Keep both thoughts; do not drop the first."""

from __future__ import annotations

from core.coach_service import _norm


def merge_pending_stt(prev: str, nxt: str) -> str:
    """Revise if nxt is the same utterance; otherwise keep both lines."""
    left = (prev or "").strip()
    right = (nxt or "").strip()
    if not left:
        return right
    if not right:
        return left
    a, b = _norm(left), _norm(right)
    if not a:
        return right
    if not b:
        return left
    if b.startswith(a) or a.startswith(b):
        return left if len(a) >= len(b) else right
    aw, bw = a.split(), b.split()
    overlap = set(aw[-5:]) & set(bw[:6])
    if len(overlap) >= 2:
        if a in b:
            return right
        if b in a:
            return left
        return f"{left} {right}".strip()
    if len(bw) <= 2 and len(aw) >= 4:
        return left
    if len(aw) <= 2 and len(bw) >= 4:
        return right
    return f"{left} {right}".strip()
