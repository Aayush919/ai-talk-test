"""Dedicated call debug log.

  logs/ai-talk.log          — every session, all events
  logs/call-<session>.log   — one talk, easy to read
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from core.text_clean import clean_speech_text

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
MAIN_LOG = LOG_DIR / "ai-talk.log"
_MAX_BYTES = 8 * 1024 * 1024
_lock = threading.Lock()


def _ensure_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_path() -> Path:
    _ensure_dir()
    return MAIN_LOG


def session_log_path(session_id: str) -> Path:
    _ensure_dir()
    sid = (session_id or "unknown").strip() or "unknown"
    return LOG_DIR / f"call-{sid}.log"


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _safe(text: object) -> str:
    return clean_speech_text(str(text)).replace("\n", " ").replace("\r", " ")


def _rotate_if_needed(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size > _MAX_BYTES:
            bak = path.with_suffix(path.suffix + ".1")
            if bak.exists():
                bak.unlink()
            path.replace(bak)
    except OSError:
        pass


def _fmt_extra(extra: dict[str, Any] | None) -> str:
    if not extra:
        return ""
    bits = []
    for key, val in extra.items():
        if val is None or val == "":
            continue
        bits.append(f"{key}={_safe(val)}")
    return ("  " + "  ".join(bits)) if bits else ""


def write(
    level: str,
    kind: str,
    msg: str = "",
    *,
    session_id: str = "",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Append one line to main + per-session log files."""
    _ensure_dir()
    sid = session_id or "-"
    line = (
        f"{_stamp()}  {level.upper():5}  sid={sid}  {kind:<10}  "
        f"{_safe(msg)}{_fmt_extra(extra)}"
    )

    with _lock:
        _rotate_if_needed(MAIN_LOG)
        with MAIN_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
        if session_id:
            with session_log_path(session_id).open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()

    try:
        print(line)
    except Exception:
        pass
    return MAIN_LOG


def info(
    kind: str,
    msg: str = "",
    *,
    session_id: str = "",
    extra: dict[str, Any] | None = None,
    **_ignored: object,
) -> None:
    write("INFO", kind, msg, session_id=session_id, extra=extra)


def warn(
    kind: str,
    msg: str = "",
    *,
    session_id: str = "",
    extra: dict[str, Any] | None = None,
    **_ignored: object,
) -> None:
    write("WARN", kind, msg, session_id=session_id, extra=extra)


def error(
    kind: str,
    msg: str = "",
    *,
    session_id: str = "",
    extra: dict[str, Any] | None = None,
    **_ignored: object,
) -> None:
    write("ERROR", kind, msg, session_id=session_id, extra=extra)


def lat(
    kind: str,
    msg: str = "",
    *,
    session_id: str = "",
    extra: dict[str, Any] | None = None,
    **_ignored: object,
) -> None:
    write("LAT", kind, msg, session_id=session_id, extra=extra)
