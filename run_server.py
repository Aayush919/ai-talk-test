"""Keep API running (no auto-reload). Seed runs once-only in app lifespan."""

from __future__ import annotations

import socket

import uvicorn


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    if not _port_free(host, port):
        print(f"[ai-talk] Port {port} busy.")
        print("Close old server first (Ctrl+C in that terminal), OR run:")
        print("  netstat -ano | findstr :8000")
        print("  taskkill /PID <pid> /F")
        raise SystemExit(1)

    print(f"[ai-talk] http://{host}:{port}")
    print("[ai-talk] reload=OFF — stays up until Ctrl+C")
    print("[ai-talk] seed=once-only (already done => skip)")
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
