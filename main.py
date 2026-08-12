"""CLI entry retired — use the realtime web coach instead."""

from __future__ import annotations


def main() -> None:
    print("AI Talk is realtime-only now.")
    print("Start the live coach with:")
    print("  .\\.venv\\Scripts\\python run_server.py")
    print("Then open http://127.0.0.1:8000")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
