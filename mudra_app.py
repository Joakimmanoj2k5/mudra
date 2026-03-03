"""Backward-compatible launcher for MUDRA.

This keeps the original entrypoint name while running the modular v2 app.
"""

from ui.app import main


if __name__ == "__main__":
    raise SystemExit(main())
