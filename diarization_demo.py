"""Compatibility entry point for the modular standalone diarization demo."""

from sim.diarization.cli import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
