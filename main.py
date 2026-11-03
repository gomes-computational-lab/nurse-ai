from __future__ import annotations

import sys
from collections.abc import Callable


def _select_app(arguments: list[str]) -> tuple[Callable[[], int | None], list[str]]:
    if "--diarization" in arguments:
        if "--voice" in arguments:
            raise SystemExit("Choose either --diarization or --voice.")
        from diarization_demo import main as app_main

        return app_main, [argument for argument in arguments if argument != "--diarization"]

    if "--voice" in arguments:
        from sim.voice_app import main as app_main

        return app_main, [argument for argument in arguments if argument != "--voice"]

    from sim.app import main as app_main

    return app_main, arguments


def main() -> int | None:
    app_main, arguments = _select_app(sys.argv)
    original_arguments = sys.argv
    sys.argv = arguments
    try:
        return app_main()
    finally:
        sys.argv = original_arguments


if __name__ == "__main__":
    raise SystemExit(main())
