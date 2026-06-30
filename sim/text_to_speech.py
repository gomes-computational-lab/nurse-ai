from __future__ import annotations

import platform
import shutil
import subprocess


_non_macos_warning_printed = False
_say_failure_warning_printed = False


def speak_text(text: str) -> None:
    global _non_macos_warning_printed
    global _say_failure_warning_printed

    if not text.strip():
        return

    if platform.system() != "Darwin":
        if not _non_macos_warning_printed:
            print("TTS unavailable: this prototype uses macOS `say`. Continuing with text output only.")
            _non_macos_warning_printed = True
        return

    say_path = shutil.which("say")
    if not say_path:
        if not _say_failure_warning_printed:
            print("TTS unavailable: `say` was not found on this system. Continuing with text output only.")
            _say_failure_warning_printed = True
        return

    try:
        subprocess.run([say_path, text], check=True)
    except Exception as exc:
        if not _say_failure_warning_printed:
            print(f"TTS warning: could not speak response with `say` ({exc}). Continuing without audio.")
            _say_failure_warning_printed = True
