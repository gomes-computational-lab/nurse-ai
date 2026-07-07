from __future__ import annotations

import platform
import shutil
import subprocess
import threading


_non_macos_warning_printed = False
_say_failure_warning_printed = False
_speech_lock = threading.Lock()
_active_process: subprocess.Popen[str] | None = None


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
        process = subprocess.Popen([say_path, text])
    except Exception as exc:
        if not _say_failure_warning_printed:
            print(f"TTS warning: could not speak response with `say` ({exc}). Continuing without audio.")
            _say_failure_warning_printed = True
        return

    with _speech_lock:
        global _active_process
        previous = _active_process
        _active_process = process

    if previous is not None and previous.poll() is None:
        previous.terminate()

    threading.Thread(target=_monitor_speech_process, args=(process,), daemon=True).start()


def stop_speaking() -> None:
    with _speech_lock:
        process = _active_process

    if process is not None and process.poll() is None:
        process.terminate()


def _monitor_speech_process(process: subprocess.Popen[str]) -> None:
    global _active_process
    global _say_failure_warning_printed

    try:
        return_code = process.wait()
    except Exception as exc:
        if not _say_failure_warning_printed:
            print(f"TTS warning: could not monitor speech playback ({exc}). Continuing without audio.")
            _say_failure_warning_printed = True
        return

    with _speech_lock:
        if _active_process is process:
            _active_process = None

    if return_code not in (0, -15):
        if not _say_failure_warning_printed:
            print("TTS warning: speech playback ended unexpectedly. Continuing without audio.")
            _say_failure_warning_printed = True
