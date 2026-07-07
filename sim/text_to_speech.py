from __future__ import annotations

import platform
import queue
import re
import shutil
import subprocess
import threading


_non_macos_warning_printed = False
_say_failure_warning_printed = False
_speech_lock = threading.Lock()
_active_process: subprocess.Popen[str] | None = None
_speech_queue: queue.Queue[tuple[int, str] | None] = queue.Queue()
_current_session_id = 0
_worker_started = False
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(.+?[.!?](?:\s+|$))", re.DOTALL)


class SpeechStream:
    def __init__(self, session_id: int, enabled: bool):
        self._session_id = session_id
        self._enabled = enabled
        self._buffer = ""

    def add_chunk(self, chunk: str) -> None:
        if not self._enabled or not chunk:
            return

        self._buffer += chunk
        self._flush_complete_segments()

    def finish(self) -> None:
        if not self._enabled:
            return

        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            _enqueue_speech(self._session_id, remaining)

    def _flush_complete_segments(self) -> None:
        while True:
            match = _SENTENCE_BOUNDARY_PATTERN.match(self._buffer)
            if match is None:
                return

            segment = match.group(1).strip()
            self._buffer = self._buffer[match.end() :]
            if segment:
                _enqueue_speech(self._session_id, segment)


def speak_text(text: str) -> None:
    if not text.strip():
        return

    stream = create_speech_stream()
    stream.add_chunk(text)
    stream.finish()


def create_speech_stream() -> SpeechStream:
    if not _tts_available():
        return SpeechStream(0, enabled=False)

    session_id = _begin_speech_session()
    _ensure_worker_started()
    return SpeechStream(session_id, enabled=True)


def stop_speaking() -> None:
    global _current_session_id

    with _speech_lock:
        _current_session_id += 1
        process = _active_process

    _drain_speech_queue()

    if process is not None and process.poll() is None:
        process.terminate()


def _tts_available() -> bool:
    global _non_macos_warning_printed
    global _say_failure_warning_printed

    if platform.system() != "Darwin":
        if not _non_macos_warning_printed:
            print("TTS unavailable: this prototype uses macOS `say`. Continuing with text output only.")
            _non_macos_warning_printed = True
        return False

    say_path = shutil.which("say")
    if not say_path:
        if not _say_failure_warning_printed:
            print("TTS unavailable: `say` was not found on this system. Continuing with text output only.")
            _say_failure_warning_printed = True
        return False

    return True


def _begin_speech_session() -> int:
    global _current_session_id

    with _speech_lock:
        _current_session_id += 1
        session_id = _current_session_id

    _drain_speech_queue()
    return session_id


def _ensure_worker_started() -> None:
    global _worker_started

    with _speech_lock:
        if _worker_started:
            return
        _worker_started = True

    threading.Thread(target=_speech_worker, daemon=True).start()


def _enqueue_speech(session_id: int, text: str) -> None:
    _speech_queue.put((session_id, text))


def _drain_speech_queue() -> None:
    while True:
        try:
            _speech_queue.get_nowait()
        except queue.Empty:
            return


def _speech_worker() -> None:
    while True:
        item = _speech_queue.get()
        if item is None:
            return

        session_id, text = item
        with _speech_lock:
            current_session_id = _current_session_id

        if session_id != current_session_id or not text.strip():
            continue

        _speak_segment(session_id, text)


def _speak_segment(session_id: int, text: str) -> None:
    global _say_failure_warning_printed
    global _active_process

    say_path = shutil.which("say")
    if not say_path:
        return

    try:
        process = subprocess.Popen([say_path, text])
    except Exception as exc:
        if not _say_failure_warning_printed:
            print(f"TTS warning: could not speak response with `say` ({exc}). Continuing without audio.")
            _say_failure_warning_printed = True
        return

    with _speech_lock:
        _active_process = process

    try:
        return_code = process.wait()
    except Exception as exc:
        if not _say_failure_warning_printed:
            print(f"TTS warning: could not monitor speech playback ({exc}). Continuing without audio.")
            _say_failure_warning_printed = True
        return
    finally:
        with _speech_lock:
            if _active_process is process:
                _active_process = None

    if return_code not in (0, -15):
        with _speech_lock:
            current_session_id = _current_session_id

        if session_id == current_session_id and not _say_failure_warning_printed:
            print("TTS warning: speech playback ended unexpectedly. Continuing without audio.")
            _say_failure_warning_printed = True
