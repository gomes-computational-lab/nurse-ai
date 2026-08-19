from __future__ import annotations

import asyncio
import importlib
import os
import queue
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

SpeechStatus = Literal["pending", "completed", "interrupted", "failed"]

_tts_unavailable_warning_printed = False
_tts_failure_warning_printed = False
_speech_lock = threading.RLock()
_backend_lock = threading.Lock()
_synthesis_queue: queue.Queue[tuple[int, str]] = queue.Queue(maxsize=2)
_playback_queue: queue.Queue[tuple[int, "_PreparedAudio"]] = queue.Queue(maxsize=2)
_current_session_id = 0
_workers_started = False
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"[.!?](?:[\"'”’\)\]]*)\s+")
_NON_TERMINAL_ABBREVIATION_PATTERN = re.compile(
    r"(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|e\.g|i\.e)|\b[A-Z])\.$",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_PATTERN = re.compile(r"[,;:](?:[\"'”’\)\]]*)\s+")
_NATURAL_SPLIT_PATTERN = re.compile(r"[,;:]\s+|\s+")
_SPEECH_WHITESPACE_PATTERN = re.compile(r"\s+")
_MIN_FIRST_SEGMENT_CHARS = 28
_FIRST_SEGMENT_TARGET_CHARS = max(
    _MIN_FIRST_SEGMENT_CHARS,
    int(os.environ.get("TTS_FIRST_SEGMENT_CHARS", "64")),
)
_MAX_SEGMENT_CHARS = 160
_MIN_FALLBACK_SPLIT_CHARS = 80
_speech_sessions: dict[int, dict[str, object]] = {}
_TTS_VOICE = os.environ.get("TTS_VOICE", "en-US-AriaNeural")
_TTS_RATE = os.environ.get("TTS_RATE", "+0%")
_TTS_UNAVAILABLE = object()
_backend: "_SpeechBackend | object | None" = None


@dataclass(frozen=True)
class SpeechMetrics:
    status: SpeechStatus
    first_audio_started_at: float | None
    completed_at: float | None
    segments_started: int
    first_segment_submitted_at: float | None = None


@dataclass(frozen=True)
class _PreparedAudio:
    path: Path


@dataclass
class _LoadedAudio:
    prepared: _PreparedAudio
    playable: object


class _SpeechBackend(Protocol):
    def synthesize(self, text: str, cancel_event: threading.Event) -> _PreparedAudio | None:
        ...

    def load(self, audio: _PreparedAudio) -> object:
        ...

    def play(self, playable: object) -> None:
        ...

    def queue(self, playable: object) -> None:
        ...

    def current(self) -> object | None:
        ...

    def is_busy(self) -> bool:
        ...

    def stop(self) -> None:
        ...


class SpeechStream:
    def __init__(self, session_id: int, enabled: bool):
        self._session_id = session_id
        self._enabled = enabled
        self._buffer = ""
        self._lock = threading.Lock()
        self._finished = False
        self._segments_emitted = 0

    def add_chunk(self, chunk: str) -> None:
        if not self._enabled or not chunk:
            return

        with self._lock:
            if self._finished:
                return
            self._buffer += chunk
            ready = self._drain_ready_segments_locked()

        for segment in ready:
            _enqueue_speech(self._session_id, segment)

    def finish(self) -> None:
        if not self._enabled:
            return

        with self._lock:
            if self._finished:
                return
            self._finished = True
            ready = self._drain_ready_segments_locked()
            remaining = self._buffer.strip()
            self._buffer = ""

        for segment in ready:
            _enqueue_speech(self._session_id, segment)
        if remaining:
            _enqueue_speech(self._session_id, remaining)
        _mark_session_finished(self._session_id)

    def wait_until_done(self, timeout: float | None = None) -> bool:
        if not self._enabled:
            return True

        with _speech_lock:
            state = _speech_sessions.get(self._session_id)
            if state is None:
                return True
            done_event = state["done_event"]

        return done_event.wait(timeout)

    def metrics(self) -> SpeechMetrics:
        if not self._enabled:
            return SpeechMetrics(
                status="failed",
                first_audio_started_at=None,
                completed_at=time.perf_counter(),
                segments_started=0,
                first_segment_submitted_at=None,
            )

        with _speech_lock:
            state = _speech_sessions.get(self._session_id)
            if state is None:
                return SpeechMetrics(
                    status="failed",
                    first_audio_started_at=None,
                    completed_at=None,
                    segments_started=0,
                    first_segment_submitted_at=None,
                )
            return SpeechMetrics(
                status=state["status"],
                first_audio_started_at=state["first_audio_started_at"],
                completed_at=state["completed_at"],
                segments_started=int(state["segments_started"]),
                first_segment_submitted_at=state["first_segment_submitted_at"],
            )

    def _drain_ready_segments_locked(self) -> list[str]:
        segments: list[str] = []
        while self._buffer:
            sentence_end = _sentence_split_index(self._buffer)
            if sentence_end is not None:
                segment = self._buffer[:sentence_end].strip()
                self._buffer = self._buffer[sentence_end:].lstrip()
                if segment:
                    segments.append(segment)
                    self._segments_emitted += 1
                continue

            if self._segments_emitted == 0:
                first_segment_end = _first_segment_split_index(self._buffer)
                if first_segment_end is not None:
                    segment = self._buffer[:first_segment_end].strip()
                    self._buffer = self._buffer[first_segment_end:].lstrip()
                    if segment:
                        segments.append(segment)
                        self._segments_emitted += 1
                    continue

            if len(self._buffer) < _MAX_SEGMENT_CHARS:
                break

            split_at = _fallback_split_index(self._buffer)
            segment = self._buffer[:split_at].strip()
            self._buffer = self._buffer[split_at:].lstrip()
            if segment:
                segments.append(segment)
                self._segments_emitted += 1
        return segments

    # Kept as a small test hook for callers that previously forced timer flushes.
    def _flush_ready_segments(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            ready = self._drain_ready_segments_locked()
        for segment in ready:
            _enqueue_speech(self._session_id, segment)


class _EdgeTTSBackend:
    def __init__(self, edge_tts_module, pygame_module):
        self._communicate = edge_tts_module.Communicate
        self._pygame = pygame_module
        self._mixer = pygame_module.mixer
        self._channel_lock = threading.Lock()
        if not self._mixer.get_init():
            self._mixer.init(frequency=24000, channels=1)
        self._channel = self._mixer.find_channel(force=True)

    def synthesize(self, text: str, cancel_event: threading.Event) -> _PreparedAudio | None:
        if cancel_event.is_set():
            return None

        audio_bytes = asyncio.run(self._synthesize_audio(text, cancel_event))
        if not audio_bytes or cancel_event.is_set():
            return None
        return _PreparedAudio(_write_temp_audio_file(audio_bytes))

    def load(self, audio: _PreparedAudio) -> object:
        return self._mixer.Sound(str(audio.path))

    def play(self, playable: object) -> None:
        with self._channel_lock:
            self._channel.play(playable)

    def queue(self, playable: object) -> None:
        with self._channel_lock:
            self._channel.queue(playable)

    def current(self) -> object | None:
        with self._channel_lock:
            return self._channel.get_sound()

    def is_busy(self) -> bool:
        with self._channel_lock:
            return bool(self._channel.get_busy())

    def stop(self) -> None:
        with self._channel_lock:
            try:
                self._channel.stop()
            except Exception:
                return

    async def _synthesize_audio(self, text: str, cancel_event: threading.Event) -> bytes:
        synthesis_task = asyncio.create_task(self._collect_audio(text))
        cancellation_task = asyncio.create_task(self._wait_for_cancellation(cancel_event))
        done, _ = await asyncio.wait(
            {synthesis_task, cancellation_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if cancellation_task in done:
            synthesis_task.cancel()
            await asyncio.gather(synthesis_task, return_exceptions=True)
            return b""

        cancellation_task.cancel()
        await asyncio.gather(cancellation_task, return_exceptions=True)
        return await synthesis_task

    async def _collect_audio(self, text: str) -> bytes:
        communicate = self._communicate(text, _TTS_VOICE, rate=_TTS_RATE)
        audio_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)

    async def _wait_for_cancellation(self, cancel_event: threading.Event) -> None:
        while not cancel_event.is_set():
            await asyncio.sleep(0.02)


def _sentence_split_index(text: str) -> int | None:
    for match in _SENTENCE_BOUNDARY_PATTERN.finditer(text):
        punctuation_index = match.start()
        if text[punctuation_index] == "." and _is_non_terminal_period(text, punctuation_index):
            continue
        return match.end()
    return None


def _is_non_terminal_period(text: str, punctuation_index: int) -> bool:
    prefix = text[: punctuation_index + 1]
    return _NON_TERMINAL_ABBREVIATION_PATTERN.search(prefix) is not None


def _first_segment_split_index(text: str) -> int | None:
    for match in _CLAUSE_BOUNDARY_PATTERN.finditer(text):
        if match.end() >= _MIN_FIRST_SEGMENT_CHARS:
            return match.end()

    if len(text) < _FIRST_SEGMENT_TARGET_CHARS:
        return None

    window = text[:_FIRST_SEGMENT_TARGET_CHARS]
    candidates = [match.end() for match in _NATURAL_SPLIT_PATTERN.finditer(window)]
    usable = [index for index in candidates if index >= _MIN_FIRST_SEGMENT_CHARS]
    return usable[-1] if usable else _FIRST_SEGMENT_TARGET_CHARS


def _fallback_split_index(text: str) -> int:
    window = text[:_MAX_SEGMENT_CHARS]
    candidates = [match.end() for match in _NATURAL_SPLIT_PATTERN.finditer(window)]
    usable = [index for index in candidates if index >= _MIN_FALLBACK_SPLIT_CHARS]
    return usable[-1] if usable else _MAX_SEGMENT_CHARS


def speak_text(text: str) -> None:
    if not text.strip():
        return

    stream = create_speech_stream()
    stream.add_chunk(text)
    stream.finish()


def create_speech_stream(
    *,
    on_playback_start: Callable[[], None] | None = None,
    on_complete: Callable[[SpeechStatus], None] | None = None,
) -> SpeechStream:
    if not _tts_available():
        return SpeechStream(0, enabled=False)

    session_id = _begin_speech_session(
        on_playback_start=on_playback_start,
        on_complete=on_complete,
    )
    _ensure_workers_started()
    return SpeechStream(session_id, enabled=True)


def stop_speaking() -> None:
    global _current_session_id

    with _speech_lock:
        session_id = _current_session_id
        state = _speech_sessions.get(session_id)
        if state is not None and not state["done_event"].is_set():
            if state["status"] == "pending":
                state["status"] = "interrupted"
            state["finish_requested"] = True
            cancel_event = state["cancel_event"]
            cancel_event.set()
        _current_session_id += 1

    backend = _peek_tts_backend()
    if backend is not None:
        backend.stop()

    _drain_synthesis_queue()
    _drain_playback_queue()
    if state is not None:
        _complete_session_if_ready(session_id)


def _tts_available() -> bool:
    return _get_tts_backend() is not None


def _get_tts_backend() -> _SpeechBackend | None:
    global _backend

    with _backend_lock:
        if _backend is _TTS_UNAVAILABLE:
            return None
        if _backend is not None:
            return _backend

        try:
            backend = _create_tts_backend()
        except Exception as exc:
            _warn_tts_unavailable_once(
                f"TTS unavailable: could not initialize cross-platform speech ({exc}). Continuing with text output only."
            )
            _backend = _TTS_UNAVAILABLE
            return None

        if backend is None:
            _backend = _TTS_UNAVAILABLE
            return None

        _backend = backend
        return backend


def _peek_tts_backend() -> _SpeechBackend | None:
    with _backend_lock:
        if _backend in (None, _TTS_UNAVAILABLE):
            return None
        return _backend


def _create_tts_backend() -> _SpeechBackend | None:
    try:
        edge_tts_module = importlib.import_module("edge_tts")
        pygame_module = importlib.import_module("pygame")
    except ImportError:
        _warn_tts_unavailable_once(
            "TTS unavailable: install `edge-tts` and `pygame` for cross-platform speech output. Continuing with text output only."
        )
        return None

    try:
        return _EdgeTTSBackend(edge_tts_module, pygame_module)
    except Exception as exc:
        _warn_tts_unavailable_once(
            f"TTS unavailable: could not initialize audio playback ({exc}). Continuing with text output only."
        )
        return None


def _begin_speech_session(
    *,
    on_playback_start: Callable[[], None] | None = None,
    on_complete: Callable[[SpeechStatus], None] | None = None,
) -> int:
    global _current_session_id

    stop_speaking()
    with _speech_lock:
        _current_session_id += 1
        session_id = _current_session_id
        _speech_sessions[session_id] = {
            "pending_segments": 0,
            "finish_requested": False,
            "done_event": threading.Event(),
            "cancel_event": threading.Event(),
            "status": "pending",
            "first_audio_started_at": None,
            "first_segment_submitted_at": None,
            "completed_at": None,
            "segments_started": 0,
            "on_playback_start": on_playback_start,
            "on_complete": on_complete,
        }
    return session_id


def _ensure_workers_started() -> None:
    global _workers_started

    with _speech_lock:
        if _workers_started:
            return
        _workers_started = True

    threading.Thread(target=_synthesis_worker, daemon=True).start()
    threading.Thread(target=_playback_worker, daemon=True).start()


def _enqueue_speech(session_id: int, text: str) -> None:
    spoken_text = _normalize_speech_text(text)
    if not spoken_text:
        return

    with _speech_lock:
        state = _speech_sessions.get(session_id)
        if state is None or state["cancel_event"].is_set():
            return
        if state["first_segment_submitted_at"] is None:
            state["first_segment_submitted_at"] = time.perf_counter()
        state["pending_segments"] = int(state["pending_segments"]) + 1

    while True:
        if not _session_is_current(session_id):
            _mark_segment_done(session_id)
            return
        try:
            _synthesis_queue.put((session_id, spoken_text), timeout=0.05)
            return
        except queue.Full:
            continue


def _synthesis_worker() -> None:
    while True:
        session_id, text = _synthesis_queue.get()
        if not _session_is_current(session_id):
            _mark_segment_done(session_id)
            continue

        backend = _get_tts_backend()
        cancel_event = _session_cancel_event(session_id)
        if backend is None or cancel_event is None:
            _mark_session_failed(session_id)
            _mark_segment_done(session_id)
            continue

        try:
            prepared = backend.synthesize(text, cancel_event)
        except Exception as exc:
            _warn_tts_failure_once(f"TTS warning: could not synthesize response ({exc}). Continuing without audio.")
            _mark_session_failed(session_id)
            _mark_segment_done(session_id)
            continue

        if prepared is None:
            if not cancel_event.is_set():
                _mark_session_failed(session_id)
            _mark_segment_done(session_id)
            continue

        while True:
            if not _session_is_current(session_id):
                _cleanup_prepared(prepared)
                _mark_segment_done(session_id)
                break
            try:
                _playback_queue.put((session_id, prepared), timeout=0.05)
                break
            except queue.Full:
                continue


def _playback_worker() -> None:
    carry: tuple[int, _PreparedAudio] | None = None
    while True:
        item = carry if carry is not None else _playback_queue.get()
        carry = None
        session_id, prepared = item
        if not _session_is_current(session_id):
            _cleanup_prepared(prepared)
            _mark_segment_done(session_id)
            continue

        backend = _get_tts_backend()
        if backend is None:
            _cleanup_prepared(prepared)
            _mark_session_failed(session_id)
            _mark_segment_done(session_id)
            continue

        active: _LoadedAudio | None = None
        queued: _LoadedAudio | None = None
        loading_prepared: _PreparedAudio | None = prepared
        try:
            active = _LoadedAudio(prepared=prepared, playable=backend.load(prepared))
            loading_prepared = None
            backend.play(active.playable)
            _mark_segment_started(session_id)

            while active is not None:
                if not _session_is_current(session_id):
                    backend.stop()
                    _finish_loaded_segment(session_id, active)
                    if queued is not None:
                        _finish_loaded_segment(session_id, queued)
                    break

                if queued is None:
                    try:
                        next_item = _playback_queue.get_nowait()
                    except queue.Empty:
                        next_item = None

                    if next_item is not None:
                        next_session_id, next_prepared = next_item
                        if next_session_id != session_id:
                            carry = next_item
                        else:
                            loading_prepared = next_prepared
                            queued = _LoadedAudio(
                                prepared=next_prepared,
                                playable=backend.load(next_prepared),
                            )
                            loading_prepared = None
                            backend.queue(queued.playable)

                current = backend.current()
                if queued is not None and current is queued.playable:
                    _finish_loaded_segment(session_id, active)
                    active = queued
                    queued = None
                    _mark_segment_started(session_id)
                    continue

                if not backend.is_busy():
                    _finish_loaded_segment(session_id, active)
                    active = None
                    if queued is not None:
                        backend.play(queued.playable)
                        active = queued
                        queued = None
                        _mark_segment_started(session_id)
                        continue
                    break

                time.sleep(0.01)
        except Exception as exc:
            backend.stop()
            _mark_session_failed(session_id)
            if loading_prepared is not None:
                _cleanup_prepared(loading_prepared)
                _mark_segment_done(session_id)
            if active is not None:
                _finish_loaded_segment(session_id, active)
            if queued is not None:
                _finish_loaded_segment(session_id, queued)
            _warn_tts_failure_once(f"TTS warning: could not play response ({exc}). Continuing without audio.")


def _finish_loaded_segment(session_id: int, loaded: _LoadedAudio) -> None:
    _cleanup_prepared(loaded.prepared)
    _mark_segment_done(session_id)


def _drain_synthesis_queue() -> None:
    while True:
        try:
            session_id, _ = _synthesis_queue.get_nowait()
        except queue.Empty:
            return
        _mark_segment_done(session_id)


def _drain_playback_queue() -> None:
    while True:
        try:
            session_id, prepared = _playback_queue.get_nowait()
        except queue.Empty:
            return
        _cleanup_prepared(prepared)
        _mark_segment_done(session_id)


def _session_is_current(session_id: int) -> bool:
    with _speech_lock:
        state = _speech_sessions.get(session_id)
        return bool(
            state is not None
            and session_id == _current_session_id
            and not state["cancel_event"].is_set()
        )


def _session_cancel_event(session_id: int) -> threading.Event | None:
    with _speech_lock:
        state = _speech_sessions.get(session_id)
        return state["cancel_event"] if state is not None else None


def _normalize_speech_text(text: str) -> str:
    return _SPEECH_WHITESPACE_PATTERN.sub(" ", text).strip()


def _write_temp_audio_file(audio_bytes: bytes) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_file.write(audio_bytes)
        return Path(temp_file.name)


def _cleanup_prepared(audio: _PreparedAudio) -> None:
    try:
        audio.path.unlink(missing_ok=True)
    except OSError:
        pass


def _warn_tts_unavailable_once(message: str) -> None:
    global _tts_unavailable_warning_printed

    if _tts_unavailable_warning_printed:
        return
    print(message)
    _tts_unavailable_warning_printed = True


def _warn_tts_failure_once(message: str) -> None:
    global _tts_failure_warning_printed

    if _tts_failure_warning_printed:
        return
    print(message)
    _tts_failure_warning_printed = True


def _mark_session_finished(session_id: int) -> None:
    with _speech_lock:
        state = _speech_sessions.get(session_id)
        if state is None:
            return
        state["finish_requested"] = True
    _complete_session_if_ready(session_id)


def _mark_session_failed(session_id: int) -> None:
    with _speech_lock:
        state = _speech_sessions.get(session_id)
        if state is not None and state["status"] == "pending":
            state["status"] = "failed"


def _mark_segment_started(session_id: int) -> None:
    callback: Callable[[], None] | None = None
    with _speech_lock:
        state = _speech_sessions.get(session_id)
        if state is None:
            return
        state["segments_started"] = int(state["segments_started"]) + 1
        if state["first_audio_started_at"] is None:
            state["first_audio_started_at"] = time.perf_counter()
            callback = state["on_playback_start"]
    if callback is not None:
        callback()


def _mark_segment_done(session_id: int) -> None:
    with _speech_lock:
        state = _speech_sessions.get(session_id)
        if state is None:
            return
        state["pending_segments"] = max(0, int(state["pending_segments"]) - 1)
    _complete_session_if_ready(session_id)


def _complete_session_if_ready(session_id: int) -> None:
    callback: Callable[[SpeechStatus], None] | None = None
    status: SpeechStatus | None = None
    with _speech_lock:
        state = _speech_sessions.get(session_id)
        if state is None:
            return
        if not bool(state["finish_requested"]) or int(state["pending_segments"]) != 0:
            return
        done_event = state["done_event"]
        if done_event.is_set():
            return
        if state["status"] == "pending":
            state["status"] = "completed"
        state["completed_at"] = time.perf_counter()
        status = state["status"]
        callback = state["on_complete"]
        done_event.set()
    if callback is not None and status is not None:
        callback(status)
