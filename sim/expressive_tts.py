from __future__ import annotations

import asyncio
import importlib
import io
import ipaddress
import json
import os
import re
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from sim.voice_delivery import (
    DEFAULT_DELIVERY,
    DeliveryStyle,
    UnsafeVoiceResponseError,
    validate_spoken_text,
)


TTSProviderName = Literal["zonos2", "chatterbox_nano", "edge"]
SUPPORTED_TTS_PROVIDERS = ("zonos2", "chatterbox_nano", "edge")
DEFAULT_TTS_PROVIDER: TTSProviderName = "zonos2"
DEFAULT_ZONOS2_URL = "http://localhost:1919"
DEFAULT_VOICE = "default"
SUPPORTED_VOICE_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}
APPROVED_CLINICAL_CUES = {"sigh", "cough", "sniffle", "groan", "yawn"}
_CUE_PATTERN = re.compile(r"\[([a-z_]+)\]", re.IGNORECASE)


class LocalTTSError(RuntimeError):
    pass


class LocalTTSUnavailable(LocalTTSError):
    pass


@dataclass(frozen=True)
class TTSConfig:
    provider: TTSProviderName = DEFAULT_TTS_PROVIDER
    voice: str = DEFAULT_VOICE
    zonos2_url: str = DEFAULT_ZONOS2_URL
    voice_catalog_dir: str = "voices"
    allow_online_edge_fallback: bool = False
    timeout_seconds: float = 45.0

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_TTS_PROVIDERS:
            supported = ", ".join(SUPPORTED_TTS_PROVIDERS)
            raise ValueError(
                f"Unsupported TTS provider {self.provider!r}; expected one of: {supported}"
            )


@dataclass(frozen=True)
class SynthesizedAudio:
    data: bytes
    mime_type: str
    provider: TTSProviderName
    fallback_from: TTSProviderName | None = None


@dataclass(frozen=True)
class ProviderCapability:
    provider: TTSProviderName
    available: bool
    detail: str
    local: bool


@dataclass
class _ChatterboxRuntime:
    model: object
    torchaudio: object
    inference_lock: threading.Lock
    voice_conditionings: dict[tuple[str, int, int], object] = field(
        default_factory=dict
    )


_CHATTERBOX_RUNTIME_LOCK = threading.Lock()
_CHATTERBOX_RUNTIMES: dict[str, _ChatterboxRuntime] = {}


class _Provider(Protocol):
    name: TTSProviderName

    def synthesize(
        self,
        text: str,
        style: DeliveryStyle,
        voice: str,
        cancel_event: threading.Event,
    ) -> SynthesizedAudio: ...

    def capability(self) -> ProviderCapability: ...


def list_approved_voices(catalog_dir: str | Path = "voices") -> dict[str, Path]:
    root = Path(catalog_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / root
    if not root.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_VOICE_SUFFIXES
    }


def zonos2_delivery_parameters(style: DeliveryStyle) -> dict[str, object]:
    valence_arousal = {
        "neutral": (0.0, 0.0),
        "anxious": (-0.30, 0.28),
        "fearful": (-0.45, 0.38),
        "frustrated": (-0.38, 0.30),
        "confused": (-0.18, 0.08),
        "relieved": (0.30, -0.20),
        "sad": (-0.42, -0.30),
        "in_pain": (-0.48, 0.34),
        "tired": (-0.22, -0.38),
    }
    valence, arousal = valence_arousal[style.emotion]
    return {
        "emotion_enabled": style.emotion != "neutral",
        "emotion_valence": valence,
        "emotion_arousal": arousal,
        "emotion_strength": {1: 0.55, 2: 0.75, 3: 0.95}[style.intensity],
        "emotion_cfg_scale": 1.2 if style.emotion != "neutral" else 1.0,
        "accurate_mode": True,
        "speaking_rate_enabled": True,
        "speed": {"slow": 0.92, "normal": 1.0, "fast": 1.08}[style.pace],
    }


def chatterbox_delivery_parameters(style: DeliveryStyle) -> dict[str, float]:
    exaggeration = {1: 0.38, 2: 0.48, 3: 0.58}[style.intensity]
    if style.emotion == "neutral":
        exaggeration = 0.35
    pace_adjustment = {"slow": -0.10, "normal": 0.0, "fast": 0.08}[style.pace]
    return {
        "exaggeration": max(0.25, min(0.65, exaggeration + pace_adjustment)),
        "cfg_weight": {"slow": 0.30, "normal": 0.45, "fast": 0.52}[style.pace],
    }


def sanitize_chatterbox_cues(text: str) -> str:
    """Keep only the small reviewed cue vocabulary supported by the simulation."""

    def replace(match: re.Match[str]) -> str:
        cue = match.group(1).lower()
        return f"[{cue}]" if cue in APPROVED_CLINICAL_CUES else ""

    return " ".join(_CUE_PATTERN.sub(replace, text).split())


class TTSService:
    """Local-first TTS service with bounded fallback and reusable model instances."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self._providers: dict[TTSProviderName, _Provider] = {}
        self._lock = threading.Lock()

    def synthesize(
        self,
        text: str,
        style: DeliveryStyle | None = None,
        cancel_event: threading.Event | None = None,
    ) -> SynthesizedAudio:
        if not text.strip():
            raise LocalTTSError("Patient speech was empty.")
        try:
            safe_text = validate_spoken_text(text)
        except UnsafeVoiceResponseError as exc:
            raise LocalTTSError(
                "Patient speech contained internal metadata and was not synthesized."
            ) from exc
        clean_text = " ".join(safe_text.split())
        style = style or DEFAULT_DELIVERY
        cancel_event = cancel_event or threading.Event()
        attempted: list[str] = []
        first = self.config.provider

        for provider_name in self._fallback_order():
            if cancel_event.is_set():
                raise LocalTTSError("Speech synthesis was cancelled.")
            try:
                result = self._provider(provider_name).synthesize(
                    clean_text, style, self.config.voice, cancel_event
                )
                if provider_name != first:
                    return SynthesizedAudio(
                        data=result.data,
                        mime_type=result.mime_type,
                        provider=result.provider,
                        fallback_from=first,
                    )
                return result
            except LocalTTSError as exc:
                attempted.append(f"{provider_name}: {exc}")

        raise LocalTTSUnavailable(
            "; ".join(attempted) or "No TTS provider is configured."
        )

    def capabilities(self) -> list[ProviderCapability]:
        capabilities = []
        for name in self._fallback_order():
            try:
                capabilities.append(self._provider(name).capability())
            except LocalTTSError as exc:
                capabilities.append(
                    ProviderCapability(name, False, str(exc), name != "edge")
                )
        return capabilities

    def _fallback_order(self) -> list[TTSProviderName]:
        if self.config.provider == "zonos2":
            order: list[TTSProviderName] = ["zonos2", "chatterbox_nano"]
        elif self.config.provider == "chatterbox_nano":
            order = ["chatterbox_nano"]
        elif self.config.provider == "edge":
            order = ["edge"]
        else:
            raise LocalTTSError(f"Unsupported TTS provider: {self.config.provider!r}")
        if self.config.allow_online_edge_fallback and "edge" not in order:
            order.append("edge")
        return order

    def _provider(self, name: TTSProviderName) -> _Provider:
        with self._lock:
            if name not in self._providers:
                if name == "zonos2":
                    self._providers[name] = _Zonos2Provider(self.config)
                elif name == "chatterbox_nano":
                    self._providers[name] = _ChatterboxNanoProvider(self.config)
                elif name == "edge":
                    self._providers[name] = _EdgeProvider()
                else:
                    raise LocalTTSError(f"Unsupported TTS provider: {name!r}")
            return self._providers[name]


class _Zonos2Provider:
    name: TTSProviderName = "zonos2"

    def __init__(self, config: TTSConfig):
        self._url = config.zonos2_url.rstrip("/")
        self._timeout = config.timeout_seconds
        self._speaker_lock = threading.Lock()
        self._server_speakers: list[dict[str, object]] | None = None
        _validate_local_or_approved_url(self._url)

    def synthesize(self, text, style, voice, cancel_event) -> SynthesizedAudio:
        if cancel_event.is_set():
            raise LocalTTSError("cancelled")
        payload = {
            "text": text,
            "speaker_embedding_id": self._resolve_server_speaker_id(voice),
            "stream": False,
            **zonos2_delivery_parameters(style),
        }
        request = urllib.request.Request(
            f"{self._url}/tts/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                pcm = response.read()
                response_headers = {
                    "content_type": response.headers.get("Content-Type", ""),
                    "sample_rate": response.headers.get("X-Audio-Sample-Rate", ""),
                    "channels": response.headers.get("X-Audio-Channels", ""),
                    "format": response.headers.get("X-Audio-Format", ""),
                }
        except (OSError, ValueError, urllib.error.URLError) as exc:
            raise LocalTTSUnavailable(
                f"university ZONOS2 server unavailable ({exc})"
            ) from exc
        sample_rate = _validate_zonos2_pcm_response(pcm, response_headers)
        return SynthesizedAudio(
            _pcm_f32_to_wav(pcm, sample_rate), "audio/wav", self.name
        )

    def _resolve_server_speaker_id(self, voice: str) -> str:
        speakers = self._get_server_speakers()
        if voice == DEFAULT_VOICE:
            if not speakers:
                raise LocalTTSUnavailable(
                    "ZONOS2 has no university-approved default voices installed"
                )
            return str(speakers[0]["id"])

        normalized_voice = voice.casefold()
        matches = []
        for speaker in speakers:
            aliases = {
                str(speaker["id"]).casefold(),
                str(speaker.get("label", "")).casefold(),
                Path(str(speaker.get("original_name", ""))).stem.casefold(),
            }
            if normalized_voice in aliases:
                matches.append(str(speaker["id"]))

        unique_matches = list(dict.fromkeys(matches))
        if len(unique_matches) == 1:
            return unique_matches[0]
        if len(unique_matches) > 1:
            raise LocalTTSUnavailable(
                f"ZONOS2 voice {voice!r} matches more than one approved server voice"
            )
        raise LocalTTSUnavailable(
            f"approved ZONOS2 voice {voice!r} is not installed on the university server"
        )

    def _get_server_speakers(self) -> list[dict[str, object]]:
        with self._speaker_lock:
            if self._server_speakers is not None:
                return self._server_speakers

            request = urllib.request.Request(f"{self._url}/tts/speakers", method="GET")
            try:
                with urllib.request.urlopen(
                    request, timeout=min(self._timeout, 3.0)
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except (OSError, ValueError, urllib.error.URLError) as exc:
                raise LocalTTSUnavailable(
                    f"could not load approved voices from ZONOS2 ({exc})"
                ) from exc

            if not isinstance(payload, dict):
                raise LocalTTSError("ZONOS2 returned an invalid speaker catalog")
            raw_speakers = payload.get("speakers")
            if not isinstance(raw_speakers, list):
                raise LocalTTSError("ZONOS2 returned an invalid speaker catalog")

            approved_speakers = []
            for speaker in raw_speakers:
                if not isinstance(speaker, dict) or not isinstance(
                    speaker.get("id"), str
                ):
                    continue
                if (
                    speaker.get("scope") == "default"
                    or speaker.get("is_default") is True
                ):
                    approved_speakers.append(speaker)

            self._server_speakers = approved_speakers
            return approved_speakers

    def capability(self) -> ProviderCapability:
        request = urllib.request.Request(f"{self._url}/tts/capabilities", method="GET")
        try:
            with urllib.request.urlopen(
                request, timeout=min(self._timeout, 3.0)
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            emotions = ", ".join(payload.get("emotion_names", [])) or "valence/arousal"
            return ProviderCapability(
                self.name, True, f"Local server; emotion: {emotions}", True
            )
        except Exception as exc:
            return ProviderCapability(
                self.name, False, f"Server unavailable: {exc}", True
            )


class _ChatterboxNanoProvider:
    name: TTSProviderName = "chatterbox_nano"

    def __init__(self, config: TTSConfig):
        self._catalog = list_approved_voices(config.voice_catalog_dir)

    def synthesize(self, text, style, voice, cancel_event) -> SynthesizedAudio:
        prompt = self._catalog.get(voice)
        if prompt is None:
            raise LocalTTSUnavailable(
                f"approved voice '{voice}' is not installed in the voice catalog"
            )
        runtime = _get_chatterbox_runtime()
        if cancel_event.is_set():
            raise LocalTTSError("cancelled")
        params = chatterbox_delivery_parameters(style)
        try:
            with runtime.inference_lock:
                if cancel_event.is_set():
                    raise LocalTTSError("cancelled")
                _activate_chatterbox_voice(
                    runtime,
                    prompt,
                    exaggeration=params["exaggeration"],
                )
                audio = runtime.model.generate(
                    sanitize_chatterbox_cues(text),
                    **params,
                )
            with tempfile.NamedTemporaryFile(suffix=".wav") as output:
                runtime.torchaudio.save(output.name, audio.cpu(), runtime.model.sr)
                output.seek(0)
                data = output.read()
        except LocalTTSError:
            raise
        except Exception as exc:
            raise LocalTTSError(f"Chatterbox Nano failed ({exc})") from exc
        if cancel_event.is_set():
            raise LocalTTSError("cancelled")
        return SynthesizedAudio(data, "audio/wav", self.name)

    def capability(self) -> ProviderCapability:
        if not self._catalog:
            return ProviderCapability(
                self.name, False, "No approved reference voices are installed", True
            )
        try:
            _configure_chatterbox_import_cache()
            importlib.import_module("chatterbox.tts_turbo")
            _require_perth_watermarker()
        except ImportError:
            return ProviderCapability(
                self.name, False, "Install optional Chatterbox dependencies", True
            )
        except Exception as exc:
            return ProviderCapability(
                self.name, False, f"Chatterbox import failed: {exc}", True
            )
        return ProviderCapability(self.name, True, "On-device model available", True)


def _configure_chatterbox_import_cache() -> None:
    if "NUMBA_CACHE_DIR" in os.environ:
        return
    cache_dir = Path(tempfile.gettempdir()) / "nursing-ai-sim-numba-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)


def _require_perth_watermarker() -> None:
    perth = importlib.import_module("perth")
    if not callable(getattr(perth, "PerthImplicitWatermarker", None)):
        raise LocalTTSUnavailable(
            "resemble-perth 1.1.0 or newer is required; run "
            "`python -m pip install -r requirements-tts.txt`"
        )


def preload_chatterbox_model(
    *,
    voice: str = DEFAULT_VOICE,
    voice_catalog_dir: str | Path = "voices",
) -> None:
    """Load Chatterbox and cache the selected approved voice when available."""
    preload_chatterbox_runtime()
    if voice == DEFAULT_VOICE:
        return
    prepare_chatterbox_voice(voice=voice, voice_catalog_dir=voice_catalog_dir)


def preload_chatterbox_runtime() -> None:
    """Load and retain the shared Chatterbox runtime."""
    _get_chatterbox_runtime()


def prepare_chatterbox_voice(
    *,
    voice: str = DEFAULT_VOICE,
    voice_catalog_dir: str | Path = "voices",
) -> None:
    """Prepare and retain the conditioning for one approved patient voice."""
    runtime = _get_chatterbox_runtime()
    if voice == DEFAULT_VOICE:
        return

    prompt = list_approved_voices(voice_catalog_dir).get(voice)
    if prompt is None:
        raise LocalTTSUnavailable(
            f"approved voice '{voice}' is not installed in the voice catalog"
        )
    exaggeration = chatterbox_delivery_parameters(DEFAULT_DELIVERY)["exaggeration"]
    with runtime.inference_lock:
        _activate_chatterbox_voice(runtime, prompt, exaggeration=exaggeration)


def _activate_chatterbox_voice(
    runtime: _ChatterboxRuntime,
    prompt: Path,
    *,
    exaggeration: float,
) -> None:
    """Select cached voice conditioning; caller must hold the inference lock."""
    resolved_prompt = prompt.resolve()
    stat = resolved_prompt.stat()
    cache_key = (str(resolved_prompt), stat.st_mtime_ns, stat.st_size)
    conditionals = runtime.voice_conditionings.get(cache_key)

    if conditionals is None:
        runtime.model.prepare_conditionals(
            str(resolved_prompt),
            exaggeration=exaggeration,
        )
        conditionals = getattr(runtime.model, "conds", None)
        if conditionals is None:
            raise LocalTTSUnavailable(
                f"Chatterbox did not prepare approved voice '{prompt.stem}'"
            )
        prompt_key = str(resolved_prompt)
        runtime.voice_conditionings = {
            key: value
            for key, value in runtime.voice_conditionings.items()
            if key[0] != prompt_key
        }
        runtime.voice_conditionings[cache_key] = conditionals
    else:
        runtime.model.conds = conditionals

    emotion = getattr(getattr(conditionals, "t3", None), "emotion_adv", None)
    fill_emotion = getattr(emotion, "fill_", None)
    if callable(fill_emotion):
        fill_emotion(exaggeration)


def _get_chatterbox_runtime() -> _ChatterboxRuntime:
    _configure_chatterbox_import_cache()
    try:
        torch = importlib.import_module("torch")
        torchaudio = importlib.import_module("torchaudio")
        module = importlib.import_module("chatterbox.tts_turbo")
    except ImportError as exc:
        raise LocalTTSUnavailable(
            "install Chatterbox with `pip install -r requirements-tts.txt`"
        ) from exc
    _require_perth_watermarker()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with _CHATTERBOX_RUNTIME_LOCK:
        existing = _CHATTERBOX_RUNTIMES.get(device)
        if existing is not None:
            return existing
        try:
            model = module.ChatterboxTurboTTS.from_pretrained(device=device)
        except Exception as exc:
            raise LocalTTSUnavailable(
                f"could not load Chatterbox Nano ({exc})"
            ) from exc

        runtime = _ChatterboxRuntime(
            model=model,
            torchaudio=torchaudio,
            inference_lock=threading.Lock(),
        )
        _CHATTERBOX_RUNTIMES[device] = runtime
        return runtime


class _EdgeProvider:
    name: TTSProviderName = "edge"

    def __init__(self):
        self._voice = os.environ.get("TTS_VOICE", "en-US-AriaNeural")
        self._rate = os.environ.get("TTS_RATE", "+0%")

    def synthesize(self, text, style, voice, cancel_event) -> SynthesizedAudio:
        del style
        try:
            edge_tts = importlib.import_module("edge_tts")
        except ImportError as exc:
            raise LocalTTSUnavailable("edge-tts is not installed") from exc

        async def collect() -> bytes:
            selected_voice = voice if voice != DEFAULT_VOICE else self._voice
            communicate = edge_tts.Communicate(text, selected_voice, rate=self._rate)
            chunks = []
            async for chunk in communicate.stream():
                if cancel_event.is_set():
                    return b""
                if chunk.get("type") == "audio":
                    chunks.append(chunk["data"])
            return b"".join(chunks)

        try:
            data = asyncio.run(collect())
        except Exception as exc:
            raise LocalTTSError(f"legacy Edge TTS failed ({exc})") from exc
        if not data:
            raise LocalTTSError("legacy Edge TTS returned no audio")
        return SynthesizedAudio(data, "audio/mpeg", self.name)

    def capability(self) -> ProviderCapability:
        try:
            importlib.import_module("edge_tts")
            return ProviderCapability(self.name, True, "Online legacy provider", False)
        except ImportError:
            return ProviderCapability(
                self.name, False, "edge-tts is not installed", False
            )


def _validate_local_or_approved_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""
    approved = {
        item.strip()
        for item in os.environ.get("TTS_ALLOWED_HOSTS", "").split(",")
        if item.strip()
    }
    allowed = (
        hostname in {"localhost", "::1"}
        or hostname.endswith(".local")
        or hostname in approved
    )
    try:
        allowed = allowed or ipaddress.ip_address(hostname).is_private
    except ValueError:
        pass
    if parsed.scheme not in {"http", "https"} or not allowed:
        raise LocalTTSUnavailable(
            "ZONOS2 URL must be localhost, a private IP, a .local host, or listed in TTS_ALLOWED_HOSTS"
        )


def _validate_zonos2_pcm_response(pcm: bytes, headers: dict[str, str]) -> int:
    if not pcm:
        raise LocalTTSError("ZONOS2 returned no audio")

    content_type = headers["content_type"].partition(";")[0].strip().lower()
    if content_type != "audio/pcm":
        raise LocalTTSError(
            f"ZONOS2 returned unexpected content type {content_type or 'missing'!r}"
        )

    audio_format = headers["format"].strip().lower()
    if audio_format not in {"float32", "float32le", "f32le"}:
        raise LocalTTSError(
            f"ZONOS2 returned unexpected audio format {audio_format or 'missing'!r}"
        )
    if headers["channels"].strip() != "1":
        raise LocalTTSError("ZONOS2 response must contain mono audio")
    if len(pcm) % 4:
        raise LocalTTSError("ZONOS2 returned an incomplete float32 PCM frame")

    try:
        sample_rate = int(headers["sample_rate"])
    except ValueError as exc:
        raise LocalTTSError("ZONOS2 returned an invalid audio sample rate") from exc
    if sample_rate <= 0:
        raise LocalTTSError("ZONOS2 returned an invalid audio sample rate")
    return sample_rate


def _pcm_f32_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    samples = np.frombuffer(pcm, dtype="<f4")
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)
    return output.getvalue()
