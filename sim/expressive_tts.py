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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from sim.voice_delivery import DEFAULT_DELIVERY, DeliveryStyle


TTSProviderName = Literal["zonos2", "chatterbox_nano", "edge"]
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
        clean_text = " ".join(text.split())
        if not clean_text:
            raise LocalTTSError("Patient speech was empty.")
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
        else:
            order = ["edge"]
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
                else:
                    self._providers[name] = _EdgeProvider()
            return self._providers[name]


class _Zonos2Provider:
    name: TTSProviderName = "zonos2"

    def __init__(self, config: TTSConfig):
        self._url = config.zonos2_url.rstrip("/")
        self._timeout = config.timeout_seconds
        _validate_local_or_approved_url(self._url)

    def synthesize(self, text, style, voice, cancel_event) -> SynthesizedAudio:
        if cancel_event.is_set():
            raise LocalTTSError("cancelled")
        payload = {
            "model": "zonos2",
            "input": text,
            "voice": voice,
            "response_format": "pcm",
            **zonos2_delivery_parameters(style),
        }
        request = urllib.request.Request(
            f"{self._url}/v1/audio/speech",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                pcm = response.read()
                sample_rate = int(response.headers.get("X-Audio-Sample-Rate", "44100"))
        except (OSError, ValueError, urllib.error.URLError) as exc:
            raise LocalTTSUnavailable(
                f"university ZONOS2 server unavailable ({exc})"
            ) from exc
        if not pcm:
            raise LocalTTSError("ZONOS2 returned no audio")
        return SynthesizedAudio(
            _pcm_f32_to_wav(pcm, sample_rate), "audio/wav", self.name
        )

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
        self._model = None
        self._torch = None
        self._torchaudio = None

    def synthesize(self, text, style, voice, cancel_event) -> SynthesizedAudio:
        prompt = self._catalog.get(voice)
        if prompt is None:
            raise LocalTTSUnavailable(
                f"approved voice '{voice}' is not installed in the voice catalog"
            )
        self._load_model()
        if cancel_event.is_set():
            raise LocalTTSError("cancelled")
        params = chatterbox_delivery_parameters(style)
        try:
            audio = self._model.generate(
                sanitize_chatterbox_cues(text),
                audio_prompt_path=str(prompt),
                **params,
            )
            with tempfile.NamedTemporaryFile(suffix=".wav") as output:
                self._torchaudio.save(output.name, audio.cpu(), self._model.sr)
                output.seek(0)
                data = output.read()
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
            importlib.import_module("chatterbox.tts_turbo")
        except ImportError:
            return ProviderCapability(
                self.name, False, "Install optional Chatterbox dependencies", True
            )
        return ProviderCapability(self.name, True, "On-device model available", True)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            self._torch = importlib.import_module("torch")
            self._torchaudio = importlib.import_module("torchaudio")
            module = importlib.import_module("chatterbox.tts_turbo")
        except ImportError as exc:
            raise LocalTTSUnavailable(
                "install Chatterbox with `pip install -r requirements-tts.txt`"
            ) from exc
        device = "cuda" if self._torch.cuda.is_available() else "cpu"
        try:
            self._model = module.ChatterboxTurboTTS.from_pretrained(
                device=device, nano=True
            )
        except Exception as exc:
            raise LocalTTSUnavailable(
                f"could not load Chatterbox Nano ({exc})"
            ) from exc


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
