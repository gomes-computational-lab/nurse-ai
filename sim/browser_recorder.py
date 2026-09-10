from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import streamlit as st


SILENCE_SECONDS = 5
MAX_RECORDING_SECONDS = 120


@dataclass(frozen=True)
class BrowserRecording:
    audio: bytes
    mime_type: str

    @property
    def file_suffix(self) -> str:
        if "ogg" in self.mime_type:
            return ".ogg"
        if "mp4" in self.mime_type:
            return ".mp4"
        return ".webm"


_RECORDER_HTML = """
<div class="recorder" role="status" aria-live="polite">
  <span class="indicator" aria-hidden="true"></span>
  <span class="message">Preparing the microphone…</span>
  <button class="retry" type="button" hidden>Start recording</button>
</div>
"""

_RECORDER_CSS = """
.recorder {
  align-items: center;
  background: var(--st-secondary-background-color);
  border: 1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent);
  border-radius: var(--st-border-radius);
  display: flex;
  gap: 0.65rem;
  min-height: 2.75rem;
  padding: 0.6rem 0.8rem;
}

.indicator {
  background: color-mix(in srgb, var(--st-text-color) 35%, transparent);
  border-radius: 50%;
  flex: 0 0 auto;
  height: 0.7rem;
  width: 0.7rem;
}

.recorder[data-state="recording"] .indicator {
  animation: pulse 1.2s ease-in-out infinite;
  background: #e5484d;
}

.recorder[data-state="waiting"] .indicator {
  background: var(--st-primary-color);
}

.retry {
  background: var(--st-primary-color);
  border: 0;
  border-radius: var(--st-button-border-radius, var(--st-border-radius));
  color: white;
  cursor: pointer;
  font: inherit;
  margin-left: auto;
  padding: 0.35rem 0.65rem;
}

@keyframes pulse {
  50% { opacity: 0.35; transform: scale(0.85); }
}
"""

_RECORDER_JS = r"""
const controllers = new WeakMap()

function bytesToBase64(bytes) {
  const blockSize = 0x8000
  let binary = ""
  for (let offset = 0; offset < bytes.length; offset += blockSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + blockSize))
  }
  return btoa(binary)
}

export default function(component) {
  const { data, parentElement, setTriggerValue } = component
  const root = parentElement.querySelector(".recorder")
  const message = parentElement.querySelector(".message")
  const retry = parentElement.querySelector(".retry")
  if (!root || !message || !retry) return

  const turnId = String(data.turn_id)
  const existing = controllers.get(parentElement)
  if (existing && existing.turnId === turnId) return existing.cleanup
  if (existing) existing.cleanup()

  if (!data.active) {
    root.dataset.state = "complete"
    message.textContent = "Recording complete. Review the transcription below."
    retry.hidden = true
    return
  }

  let cancelled = false
  let stream = null
  let recorder = null
  let audioContext = null
  let animationFrame = null
  let maxTimer = null
  let patientAudio = null

  const setStatus = (state, text) => {
    root.dataset.state = state
    message.textContent = text
  }

  const releaseMicrophone = () => {
    if (animationFrame !== null) cancelAnimationFrame(animationFrame)
    animationFrame = null
    if (maxTimer !== null) clearTimeout(maxTimer)
    maxTimer = null
    if (audioContext) audioContext.close().catch(() => {})
    audioContext = null
    if (stream) stream.getTracks().forEach(track => track.stop())
    stream = null
  }

  const fail = (error) => {
    if (cancelled) return
    releaseMicrophone()
    const detail = error && error.message ? ` (${error.message})` : ""
    setStatus("error", `Automatic recording could not start${detail}.`)
    retry.hidden = false
    setTriggerValue("error", `Automatic recording could not start${detail}.`)
  }

  const getMicrophone = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("microphone capture is not supported in this browser")
    }
    if (!stream) {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
    }
    return stream
  }

  const startRecording = async () => {
    if (cancelled || (recorder && recorder.state === "recording")) return
    retry.hidden = true
    try {
      const microphone = await getMicrophone()
      if (cancelled) return

      const preferredTypes = [
        "audio/webm;codecs=opus",
        "audio/ogg;codecs=opus",
        "audio/mp4",
      ]
      const mimeType = preferredTypes.find(type => MediaRecorder.isTypeSupported(type)) || ""
      recorder = mimeType
        ? new MediaRecorder(microphone, { mimeType })
        : new MediaRecorder(microphone)

      const chunks = []
      recorder.ondataavailable = event => {
        if (event.data && event.data.size > 0) chunks.push(event.data)
      }
      recorder.onerror = event => fail(event.error || new Error("recording failed"))
      recorder.onstop = async () => {
        if (cancelled) return
        setStatus("processing", "Recording complete. Transcribing…")
        const blob = new Blob(chunks, { type: recorder.mimeType || mimeType || "audio/webm" })
        const bytes = new Uint8Array(await blob.arrayBuffer())
        releaseMicrophone()
        setTriggerValue("recording", {
          audio_base64: bytesToBase64(bytes),
          mime_type: blob.type || "audio/webm",
        })
      }

      audioContext = new AudioContext()
      await audioContext.resume()
      const source = audioContext.createMediaStreamSource(microphone)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 2048
      source.connect(analyser)
      const samples = new Float32Array(analyser.fftSize)
      const silenceThreshold = Number(data.silence_threshold || 0.018)
      const silenceMs = Number(data.silence_seconds) * 1000
      let quietSince = performance.now()

      const detectSilence = now => {
        if (cancelled || !recorder || recorder.state !== "recording") return
        analyser.getFloatTimeDomainData(samples)
        let energy = 0
        for (const sample of samples) energy += sample * sample
        const rms = Math.sqrt(energy / samples.length)
        if (rms >= silenceThreshold) quietSince = now
        const quietFor = now - quietSince
        const secondsLeft = Math.max(0, Math.ceil((silenceMs - quietFor) / 1000))
        setStatus(
          "recording",
          rms >= silenceThreshold
            ? "Recording…"
            : `Recording… stopping after ${secondsLeft}s of silence`,
        )
        if (quietFor >= silenceMs) {
          recorder.stop()
          return
        }
        animationFrame = requestAnimationFrame(detectSilence)
      }

      recorder.start(250)
      setStatus("recording", "Recording… stopping after 5s of silence")
      animationFrame = requestAnimationFrame(detectSilence)
      maxTimer = setTimeout(() => {
        if (recorder && recorder.state === "recording") recorder.stop()
      }, Number(data.max_recording_seconds) * 1000)
    } catch (error) {
      fail(error)
    }
  }

  const beginTurn = async () => {
    // Request access while the patient speaks; MediaRecorder starts only after playback ends.
    const microphonePromise = getMicrophone()
    if (!data.patient_audio_base64) {
      setStatus("waiting", "Patient finished. Starting the microphone…")
      await microphonePromise
      await startRecording()
      return
    }

    setStatus("waiting", "Listening to the patient…")
    patientAudio = new Audio(`data:audio/mpeg;base64,${data.patient_audio_base64}`)
    patientAudio.onended = async () => {
      setStatus("waiting", "Patient finished. Starting the microphone…")
      try {
        await microphonePromise
        await startRecording()
      } catch (error) {
        fail(error)
      }
    }
    patientAudio.onerror = () => fail(new Error("patient audio could not be played"))
    try {
      await patientAudio.play()
    } catch (error) {
      setStatus("error", "Autoplay was blocked. Select Start recording after the patient finishes.")
      retry.hidden = false
    }
  }

  retry.onclick = () => startRecording()

  const cleanup = () => {
    cancelled = true
    retry.onclick = null
    if (patientAudio) {
      patientAudio.pause()
      patientAudio.src = ""
    }
    if (recorder && recorder.state === "recording") recorder.stop()
    releaseMicrophone()
    controllers.delete(parentElement)
  }

  controllers.set(parentElement, { turnId, cleanup })
  beginTurn().catch(fail)
  return cleanup
}
"""

_AUTOMATIC_RECORDER = st.components.v2.component(
    "automatic_silence_recorder",
    html=_RECORDER_HTML,
    css=_RECORDER_CSS,
    js=_RECORDER_JS,
)


def ensure_browser_recorder_registered() -> None:
    """Register in the current Streamlit runtime (including isolated AppTest runs)."""
    st.components.v2.component(
        "automatic_silence_recorder",
        html=_RECORDER_HTML,
        css=_RECORDER_CSS,
        js=_RECORDER_JS,
    )


def automatic_silence_recorder(
    *,
    turn_id: str,
    patient_audio: bytes | None,
    key: str,
    active: bool = True,
    silence_seconds: int = SILENCE_SECONDS,
) -> tuple[BrowserRecording | None, str | None]:
    """Play the patient turn, then capture audio until trailing silence."""
    encoded_patient_audio = (
        base64.b64encode(patient_audio).decode("ascii") if patient_audio else None
    )
    result = _AUTOMATIC_RECORDER(
        key=key,
        data={
            "turn_id": turn_id,
            "active": active,
            "patient_audio_base64": encoded_patient_audio,
            "silence_seconds": silence_seconds,
            "max_recording_seconds": MAX_RECORDING_SECONDS,
            "silence_threshold": 0.018,
        },
        on_recording_change=lambda: None,
        on_error_change=lambda: None,
    )

    payload: Any = getattr(result, "recording", None)
    error: str | None = getattr(result, "error", None)
    if not isinstance(payload, dict):
        return None, error

    encoded_audio = payload.get("audio_base64")
    if not encoded_audio:
        return None, error
    try:
        audio = base64.b64decode(encoded_audio, validate=True)
    except (ValueError, TypeError):
        return None, "The browser returned an invalid audio recording."

    return BrowserRecording(
        audio=audio,
        mime_type=str(payload.get("mime_type") or "audio/webm"),
    ), error
