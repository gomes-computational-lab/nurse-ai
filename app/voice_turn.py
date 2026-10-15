"""Browser-side pause detection and playback for one conversational turn."""

from collections.abc import Callable

import streamlit as st


# Keep capture in the browser: sounddevice would use the server's microphone.
VOICE_TURN_JS = """
export default function ({ data, parentElement, setStateValue }) {
    const status = parentElement.querySelector('#status');
    const enable = parentElement.querySelector('#enable');
    let disposed = false;
    let submitted = false;
    let stream;
    let context;
    let source;
    let timer;
    let recorder;
    let utterance;

    function releaseMicrophone() {
        clearInterval(timer);
        if (recorder && recorder.state !== 'inactive') recorder.stop();
        stream?.getTracks().forEach(track => track.stop());
        source?.disconnect();
        if (context && context.state !== 'closed') context.close().catch(() => {});
    }

    function offerRetry(message, action) {
        if (disposed) return;
        status.textContent = message;
        enable.hidden = false;
        enable.onclick = () => {
            enable.hidden = true;
            action();
        };
    }

    async function listen() {
        if (disposed || submitted) return;
        try {
            status.textContent = 'Connecting to microphone…';
            stream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true }
            });
            if (disposed) { releaseMicrophone(); return; }
            context = new AudioContext();
            if (context.state === 'suspended') {
                offerRetry('Click Enable voice to allow microphone processing.', async () => {
                    try {
                        await context.resume();
                        beginCapture();
                    } catch (error) {
                        releaseMicrophone();
                        offerRetry('Microphone processing failed: ' + error.message, listen);
                    }
                });
            } else {
                beginCapture();
            }
        } catch (error) {
            releaseMicrophone();
            offerRetry('Microphone unavailable: ' + error.message, listen);
        }
    }

    function beginCapture() {
        if (disposed || submitted) return;
        source = context.createMediaStreamSource(stream);
        const analyser = context.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        const samples = new Float32Array(analyser.fftSize);

        // Each clip is a complete media container, including its own header.
        // Idle clips are discarded locally; silence never becomes an AI turn.
        function newClip() {
            if (disposed || submitted) return;
            const chunks = [];
            let send = false;
            let recycle = false;
            let failed = false;
            let voicedMs = 0;
            let lastVoiceAt = 0;
            let previousTick = performance.now();
            const startedAt = previousTick;
            const clip = new MediaRecorder(stream);
            recorder = clip;
            clip.ondataavailable = event => {
                if (event.data.size) chunks.push(event.data);
            };
            clip.onerror = () => {
                failed = true;
                releaseMicrophone();
                offerRetry('Recording failed. Click Enable voice to retry.', listen);
            };
            clip.onstop = () => {
                if (disposed || failed) return;
                if (!send) {
                    if (recycle) {
                        newClip();
                    } else {
                        releaseMicrophone();
                        offerRetry('Microphone disconnected. Click Enable voice to reconnect.', listen);
                    }
                    return;
                }
                releaseMicrophone();
                const reader = new FileReader();
                reader.onload = () => {
                    if (disposed) return;
                    setStateValue('audio', {
                        base64: reader.result.split(',')[1],
                        mime_type: clip.mimeType,
                    });
                };
                reader.onerror = () => {
                    submitted = false;
                    offerRetry('Could not read the recording. Please try again.', listen);
                };
                reader.readAsDataURL(new Blob(chunks, { type: clip.mimeType }));
            };
            clip.start(250);
            status.textContent = 'Listening — speak, then pause for a reply.';
            timer = setInterval(() => {
                if (disposed || submitted) return;
                analyser.getFloatTimeDomainData(samples);
                const rms = Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);
                const now = performance.now();
                const tickMs = Math.min(now - previousTick, 60);
                previousTick = now;
                if (rms >= data.threshold) {
                    voicedMs += tickMs;
                    lastVoiceAt = now;
                }
                const hasSpeech = voicedMs >= 250;
                const paused = lastVoiceAt > 0 && now - lastVoiceAt >= data.pause_ms;
                const limitReached = now - startedAt >= 60000;
                if (hasSpeech && (paused || limitReached)) {
                    send = true;
                    submitted = true;
                    clearInterval(timer);
                    status.textContent = 'The family member is responding…';
                    clip.stop();
                } else if (paused || (!lastVoiceAt && now - startedAt >= 10000)) {
                    recycle = true;
                    clearInterval(timer);
                    clip.stop();
                }
            }, 30);
        }
        newClip();
    }

    function playReply() {
        if (disposed) return;
        if (!data.reply || !window.speechSynthesis) { listen(); return; }
        status.textContent = 'The family member is speaking…';
        utterance = new SpeechSynthesisUtterance(data.reply);
        utterance.lang = 'en-US';
        utterance.onend = () => { if (!disposed) listen(); };
        utterance.onerror = () => {
            offerRetry('Voice playback was blocked or unavailable. Click Enable voice to continue listening; the reply is shown above.', listen);
        };
        window.speechSynthesis.speak(utterance);
    }

    playReply();
    return () => {
        disposed = true;
        releaseMicrophone();
        if (utterance) {
            utterance.onend = null;
            utterance.onerror = null;
            window.speechSynthesis.cancel();
        }
    };
}
"""

_VOICE_TURN = st.components.v2.component(
    "automatic_voice_turn",
    html="""
    <p id="status" role="status">Preparing voice conversation…</p>
    <button id="enable" type="button" hidden>Enable voice</button>
    """,
    js=VOICE_TURN_JS,
)


def voice_turn(
    *,
    key: str,
    reply: str,
    pause_ms: int,
    threshold: float,
    on_audio_change: Callable[[], None],
):
    return _VOICE_TURN(
        key=key,
        data={"reply": reply, "pause_ms": pause_ms, "threshold": threshold},
        on_audio_change=on_audio_change,
    )
