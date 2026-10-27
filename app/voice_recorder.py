"""Browser microphone recording for an entire simulation."""

import streamlit as st

_RECORDER = st.components.v2.component(
    "simulation_voice_recorder",
    html="""
    <p id="status" role="status">Requesting microphone access…</p>
    <button id="retry" type="button" hidden>Retry microphone access</button>
    <button id="end" type="button" disabled>End and save</button>
    """,
    js="""
export default function ({ parentElement, setStateValue }) {
    const status = parentElement.querySelector('#status');
    const retry = parentElement.querySelector('#retry');
    const end = parentElement.querySelector('#end');
    let stream;
    let recorder;
    let disposed = false;
    let finished = false;
    const release = () => stream?.getTracks().forEach(track => track.stop());

    async function start() {
        retry.hidden = true;
        status.textContent = 'Requesting microphone access…';
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            if (disposed) { release(); return; }
            recorder = new MediaRecorder(stream);
            const chunks = [];
            recorder.ondataavailable = event => {
                if (event.data.size) chunks.push(event.data);
            };
            recorder.onerror = () => {
                status.textContent = 'Recording failed. End the simulation to save captured audio.';
                end.disabled = false;
            };
            recorder.onstop = async () => {
                release();
                if (disposed) return;
                finished = true;
                end.disabled = true;
                status.textContent = 'Preparing recording…';
                const blob = new Blob(chunks, { type: recorder.mimeType });
                const reader = new FileReader();
                reader.onload = () => {
                    if (disposed) return;
                    status.textContent = 'Recording stopped. Saving simulation…';
                    setStateValue('recording', {
                        audio: reader.result.split(',')[1],
                        mime_type: blob.type,
                    });
                };
                reader.onerror = () => {
                    status.textContent = 'Could not read recording. Try ending again.';
                    finished = false;
                    end.disabled = false;
                };
                reader.readAsDataURL(blob);
            };
            recorder.start(1000);
            status.textContent = 'Recording continuously until you end the simulation.';
            end.disabled = false;
        } catch (error) {
            release();
            status.textContent = 'Microphone unavailable: ' + error.message;
            retry.hidden = false;
            end.disabled = false;
        }
    }

    retry.onclick = start;
    end.onclick = () => {
        if (finished) return;
        end.disabled = true;
        if (recorder && recorder.state !== 'inactive') {
            status.textContent = 'Stopping recording…';
            recorder.stop();
        } else if (recorder) {
            recorder.onstop();
        } else {
            finished = true;
            setStateValue('recording', { audio: '', mime_type: '' });
        }
    };
    start();
    return () => {
        disposed = true;
        if (recorder && recorder.state !== 'inactive') recorder.stop();
        release();
    };
}
""",
)


def record_simulation(*, key: str):
    """Return completed audio only after the user ends the simulation."""
    return _RECORDER(key=key, on_recording_change=lambda: None)
