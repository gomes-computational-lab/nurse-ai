import base64
import tempfile
from pathlib import Path

import streamlit as st

from app.voice_turn import voice_turn
from sim.ollama_client import OllamaClient
from sim.scenarios import list_scenarios
from sim.session import SimulationSession
from sim.speech_to_text import transcribe_audio
from sim.storage import save_result


st.title("Family Member Simulation")

if "simulation" not in st.session_state:
    st.session_state.simulation = None

if "recording_number" not in st.session_state:
    st.session_state.recording_number = 0

st.session_state.setdefault("pending_audio", None)
st.session_state.setdefault("pending_text", None)
st.session_state.setdefault("voice_reply", "")


def receive_audio():
    key = f"recording_{st.session_state.recording_number}"
    st.session_state.pending_audio = st.session_state[key]["audio"]
    st.session_state.pending_text = None


if saved_message := st.session_state.pop("saved_message", None):
    st.success(saved_message)


# Choose a scenario and start a conversation.
if st.session_state.simulation is None:
    scenarios = list_scenarios()

    if not scenarios:
        st.error("No scenarios were found in the scenarios folder.")
        st.stop()

    scenario = st.selectbox(
        "Choose a scenario",
        scenarios,
        format_func=lambda item: item.title,
    )
    st.slider(
        "Pause before responding (seconds)",
        min_value=0.4,
        max_value=2.0,
        value=0.7,
        step=0.1,
        key="pause_seconds",
    )
    st.selectbox(
        "Microphone sensitivity",
        ["Normal", "High (quiet voice)", "Low (noisy room)"],
        key="microphone_sensitivity",
    )

    if st.button("Start simulation", type="primary"):
        with st.spinner("Starting..."):
            client = OllamaClient(model="llama3.1")
            simulation = SimulationSession(scenario, client)
            opening = simulation.opening(response_mode="voice")

        st.session_state.simulation = simulation
        st.session_state.voice_reply = opening
        st.session_state.voice_pause_ms = round(st.session_state.pause_seconds * 1000)
        st.session_state.voice_threshold = {
            "Normal": 0.015,
            "High (quiet voice)": 0.008,
            "Low (noisy room)": 0.03,
        }[st.session_state.microphone_sensitivity]
        st.rerun()

else:
    simulation = st.session_state.simulation

    # Handle stopping before mounting the microphone again. Unmounting releases
    # the browser's microphone and cancels any spoken reply.
    if st.button("End and save"):
        saved_path = save_result(simulation.scenario, simulation.transcript)
        st.session_state.simulation = None
        st.session_state.pending_audio = None
        st.session_state.pending_text = None
        st.session_state.voice_reply = ""
        st.session_state.recording_number += 1
        st.session_state.saved_message = f"Conversation saved to {saved_path.name}"
        st.rerun()

    # A pause in the browser queues a turn through receive_audio. Process it
    # before mounting the next recorder, so it cannot capture during generation.
    if st.session_state.pending_audio is not None:
        try:
            with st.spinner("The family member is responding..."):
                if st.session_state.pending_text is None:
                    recording = st.session_state.pending_audio
                    suffix = ".mp4" if "mp4" in recording["mime_type"] else ".webm"
                    if "ogg" in recording["mime_type"]:
                        suffix = ".ogg"
                    with tempfile.TemporaryDirectory() as temporary_folder:
                        audio_path = Path(temporary_folder) / f"recording{suffix}"
                        audio_path.write_bytes(base64.b64decode(recording["base64"], validate=True))
                        st.session_state.pending_text = transcribe_audio(audio_path)

                transcript = st.session_state.pending_text
                reply = ""
                if transcript:
                    # respond() appends the student message before calling the
                    # model; restore history if generation fails before retrying.
                    history = list(simulation.transcript)
                    try:
                        reply = simulation.respond(transcript, response_mode="voice")
                    except Exception:
                        simulation.transcript[:] = history
                        raise
                else:
                    st.session_state.voice_notice = "No speech was detected. Listening again."

                st.session_state.voice_reply = reply
                st.session_state.pending_audio = None
                st.session_state.pending_text = None
                st.session_state.recording_number += 1
        except Exception as exc:
            st.error(f"Could not process your voice message: {exc}")
            st.button("Retry voice message")
            st.stop()

    # Display history already stored by your backend.
    for message in simulation.transcript:
        role = "user" if message.role == "student" else "assistant"
        with st.chat_message(role):
            st.write(message.content)

    if notice := st.session_state.pop("voice_notice", None):
        st.info(notice)
    st.caption(
        "Speak, then pause for an automatic reply. Listening resumes after the reply. "
        "End and save stops listening and saves completed turns."
    )
    voice_turn(
        key=f"recording_{st.session_state.recording_number}",
        reply=st.session_state.voice_reply,
        pause_ms=st.session_state.voice_pause_ms,
        threshold=st.session_state.voice_threshold,
        on_audio_change=receive_audio,
    )
