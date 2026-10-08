import tempfile
from pathlib import Path

import streamlit as st

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

    if st.button("Start simulation", type="primary"):
        with st.spinner("Starting..."):
            client = OllamaClient(model="llama3.1")
            simulation = SimulationSession(scenario, client)
            simulation.opening(response_mode="voice")

        st.session_state.simulation = simulation
        st.rerun()

else:
    simulation = st.session_state.simulation

    # Display history already stored by your backend.
    for message in simulation.transcript:
        role = "user" if message.role == "student" else "assistant"
        with st.chat_message(role):
            st.write(message.content)

    recording = st.audio_input(
        "Speak to the family member",
        key=f"recording_{st.session_state.recording_number}",
    )

    if st.button("Send message", disabled=recording is None):
        with st.spinner("The family member is responding..."):
            # Your transcriber accepts a file path.
            with tempfile.TemporaryDirectory() as temporary_folder:
                audio_path = Path(temporary_folder) / "recording.wav"
                audio_path.write_bytes(recording.getvalue())
                transcript = transcribe_audio(audio_path)

            if transcript:
                simulation.respond(transcript, response_mode="voice")
                st.session_state.recording_number += 1
                st.rerun()
            else:
                st.warning("No speech was detected. Please record again.")

    if st.button("End and save"):
        saved_path = save_result(
            simulation.scenario,
            simulation.transcript,
        )
        st.session_state.simulation = None
        st.session_state.recording_number += 1
        st.success(f"Conversation saved to {saved_path.name}")
        st.button("Start another simulation", on_click=st.rerun)

