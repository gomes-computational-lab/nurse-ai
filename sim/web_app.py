from __future__ import annotations

import os
from typing import Any, Callable

import streamlit as st

from sim.browser_recorder import (
    automatic_silence_recorder,
    ensure_browser_recorder_registered,
)
from sim.evaluator import evaluate_transcript
from sim.ollama_client import OllamaClient, OllamaError
from sim.scenarios import list_scenarios
from sim.session import SimulationSession
from sim.speech_to_text import (
    DEFAULT_STT_MODEL,
    SpeechToTextError,
    transcribe_audio_bytes,
)
from sim.storage import save_result
from sim.text_to_speech import TextToSpeechError, synthesize_speech_bytes


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
DEFAULT_HOST = "http://localhost:11434"
STATE_DEFAULTS: dict[str, Any] = {
    "simulation_session": None,
    "simulation_client": None,
    "simulation_scenario": None,
    "simulation_config": None,
    "patient_audio": {},
    "draft_text": "",
    "composer_version": 0,
    "recorder_version": 0,
    "processed_recording_turn": None,
    "feedback": None,
    "saved_path": None,
    "simulation_ended": False,
    "tts_warning": None,
}


def main() -> None:
    st.set_page_config(page_title="Nursing AI Simulation", page_icon="🩺", layout="centered")
    ensure_browser_recorder_registered()
    _initialize_state()

    st.title("Nursing AI Simulation")
    st.caption("Practice a patient conversation with typed or recorded responses.")

    config = _render_sidebar()
    session: SimulationSession | None = st.session_state.simulation_session

    if session is None:
        st.info("Choose a scenario and start a simulation.")
        _render_scenario_preview(config["scenario"])
        return

    scenario = st.session_state.simulation_scenario
    st.subheader(scenario.title)
    st.caption(f"{scenario.setting} · Patient role: {scenario.role}")
    _render_transcript()

    if st.session_state.tts_warning:
        st.warning(st.session_state.tts_warning)

    if st.session_state.simulation_ended:
        _render_feedback()
        return

    _render_composer()


def _initialize_state() -> None:
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, dict) else value


def _render_sidebar() -> dict[str, Any]:
    scenarios = list_scenarios()
    if not scenarios:
        st.error("No scenarios are available.")
        st.stop()

    active = st.session_state.simulation_session is not None
    with st.sidebar:
        st.header("Simulation setup")
        scenario = st.selectbox(
            "Scenario",
            scenarios,
            format_func=lambda item: item.title,
            disabled=active,
        )
        model = st.text_input("Ollama model", value=DEFAULT_MODEL, disabled=active)
        host = st.text_input("Ollama host", value=DEFAULT_HOST, disabled=active)
        stt_model = st.text_input(
            "Whisper model",
            value=DEFAULT_STT_MODEL,
            disabled=active,
        )
        patient_audio = st.toggle("Patient browser audio", value=True, disabled=active)

        config = {
            "scenario": scenario,
            "model": model.strip(),
            "host": host.strip(),
            "stt_model": stt_model.strip(),
            "patient_audio": patient_audio,
        }

        if not active:
            if st.button("Start simulation", type="primary", width="stretch"):
                if _start_simulation(config):
                    st.rerun()
        else:
            if st.button("End simulation", type="primary", width="stretch"):
                if _end_simulation():
                    st.rerun()
            if st.button("New simulation", width="stretch"):
                _reset_simulation()
                st.rerun()

        if st.session_state.saved_path:
            st.caption(f"Saved to `{st.session_state.saved_path}`")

    return config


def _render_scenario_preview(scenario) -> None:
    st.subheader(scenario.title)
    st.write(scenario.setting)
    with st.expander("Learning objectives"):
        for objective in scenario.learning_objectives:
            st.markdown(f"- {objective}")


def _start_simulation(config: dict[str, Any]) -> bool:
    scenario = config["scenario"]
    client = OllamaClient(model=config["model"], host=config["host"])
    session = SimulationSession(scenario, client)

    st.session_state.simulation_client = client
    st.session_state.simulation_session = session
    st.session_state.simulation_scenario = scenario
    st.session_state.simulation_config = {
        key: value for key, value in config.items() if key != "scenario"
    }

    try:
        with st.spinner("Starting the simulation..."):
            response = _stream_patient_response(
                lambda on_chunk: session.opening(on_chunk, response_mode="voice")
            )
        _add_patient_audio(len(session.transcript) - 1, response)
        return True
    except OllamaError as exc:
        _reset_simulation()
        st.error(str(exc))
        return False


def _render_transcript() -> None:
    session: SimulationSession = st.session_state.simulation_session
    audio_by_message: dict[int, bytes] = st.session_state.patient_audio
    for index, message in enumerate(session.transcript):
        chat_role = "user" if message.role == "student" else "assistant"
        avatar = "🧑‍⚕️" if message.role == "student" else "🧑"
        with st.chat_message(chat_role, avatar=avatar):
            st.markdown(message.content)
            audio = audio_by_message.get(index)
            if audio:
                st.audio(
                    audio,
                    format="audio/mpeg",
                    autoplay=False,
                )

def _render_composer() -> None:
    st.divider()
    st.markdown("#### Your response")

    session: SimulationSession = st.session_state.simulation_session
    patient_message_index = len(session.transcript) - 1
    turn_id = f"{patient_message_index}:{st.session_state.recorder_version}"
    patient_audio = st.session_state.patient_audio.get(patient_message_index)
    already_processed = st.session_state.processed_recording_turn == turn_id
    recording, recorder_error = automatic_silence_recorder(
        turn_id=turn_id,
        patient_audio=patient_audio,
        key=f"student_recorder_{turn_id}",
        active=not already_processed,
    )
    if recorder_error:
        st.warning(recorder_error)

    if recording is not None and not already_processed:
        st.session_state.processed_recording_turn = turn_id
        try:
            with st.spinner("Transcribing..."):
                transcript = transcribe_audio_bytes(
                    recording.audio,
                    model_name=st.session_state.simulation_config["stt_model"],
                    suffix=recording.file_suffix,
                )
            if transcript:
                st.session_state.draft_text = transcript
                st.session_state.composer_version += 1
                st.rerun()
            else:
                st.warning("No speech was detected.")
                _render_record_again(turn_id)
        except SpeechToTextError as exc:
            st.error(str(exc))
            _render_record_again(turn_id)

    draft = st.text_area(
        "Review or type your response",
        value=st.session_state.draft_text,
        key=f"student_draft_{st.session_state.composer_version}",
        placeholder="Your recording is transcribed here automatically, or you can type a response.",
    )
    if st.button("Send response", type="primary", disabled=not draft.strip()):
        _submit_student_response(draft.strip())
        st.session_state.draft_text = ""
        st.session_state.composer_version += 1
        st.session_state.recorder_version += 1
        st.session_state.processed_recording_turn = None
        st.rerun()


def _render_record_again(turn_id: str) -> None:
    if st.button("Record again", key=f"record_again_{turn_id}"):
        st.session_state.recorder_version += 1
        st.session_state.processed_recording_turn = None
        st.rerun()


def _submit_student_response(student_text: str) -> None:
    session: SimulationSession = st.session_state.simulation_session
    try:
        with st.spinner("Patient is responding..."):
            response = _stream_patient_response(
                lambda on_chunk: session.respond(
                    student_text,
                    on_chunk,
                    response_mode="voice",
                )
            )
        _add_patient_audio(len(session.transcript) - 1, response)
    except OllamaError as exc:
        st.error(str(exc))


def _stream_patient_response(
    generate: Callable[[Callable[[str], None]], str],
) -> str:
    placeholder = st.empty()
    chunks: list[str] = []

    def on_chunk(chunk: str) -> None:
        chunks.append(chunk)
        placeholder.markdown("".join(chunks) + " ▌")

    response = generate(on_chunk)
    placeholder.markdown(response)
    return response


def _add_patient_audio(message_index: int, response: str) -> None:
    if not st.session_state.simulation_config["patient_audio"]:
        return
    try:
        with st.spinner("Preparing patient audio..."):
            audio = synthesize_speech_bytes(response)
        if audio:
            st.session_state.patient_audio[message_index] = audio
            st.session_state.tts_warning = None
    except TextToSpeechError as exc:
        st.session_state.tts_warning = f"{exc} Continuing with text only."


def _end_simulation() -> bool:
    if st.session_state.simulation_ended:
        return True
    session: SimulationSession = st.session_state.simulation_session
    scenario = st.session_state.simulation_scenario
    client: OllamaClient = st.session_state.simulation_client
    try:
        with st.spinner("Evaluating student performance..."):
            feedback = evaluate_transcript(scenario, session.transcript, client)
        path = save_result(scenario, session.transcript, feedback)
        st.session_state.feedback = feedback
        st.session_state.saved_path = str(path)
        st.session_state.simulation_ended = True
        return True
    except OllamaError as exc:
        st.error(str(exc))
        return False


def _render_feedback() -> None:
    feedback = st.session_state.feedback
    if not feedback:
        return

    st.divider()
    st.header("Feedback")
    score = feedback.get("overall_score")
    st.metric("Overall score", f"{score}/5" if score is not None else "Unavailable")
    st.write(feedback.get("summary", ""))

    for name, criterion in feedback.get("criteria", {}).items():
        with st.expander(f"{name}: {criterion.get('score', '—')}/5"):
            st.markdown(f"**Evidence:** {criterion.get('evidence', '')}")
            st.markdown(f"**Coaching:** {criterion.get('coaching', '')}")

    _render_feedback_list("Strengths", feedback.get("strengths", []))
    _render_feedback_list("Improvements", feedback.get("improvements", []))
    _render_feedback_list("Safety concerns", feedback.get("safety_concerns", []))


def _render_feedback_list(title: str, items: list[str]) -> None:
    if not items:
        return
    st.subheader(title)
    for item in items:
        st.markdown(f"- {item}")


def _reset_simulation() -> None:
    for key, value in STATE_DEFAULTS.items():
        st.session_state[key] = value.copy() if isinstance(value, dict) else value
