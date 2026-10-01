from __future__ import annotations

import os
from typing import Any, Callable

import streamlit as st

from sim.browser_recorder import (
    automatic_silence_recorder,
    ensure_browser_recorder_registered,
)
from sim.evaluator import evaluate_transcript
from sim.expressive_tts import (
    DEFAULT_ZONOS2_URL,
    LocalTTSError,
    SynthesizedAudio,
    TTSConfig,
    TTSService,
    list_approved_voices,
    preload_chatterbox_runtime,
    prepare_chatterbox_voice,
)
from sim.ollama_client import OllamaClient, OllamaError
from sim.scenarios import list_scenarios
from sim.session import SimulationSession
from sim.speech_to_text import (
    DEFAULT_STT_MODEL,
    SpeechToTextError,
    preload_speech_to_text_model,
    transcribe_audio_bytes,
)
from sim.storage import save_result
from sim.text_to_speech import TextToSpeechError, synthesize_patient_audio


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_PATIENT_VOICE = "child_female_8yo"
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
    "scoring_skipped": False,
    "tts_warning": None,
    "tts_provider_status": None,
    "prepared_speech_models": None,
}


def main() -> None:
    st.set_page_config(
        page_title="Nursing AI Simulation", page_icon="🩺", layout="centered"
    )
    ensure_browser_recorder_registered()
    _initialize_state()

    st.title("Nursing AI Simulation")
    st.caption("Practice a patient conversation with typed or recorded responses.")

    config = _render_sidebar()
    session: SimulationSession | None = st.session_state.simulation_session

    if session is None:
        st.info("Choose a scenario and start a simulation.")
        _render_getting_started()
        _render_scenario_preview(config["scenario"])
        return

    scenario = st.session_state.simulation_scenario
    st.subheader(scenario.title)
    st.caption(f"{scenario.setting} · Patient role: {scenario.role}")
    _render_transcript()

    if st.session_state.tts_warning:
        st.warning(st.session_state.tts_warning)
    if st.session_state.tts_provider_status:
        st.caption(st.session_state.tts_provider_status)

    if st.session_state.simulation_ended:
        if st.session_state.scoring_skipped:
            st.info(
                "Simulation ended without scoring. The transcript was saved, but no "
                "performance evaluation was generated.",
                icon=":material/check_circle:",
            )
        else:
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
        st.header("Start a simulation")
        scenario = st.selectbox(
            "Scenario",
            scenarios,
            format_func=lambda item: item.title,
            disabled=active,
        )
        patient_audio = st.toggle(
            "Patient speaks aloud",
            value=True,
            disabled=active,
            help="Turn this off to use text-only patient responses.",
        )
        catalog_dir = os.environ.get("TTS_VOICE_CATALOG", "voices")
        approved_voices = list_approved_voices(catalog_dir)
        voice_options = [
            DEFAULT_PATIENT_VOICE,
            *(voice for voice in approved_voices if voice != DEFAULT_PATIENT_VOICE),
        ]
        voice = DEFAULT_PATIENT_VOICE
        if patient_audio:
            voice = st.selectbox(
                "Patient voice",
                voice_options,
                format_func=_format_voice_name,
                disabled=active,
                help="Only approved voices stored on this computer are available.",
            )

        voice_ready = voice in approved_voices
        if patient_audio and not voice_ready:
            st.warning(
                "Ruth's voice is not installed on this computer. Ask the simulation "
                "administrator to complete the local voice setup, or turn off patient speech."
            )

        with st.expander("Before you begin", icon=":material/help:"):
            st.markdown(
                "1. Choose a scenario.\n"
                "2. Leave **Patient speaks aloud** on to hear Ruth.\n"
                "3. Select **Load voice and recording**.\n"
                "4. Select **Start simulation** and allow microphone access when asked."
            )

        config = {
            "scenario": scenario,
            "model": DEFAULT_MODEL,
            "host": DEFAULT_HOST,
            "stt_model": DEFAULT_STT_MODEL,
            "patient_audio": patient_audio,
            "tts_provider": "chatterbox_nano",
            "tts_voice": voice,
            "zonos2_url": DEFAULT_ZONOS2_URL,
            "voice_catalog_dir": catalog_dir,
            "allow_online_edge_fallback": False,
        }

        if not active:
            preload_key = _speech_model_preload_key(config)
            speech_ready = st.session_state.prepared_speech_models == preload_key
            if st.button(
                "Load voice and recording",
                width="stretch",
                disabled=speech_ready or (patient_audio and not voice_ready),
                help=(
                    "Loads transcription and the local patient voice before the simulation starts."
                    if patient_audio
                    else "Loads transcription before the simulation starts."
                ),
            ):
                if _ensure_speech_models_ready(config):
                    st.rerun()
            if speech_ready:
                st.caption(":green[●] Voice and recording ready")

            if st.button(
                "Start simulation",
                type="primary",
                width="stretch",
                disabled=patient_audio and not voice_ready,
            ):
                if _ensure_speech_models_ready(config) and _start_simulation(config):
                    st.rerun()
        else:
            if st.button("End and score simulation", type="primary", width="stretch"):
                if _end_simulation():
                    st.rerun()
            if st.button("End without scoring", width="stretch"):
                if _end_simulation_without_scoring():
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


def _render_getting_started() -> None:
    with st.container(border=True):
        st.subheader("How it works")
        st.markdown(
            "1. **Choose a scenario** and select **Start simulation**.\n"
            "2. **Listen to Ruth.** Recording begins automatically when she finishes.\n"
            "3. **Begin speaking within five seconds.** When you finish, stay quiet for three seconds.\n"
            "4. **Review your response** and select **Send response**."
        )
        st.caption(
            "Ruth's voice is AI-generated and runs locally. You can always type or edit your response."
        )


def _format_voice_name(voice: str) -> str:
    if voice == DEFAULT_PATIENT_VOICE:
        return "Ruth — child voice"
    return voice.replace("_", " ").replace("-", " ").title()


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


def _speech_model_preload_key(
    config: dict[str, Any],
) -> tuple[str, bool, str, str]:
    include_chatterbox = bool(config["patient_audio"])
    voice = config["tts_voice"] if include_chatterbox else ""
    catalog_dir = config["voice_catalog_dir"] if include_chatterbox else ""
    return config["stt_model"], include_chatterbox, voice, catalog_dir


def _ensure_speech_models_ready(config: dict[str, Any]) -> bool:
    preload_key = _speech_model_preload_key(config)
    if st.session_state.prepared_speech_models == preload_key:
        return True

    stt_model, include_chatterbox, voice, catalog_dir = preload_key
    description = "voice and recording" if include_chatterbox else "recording"
    progress = st.status(f"Getting {description} ready...", expanded=True)
    try:
        _preload_local_speech_models(
            stt_model=stt_model,
            include_chatterbox=include_chatterbox,
            chatterbox_voice=voice,
            voice_catalog_dir=catalog_dir,
            _on_stage=progress.write,
        )
    except (SpeechToTextError, LocalTTSError) as exc:
        progress.update(
            label=f"Could not prepare {description}", state="error", expanded=True
        )
        st.error(f"Could not prepare the simulation: {exc}")
        return False

    progress.write("Ready")
    progress.update(
        label=f"{description.capitalize()} ready", state="complete", expanded=False
    )
    st.session_state.prepared_speech_models = preload_key
    return True


def _preload_local_speech_models(
    *,
    stt_model: str,
    include_chatterbox: bool,
    chatterbox_voice: str,
    voice_catalog_dir: str,
    _on_stage: Callable[[str], None] | None = None,
) -> bool:
    report = _on_stage or (lambda stage: None)
    report("Loading transcription model")
    _preload_speech_to_text_resource(stt_model)
    if include_chatterbox:
        report("Loading patient voice")
        _preload_chatterbox_runtime_resource()
        voice_name = _format_voice_name(chatterbox_voice).split(" — ", 1)[0]
        report(f"Preparing {voice_name}’s voice")
        _prepare_chatterbox_voice_resource(chatterbox_voice, voice_catalog_dir)
    return True


@st.cache_resource(max_entries=2, show_spinner=False)
def _preload_speech_to_text_resource(model_name: str) -> bool:
    preload_speech_to_text_model(model_name=model_name)
    return True


@st.cache_resource(max_entries=1, show_spinner=False)
def _preload_chatterbox_runtime_resource() -> bool:
    preload_chatterbox_runtime()
    return True


@st.cache_resource(max_entries=8, show_spinner=False)
def _prepare_chatterbox_voice_resource(voice: str, voice_catalog_dir: str) -> bool:
    prepare_chatterbox_voice(voice=voice, voice_catalog_dir=voice_catalog_dir)
    return True


def _render_transcript() -> None:
    session: SimulationSession = st.session_state.simulation_session
    audio_by_message: dict[int, SynthesizedAudio] = st.session_state.patient_audio
    for index, message in enumerate(session.transcript):
        chat_role = "user" if message.role == "student" else "assistant"
        avatar = "🧑‍⚕️" if message.role == "student" else "🧑"
        with st.chat_message(chat_role, avatar=avatar):
            st.markdown(message.content)
            audio = audio_by_message.get(index)
            if audio:
                st.audio(
                    audio.data,
                    format=audio.mime_type,
                    autoplay=False,
                )


def _render_composer() -> None:
    st.divider()
    st.markdown("#### Your response")

    session: SimulationSession = st.session_state.simulation_session
    patient_audio_enabled = st.session_state.simulation_config["patient_audio"]
    if patient_audio_enabled:
        st.info(
            "Wait for the patient to finish speaking. Recording starts automatically; "
            "begin within five seconds, then remain quiet for three seconds when finished.",
            icon=":material/mic:",
        )
    else:
        st.info(
            "Recording starts automatically when the patient's text response is ready; "
            "begin within five seconds, then remain quiet for three seconds when finished.",
            icon=":material/mic:",
        )

    with st.expander(
        "Voice controls and troubleshooting",
        expanded=len(session.transcript) == 1,
        icon=":material/info:",
    ):
        st.markdown(
            "- Allow microphone access when prompted.\n"
            "- A pulsing red dot means recording is active.\n"
            "- Begin speaking within five seconds after the patient finishes.\n"
            "- After speech begins, pauses shorter than three seconds are okay.\n"
            "- Three seconds of silence ends recording and starts transcription automatically.\n"
            "- Edit the transcription if needed, then select **Send response**.\n"
            "- If autoplay is blocked, play the patient audio above and select **Start recording**."
        )

    patient_message_index = len(session.transcript) - 1
    turn_id = f"{patient_message_index}:{st.session_state.recorder_version}"
    patient_audio = st.session_state.patient_audio.get(patient_message_index)
    already_processed = st.session_state.processed_recording_turn == turn_id
    recording, recorder_error = automatic_silence_recorder(
        turn_id=turn_id,
        patient_audio=patient_audio.data if patient_audio else None,
        patient_audio_mime_type=patient_audio.mime_type
        if patient_audio
        else "audio/mpeg",
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
        if _submit_student_response(draft.strip()):
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


def _submit_student_response(student_text: str) -> bool:
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
        return True
    except OllamaError as exc:
        st.session_state.draft_text = student_text
        st.error(str(exc))
        return False


def _stream_patient_response(
    generate: Callable[[Callable[[str], None]], str],
) -> str:
    placeholder = st.empty()
    chunks: list[str] = []

    def on_chunk(chunk: str) -> None:
        chunks.append(chunk)
        placeholder.markdown("".join(chunks) + " ▌")

    try:
        response = generate(on_chunk)
    except Exception:
        placeholder.empty()
        raise
    placeholder.markdown(response)
    return response


def _add_patient_audio(message_index: int, response: str) -> None:
    if not st.session_state.simulation_config["patient_audio"]:
        return
    try:
        with st.spinner("Preparing patient audio..."):
            config = st.session_state.simulation_config
            service = _get_tts_service(
                config["tts_provider"],
                config["tts_voice"],
                config["zonos2_url"],
                config["voice_catalog_dir"],
                config["allow_online_edge_fallback"],
            )
            session: SimulationSession = st.session_state.simulation_session
            audio = synthesize_patient_audio(
                response,
                delivery=session.transcript[message_index].delivery,
                service=service,
            )
        if audio.data:
            st.session_state.patient_audio[message_index] = audio
            st.session_state.tts_warning = None
            voice_name = _format_voice_name(config["tts_voice"])
            st.session_state.tts_provider_status = (
                f"Patient voice: {voice_name} · generated locally"
            )
    except TextToSpeechError as exc:
        st.session_state.tts_warning = f"{exc} Continuing with text only."


@st.cache_resource(max_entries=8)
def _get_tts_service(
    provider: str,
    voice: str,
    zonos2_url: str,
    voice_catalog_dir: str,
    allow_online_edge_fallback: bool,
) -> TTSService:
    return TTSService(
        TTSConfig(
            provider=provider,  # type: ignore[arg-type]
            voice=voice,
            zonos2_url=zonos2_url,
            voice_catalog_dir=voice_catalog_dir,
            allow_online_edge_fallback=allow_online_edge_fallback,
        )
    )


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
        st.session_state.scoring_skipped = False
        return True
    except OllamaError as exc:
        st.error(str(exc))
        return False


def _end_simulation_without_scoring() -> bool:
    if st.session_state.simulation_ended:
        return True
    session: SimulationSession = st.session_state.simulation_session
    scenario = st.session_state.simulation_scenario
    try:
        path = save_result(scenario, session.transcript)
    except OSError as exc:
        st.error(f"Could not save the transcript: {exc}")
        return False
    st.session_state.feedback = None
    st.session_state.saved_path = str(path)
    st.session_state.simulation_ended = True
    st.session_state.scoring_skipped = True
    return True


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
