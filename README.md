# Nursing AI Simulation

A local terminal prototype for conversational nursing simulations using Ollama.

The project uses `main.py` for both terminal paths:

- `python3 main.py` starts the original typed conversation flow
- `python3 main.py --voice` starts the local voice flow using microphone input, faster-whisper transcription, Ollama, and spoken patient responses

## Requirements

- Python 3.10+
- Ollama running locally or reachable through an SSH tunnel
- At least one chat-capable Ollama model installed
- Python dependencies from `requirements.txt`

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Start Ollama and install a model:

```bash
ollama serve
ollama pull llama3.1
```

## Configuring Ollama

The Ollama model and base URL can be configured with shell environment variables:

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11435
export OLLAMA_MODEL=qwen3:4b
```

Command-line arguments take precedence over environment variables, which take precedence over the built-in defaults:

```bash
python3 main.py --host http://127.0.0.1:11435 --model qwen3:14b
```

The project includes `.env.example` as a template, but it does not parse `.env` files or require an environment-loading dependency. To use a local `.env` file with a compatible shell:

```bash
cp .env.example .env
set -a
source .env
set +a
```

To connect from a Mac to Ollama listening on the loopback interface of `stormbreaker`, use local port `11435` so a local Ollama instance can continue using `11434`:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L 11435:127.0.0.1:11434 \
  <username>@stormbreaker
```

Verify server reachability, model availability, and one minimal inference request:

```bash
python3 main.py --check-llm
```

The check reports the configured URL and model, Ollama version, health-check latency, inference latency, and total elapsed time. If the server, tunnel, or configured model is unavailable, it exits with a targeted diagnostic.

If `faster-whisper` does not install cleanly on Python 3.14, use a Python 3.11 or 3.12 virtual environment for the voice demo.

## Run The Text Simulation

```bash
python3 main.py
```

Use a specific model:

```bash
python3 main.py --model llama3.1
```

List available scenarios:

```bash
python3 main.py --list
```

Run a specific scenario:

```bash
python3 main.py --scenario post_op_pain
```

## AI Family Member Prototype

Scenarios can configure either a `patient` or `family_member` AI role. The Ruth Lawson prototype provides phase-specific family-member behavior and lightweight emotional-state guidance:

```bash
python3 main.py --scenario ruth_family_member --phase 1
python3 main.py --scenario ruth_family_member --phase 2
```

The prototype uses the existing turn-based text and audio pipeline. Continuous room listening, speaker identification, multi-speaker diarization, interruption handling, and automatic turn-taking are intentionally deferred to a later phase.

## Participants, Learners, and Simulation Time

Scenario files can optionally distinguish human, AI, learner, and facilitator participants. In the Ruth Lawson scenario:

- Ruth Lawson is the human patient
- Alex is the AI-controlled family member
- student nurses are learner participants
- the simulation technician/facilitator is a human facilitator

Participant types and roles are scenario data. Declaring Ruth as a participant does not make her an AI-controlled agent; `agent_role` continues to identify which participant the LLM portrays.

Scenarios can also define the allowed learner configuration. Ruth supports either one or two student nurses:

```json
"learner_configuration": {
  "min_nurses": 1,
  "max_nurses": 2,
  "supported_roles": [
    "nurse_primary",
    "nurse_secondary"
  ]
}
```

The typed application asks for the number of nurses before starting a scenario that offers a choice. With two nurses, use these commands to select who is entering text:

- `/primary` selects `nurse_primary`
- `/secondary` selects `nurse_secondary`

The active role appears in the input prompt. Learner transcript messages record the optional `speaker_role`, but no student names or other learner identity information are requested. Voice mode remains a single-speaker workflow and records its learner turns as `nurse_primary`; automatic speaker recognition and diarization are not implemented.

Scenarios may define a fictional clinical timeline independently of the real execution date:

```json
"simulation_timeline": {
  "simulation_date": "2025-01-15",
  "clinical_day": 7,
  "time_of_day": "morning"
}
```

When present, this timeline is authoritative for LLM references to today, yesterday, tomorrow, length of stay, and clinical day. Saved results keep the simulation timeline separate from the real-world `execution_saved_at` timestamp. They also record the selected nurse count and learner roles.

Existing scenario JSON remains compatible. A scenario without these fields defaults to one `nurse_primary`, no explicit participant list, and no fictional timeline.

## Run The Voice Demo

```bash
python3 main.py --voice
```

Run a specific scenario:

```bash
python3 main.py --voice --scenario post_op_pain
```

Change the Ollama model or faster-whisper model:

```bash
python3 main.py --voice --model llama3.1 --stt-model tiny.en
```

Voice demo behavior:

- wait for the patient audio to finish, then press `Enter` to start speaking
- recording and speech-to-text do not start until `Enter` is pressed
- recording stops after 700 ms of trailing silence
- type `/end` and press Enter to evaluate and save the transcript
- type `/quit` and press Enter to exit without evaluation

Use manual recording controls when automatic end-of-speech detection is not a good fit:

```bash
python3 main.py --voice --manual-stop
```

Automatic recording can be tuned with `--end-silence-ms` and `--max-recording-seconds`. It uses 30 ms WebRTC VAD frames, keeps 300 ms of audio before detected speech, waits up to 10 seconds for speech, and limits a turn to 60 seconds by default. If WebRTC VAD cannot initialize, the demo prints a warning and falls back to manual stop.

Patient responses now use a cross-platform TTS path powered by `edge-tts` with local playback through `pygame`. Spoken output requires internet access for synthesis. If the TTS dependencies are missing or speech playback fails, the demo continues with printed output only and shows a one-time warning.

Generated text is sent to TTS incrementally while Ollama is still responding. Synthesis of upcoming segments overlaps current playback, and punctuation is preserved for more natural pacing. Voice responses are limited to one to three short spoken sentences so audio can start sooner.

To reduce the delay before speech, the first TTS segment starts at a natural clause or at roughly 64 generated characters. Later segments remain longer for smoother prosody, and their synthesis overlaps current playback.

Optional TTS environment variables:

- `TTS_VOICE` defaults to `en-US-AriaNeural`
- `TTS_RATE` defaults to `+0%`
- `TTS_FIRST_SEGMENT_CHARS` defaults to `64`; lower values start synthesis sooner but can sound less smooth

Microphone capture, end-of-speech detection, and transcription remain local. Ollama generation uses the configured service, which may be local or reached through an SSH tunnel. Edge TTS synthesis uses the network.

Transcripts and feedback are saved in `transcripts/`. Voice results also include recording, endpoint, transcription, first-text, first-audio, completion, and interruption latency metrics.

### Understanding Voice Metrics

Every new transcript includes three aids under `latency`:

- `metric_definitions` gives the unit and plain-language meaning of every recorded metric
- `summary` reports medians across student turns and counts completed, interrupted, and failed audio responses
- `schema_version` identifies the metric format; the clearer metric names below are version 2

The most useful metrics are:

| Metric | Meaning |
| --- | --- |
| `speech_end_to_first_audio_seconds` | Main conversational latency: from the student's last detected speech until patient audio starts. Lower is better. |
| `speech_end_to_first_tts_segment_seconds` | Time until enough generated text is ready to begin the first TTS request. |
| `first_tts_segment_to_first_audio_seconds` | Edge TTS network synthesis and decoding delay after that first segment is submitted. |
| `speech_end_to_first_text_seconds` | From the student's last detected speech until the first patient text arrives. |
| `speech_end_to_transcript_ready_seconds` | Endpoint detection and speech-to-text time combined. |
| `speech_to_text_processing_seconds` | Time faster-whisper spends transcribing the retained audio. |
| `llm_time_to_first_text_seconds` | Time from sending the Ollama request until its first text chunk. |
| `llm_response_generation_seconds` | Time for Ollama to finish the complete response. |
| `end_of_speech_detection_delay_seconds` | Silence wait before automatic recording stop, normally close to 0.7 seconds. |
| `turn_start_to_text_complete_seconds` | Full turn time from starting recording until all patient text is ready. |
| `audio_status` | `completed`, `interrupted`, or `failed`. |

`captured_audio_duration_seconds` includes pre-roll and trailing silence, while `detected_speech_duration_seconds` counts only frames classified as speech. `llm_prompt_characters` is a prompt-size measurement, not a duration.

## Project Layout

```text
main.py                 Typed and --voice terminal entry point
voice_demo.py           Backward-compatible voice launcher
sim/
  app.py                Typed terminal app flow
  audio.py              Microphone recording helpers
  evaluator.py          Student-response analysis
  models.py             Scenarios, participants, learners, timelines, and messages
  ollama_client.py      Local Ollama HTTP client
  scenarios.py          Scenario loading, typed conversion, and validation
  session.py            Per-run learner selection, prompts, and transcript state
  speech_to_text.py     Local transcription helpers
  storage.py            Transcript, run metadata, and feedback persistence
  terminal_ui.py        Shared terminal helpers
  text_to_speech.py     Cross-platform TTS helpers
  voice_app.py          Voice demo flow
scenarios/
  post_op_pain.json     Sample nursing scenario
  ruth_family_member.json  Two-phase family-member scenario
transcripts/            Generated at runtime
```

## API Path Later

The important pieces are already separated:

- `SimulationSession` can become a request/session service
- `OllamaClient` can be reused by FastAPI
- scenario JSON files can be loaded by an instructor dashboard
- evaluator output is JSON-friendly

## Ollama Tests

The normal test suite mocks the Ollama HTTP endpoint and does not require a running server or SSH tunnel:

```bash
python3 -m unittest discover -s tests -v
```

An optional integration test can be explicitly enabled after establishing the tunnel and exporting `OLLAMA_BASE_URL` and `OLLAMA_MODEL`:

```bash
RUN_OLLAMA_INTEGRATION=1 python3 -m unittest discover -s tests -p 'test_ollama_integration.py' -v
```
