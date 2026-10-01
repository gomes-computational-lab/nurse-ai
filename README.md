# Nursing AI Simulation

A local terminal prototype for conversational nursing simulations using Ollama.

The project uses `main.py` for both terminal paths:

- `python3 main.py` starts the original typed conversation flow
- `python3 main.py --voice` starts the local voice flow using microphone input, faster-whisper transcription, Ollama, and spoken patient responses

It also includes `streamlit_app.py`, a browser interface with typed input, browser microphone
capture, editable transcription, and optional patient audio playback.

## Requirements

- Python 3.10+
- Ollama running locally
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

If `faster-whisper` does not install cleanly on Python 3.14, use a Python 3.11 or 3.12 virtual environment for the voice demo.

### Local patient voice setup

Patient speech is local-first and uses only permissively licensed components:

- **Chatterbox Nano (MIT)** is the current patient-voice engine. It runs on the local machine hosting Streamlit. Install it separately because its PyTorch dependencies are large:

```bash
python3 -m pip install -r requirements-tts.txt
```

Chatterbox downloads its model during first setup and can run offline afterward. Add only licensed or explicitly consented reference recordings to `voices/`; the filename becomes the selectable voice ID. No reference recordings are committed with this project.
Use Python 3.11 for the optional Chatterbox environment, matching the upstream project's tested configuration.

#### Cross-platform local installation

“Local machine” means the Windows, Linux, or macOS computer running Streamlit. Create the
same Python 3.11 Conda environment on any supported platform:

```bash
conda create -n nursing-ai-sim python=3.11 -y
conda activate nursing-ai-sim
python -m pip install -r requirements.txt
python -m pip install -r requirements-tts.txt
```

FFmpeg is only required to generate or convert reference voices. Install it using the option
for the local operating system:

| Platform | Command |
| --- | --- |
| macOS with Homebrew | `brew install ffmpeg` |
| Ubuntu or Debian | `sudo apt update && sudo apt install ffmpeg` |
| Windows PowerShell with WinGet | `winget install --id Gyan.FFmpeg -e` |
| Any platform with Conda | `conda install -n nursing-ai-sim -c conda-forge ffmpeg` |

After installation, verify the active environment before launching the app:

```bash
python --version
ffmpeg -version
python -c "import chatterbox, faster_whisper, streamlit, torch; print('Local speech dependencies are ready')"
streamlit run streamlit_app.py
```

All project paths are repository-relative. Run these commands from the project root; no
macOS-specific `/Users/...` path is required.

The browser interface always uses Chatterbox Nano with online fallback disabled. The ZONOS2
backend is retained only for a possible future university deployment and is not shown in the
browser interface. The legacy Edge provider remains available to terminal tools for compatibility,
but it is online and is never an automatic fallback.

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

## Run The Browser App

Start Ollama, then launch the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

Choose the scenario, whether the patient should speak aloud, and an approved patient voice in
the sidebar. Ruth's child voice and Chatterbox Nano are selected automatically. Ollama,
transcription, local addresses, and voice-engine settings are managed internally and are not
shown to learners. Use **Load voice and recording** to prepare transcription, the patient voice
model, and Ruth's voice before starting. After starting the simulation, the browser automatically
starts recording
when the patient finishes speaking. The nurse has five seconds to begin; after speech is
detected, three seconds of silence stops the recording. Recorded responses are transcribed
into an editable draft; you can also type a response directly.
Review the text and select **Send response** when it is ready.

The browser will ask for microphone permission on the first turn. Browser capture uses
the browser's supported Opus audio format for speech recognition. Local patient audio is
synthesized as WAV and played by the browser. Browser autoplay policies may require you to
press the player's play button. If transcription, Ollama, or
TTS fails, the page reports the error without crashing the active simulation.

Use **End and score simulation** to generate feedback and save the result, **End without
scoring** to save only the transcript, or **New simulation** to clear the current browser
session and choose new settings.

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

Select the local patient voice provider:

```bash
python3 main.py --voice --tts-provider zonos2 --zonos2-url http://10.0.0.20:1919 --tts-voice patient-a
python3 main.py --voice --tts-provider chatterbox_nano --tts-voice patient-a
```

For ZONOS2, the application reads `/tts/speakers` and accepts only voices preloaded in the
server's approved default-voice directory. The configured voice can match the server speaker
ID, label, or reference filename stem; `default` selects the first approved server voice.
The expressive request uses the full `/tts/generate` contract and validates that the response
is mono float32 PCM before converting it to WAV.

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

Ollama streams patient wording to the screen immediately, while a hidden validated delivery header carries emotion, intensity, and pace. The header is removed from displayed text, spoken wording, conversation context, and evaluator input. The complete one-to-three-sentence response is then synthesized as one utterance for smoother prosody.

Optional TTS environment variables:

- `LOCAL_TTS_PROVIDER` is `zonos2`, `chatterbox_nano`, or `edge`; default `zonos2`
- `LOCAL_TTS_VOICE` defaults to `default`
- `ZONOS2_URL` defaults to `http://localhost:1919`
- `TTS_VOICE_CATALOG` defaults to `voices`
- `TTS_ALLOWED_HOSTS` authorizes comma-separated university hostnames in addition to localhost, private IPs, and `.local` hosts
- `ALLOW_ONLINE_EDGE_TTS=1` explicitly permits online Edge fallback
- `TTS_VOICE` and `TTS_RATE` configure the legacy Edge voice

Microphone capture, endpoint detection, transcription, Ollama, and local TTS remain on university-controlled hardware. The UI clearly discloses that patient speech is AI-generated and reports the provider or fallback used.

Benchmark either local provider and create numbered files for a blinded listening review:

```bash
python3 -m scripts.benchmark_tts --provider zonos2 --voice patient-a --zonos2-url http://10.0.0.20:1919
python3 -m scripts.benchmark_tts --provider chatterbox_nano --voice patient-a
```

Results are written under `output/tts_benchmark/` with first-response latency, subsequent median latency, real-time factor, peak memory, anonymous sample filenames, and a separate listening key. Reject any voice mapping that sounds theatrical, introduces words, or obscures clinical information.

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
| `speech_end_to_first_tts_segment_seconds` | Time until the complete patient utterance is ready for its TTS request. |
| `first_tts_segment_to_first_audio_seconds` | Local synthesis and audio decoding delay after the utterance is submitted. |
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
streamlit_app.py        Browser simulation entry point
sim/
  app.py                Typed terminal app flow
  audio.py              Microphone recording helpers
  evaluator.py          Student-response analysis
  expressive_tts.py     Local TTS providers, fallback policy, and emotion mappings
  models.py             Shared data structures
  ollama_client.py      Local Ollama HTTP client
  scenarios.py          Scenario loading
  speech_to_text.py     Local transcription helpers
  terminal_ui.py        Shared terminal helpers
  text_to_speech.py     Cross-platform TTS helpers
  voice_app.py          Voice demo flow
  voice_delivery.py     Hidden delivery metadata parsing and validation
  web_app.py            Streamlit browser flow
voices/                 Approved local voice reference catalog
scenarios/
  post_op_pain.json     Sample nursing scenario
transcripts/            Generated at runtime
```

## API Path Later

The important pieces are already separated:

- `SimulationSession` can become a request/session service
- `OllamaClient` can be reused by FastAPI
- scenario JSON files can be loaded by an instructor dashboard
- evaluator output is JSON-friendly
