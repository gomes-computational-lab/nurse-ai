# Nursing AI Simulation

A local terminal prototype for conversational nursing simulations using Ollama.

The project uses `main.py` for both terminal paths:

- `python3 main.py` starts the original typed conversation flow
- `python3 main.py --voice` starts the local voice flow using microphone input, faster-whisper transcription, Ollama, and spoken patient responses

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

Microphone capture, end-of-speech detection, transcription, and Ollama generation remain local. Edge TTS synthesis uses the network.

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
  models.py             Shared data structures
  ollama_client.py      Local Ollama HTTP client
  scenarios.py          Scenario loading
  speech_to_text.py     Local transcription helpers
  terminal_ui.py        Shared terminal helpers
  text_to_speech.py     Cross-platform TTS helpers
  voice_app.py          Voice demo flow
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
