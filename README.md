# Nursing AI Simulation

A local terminal prototype for conversational nursing simulations using Ollama.

The project now includes two terminal paths:

- `main.py` for the original typed conversation flow
- `voice_demo.py` for a local voice loop using microphone input, faster-whisper transcription, Ollama, and spoken patient responses

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
python3 voice_demo.py
```

Run a specific scenario with a shorter recording window:

```bash
python3 voice_demo.py --scenario post_op_pain --record-seconds 4
```

Change the Ollama model or faster-whisper model:

```bash
python3 voice_demo.py --model llama3.1 --stt-model tiny.en
```

Voice demo behavior:

- press Enter to record a fixed-length microphone clip
- type `/end` and press Enter to evaluate and save the transcript
- type `/quit` and press Enter to exit without evaluation

On macOS, patient responses are spoken with the built-in `say` command. On other systems, the demo continues with printed output only and shows a one-time TTS warning.

This is an early prototype. Recording is fixed-length, fully local, and non-streaming.

Transcripts and feedback are saved in `transcripts/`.

## Project Layout

```text
main.py                 Typed terminal entry point
voice_demo.py           Voice demo entry point
sim/
  app.py                Typed terminal app flow
  audio.py              Microphone recording helpers
  evaluator.py          Student-response analysis
  models.py             Shared data structures
  ollama_client.py      Local Ollama HTTP client
  scenarios.py          Scenario loading
  speech_to_text.py     Local transcription helpers
  terminal_ui.py        Shared terminal helpers
  text_to_speech.py     Local TTS helpers
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
