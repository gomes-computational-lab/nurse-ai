<<<<<<< HEAD
# nurse-ai
NURSE-AI (Nursing User-centered Realistic Simulation with Emotion-Aware AI) is an agentic multi-agent simulation platform for immersive nursing education, integrating local LLMs, clinical physiology models, and AI-driven evaluation to support realistic communication, decision-making, and patient-centered training.
=======
# Nursing AI Simulation

A local terminal prototype for conversational nursing simulations using Ollama.

The current version runs entirely in the terminal:

- a simulated patient or family member responds to the student
- the conversation is saved as a transcript
- an evaluator model scores the student's performance using a nursing rubric

The code is organized so the same simulation engine can later be used from an API.

## Requirements

- Python 3.10+
- Ollama running locally
- At least one chat-capable Ollama model installed

Example:

```bash
ollama serve
ollama pull llama3.1
```

## Run

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

## During a Simulation

Type your nursing-student response and press Enter.

Commands:

- `/help` shows commands
- `/end` ends the simulation and generates feedback
- `/quit` exits without feedback

Transcripts and feedback are saved in `transcripts/`.

## Project Layout

```text
main.py                 Terminal entry point
sim/
  app.py                Terminal app flow
  evaluator.py          Student-response analysis
  models.py             Shared data structures
  ollama_client.py      Local Ollama HTTP client
  scenarios.py          Scenario loading
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
>>>>>>> 1e77b8c (Initial nursing AI simulation)
