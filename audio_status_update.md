# Audio Input/Output Status Update

My work was to move the nursing simulation from typed interaction to a local voice loop, so the student can speak and the patient can respond out loud.

To do that, I added a separate voice entry point with `voice_demo.py` and connected four main pieces. For audio input, I used `sounddevice` to record a temporary 16 kHz mono WAV clip from the microphone. For speech-to-text, I used `faster-whisper`, with the default model set to `tiny.en`, running locally on CPU. For the patient response, I reused the existing Ollama conversation pipeline, so the simulation still runs through the same scenario and session logic. For audio output, I used the macOS `say` command so the patient response is spoken back out loud. I also kept the transcript and evaluation saving in JSON, so each run can still be reviewed afterward.

Right now, the prototype works like this: I start it with `python3 voice_demo.py`, it loads the scenario, gets the opening patient line from Ollama, prints it, and speaks it. After that, each turn starts when the user presses Enter. The app records a fixed-length microphone clip, transcribes it locally, passes that text in as the student response, then generates and speaks the next patient reply. If the user types `/end`, the app evaluates the session and saves the transcript plus feedback. If the user types `/quit`, it exits and saves just the transcript.

The main limitation is that this is still a local prototype, not a production voice assistant. Recording is fixed-length and non-streaming. Transcription is fully local and CPU-based, so speed and accuracy depend on the machine and model. Spoken output currently depends on macOS `say`, so other systems fall back to text only. The full loop does work, and the saved runs show end-to-end conversations completing, but response quality still depends a lot on the Ollama model and transcription quality. I also saw some drift in patient behavior, so consistency still needs improvement. The evaluation path is model-generated JSON now, but it still needs more validation before I would treat it as reliable scoring.

## Quick Reference

- Run command: `python3 voice_demo.py`
- Scenario selection: `--scenario`
- Recording length: `--record-seconds` with a default of `5.0`
- Whisper model: `--stt-model`
- Ollama model: `--model`

## Stack Used

- Local Ollama server with an installed chat model
- Python packages: `faster-whisper`, `sounddevice`, `numpy`
- macOS `say` for spoken output

## Current Behavior

- The voice flow is implemented in `voice_demo.py` and `sim/voice_app.py`
- Microphone audio is recorded as a temporary WAV file
- Transcription runs locally through `sim/speech_to_text.py`
- Patient replies are generated through the existing Ollama session flow
- Spoken playback runs through `sim/text_to_speech.py`
- Transcripts and feedback are saved under `transcripts/`

## Evidence Behind This Update

- The voice loop, commands, dependencies, and limitations are documented in `README.md`
- End-to-end transcript artifacts were saved in:
  - `transcripts/20260622_173951_post_op_pain.json`
  - `transcripts/20260625_154423_post_op_pain.json`
- `TESTING_NOTES.md` still reflects manual prototype testing rather than a fully verified system
