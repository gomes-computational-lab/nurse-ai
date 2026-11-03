# Live two-speaker diarization and STT

`python main.py --diarization` prints timestamps, speaker IDs, and local Whisper text
from a microphone. The implementation lives in `diarization_demo.py`,
which can also be run directly. Pyannote Community-1 supplies speaker turns.
Session voice profiles automatically match IDs across windows. This standalone
experiment does not call Ollama or decide when an LLM should speak.

## Installation

Use Python 3.12 in a separate environment from the simulation app. From the repository
root, install the pinned dependencies.

### Windows with NVIDIA CUDA (PowerShell)

```powershell
py -3.12 -m venv .venv\diarization
.\.venv\diarization\Scripts\python.exe -m pip install --upgrade pip
.\.venv\diarization\Scripts\python.exe -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\diarization\Scripts\python.exe -m pip install -r .\demos\diarization\requirements.txt
```

CUDA has been used locally on an RTX 5060 and needs a compatible NVIDIA driver.
For CPU use, omit the CUDA install command and install the requirements.

### Apple Silicon Mac (M1 and newer)

Use native ARM64 Python 3.12 rather than Python running under Rosetta:

```bash
python3.12 -m venv .venv/diarization
source .venv/diarization/bin/activate
python -m pip install --upgrade pip
python -m pip install -r demos/diarization/requirements.txt
```

Dependency resolution has been checked for macOS ARM64/Python 3.12. Installation,
microphone capture, and model execution still need testing on actual Mac hardware.
The code selects CPU on macOS; Apple's MPS GPU backend is not used. Live processing
may fall behind. Intel Macs cannot use this pinned stack because PyTorch 2.9.1 has
no macOS x86_64 wheel; they need a separately tested setup.

Below, `python` assumes the demo environment is activated. On Windows, replace it
with `.\.venv\diarization\Scripts\python.exe` if it is not activated.

### Model access

Accept the conditions for
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
using your own Hugging Face account. Set a read token with model access as `HF_TOKEN`
in the environment, or add it to a local `.env` in the repository root:

```dotenv
HF_TOKEN=hf_YOUR_READ_TOKEN
```

An environment value takes precedence over `.env`. Do not commit tokens. The first
run downloads Pyannote and Whisper weights and needs internet; later runs can reuse
cached weights. Pyannote metrics are disabled by default. Microphone audio is passed
as tensors and does not require FFmpeg.

## Microphone and live output

List inputs and check the selected input before loading models:

```bash
python main.py --diarization --list-devices
python main.py --diarization --check-mic
```

Add `--device ID` to select an input from the list; IDs vary by computer. On macOS,
allow microphone access for Terminal or the app hosting the terminal. The check
runs for ten seconds without models or a token. Nearly silent levels indicate an
input, mute, permission, or positioning problem.

For timestamps, tracked speaker IDs, and text:

```bash
python main.py --diarization --window 15 --step 3
```

Illustrative output:

```text
[00:02.100-00:05.700] SPEAKER_00: Good morning. How are you feeling?
[00:06.000-00:08.200] SPEAKER_01: I feel better today.
```

`--window 15` analyzes the latest fifteen seconds. `--step 3` waits for at least
three seconds of additional captured audio between snapshots. The first result
waits for a full window; processing adds delay. The CLI default window is six
seconds, so specify fifteen explicitly to reproduce these commands. Capture
continues during inference; losing audio when processing falls behind produces a
gap warning. Raw microphone audio is not saved.

The demo automatically learns up to two voice profiles for the current run. IDs
have no role meaning; uncertain matches appear as `SPEAKER_UNKNOWN`. The IDs track
acoustic voices without estimating which person is a nurse or patient.

Stop with Ctrl+C. To save normal output in PowerShell:

```powershell
New-Item -ItemType Directory -Force .\transcripts | Out-Null
$log = ".\transcripts\diarization-$(Get-Date -Format yyyyMMdd-HHmmss).txt"
.\.venv\diarization\Scripts\python.exe .\main.py --diarization --window 15 --step 3 |
    Tee-Object -FilePath $log
```

Normal analysis diagnostics go to stderr. Keep them visible when troubleshooting.
For only timestamped speaker/text lines, append
`| Where-Object { $_ -match '^\[\d.*\]\s+SPEAKER_' }` to the PowerShell pipeline.

## Inspect diarization and STT separately

```bash
python main.py --diarization --window 15 --step 3 --inspect
```

Inspection prints `[TURN ...] SPEAKER_00 (model=SPEAKER_01)` intervals and independent
`[STT ...]` text, plus input levels and speaker/profile counts. `model=` is the local
Pyannote label, which can change between windows; the main ID is the session label.

Normal output assigns recognized words to their most-overlapping speaker turn and
discards words outside detected turns. Inspection keeps STT independent of those
turns to distinguish transcription failures from segmentation or matching failures.
"Speech detected, but no new words recognized" means no aligned words were ready
to print in the newly emitted time interval. Repeated vocabulary is not removed.

English STT is always enabled and defaults to `tiny.en` on CPU with int8 computation.
`--stt-model base.en` uses a larger English model at additional processing cost.
Each update prints newly recognized speech immediately after processing. Ctrl+C
closes capture without running a final transcription pass on the buffered tail.

## Limitations and checks

The pipeline allows one or two speakers per window, including windows where only
one participant speaks. This does not guarantee that both voices are distinguished.
Profile matching uses heuristic cosine thresholds; brief replies, similar voices,
overlap, noise, and phone playback can cause merged or uncertain labels. Emitted
lines are not revised later. This demo has no validated accuracy target.

Run the tests without downloading models or using a microphone:

```bash
python -m unittest discover -s tests -p 'test_diarization*.py' -v
```

Tests cover label alignment, voice matching, word attribution, independent inspection,
live launch, and stopping using mocked model outputs and capture. Live accuracy and
performance require representative conversations.
