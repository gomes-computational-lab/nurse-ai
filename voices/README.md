# Approved patient voices

Place only university-approved, consented reference recordings in this directory. Supported
formats are WAV, MP3, FLAC, M4A, OGG, and Opus. The filename becomes the voice ID shown in the
application, for example `patient-a.wav` appears as `patient-a`.

Do not add patient recordings, staff recordings, celebrity voices, or any recording without a
documented license or explicit consent. Audio files are intentionally not included in this
repository.

To generate the approved synthetic child reference named Ruth with Microsoft Edge TTS, activate
the project environment and run the script from the repository root. These commands work in
macOS or Linux shells and Windows PowerShell:

```bash
conda activate nursing-ai-sim
python scripts/generate_voice.py
```

This one-time generation command uses the network. It creates `child_female_8yo.wav` as mono,
24 kHz, 16-bit PCM audio and removes its temporary MP3. Simulation dialogue continues to use
local Chatterbox Nano with the voice ID `child_female_8yo`.

The generator requires FFmpeg. Install it with one of these commands, then confirm that
`ffmpeg -version` works in the same terminal:

| Platform | Command |
| --- | --- |
| macOS with Homebrew | `brew install ffmpeg` |
| Ubuntu or Debian | `sudo apt update && sudo apt install ffmpeg` |
| Windows PowerShell with WinGet | `winget install --id Gyan.FFmpeg -e` |
| Any platform with Conda | `conda install -n nursing-ai-sim -c conda-forge ffmpeg` |
