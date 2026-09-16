# Approved patient voices

Place only university-approved, consented reference recordings in this directory. Supported
formats are WAV, MP3, FLAC, M4A, OGG, and Opus. The filename becomes the voice ID shown in the
application, for example `patient-a.wav` appears as `patient-a`.

Do not add patient recordings, staff recordings, celebrity voices, or any recording without a
documented license or explicit consent. Audio files are intentionally not included in this
repository.

To generate the approved synthetic child reference named Ruth with Microsoft Edge TTS, run:

```bash
conda run -n nursing-ai-sim python scripts/generate_voice.py
```

This one-time generation command uses the network. It creates `child_female_8yo.wav` as mono,
24 kHz, 16-bit PCM audio and removes its temporary MP3. Simulation dialogue continues to use
local Chatterbox Nano with the voice ID `child_female_8yo`.
