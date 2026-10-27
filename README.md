<div align="center">

# 🎙️ AudioDeNoiser

**Instant speech restoration for traffic, wind, and city noise.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%2011%20%7C%2010-0078D6?logo=windows&logoColor=white)
![Licence](https://img.shields.io/badge/Licence-Proprietary%20%2F%20All%20Rights%20Reserved-B22222)
![Version](https://img.shields.io/badge/Version-1.0-F2A93B)

</div>

---

## Overview

AudioDeNoiser is a specialised desktop audio restoration tool for Windows 11, built with Python 3.11 and CustomTkinter. It cleans speech recorded outdoors, stripping away traffic, wind and general city noise, and lets you compare the result against the original instantly.

- **DeepFilterNet 3** does the heavy lifting: a state-of-the-art neural speech-enhancement model running locally on the CPU, so nothing leaves your machine.
- **48 kHz processing with polyphase resampling.** Every file is resampled to the model's native 48 kHz with high-quality polyphase filtering, processed in chunks, then resampled back to its original rate.
- **Instant A/B auditioning.** Original and denoised audio are both held in memory, so you can flip between them during playback with no reload and no gap.

## Key features

| Feature | Detail |
| --- | --- |
| Wind and traffic rumble attenuation | Optional Butterworth high-pass filter at 60, 80, 100 or 120 Hz, applied before denoising |
| Adjustable noise reduction | Attenuation limit from 6 dB up to maximum, so you can keep some natural ambience |
| Instant A/B comparison | Switch between original and denoised audio mid-playback with a single key |
| Chunked processing | 30 s segments with a 2 s pre-roll and 0.25 s crossfades, giving a real progress bar, a working Cancel button and flat memory use on long recordings |
| Mastering-ready output | Optional peak normalisation to -1.0 dBFS and 24-bit PCM WAV output |
| Broad input support | WAV, MP3, FLAC and OGG |
| Distinctive interface | CustomTkinter UI styled in streetlight amber on wet asphalt |

## How the processing chain works

1. Decode WAV, MP3, FLAC or OGG with `soundfile` and resample to 48 kHz.
2. Optionally high-pass filter to remove wind and traffic rumble below 60 to 120 Hz.
3. Run DeepFilterNet 3 in 30-second chunks, each with a 2-second pre-roll so the model has settled before its output is used, joined with 0.25 s crossfades.
4. Resample back to the source rate, optionally peak-normalise, and save a 24-bit WAV named `<name>_denoised.wav` in the output folder.

## Prerequisites

- Windows 11 or Windows 10, 64-bit
- **Python 3.11** (DeepFilterNet 0.5.6 only ships Windows wheels for Python 3.8 to 3.11). It installs alongside any newer Python without changing it.
- About 300 MB of free space for the private environment and an internet connection for first-time setup

## Setup

### Automatic (recommended)

Double-click **`AudioDeNoiser.bat`**. On first run it:

1. Finds Python 3.11 with the `py` launcher (and tells you how to install it if it's missing).
2. Creates a private virtual environment in `.venv` next to the app.
3. Installs CPU PyTorch, DeepFilterNet and the other dependencies.
4. Launches AudioDeNoiser.

Later runs skip straight to launching. You can also drag an audio file onto the `.bat` to open it directly. If setup is interrupted, just run the `.bat` again and it carries on.

If Python 3.11 isn't installed:

```powershell
winget install Python.Python.3.11
```

### Manual

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install deepfilternet==0.5.6 numpy==1.26.4 scipy==1.17.1 soundfile==0.14.0 sounddevice==0.5.6 customtkinter==6.0.0
python audio_denoiser.py [optional audio file]
```

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| `Space` | Play / pause |
| `A` | Audition the original |
| `B` | Audition the denoised version |
| `Ctrl+O` | Open an audio file |
| `Ctrl+D` | Denoise the current file |

## Configuration

Settings are saved automatically to `%APPDATA%\AudioDeNoiser\settings.json`, and errors are logged to `error.log` in the same folder. Neither is tracked in Git. [`settings.example.json`](settings.example.json) shows the structure:

```json
{
  "output_dir": "D:\\Audio\\Denoised",
  "atten_index": 9,
  "highpass": "80 Hz",
  "post_filter": false,
  "normalise": true,
  "last_dir": ""
}
```

| Key | Meaning |
| --- | --- |
| `output_dir` | Where denoised files are saved. Falls back to `Music\Denoised` if unavailable |
| `atten_index` | Noise-reduction strength, 0 to 9, where 9 is maximum |
| `highpass` | Rumble filter: `Off`, `60 Hz`, `80 Hz`, `100 Hz` or `120 Hz` |
| `post_filter` | DeepFilterNet's extra post-filter for heavy noise |
| `normalise` | Peak-normalise the saved file to -1.0 dBFS |
| `last_dir` | Last folder a file was opened from |

## Project structure

```text
AudioDeNoiser/
├── audio_denoiser.py       # Desktop GUI and DSP pipeline
├── AudioDeNoiser.bat       # Environment bootstrapper and launcher
├── settings.example.json   # Settings template
├── notes/01_Active/        # Design notes
├── LICENSE                 # Proprietary licence
└── README.md
```

## Legal and copyright notice

**Copyright © 2026-present Mark Hawksworth ([hawkers63](https://github.com/hawkers63)). All rights reserved.**

This is proprietary software. No licence is granted to copy, distribute, modify, reverse engineer, sublicense or commercially exploit any part of it without prior written consent. See [`LICENSE`](LICENSE) for the full terms.

DeepFilterNet, PyTorch and the other third-party libraries are not distributed with this repository and remain under their own licences.
