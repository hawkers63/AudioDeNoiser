<div align="center">

# 🎙️ AudioDeNoiser

**Instant speech restoration for traffic, wind, and city noise.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%2011%20%7C%2010-0078D6?logo=windows&logoColor=white)
![Licence](https://img.shields.io/badge/Licence-Proprietary%20%2F%20All%20Rights%20Reserved-B22222)
![Version](https://img.shields.io/badge/Version-1.6.0-F2A93B)

</div>

---

## Overview

AudioDeNoiser is a specialised desktop audio restoration tool for Windows 11, built with Python 3.11 and CustomTkinter. It cleans speech recorded outdoors, stripping away traffic, wind and general city noise, and lets you compare the result against the original instantly.

- **DeepFilterNet 3** does the heavy lifting: a state-of-the-art neural speech-enhancement model running locally on the CPU, so nothing leaves your machine.
- **48 kHz processing with polyphase resampling.** Every file is resampled to the model's native 48 kHz with high-quality polyphase filtering, processed in chunks, then resampled back to its original rate.
- **Instant, fair A/B auditioning.** Original and denoised audio are both held in memory, so you can flip between them during playback with no reload and no gap, and they can be played at matched loudness so you hear the noise difference rather than a level difference.

## Key features

| Feature | Detail |
| --- | --- |
| Scene presets | Manual keeps the 1.0 chain. Heavy Wind & Gusts, Highway / Traffic Drone, Pedestrian Street, Conversation (1–5 m) and Subtle Podcast Clean precondition each chunk for wind, road rumble, idle tones and hum. Save preset… adds the controls on screen to that list under a name you choose |
| Dialogue lift | Off, Gentle or Clear. Raises consonants a few metres away after the model. Off, the default everywhere except Conversation, leaves the audio untouched |
| Subtitles | Optional English transcript. Read speech transcribes the buffer you are hearing, without denoising first. The switch transcribes the denoised audio after Denoise, without waiting for a WAV. Words follow the playhead, and Save text writes an SRT and a plain transcript |
| Save denoised | Denoise keeps the result in memory. Save denoised… writes the WAV when you ask, at the source rate and bit depth. Closing, opening another file, or denoising again asks before an unsaved pass is dropped |
| Waveform range | Shift-drag marks a stretch of the waveform. Loop plays it until Stop. Cut removes it from this session only. Save selection writes that stretch on its own, at the source rate and bit depth |
| Transient guard | Heavy Wind and Highway repair isolated clicks, ticks and knocks (autoregressive interpolation) and low pops in pauses before the model, which otherwise passes them. Consonant bursts are left alone |
| Floor gate | Heavy Wind, Highway and Pedestrian Street lower the steady wash the model leaves under its blend limit, by up to 10, 12 or 8 dB, with voiced harmonics protected |
| Wind and traffic rumble attenuation | On Manual, an optional Butterworth high-pass at 60, 80, 100 or 120 Hz. A named scene replaces that switch with its own low-end stage |
| Adjustable noise reduction | Attenuation limit from 6 dB up to maximum, so you can keep some natural ambience |
| Instant A/B comparison | Switch between original and denoised audio mid-playback with a single key. Optionally loudness-matched (ITU-R BS.1770-4) for playback only |
| Chunked processing | 30 s segments with a 2 s pre-roll and 0.25 s crossfades, giving a real progress bar, a working Cancel button and flat memory use on long recordings |
| Mastering-ready output | Optional peak normalisation to -1.0 dBFS, applied when you save. WAV output at the source's bit depth (16, 24 or 32-bit PCM, or float); 16-bit is rounded with TPDF dither. MP3 and OGG sources are saved as 24-bit |
| Broad input support | WAV, MP3, FLAC and OGG |
| Distinctive interface | CustomTkinter UI styled in streetlight amber on wet asphalt |

## How the processing chain works

1. Decode WAV, MP3, FLAC or OGG with `soundfile` and resample to 48 kHz.
2. **Manual** is the 1.0 chain: optionally high-pass filter to remove wind and traffic rumble below 60 to 120 Hz, then hand the audio to the model. A **named scene** does not use that switch. Inside each chunk, Heavy Wind and Highway first repair isolated clicks and low pops (the transient guard). Every named scene then runs an infrasonic filter, ducks gust energy under the scene's split while the fragment is unvoiced, and pulls down a steady idle tone or a confirmed 50/100 Hz hum.
3. Run DeepFilterNet 3 in 30-second chunks, each with a 2-second pre-roll so the model has settled before its output is used, joined with 0.25 s crossfades. After the model, Heavy Wind and Highway soften isolated peaks in the hiss band, and Heavy Wind, Highway, Pedestrian Street and Conversation lower the steady residual floor (the floor gate). If Dialogue is Gentle or Clear, the consonant band is then lifted. All of that happens before the crossfade.
4. The denoised audio stays in the session. **Save denoised…** resamples it back to the source rate, optionally peak-normalises, and writes a WAV at the source's bit depth (24-bit for MP3 and OGG). The suggested name is `<name>_denoised.wav` in the Save to folder. Denoise itself does not write that file.
5. For A/B playback, the integrated loudness of both versions is measured (ITU-R BS.1770-4) and, if **Match A/B loudness** is on, the denoised copy plays at the original's loudness. The saved file is never touched by this.
6. Shift-drag on the waveform marks a range. Loop repeats it, Cut drops it from the session (the file on disk stays as it was), and Save selection writes that range alone. Read speech transcribes whichever buffer you are hearing.

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
| Shift-drag | Mark a range on the waveform |
| `Escape` | Clear that range |

## Configuration

Settings are saved automatically to `%APPDATA%\AudioDeNoiser\settings.json`, and errors are logged to `error.log` in the same folder. Neither is tracked in Git. [`settings.example.json`](settings.example.json) shows the structure:

```json
{
  "output_dir": "D:\\Audio\\Denoised",
  "atten_index": 9,
  "highpass": "80 Hz",
  "post_filter": false,
  "normalise": true,
  "loudness_match": true,
  "last_dir": "",
  "scene": "manual",
  "manual_highpass": "80 Hz",
  "manual_atten_index": 9,
  "manual_post_filter": false,
  "dialogue_lift": "off",
  "subtitles": false
}
```

| Key | Meaning |
| --- | --- |
| `output_dir` | The folder Save denoised… opens in. Choosing another folder in the save dialog does not change this. Falls back to `Music\Denoised` if unavailable |
| `atten_index` | Noise-reduction strength, 0 to 9, where 9 is maximum |
| `highpass` | Rumble filter: `Off`, `60 Hz`, `80 Hz`, `100 Hz` or `120 Hz`. Forced to `Off` while a named scene is selected |
| `post_filter` | DeepFilterNet's extra post-filter for heavy noise |
| `normalise` | Peak-normalise the saved file to -1.0 dBFS |
| `loudness_match` | Play A and B at matched integrated loudness. Playback only; missing means on |
| `last_dir` | Last folder a file was opened from |
| `scene` | `manual`, `wind`, `traffic`, `street`, `conversation` or `podcast`. Missing or unknown means Manual |
| `dialogue_lift` | `off`, `gentle` or `clear`. Missing means the scene's own default (Clear on Conversation, Off on the others) |
| `subtitles` | After Denoise, transcribe the denoised audio and show the words during playback. It does not wait for the WAV to be saved. Missing means off. Read speech is a button, not a saved setting |
| `manual_highpass` | Rumble choice remembered while a named scene is in use. Default `80 Hz` |
| `manual_atten_index` | Slider position remembered for the return to Manual. Default `9` (maximum) |
| `manual_post_filter` | Post-filter switch remembered for the return to Manual. Default `false` |

## Project structure

```text
AudioDeNoiser/
├── audio_denoiser.py       # Desktop GUI and DSP pipeline
├── AudioDeNoiser.bat       # Environment bootstrapper and launcher
├── settings.example.json   # Settings template
├── ROADMAP.md              # Scope of planned work
├── CHANGELOG.md            # Release history
├── LICENSE                 # Proprietary licence
└── README.md
```

## Releases

Builds and release notes are published on the [Releases page](https://github.com/hawkers63/AudioDeNoiser/releases).

| Version | Date | Summary |
| --- | --- | --- |
| [1.6.0](https://github.com/hawkers63/AudioDeNoiser/releases/tag/v1.6.0) | 29 October 2026 | Save preset… adds the controls on screen to the scene list under a name you choose. The denoiser is unchanged |
| [1.5.0](https://github.com/hawkers63/AudioDeNoiser/releases/tag/v1.5.0) | 29 October 2026 | Save denoised… writes the WAV when you ask. Denoise keeps the result in memory. The denoiser is unchanged |
| [1.4.0](https://github.com/hawkers63/AudioDeNoiser/releases/tag/v1.4.0) | 29 October 2026 | Mark a waveform range to loop it, cut it from the session, or save it on its own. Read speech transcribes the buffer you are hearing. The denoiser is unchanged |
| [1.3.0](https://github.com/hawkers63/AudioDeNoiser/releases/tag/v1.3.0) | 28 October 2026 | Conversation (1–5 m), a dialogue lift for talk a few metres away, and optional playback subtitles. Dialogue Off keeps the 1.2.1 sound |
| [1.2.1](https://github.com/hawkers63/AudioDeNoiser/releases/tag/v1.2.1) | 28 October 2026 | Heavy Wind & Gusts and Highway / Traffic Drone render 11-23 % faster; output byte-identical to 1.2.0 |
| [1.2.0](https://github.com/hawkers63/AudioDeNoiser/releases/tag/v1.2.0) | 28 October 2026 | Transient guard (click and pop repair), floor gate, loudness-matched A/B and source bit-depth export with dithered 16-bit. Fixes the 1.1.0 buffer-edge spike. Manual matches 1.1 exactly |
| [1.1.0](https://github.com/hawkers63/AudioDeNoiser/releases/tag/v1.1.0) | 28 October 2026 | Wind and traffic scene presets: Heavy Wind & Gusts, Highway / Traffic Drone, Pedestrian Street and Subtle Podcast Clean. Manual matches 1.0 exactly |
| [1.0.0](https://github.com/hawkers63/AudioDeNoiser/releases/tag/v1.0.0) | 27 October 2026 | First release: DeepFilterNet 3 restoration, rumble filter, chunked processing, A/B auditioning and 24-bit output |

See [`CHANGELOG.md`](CHANGELOG.md) for full details and [`ROADMAP.md`](ROADMAP.md) for what's planned next.

## Legal and copyright notice

**Copyright © 2026-present Mark Hawksworth ([hawkers63](https://github.com/hawkers63)). All rights reserved.**

This is proprietary software. No licence is granted to copy, distribute, modify, reverse engineer, sublicense or commercially exploit any part of it without prior written consent. See [`LICENSE`](LICENSE) for the full terms.

DeepFilterNet, PyTorch and the other third-party libraries are not distributed with this repository and remain under their own licences.
