# AudioDeNoiser Roadmap

This file sets out the scope of work for AudioDeNoiser: what has shipped, what is planned next, and the rules every change follows. Detailed design notes are kept locally in `notes/` and are not published.

Copyright (c) 2026-present Mark Hawksworth (hawkers63). All rights reserved.

## Status at a glance

| Version | Theme | Status |
| --- | --- | --- |
| 1.0.0 | Baseline desktop app, documentation and licence | Released 27 October 2026 |
| 1.1.0 | Wind and traffic scene presets | Released 28 October 2026 (tagged locally; push and GitHub Release pending Mark's approval). Mark's listening check pending |
| Later | To be agreed with Mark | Not scheduled |

## 1.0.0 - Baseline (released)

- DeepFilterNet 3 speech restoration with 48 kHz polyphase resampling.
- Optional Butterworth rumble filter at 60, 80, 100 or 120 Hz.
- Chunked processing: 30 s segments, 2 s pre-roll, 0.25 s crossfades, with progress and Cancel.
- Instant A/B auditioning of the original and denoised audio.
- Peak normalisation to -1.0 dBFS and 24-bit PCM WAV output.
- CustomTkinter interface in streetlight amber on wet asphalt.
- `AudioDeNoiser.bat` first-run bootstrapper for a private Python 3.11 environment.
- Repository baseline: `.gitignore`, `settings.example.json`, proprietary `LICENSE`, `README.md`.

## 1.1.0 - Wind and traffic scene presets (released 28 October 2026)

Design note: `notes/01_Active/002_wind_traffic_scene_presets.md` (local only).

Committed and tagged `v1.1.0` locally on 28 October 2026 after automated tests with the real DeepFilterNet 3 model. Pushing the commit and tag, publishing the GitHub Release and Mark's own listening check are still with Mark.

Scope:

- A scene control with **Manual** plus four presets: **Heavy Wind & Gusts**, **Highway / Traffic Drone**, **Pedestrian Street** and **Subtle Podcast Clean**.
- **Manual** must reproduce the 1.0 chain exactly, so existing results don't change.
- Per-chunk conditioning before DeepFilterNet: infrasonic removal, gust ducking on unvoiced frames only, idle-tone and confirmed 50/100 Hz hum reduction, and an optional spectral floor.
- Work runs inside the existing worker thread, honours Cancel between stages, installs nothing new and keeps memory use flat.
- Companion updates to `settings.example.json` and the README's processing description.

Done when:

- [x] The patch is applied to `audio_denoiser.py` and the app starts cleanly. Verified by constructing the app window from the venv Python with no errors; launching through `AudioDeNoiser.bat` itself was not tested.
- [x] A file processed on Manual matches the 1.0 output with the same settings. Verified bit-identical (difference 0) against `v1.0.0` with DeepFilterNet 3, at maximum and 18 dB, on the three listening clips and a 70 s multi-chunk file.
- [ ] Each preset passes the listening check described in the design note. Each preset ran cleanly on its clip with the real model and the measured band levels meet the expectations, but Mark's own listening check is still pending.
- [ ] `CHANGELOG.md` has a 1.1.0 entry (done) and Mark has approved the push and release (pending).

## Later

Further work will be added here once agreed with Mark.

## Ground rules for every change

- Work on `main` locally. Never push to GitHub without Mark's confirmation, and never force-push.
- Keep `.venv/`, runtime `settings.json`, logs and the `notes/` folder out of Git.
- Every release gets a `CHANGELOG.md` entry, a `vX.Y.Z` tag and a GitHub Release.
- Keep the proprietary copyright banner at the top of `audio_denoiser.py`.
