# AudioDeNoiser Roadmap

This file sets out the scope of work for AudioDeNoiser: what has shipped, what is planned next, and the rules every change follows. Detailed design notes are kept locally in `notes/` and are not published.

Copyright (c) 2026-present Mark Hawksworth (hawkers63). All rights reserved.

## Status at a glance

| Version | Theme | Status |
| --- | --- | --- |
| 1.0.0 | Baseline desktop app, documentation and licence | Released 27 October 2026 |
| 1.1.0 | Wind and traffic scene presets | Released 28 October 2026. Has the edge-spike bug fixed in 1.2.0 |
| 1.2.0 | Transient guard, floor gate, loudness-matched A/B, source bit depth | Released 28 October 2026. Mark's listening check passed 28 October 2026 |
| 1.2.1 | Faster Heavy Wind and Highway renders, identical output | Released 28 October 2026 |
| 1.3.0 | Dialogue lift, Conversation (1–5 m), playback subtitles | Released 28 October 2026. Listening check on real 1–5 m speech is still open |
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

Released on 28 October 2026 after automated tests with the real DeepFilterNet 3 model: `main`, the `v1.1.0` tag and the GitHub Release are on GitHub.

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
- [x] `CHANGELOG.md` has a 1.1.0 entry, and the commit and tag are pushed.

## 1.2.0 - Transient guard, floor gate and loudness-matched A/B (released 28 October 2026)

Design note: `notes/01_Active/003_transient_gate_loudness.md`, with the implementation review beside it (both local only).

Scope:

- **Transient guard** on Heavy Wind and Highway: repair isolated clicks (LSAR interpolation) and low pops in pauses on the dry chunk, before the infrasonic filter. Leave consonant bursts, glottal pulses and anything near speech onsets alone.
- **Floor gate** on Heavy Wind, Highway and Pedestrian Street: lower the steady wash left under the blend limit, after the network, with voiced harmonics protected.
- **Loudness-matched A/B** (ITU-R BS.1770-4), playback only, switchable at any time, never clipping.
- **Export at the source's bit depth**, with TPDF dither at 16-bit.
- Fix the 1.1 edge spike in the spectral stages.
- Manual stays bit-identical to 1.1. Nothing new is installed.

Done when:

- [x] The patch is applied and the app starts cleanly. Verified by driving the real window end to end with the real model: open a 16-bit file, denoise on Highway, A/B gains set and toggled live, settings saved, layout checked at default and minimum size.
- [x] Manual matches 1.1 exactly. Verified bit-identical (file and 48 kHz buffer) with DeepFilterNet 3 at maximum and 18 dB, on the three 1.1 clips and a 70 s multi-chunk file.
- [x] Each new stage measured on synthetic and TTS speech: no false repairs on speech, clicks and pops removed, gate depth and speech cost, EBU Tech 3341 loudness cases, dither statistics.
- [x] Mark's listening check on `notes/listening/04_*` and `05_*` (see `EXPECTATIONS.md`), and on a real windy and a real traffic recording. Passed 28 October 2026: works well, with some improvements noted for later. Because 1.2.1 audio is byte-identical to 1.2.0, this check holds for 1.2.1 too.
- [x] `CHANGELOG.md` 1.2.0 entry, commit, `v1.2.0` tag and GitHub Release (published 28 October 2026).

## 1.2.1 - Faster spectral floor (released 28 October 2026)

Design note: `notes/01_Active/004_faster_wind_traffic restoration_with_identical_audio.md`, with the implementation review beside it (both local only).

Scope:

- The spectral floor on Heavy Wind and Highway takes its 5 × 7 median only over its own band (2-5 kHz and 1.2-4 kHz) and the two bins either side that the median reaches, instead of all 1,025 bins.
- The output stays byte-identical to 1.2.0. No preset, setting, window or export change. Nothing new is installed.

Done when:

- [x] The change is applied and compiles cleanly (`py_compile`, pyflakes).
- [x] Every scene matches 1.2.0 exactly with DeepFilterNet 3: 48 kHz buffer, saved file (SHA-256) and the reported loudness, peak and repairs, on nine files (mono and stereo, 44.1 and 48 kHz, 16 and 24-bit, one to four chunks).
- [x] Manual still matches 1.1.0 exactly.
- [x] Time measured: the stage is 2.5-3.3 times faster, and Heavy Wind and Highway renders 11-23 % faster on this laptop.
- [x] Mark's go-ahead (28 October 2026): version 1.2.1, `CHANGELOG.md` cut, commit and `v1.2.1` tag. Push and GitHub Release are Mark's.
- [x] Listening: output is byte-identical to 1.2.0, so Mark's passed 1.2.0 listening check (28 October 2026) holds for 1.2.1.

## 1.3.0 - Dialogue lift, Conversation and subtitles (released 28 October 2026)

Design note: `notes/01_Active/005_dialogue_clarity_and_subtitles.md`, with the implementation review beside it (both local only).

Scope:

- A dialogue lift after the model (Off, Gentle 3 dB, Clear 6 dB). Off is the default on every scene shipped in 1.2.1, so those scenes stay bit-identical when the menu is left alone.
- Conversation (1–5 m): Pedestrian Street with the post-filter off, a gentler gust duck, a gate that starts at 2.2 kHz, and Dialogue on Clear.
- Optional subtitles during playback, from wav2vec2-base already in torchaudio, with Save text writing an SRT and a plain transcript. The switch defaults off. No new package.

Done when:

- [x] The change is applied and compiles cleanly (`py_compile`).
- [x] The two lift constants are measured and frozen: target tilt −23.0 dB, floor bias 1.2.
- [x] Manual matches 1.1.0 exactly (`notes/tests/integration.py parity`): 48 kHz buffer byte-identical on the three listening clips, the 70 s file and the TTS clip, at steps 9 and 4. Saved files match on the clips and the 70 s file. The TTS file is 16-bit here and was 24-bit in 1.1, which is the 1.2 export rule.
- [x] Heavy Wind, Highway, Pedestrian Street and Subtle Podcast Clean, Dialogue Off, match frozen 1.2.0 (the same audio as 1.2.1) on those three clips and the TTS file: 48 kHz buffer and saved file byte-identical, 16 of 16. Conversation rendered on the TTS file without error. That is not a listening check.
- [ ] Mark's listening check: a real conversation at about 1–5 m, in wind and in traffic, Dialogue Clear against Off, on Conversation and on Highway with the menu set to Clear. Synthetic clips and text-to-speech are not this check.

## Later

Candidates, not yet agreed:

- Voicing (`estimate_voicing`) runs twice per chunk on Heavy Wind and Highway, once on the dry chunk and once on the network's output, at about 0.77 s a call. It is now the largest cost after DeepFilterNet itself, about a quarter of a Heavy Wind render.
- On Pedestrian Street the floor gate needs its own voicing pass, which makes Street renders about 40 % slower than in 1.1. Reusing the voicing measured before the model would roughly halve that, at some cost in how well quiet harmonics are protected.
- After `denoise()` returns, Cancel is no longer checked, so a Cancel pressed while the file is being written still saves it; and a failed write leaves a `.partial.wav` behind.

Further work will be added here once agreed with Mark.

## Ground rules for every change

- Work on `main` locally. Never push to GitHub without Mark's confirmation, and never force-push.
- Keep `.venv/`, runtime `settings.json`, logs and the `notes/` folder out of Git.
- Every release gets a `CHANGELOG.md` entry, a `vX.Y.Z` tag and a GitHub Release.
- Keep the proprietary copyright banner at the top of `audio_denoiser.py`.
