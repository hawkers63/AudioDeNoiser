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
| 1.3.0 | Dialogue lift, Conversation (1–5 m), playback subtitles | Released 28 October 2026. Listening check passed 29 October 2026: works well on a real 1–5 metre take |
| 1.4.0 | Waveform range: loop, cut from the session, save the range, read speech | Released 29 October 2026. The denoiser is unchanged |
| 1.5.0 | Save the denoised audio when asked | Released 29 October 2026. The denoiser is unchanged |
| 1.6.0 | Named presets in the scene list | Released 29 October 2026. The denoiser is unchanged |
| 1.7.0 | Voice clarity on voiced frames | Released 29 October 2026. Restore consonants ships off. On the measured helmet take Mark preferred that off pass. Further samples may follow |
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
- [x] Mark's listening check: a real conversation at about 1–5 m, in wind and in traffic, Dialogue Clear against Off, on Conversation and on Highway with the menu set to Clear. Synthetic clips and text-to-speech are not this check. Passed 29 October 2026: Mark confirmed it works well on a real 1–5 metre take. He did not report a separate Clear-against-Off score for each scene.

## 1.4.0 - Waveform range and a transcript you can read (released 29 October 2026)

Design note: `notes/01_Active/006_waveform_selection_and_transcript.md` (local only).

Scope:

- Shift-drag marks a range on the waveform. Click and drag still seek. Escape clears the range.
- Loop plays that range until Stop. Cut removes it from the working session, with one Undo. The file on disk is not modified.
- Save selection writes that range alone, at the source rate and bit depth, without denoising it.
- Read speech transcribes the buffer being heard. The switch is labelled "English transcript after denoise" and still runs after Denoise. Save text no longer waits for a denoised WAV.
- `denoise()`, the scenes, the dialogue lift and `write_wav` are not edited. Nothing new is installed.

Done when:

- [x] The patch compiles, and the session checks pass with no model: cut length, both channels, cues, undo byte for byte, loop wrap, and a saved selection at the source rate. `notes/tests/session_140.py`, 29 October 2026.
- [x] Manual still matches 1.1.0 (`notes/tests/integration.py parity`): 48 kHz buffer byte-identical on the three clips, the 70 s file and the TTS clip, at steps 9 and 4. Saved files match except the TTS file, which is 16-bit here and was 24-bit in 1.1. The four released scenes, Dialogue Off, match frozen 1.2.0: 16 of 16, buffer and file. The Hazel transcript smoke passed again (6 of 8 marker words, `kit` heard as `kick`).
- [x] The window is measured: requested 721×1020. At 860×1020 and at 760×996 the log's last line is visible and the waveform is at least 130 px. The five buttons sit on one row at 760 wide.
- [x] `CHANGELOG.md` has a 1.4.0 entry, and the commit, `v1.4.0` tag and GitHub Release are published.

## 1.5.0 - Save the denoised audio when asked (released 29 October 2026)

Design note: `notes/01_Active/007_manual_save_denoised.md` (local only). Mark's go-ahead was 29 October 2026.

Scope:

- Denoise finishes in memory and does not write a WAV. **Save denoised…** asks for a name and writes the file then.
- The samples match what 1.4.0 writes on its own: source rate, source bit depth, and peak normalisation only when that switch is on at the moment of saving. Match A/B loudness stays playback-only.
- Closing the window, opening another file, or running Denoise again asks before an unsaved pass is dropped. The opened file cannot be chosen as the destination.
- The parity scripts keep writing a file, because they do not pass the new flag. The window does.
- `denoise()`, the scenes, the dialogue lift and `write_wav` stay as they are. Nothing new is installed.

Done when:

- [x] The patch is applied and the export of an unchanged pass matches the 1.4.0 write block byte for byte. `notes/tests/save_150.py`: normalise on and off, 16-bit and 24-bit, a second save of the same samples, and the length-pad branch. A denoise that omits `write_file` still writes that file.
- [x] A denoise from the window leaves no WAV until Save is used. The same test calls the worker with `write_file` false and the output folder stays empty. Saving over the opened file is refused and does not change that file. A failed write removes the partial.
- [x] Manual still matches 1.1.0 (`notes/tests/integration.py parity`): 48 kHz buffer byte-identical on the three clips, the 70 s file and the TTS clip, at steps 9 and 4. Saved files match except the TTS file, which is 16-bit here and was 24-bit in 1.1 (max abs diff about 4.5e-5). The four released scenes, Dialogue Off, match frozen 1.2.0: 16 of 16, buffer and file. `verify_core.py` exit 0, same figures as 1.4.0. Hazel smoke: 16.9 s for 17.9 s, 6 of 8 marker words, `kit` heard as `kick`.
- [x] `CHANGELOG.md` has a 1.5.0 entry, and the commit, `v1.5.0` tag and GitHub Release are published. Window remeasured: requested 721×1020. At 860×996 the waveform is 130 px; at 995 px it is 129. At 760×996 the action row and the selection row each stay on one line. Minimum size stays 760×996.

## 1.6.0 - Named presets (released 29 October 2026)

Design note: `notes/01_Active/008_user_presets_and_speech_crispness.md` (local only). Mark's go-ahead to release the presets was 29 October 2026. The consonant restore is not in this version. He scheduled it as 1.7.0.

Scope:

- **Save preset…** stores the controls on screen under a name and lists that name under the six factory scenes. Delete removes the name. Factory scenes are not deleted, and choosing one still applies that scene's own defaults.
- The record is the scene, dialogue, noise-reduction step, rumble choice, post-filter and normalise. It lives in `settings.json` under `user_presets`. Loading applies the scene, then writes the stored controls back, so a saved tweak is kept. A named scene still forces the rumble filter off.
- `denoise()`, `apply_scene`, `apply_dialogue_lift` and `write_wav` stay as they are. Manual, and Dialogue Off on the released scenes, stay bit-identical to 1.5.0. Nothing new is installed.

Done when:

- [x] Named presets round-trip through `settings.json`. A factory scene still applies its own defaults. Loading a saved Conversation tweak keeps that step, and a named scene leaves the rumble filter off. `notes/tests/presets_160.py`, 29 October 2026, fails 0.
- [x] Manual still matches 1.1.0 (`notes/tests/integration.py parity`): 48 kHz buffer byte-identical on the three clips, the 70 s file and the TTS clip, at steps 9 and 4. Saved files match except the TTS file, which is 16-bit here and was 24-bit in 1.1 (max abs diff about 4.5e-5).
- [x] The window is measured with Save preset… and Delete on the scene row. Requested size stays 721×1026. At 860×1020 the waveform is 148 px and the log is on screen. The smallest height is 1002: the waveform is 130 px there and 129 px at 1001. At 760×1002 the scene row stays on one line and Delete stays inside the window. Default size stays 860×1020.
- [x] `CHANGELOG.md` has a 1.6.0 entry, and the commit, `v1.6.0` tag and GitHub Release are published.

## 1.7.0 - Voice clarity on voiced frames (released)

Design note: `notes/01_Active/008_user_presets_and_speech_crispness.md`, section 4.2 (local only). Mark scheduled this on 29 October 2026, when he released 1.6.0. It is not in 1.6.0. Named presets do not change a single sample, so Conversation still darkens the consonants on the measured file.

The table is a measurement of one file, `D:\Audio\01.wav` against `D:\Audio\Denoised\01_denoised.wav`. It is not a listening pass. That file is the 7:11 helmet-mic cycling take. It is not the walking take, and it is not the separate five-metre take. Do not retune Conversation's attenuation or the Clear cap from this file alone. The Clear lift stands aside here because the voice is already much brighter than the frozen −23 dB close-speech target.

Every preview of this work reports voiced frames in this shape. Voicing is the app's own detector. "Change" is denoised minus original. "Against the vowel" subtracts the vowel-body change, so peak-normalise cannot look like a filter.

| Band | Change | Against the vowel |
| --- | --- | --- |
| 20–80 Hz | −21.6 dB | −14.8 dB |
| 80–200 Hz | −15.7 dB | −8.9 dB |
| 300 Hz–1 kHz, the vowel | −6.8 dB | — |
| 1.8–4 kHz, presence | −11.1 dB | −4.3 dB |
| 4–6.5 kHz, consonants | −14.2 dB | −7.4 dB |

The gaps to close, relative to the vowel, are 4.3 dB in presence and 7.4 dB in the consonant band. Those numbers describe this file. They are not new frozen constants. The neighbouring bands measured in the same pass stay in the local note.

Scope:

- A consonant restore after the dialogue lift. Default off. On voiced frames only, restore part of a presence or consonant band that the processing made darker than the dry balance against the vowel. Never brighter than the dry frame. Never on unvoiced frames, so wind and hiss stay down.
- A preview of this pair, stage on and stage off, reported in the table above, before the stage can be a default. Factory scenes stay bit-identical while it is off.
- `denoise()` is touched only when this stage is actually built.

The stage ships in 1.7.0, off unless **Restore consonants** is on. It is not part of a saved preset. The window is 860×1054, and the smallest height is 1036, measured after that switch row. A preview of this pair, Conversation at 18 dB with Clear and the post-filter off, is at `D:\Audio\Denoised\01_preview_off.wav` and `01_preview_on.wav`. `01_denoised.wav` was not overwritten. Voiced frames, same count of 587:

| Band | Off, against the vowel | On, against the vowel |
| --- | --- | --- |
| 20–80 Hz | −14.8 dB | −14.8 dB |
| 80–200 Hz | −8.9 dB | −8.9 dB |
| 300 Hz–1 kHz, the vowel | — | — |
| 1.8–4 kHz, presence | −4.3 dB | −1.2 dB |
| 4–6.5 kHz, consonants | −7.4 dB | −1.7 dB |

The off column matches the table above. The absolute "change" column on the on file sits about 1.2 dB lower because peak-normalise followed a higher peak. Pauses more than a quarter of a second from speech move about half a decibel in the consonant band. The rumble bands do not.

Mark listened to both files on 29 October 2026. He prefers `01_preview_off.wav`: it distinguishes the speech. `01_preview_on.wav` takes out too much and the voice is less clear. The spectrum had moved the consonant band towards the original; his ears do not want that pass. Restore consonants stays off. It does not become the default. Conversation and the Clear cap stay as they are. He will add more samples to the designated folders as they become available. A later pass is possible. It is not this release.

Done when:

- [x] The stage is built, off by default, and a preview of this pair is rendered with it on and off.
- [x] Mark has listened to that preview. He prefers the off pass. The table above is the measurement for both passes. The listening result stands over that table.
- [x] Manual, and Dialogue Off on the scenes shipped in 1.6.0, stay bit-identical while the stage is off.
- [x] `CHANGELOG.md` has a dated 1.7.0 entry, and the commit, `v1.7.0` tag and GitHub Release are published.

## Later

Candidates, not yet agreed:

- Voicing (`estimate_voicing`) runs twice per chunk on Heavy Wind and Highway, once on the dry chunk and once on the network's output, at about 0.77 s a call. It is now the largest cost after DeepFilterNet itself, about a quarter of a Heavy Wind render.
- On Pedestrian Street the floor gate needs its own voicing pass, which makes Street renders about 40 % slower than in 1.1. Reusing the voicing measured before the model would roughly halve that, at some cost in how well quiet harmonics are protected.
- Denoise no longer writes a file, so Cancel during Denoise cannot leave a WAV. Save denoised… is the write you just asked for, so Cancel stays disabled during it; a failed save removes its `.partial.wav`. Save selection already removes its own partial.

Further work will be added here once agreed with Mark.

## Ground rules for every change

- Work on `main` locally. Never push to GitHub without Mark's confirmation, and never force-push.
- Keep `.venv/`, runtime `settings.json`, logs and the `notes/` folder out of Git.
- Every release gets a `CHANGELOG.md` entry, a `vX.Y.Z` tag and a GitHub Release.
- Keep the proprietary copyright banner at the top of `audio_denoiser.py`.
