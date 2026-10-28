# Changelog

All notable changes to AudioDeNoiser are recorded here. Versions follow [Semantic Versioning](https://semver.org/), and each release is published on the [Releases page](https://github.com/hawkers63/AudioDeNoiser/releases).

Copyright (c) 2026-present Mark Hawksworth (hawkers63). All rights reserved.

## [Unreleased]

## [1.2.0] - 2026-10-28

Manual output is unchanged: verified bit-identical to 1.1.0 with DeepFilterNet 3, file and 48 kHz buffer, on the listening clips and a 70 s multi-chunk file.

### Added

- Transient guard on Heavy Wind & Gusts and Highway / Traffic Drone. Isolated clicks, ticks and knocks are found on the dry chunk and filled by least-squares autoregressive (LSAR) interpolation; low pops in pauses have the band under 200 Hz lowered for their few milliseconds. A click must stand clear of every other peak within 40 ms and sit on steady surroundings, so consonant bursts and glottal pulses are left alone. DeepFilterNet on its own often passes such clicks, up to 60 dB above the silenced pauses around them.
- Floor gate on Heavy Wind (10 dB, 1.2-6 kHz), Highway (12 dB, 0.8-6 kHz) and Pedestrian Street (8 dB, 1.5-6 kHz): minimum-statistics floor tracking with a Wiener-shaped gain after the network, voiced harmonics forced to unity. On the listening clips the tyre wash in pauses falls a further 11 dB on Highway and 7 dB on Street.
- Loudness-matched A/B: **Match A/B loudness** beside the A/B buttons. Both versions are measured (ITU-R BS.1770-4, checked against the EBU Tech 3341 cases) and the denoised copy plays at the original's loudness. If that would push it past full scale, both play lower together, so neither clips. Playback only, and it takes effect as soon as it is switched.
- The saved WAV keeps the source's bit depth: 16, 24 or 32-bit PCM, or 32/64-bit float, from WAV and FLAC. 16-bit is rounded with TPDF dither (libsndfile alone floors each sample). MP3, OGG and unusual WAV encodings are saved as 24-bit, as before.
- Settings key `loudness_match` (missing means on). The bit depth shows under the file name, and the log reports the original's loudness, the transient repairs, the saved depth and the A/B match.

### Changed

- Scene summaries for Heavy Wind, Highway and Pedestrian Street mention the new stages.
- The inverse STFT behind the tone, floor and gate stages is vectorised: bit-for-bit the same output, 17-35 % faster.
- Minimum window height is 804 px (was 760), so the log box is no longer cut off at the smallest size.

### Fixed

- Named scenes put a spike, up to several times full scale, into the first and last frame of each spectral stage's buffer (`_apply_spectral_gain` divided by a near-zero window sum at the edges). It reached DeepFilterNet through the tone stage and the saved file through the floor; with normalising on, Highway, Street and Podcast files were turned down by 6-26 dB to fit a spike in their last few milliseconds. The interior of every buffer is unchanged, bit for bit.

## [1.1.0] - 2026-10-28

### Added

- Scene menu: Manual, Heavy Wind & Gusts, Highway / Traffic Drone, Pedestrian Street and Subtle Podcast Clean.
- Named scenes precondition each 30 s chunk before DeepFilterNet (infrasonic filter, unvoiced gust duck, idle-tone or confirmed hum reduction). Heavy Wind and Highway also soften isolated hiss-band peaks. No new packages.
- Settings keys `scene`, `manual_highpass`, `manual_atten_index` and `manual_post_filter`. A missing `scene` stays on Manual, which is the 1.0 chain.

- `ROADMAP.md` setting out the scope of planned work.
- This changelog.

### Changed

- `denoise()` now converts one chunk at a time to a torch tensor instead of holding a copy of the whole file. Manual output is unchanged: verified bit-identical to 1.0.0 with DeepFilterNet 3 on short clips and a 70 s file.
- The window is slightly taller (860x820, minimum 760x760) to fit the scene menu.
- Design notes in `notes/` are now kept locally only and are no longer tracked in Git.

## [1.0.0] - 2026-10-27

First release of AudioDeNoiser.

### Added

- DeepFilterNet 3 speech restoration for traffic, wind and city noise, running locally on the CPU.
- 48 kHz processing with high-quality polyphase resampling back to the source rate.
- Optional Butterworth high-pass rumble filter at 60, 80, 100 or 120 Hz.
- Adjustable noise-reduction limit from 6 dB to maximum, plus DeepFilterNet's optional post-filter.
- Chunked processing in 30 s segments with a 2 s pre-roll and 0.25 s crossfades, giving a real progress bar, a working Cancel button and flat memory use.
- Instant A/B auditioning of the original and denoised audio.
- Optional peak normalisation to -1.0 dBFS and 24-bit PCM WAV output named `<name>_denoised.wav`.
- WAV, MP3, FLAC and OGG input.
- Keyboard shortcuts: `Space`, `A`, `B`, `Ctrl+O` and `Ctrl+D`.
- CustomTkinter interface styled in streetlight amber on wet asphalt.
- `AudioDeNoiser.bat` first-run setup of a private Python 3.11 environment and launcher.
- `settings.example.json`, proprietary `LICENSE` (All Rights Reserved) and full `README.md`.

[Unreleased]: https://github.com/hawkers63/AudioDeNoiser/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/hawkers63/AudioDeNoiser/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/hawkers63/AudioDeNoiser/releases/tag/v1.1.0
[1.0.0]: https://github.com/hawkers63/AudioDeNoiser/releases/tag/v1.0.0
