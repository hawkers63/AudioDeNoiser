# Changelog

All notable changes to AudioDeNoiser are recorded here. Versions follow [Semantic Versioning](https://semver.org/), and each release is published on the [Releases page](https://github.com/hawkers63/AudioDeNoiser/releases).

Copyright (c) 2026-present Mark Hawksworth (hawkers63). All rights reserved.

## [Unreleased]

### Changed

- Design notes in `notes/` are now kept locally only and are no longer tracked in Git.

### Added

- `ROADMAP.md` setting out the scope of planned work.
- This changelog.

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

[Unreleased]: https://github.com/hawkers63/AudioDeNoiser/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/hawkers63/AudioDeNoiser/releases/tag/v1.0.0
