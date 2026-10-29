# Changelog

All notable changes to AudioDeNoiser are recorded here. Versions follow [Semantic Versioning](https://semver.org/), and each release is published on the [Releases page](https://github.com/hawkers63/AudioDeNoiser/releases).

Copyright (c) 2026-present Mark Hawksworth (hawkers63). All rights reserved.

## [Unreleased]

## [1.7.0] - 2026-10-29

With Restore consonants left off, which is how it ships, Manual and every scene shipped in 1.6.0 are unchanged. The switch is not stored in a named preset. On the measured helmet take, the pass with the switch off was the clearer one. Further samples may be tried later. They are not part of this release.

### Added

- **Restore consonants**, off by default. After the dialogue lift, on voiced frames only, it puts back presence (1.8–4 kHz) and consonants (4–6.5 kHz) that processing made darker than the dry balance against the vowel. It does not go brighter than the dry frame, and it leaves pauses alone. The switch is not stored in a named preset. With the switch off, Manual and the scenes shipped in 1.6.0 are unchanged. On the measured helmet take, the pass with the switch off was the clearer one, so the switch stays off.

### Changed

- The window is 860×1054, and the smallest size is 760×1036, so the Restore consonants row, the waveform and the log all remain visible.

## [1.6.0] - 2026-10-29

Manual, and every scene shipped in 1.5.0, is unchanged. Denoise still does the same work. A named preset remembers the controls on screen. It does not change the samples. The consonant darkening measured on one outdoor take is not repaired here. That work is scheduled as 1.7.0.

### Added

- **Save preset…** on the scene row stores the controls on screen under a name and adds that name to the Scene menu. Delete removes a saved name. The six factory scenes stay. A preset remembers the scene, dialogue, noise reduction, rumble choice, post-filter and normalise. Choosing it applies the scene first, then those controls, so a change you saved is not replaced by the scene's own defaults. A named scene still forces the rumble filter off. The list is kept in `settings.json`.

## [1.5.0] - 2026-10-29

Manual, and every scene shipped in 1.4.0, is unchanged. Denoise still does the same work. What changed is when the WAV is written.

### Added

- **Save denoised…** on the action row, between Denoise and Cancel. It asks for a file name, suggests `<name>_denoised.wav` in the Save to folder, and writes that WAV at the source rate and bit depth. Choosing another folder in the dialog does not change the Save to preference. The file you opened cannot be chosen as the destination. A failed write removes its temporary file and leaves the session unsaved.
- Closing the window, opening another file, or pressing Denoise again asks before an unsaved pass is dropped. Yes writes the file and then continues. No drops the pass. Cancel leaves the session as it is. The window stays open while a save is still being written.

### Changed

- **Denoise no longer writes a WAV.** The result stays in memory, and B plays that 48 kHz buffer. **Normalise saved file to −1 dBFS** applies when you save, not when Denoise finishes, so turning it on or off after a render changes the file you would write and leaves playback as it was. Match A/B loudness is still playback only.
- The **English transcript after denoise** switch still reads the denoised audio after Denoise, and it no longer waits for a WAV. Save text writes `<name>_transcript.srt` and `.txt` until a denoised WAV has been saved, then writes beside that WAV.
- Cutting a session that has denoised audio marks that export unsaved, because the WAV on disk would be the uncut take. Undo puts the audio back and restores whether it still needed saving.

## [1.4.0] - 2026-10-29

Manual, and every scene shipped in 1.3.0, is unchanged. The tools below edit the session in memory, or write a copy. None of them runs inside `denoise()`.

### Added

- **Shift-drag** on the waveform marks a range of at least 50 ms. A plain click or drag still seeks and leaves the range where it is. Escape clears it. The range is a stippled amber band behind the playhead, with its times on the new button row (`0:12 – 0:40`).
- **Loop** plays that range until Stop. Starting playback from outside the range jumps to its start. Stop returns to the beginning of the file and leaves the range marked.
- **Cut** removes the range from this working session. The file on disk is not opened for writing. Each side of the join fades over 2 ms, with no overlap, so the length shrinks by exactly the cut. One **Undo** puts that section back, including any transcript lines. A cut that would leave under a quarter of a second is refused. A later Denoise runs on what remains.
- **Save selection** writes the marked range of the buffer you are hearing (the denoised audio when B is selected, otherwise the original) to `<name>_selection.wav` in the output folder, then `<name>_selection_2.wav` and so on. It is the source rate and bit depth, including dithered 16-bit. It does not denoise or normalise.
- **Read speech** transcribes the buffer you are hearing, without a denoise first, in the same 20-second blocks as before, with progress and Cancel. The log names that buffer and says names and street noise will be misspelt. Lines follow the playhead as they arrive. **Save text** is available as soon as one line exists. Beside the denoised WAV when one has been written, otherwise `<name>_transcript.srt` and `<name>_transcript.txt` in the output folder.

### Changed

- The subtitle switch is labelled **English transcript after denoise**. It still reads the denoised audio after Denoise has saved the WAV. The recogniser is unchanged: wav2vec2-base, English read speech, no new package.
- The window is 860×1020, and the smallest size is 760×996, so the new row, the waveform and the log all remain visible.

## [1.3.0] - 2026-10-28

Manual, Heavy Wind & Gusts, Highway / Traffic Drone, Pedestrian Street and Subtle Podcast Clean are unchanged when Dialogue is left on Off, which is their default.

### Added

- **Conversation (1–5 m)**, for talk a few metres from the microphone. It keeps Pedestrian Street's 18 dB blend and its idle-tone notch, turns the post-filter off so quiet consonants are not stripped, ducks gusts only under 110 Hz (4 dB, and 1.5 dB while the fragment is voiced), and gates the steady floor from 2.2 kHz to 7 kHz by 6 dB, above the main vowel formants. Dialogue starts on Clear.
- **Dialogue** menu: Off, Gentle (up to 3 dB) or Clear (up to 6 dB). After the model it lifts 1.8–4 kHz, and 4–6.5 kHz by half that amount, on voiced speech and on consonants within 80 ms of a vowel, and only where that band is darker than close speech. The vowel's body (300 Hz–1 kHz) is never boosted. A frame quieter than −50 dBFS is left alone, and a stretch of noise with no vowel nearby is left alone. If the boost would push the chunk past 0.99, only the added part is turned down. Off returns the samples untouched. The menu follows the scene, and the choice in front of you is what Denoise uses. Stored as `dialogue_lift` (`off`, `gentle` or `clear`). A missing key means the scene's own default.
- The close-speech balance used by that lift is −23.0 dB of presence against body, measured on the Hazel text-to-speech clip (voiced-frame median −23.1 dB, rounded to 0.5 dB). The presence-floor bias is 1.2, measured on steady noise (the band's RMS was 1.225 times the tracked minimum, rounded to 0.1), so the floor sits on the noise rather than under it. These are not a listening result.
- **Subtitles while playing**, off by default. Once the WAV is saved, the denoised audio is transcribed on this machine with wav2vec2-base, the model already included in torchaudio 2.1.2. No new package is installed. The words follow the playhead during playback. **Save text** writes `<name>_denoised.srt` and `<name>_denoised.txt` next to the WAV. The first run downloads the weights (MIT licence) to `D:\Audio\Models`, or to a `models` folder in the output directory if that drive cannot be used. There is no language model: no punctuation, no capitals, and names and street noise will be misspelt. The log says so the first time a transcript is made. Cancel during subtitles keeps the WAV and any lines already recognised. Stored as `subtitles` (missing means off). On this laptop, with the weights already on disk, the 18-second Hazel clip took 20 seconds, including loading the recogniser. Six of eight marker words came back (`kit` was heard as `kick`). That is a clean-voice smoke test, not an outdoor accuracy figure, and it is not a live microphone.

### Changed

- The window is 860×978, and the smallest size is 760×954, so the dialogue row, the subtitle line and the log all remain visible.

## [1.2.1] - 2026-10-28

### Changed

- Heavy Wind & Gusts and Highway / Traffic Drone render 11-23 % faster, about a second less per 30 s of audio, with exactly the same output. The spectral floor that softens isolated hiss-band peaks after the network took a 5 × 7 median over all 1,025 frequency bins and then kept only its own band; it now filters just that band (2-5 kHz on Heavy Wind, 1.2-4 kHz on Highway) and the two bins either side that the median reaches. The stage itself is 2.5-3.3 times faster. Verified with DeepFilterNet 3: the 48 kHz buffer and the saved file are byte-identical to 1.2.0 on every scene, mono and stereo, 44.1 and 48 kHz, 16 and 24-bit, one to four chunks.

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

[Unreleased]: https://github.com/hawkers63/AudioDeNoiser/compare/v1.7.0...HEAD
[1.7.0]: https://github.com/hawkers63/AudioDeNoiser/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/hawkers63/AudioDeNoiser/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/hawkers63/AudioDeNoiser/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/hawkers63/AudioDeNoiser/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/hawkers63/AudioDeNoiser/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/hawkers63/AudioDeNoiser/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/hawkers63/AudioDeNoiser/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/hawkers63/AudioDeNoiser/releases/tag/v1.1.0
[1.0.0]: https://github.com/hawkers63/AudioDeNoiser/releases/tag/v1.0.0
