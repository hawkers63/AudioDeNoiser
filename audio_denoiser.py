#!/usr/bin/env python3
"""
Audio DeNoiser 1.7
==================

Copyright (c) 2026-present Mark Hawksworth (hawkers63). All rights reserved.
Proprietary and confidential. Unauthorised copying, distribution, modification,
reverse engineering or commercial use of this file, in whole or in part, is
strictly prohibited without prior written consent. See LICENSE for full terms.

Removes traffic, wind and general city noise from speech recordings with
DeepFilterNet 3, and lets you compare before/after instantly.

Platform : Windows 11 (also runs on Linux/macOS for development)
Python   : 3.11  - DeepFilterNet 0.5.6 only ships Windows wheels for 3.8-3.11
Setup    : run AudioDeNoiser.bat once (creates .venv and installs everything),
           or install manually into a Python 3.11 venv:
    pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
    pip install deepfilternet==0.5.6 numpy==1.26.4 scipy==1.17.1 soundfile==0.14.0
                sounddevice==0.5.6 customtkinter==6.0.0

Usage    : python audio_denoiser.py [optional audio file]
Keys     : Space play/pause, A original, B denoised, Ctrl+O open, Ctrl+D denoise,
           Escape clears a waveform selection

How the processing chain works
------------------------------
1. Decode WAV/MP3/FLAC/OGG with soundfile and resample to 48 kHz (the model's rate).
2. Manual runs the 1.0 chain: an optional high-pass below 60-120 Hz, then the
   model. A named scene instead conditions each chunk before the model, with an
   infrasonic filter, a gust duck and an idle-tone stage. Heavy Wind and Highway
   first repair isolated clicks and low pops (the transient guard).
3. DeepFilterNet 3 in 30-second chunks, each with a 2-second pre-roll so the
   model has settled before its output is used, joined with 0.25 s crossfades.
   Chunking gives a real progress bar, a working Cancel button and flat memory use.
   After the model, Heavy Wind and Highway soften isolated spectral peaks, and
   Wind, Highway, Street and Conversation lower the steady residual floor (the
   floor gate). Conversation, and any scene whose Dialogue control is above Off,
   then lifts the consonant band on voiced frames and on consonants beside a vowel.
   Restore consonants is a separate switch, off unless it is turned on. It then
   puts back presence and air that processing made darker than the dry balance
   against the vowel, and only on voiced frames.
4. The denoised audio stays in the session. Save denoised… resamples it back to
   the source rate, optionally peak-normalises, and writes a WAV at the source's
   bit depth (24-bit for MP3 and OGG). Denoise itself does not write that file.
5. A/B playback can be loudness-matched (ITU-R BS.1770-4). The file is untouched.
   Shift-drag on the waveform marks a range: loop it, drop it from this session
   (the file on disk stays), or save that range on its own. Read speech transcribes
   the buffer you are hearing. With the transcript switch on, the denoised buffer
   is transcribed after Denoise. Save text writes an SRT and a plain transcript.
"""
from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from math import gcd
from pathlib import Path

# pythonw.exe starts without a console, so sys.stdout / sys.stderr are None.
# Some libraries write to them regardless, so give them somewhere harmless to go.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
from scipy.linalg import LinAlgError, solve_toeplitz, solveh_banded
from scipy.ndimage import maximum_filter1d, median_filter, minimum_filter1d, uniform_filter1d
from scipy import fft as sp_fft
from scipy.signal import (butter, get_window, resample_poly, sosfilt, sosfilt_zi,
                          sosfiltfilt, stft)
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import customtkinter as ctk

try:
    import sounddevice as sd
    SD_IMPORT_ERROR = ""
except Exception as exc:  # PortAudio missing or no audio subsystem
    sd = None
    SD_IMPORT_ERROR = str(exc)


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
APP_NAME = "Audio DeNoiser"
APP_VERSION = "1.7.0"

MODEL_SR = 48_000
SUPPORTED_EXTS = (".wav", ".mp3", ".flac", ".ogg")

DEFAULT_OUTPUT_DIR = Path(r"D:\Audio\Denoised")
FALLBACK_OUTPUT_DIR = Path.home() / "Music" / "Denoised"
CONFIG_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "AudioDeNoiser"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
ERROR_LOG = CONFIG_DIR / "error.log"

CHUNK_SECONDS = 30.0
PREROLL_SECONDS = 2.0
CROSSFADE_SECONDS = 0.25
NORMALISE_PEAK_DBFS = -1.0
LONG_FILE_MINUTES = 45

# Export keeps the source's bit depth when it has one. Anything else is 24-bit.
EXPORT_SUBTYPES = ("PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE")
DEPTH_LABELS = {"PCM_16": "16-bit", "PCM_24": "24-bit", "PCM_32": "32-bit",
                "FLOAT": "32-bit float", "DOUBLE": "64-bit float"}
DITHER_SEED = 1770  # fixed, so the same render always writes the same 16-bit file

# Noise-reduction limit steps for the slider. None = no limit (model default).
ATTEN_STEPS = [6, 9, 12, 15, 18, 24, 30, 40, 60, None]
HIGHPASS_CHOICES = {"Off": None, "60 Hz": 60, "80 Hz": 80, "100 Hz": 100, "120 Hz": 120}

# Palette: streetlight amber on wet asphalt.
C_BG = "#1F2429"
C_PANEL = "#283037"
C_LINE = "#3A444D"
C_LINE_HOVER = "#475360"
C_TEXT = "#ECE6DA"
C_MUTED = "#8E98A1"
C_AMBER = "#F2A93B"
C_AMBER_HOVER = "#DB9227"
C_ON_AMBER = "#1F2429"
C_CONCRETE = "#66717B"
C_DANGER = "#B4483C"
C_DANGER_HOVER = "#963A30"
FONT_FAMILY = "Segoe UI"


# --------------------------------------------------------------------------- #
#  Audio engine (no Tk in here - safe to call from worker threads)
# --------------------------------------------------------------------------- #
class Cancelled(Exception):
    """Raised inside the worker when the user presses Cancel."""


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Decode a file to float32 (channels, frames) plus its sample rate."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[0] == 0:
        raise ValueError("The file contains no audio.")
    return np.ascontiguousarray(data.T), int(sr)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High-quality polyphase resampling along the last axis."""
    if sr_in == sr_out:
        return audio
    g = gcd(sr_in, sr_out)
    out = resample_poly(audio, sr_out // g, sr_in // g, axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)


def highpass(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Zero-phase Butterworth high-pass (24 dB/octave) for wind and traffic rumble."""
    if audio.shape[-1] < sr // 10:
        return audio
    sos = butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return np.ascontiguousarray(sosfiltfilt(sos, audio, axis=-1), dtype=np.float32)


# Scene conditioner. Manual never calls this. Named scenes run it on each
# extended chunk, before DeepFilterNet, and the spectral floor runs afterwards.
GUST_HOP = 480
VOICE_FRAME = 2048
VOICE_HOP = 480
F0_MIN_HZ = 70.0
F0_MAX_HZ = 320.0
FLOOR_FRAME = 2048
FLOOR_HOP = 1024  # 50% Hann overlap, which satisfies NOLA
NOTCH_FRAME = 8192
NOTCH_HOP = 4096  # 5.86 Hz bins, 50% overlap
VOICE_HIGHBAND_HZ = 280.0
# Frames whose energy above VOICE_HIGHBAND_HZ is more than this far below the
# frame energy are rumble or silence. Their autocorrelation is meaningless.
VOICE_HIGHBAND_MARGIN_DB = 30.0
VOICING_MIN = 0.33

# Transient guard. A click is short, broadband and sits on an unchanged
# background. A consonant burst is short too, but it comes out of a closure and
# runs into aspiration or a vowel, so the sound either side of it differs. A pop
# is low, brief and unvoiced. Each detector finds the event on a short envelope,
# checks the sound around it on a longer one, then repairs at sample accuracy.
CLICK_HP_HZ = 3000.0
CLICK_HOP = 24                  # 0.5 ms blocks
CLICK_FINE = 2                  # blocks per window that finds the click (1 ms)
CLICK_COARSE = 20               # blocks per window that judges its surroundings (10 ms)
CLICK_MEDIAN_S = 0.100          # running-median background, ±50 ms
CLICK_DB = 10.0                 # how far a click must stand above the background
CLICK_EXTENT_DB = 6.0           # the click's own extent on the short envelope
CLICK_MAX_S = 0.006             # longer than this is speech, not a click
CLICK_UNIQUE_S = 0.040          # a click also tops every other peak this close...
CLICK_UNIQUE_DB = 6.0           # ...by this much: glottal pulses, even creaky ones, never do
CLICK_FLAT_DB = 6.0             # either side must stay this close to its own median
CLICK_SAMPLE_RATIO = 3.0        # samples this far above the background RMS are the click
CLICK_MARGIN = 32               # samples added either side, so the fill meets clean audio
CLICK_AR_ORDER = 48
CLICK_CONTEXT = 1024            # samples either side used to fit the AR model
CLICK_GUARD_SPAN = 512          # a fill may not be louder than this much audio either side...
CLICK_GUARD_DB = 3.0            # ...by more than this
THUMP_LP_HZ = 200.0
THUMP_HOP = 120                 # 2.5 ms blocks
THUMP_FINE = 4                  # 10 ms
THUMP_COARSE = 16               # 40 ms
THUMP_MEDIAN_S = 0.200
THUMP_DB = 14.0
THUMP_EXTENT_DB = 6.0
THUMP_MAX_S = 0.060             # longer low-band bursts are gusts, left to the gust duck
THUMP_FLAT_DB = 8.0             # low-band noise swings more than the high band
THUMP_MAX_DB = 30.0
THUMP_VOICE_GUARD_S = 0.030     # no voiced frame this close to a pop we repair
TRANS_REACH_S = 0.040           # how far either side the surroundings are judged
TRANS_FLOOR_DBFS = -90.0        # envelopes never read below this, so near-silence can't spike
TRANS_MAX_EVENTS = 512

# Noise floor gate: minimum statistics with a Wiener-shaped gain.
GATE_FRAME = 1024
GATE_HOP = 256                  # 75 % Hann overlap, 5.33 ms
GATE_SMOOTH_FRAMES = 3
GATE_FLOOR_BINS = 5             # a broadband floor is smooth across ~230 Hz
GATE_TRACK_S = 0.6              # longer than a held fricative or vowel
GATE_BIAS = 3.6                 # minimum-to-RMS on Gaussian noise, for the three lines above
GATE_OVERSUB = 1.6              # the gain is fully closed within 4 dB of the floor

# Dialogue lift. Off (0 dB) returns the audio untouched. Gentle is 3 dB and
# Clear is 6 dB, both caps on the presence band. The air band takes half.
# DIALOGUE_TARGET_TILT_DB is the presence-minus-body balance of close speech
# (Hazel TTS, dry, voiced-frame median −23.1 dB, rounded to 0.5 dB).
# DIALOGUE_FLOOR_BIAS lifts the presence-band minimum up to its RMS
# (Gaussian noise, median ratio 1.225, rounded to 0.1). See the note 005 review.
DIALOGUE_FRAME = 1024
DIALOGUE_HOP = 256
DIALOGUE_BODY = (300.0, 1000.0)
DIALOGUE_PRESENCE = (1800.0, 4000.0)
DIALOGUE_AIR = (4000.0, 6500.0)
DIALOGUE_TARGET_TILT_DB = -23.0
DIALOGUE_AIR_SHARE = 0.5
DIALOGUE_CONSONANT_MARGIN_DB = 6.0
DIALOGUE_FLATNESS_MIN = 0.25
DIALOGUE_VOICE_REACH_S = 0.080
DIALOGUE_SILENCE_DBFS = -50.0   # time-domain frame RMS; a quiet consonant still passes
DIALOGUE_FLOOR_BIAS = 1.2
DIALOGUE_EDGE_BINS = 4
DIALOGUE_CEILING = 0.99
# Consonant restore. Off leaves every sample untouched. The cap is a guard
# above one file's measured gap, not a target fitted to that file. The share
# puts back the whole darkened gap up to the dry balance, and no further.
RESTORE_MAX_DB = 12.0
RESTORE_SNAP_DB = 0.3
RESTORE_SHARE = 1.0
RESTORE_BODY_MARGIN_DB = 30.0
LIFT_GENTLE_DB = 3.0
LIFT_CLEAR_DB = 6.0
LIFT_CHOICES = {"Off": 0.0, "Gentle": LIFT_GENTLE_DB, "Clear": LIFT_CLEAR_DB}

# Subtitles. The recogniser is wav2vec2-base, already in torchaudio 2.1.
# Its convolutional front end strides by 5×2×2×2×2×2×2 = 320 samples at 16 kHz.
ASR_CHUNK_S = 20.0
ASR_OVERLAP_S = 1.0
ASR_SR = 16_000
CTC_STRIDE = 320
CUE_MAX_CHARS = 42
CUE_MAX_S = 6.0
CUE_GAP_S = 0.8
CUE_MIN_S = 1.0
CUE_TAIL_S = 0.2

# A Shift-drag shorter than this is ignored. A cut must leave at least a quarter
# of a second, which is also the shortest file the loader accepts. The join
# fades 2 ms on each kept edge and does not overlap, so the length shrinks by
# exactly the cut and cue times stay exact.
SELECTION_MIN_S = 0.05
SELECTION_MIN_KEEP_S = 0.25
JOIN_FADE_S = 0.002

_SOS_CACHE: dict[tuple, np.ndarray] = {}


def lift_label(lift_db: float) -> str:
    """Menu label for a dialogue-lift cap in decibels."""
    if lift_db <= 0.0:
        return "Off"
    if lift_db < (LIFT_GENTLE_DB + LIFT_CLEAR_DB) / 2.0:
        return "Gentle"
    return "Clear"


@dataclass(frozen=True)
class ScenePreset:
    """One outdoor-scene profile.

    ``atten_db`` is the DeepFilterNet blend limit. ``None`` on a stage switches
    that stage off. Manual leaves the existing rumble filter and slider in charge.
    """

    key: str
    label: str
    summary: str
    atten_db: int | None
    post_filter: bool
    post_filter_beta: float
    infrasonic_hz: float | None
    gust_split_hz: float | None
    gust_max_db: float
    gust_margin_db: float
    gust_voiced_cap_db: float
    gust_attack_s: float
    gust_release_s: float
    notch_lo_hz: float | None
    notch_hi_hz: float | None
    notch_max_db: float
    notch_max_count: int
    hum_hz: tuple[float, ...]
    floor_lo_hz: float | None
    floor_hi_hz: float | None
    floor_peak_db: float
    floor_max_db: float
    transient_guard: bool
    gate_db: float
    gate_lo_hz: float | None
    gate_hi_hz: float | None
    lift_db: float

    @property
    def active(self) -> bool:
        return self.key != "manual"

    @property
    def uses_floor(self) -> bool:
        return self.floor_lo_hz is not None and self.floor_hi_hz is not None

    @property
    def uses_gate(self) -> bool:
        return self.gate_db > 0.0 and self.gate_lo_hz is not None and self.gate_hi_hz is not None

    @property
    def uses_lift(self) -> bool:
        return self.lift_db > 0.0


class ConditionerState:
    """Reusable scratch, the slow low-band floor carried across chunks, and the
    transients repaired in the most recent chunk as (kind, start sample)."""

    def __init__(self) -> None:
        self.low_floor_db: float | None = None
        self.events: list[tuple[str, int]] = []
        self._scratch: dict[str, np.ndarray] = {}

    def buffer(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        """Return a float64 scratch buffer, allocating only when the shape changes."""
        current = self._scratch.get(name)
        if current is None or current.shape != shape:
            current = np.empty(shape, dtype=np.float64)
            self._scratch[name] = current
        return current


def _sos(kind: str, sr: int, hz: float, order: int = 4) -> np.ndarray:
    """Design a Butterworth SOS, cached for the rest of the process."""
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    nyquist = sr / 2
    if not 1.0 < hz < nyquist - 1.0:
        raise ValueError(f"Cutoff {hz:g} Hz is outside the usable band at {sr} Hz.")
    key = (kind, sr, round(float(hz), 3), order)
    cached = _SOS_CACHE.get(key)
    if cached is not None:
        return cached
    if kind == "high":
        sos = butter(order, hz, btype="highpass", fs=sr, output="sos")
    elif kind == "low":
        sos = butter(order, hz, btype="lowpass", fs=sr, output="sos")
    else:
        raise ValueError(f"Unknown filter kind {kind!r}.")
    sos = np.asarray(sos, dtype=np.float64)
    _SOS_CACHE[key] = sos
    return sos


def _linkwitz_riley(sr: int, hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Fourth-order Linkwitz-Riley pair: two cascaded Butterworth sections of order 2.

    The low and high outputs sum to a flat, time-aligned all-pass, so ducking
    only the low output does not comb against the high output.
    """
    low = _sos("low", sr, hz, order=2)
    high = _sos("high", sr, hz, order=2)
    return np.vstack((low, low)), np.vstack((high, high))


def _apply_sos(audio: np.ndarray, sos: np.ndarray, state: ConditionerState, name: str) -> np.ndarray:
    """Causal SOS filter with the initial state set from the first sample.

    Seeding the state from the first sample keeps a gust from producing a click
    while the filter wakes up. The 2 s chunk pre-roll then finishes the settling.
    """
    out = state.buffer(name, audio.shape)
    zi_unit = sosfilt_zi(sos)
    for channel in range(audio.shape[0]):
        initial = zi_unit * float(audio[channel, 0])
        filtered, _ = sosfilt(sos, audio[channel], zi=initial)
        out[channel] = filtered
    return out


def _safe_filtfilt(audio: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Zero-phase SOS filter. Short fragments keep the input rather than raising."""
    try:
        return sosfiltfilt(sos, audio, axis=-1)
    except ValueError:
        return audio


def _frames(mono: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Read-only framed view. The caller must copy before writing."""
    if mono.shape[0] < frame:
        return np.zeros((0, frame), dtype=np.float64)
    count = 1 + (mono.shape[0] - frame) // hop
    stride = mono.strides[0]
    view = np.lib.stride_tricks.as_strided(
        mono, shape=(count, frame), strides=(hop * stride, stride), writeable=False
    )
    return view


def estimate_voicing(mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Periodicity and f0, one value per ``VOICE_HOP`` samples.

    The spectrum is flattened with a short cepstral envelope so tyre hiss does
    not swamp a vowel, then bins below 280 Hz are removed before the
    autocorrelation. Wind and road rumble therefore cannot impersonate a voice.
    A frame whose remaining high-band energy is negligible is marked unvoiced:
    the normalised autocorrelation of near-silence is otherwise a coin toss.
    """
    empty = np.zeros(0, dtype=np.float64)
    if mono.shape[0] < VOICE_FRAME:
        return empty, empty
    frames = _frames(np.ascontiguousarray(mono, dtype=np.float64), VOICE_FRAME, VOICE_HOP)
    if len(frames) == 0:
        return empty, empty
    spectrum = np.fft.rfft(frames * np.hanning(VOICE_FRAME), axis=-1)
    magnitude = np.abs(spectrum) + 1e-8
    cepstrum = np.fft.irfft(np.log(magnitude), n=VOICE_FRAME, axis=-1)
    keep = max(8, int(0.0015 * sr))
    cepstrum[:, keep:] = 0
    envelope = np.exp(np.fft.rfft(cepstrum, n=VOICE_FRAME, axis=-1).real) + 1e-8
    cutoff = int(VOICE_HIGHBAND_HZ / (sr / VOICE_FRAME))
    total = np.mean(magnitude[:, 1:] ** 2, axis=1)
    high = np.mean(magnitude[:, cutoff:] ** 2, axis=1)
    audible = high > total * 10 ** (-VOICE_HIGHBAND_MARGIN_DB / 10)
    flat = spectrum / envelope
    flat[:, :cutoff] = 0
    excitation = np.fft.irfft(flat, n=VOICE_FRAME, axis=-1)
    exc_spectrum = np.fft.rfft(excitation, axis=-1)
    autocorr = np.fft.irfft(exc_spectrum.real ** 2 + exc_spectrum.imag ** 2, n=VOICE_FRAME, axis=-1)
    autocorr /= np.maximum(autocorr[:, :1], 1e-12)
    lag_min = max(1, int(sr / F0_MAX_HZ))
    lag_max = min(VOICE_FRAME - 1, int(sr / F0_MIN_HZ))
    search = autocorr[:, lag_min:lag_max]
    peak = np.where(audible, search.max(axis=1), 0.0)
    lag = search.argmax(axis=1) + lag_min
    f0 = np.where(audible, sr / np.maximum(lag, 1), 0.0)
    return peak.astype(np.float64), f0.astype(np.float64)


def _hop_rms_db(mono: np.ndarray, hop: int) -> np.ndarray:
    count = mono.shape[0] // hop
    if count <= 0:
        rms = math.sqrt(float(np.mean(mono * mono)) + 1e-12) if mono.size else 1e-6
        return np.array([20 * math.log10(rms)], dtype=np.float64)
    trimmed = mono[: count * hop].reshape(count, hop)
    power = np.mean(trimmed * trimmed, axis=1)
    return 20 * np.log10(np.sqrt(power + 1e-12))


def _smooth_gain(target: np.ndarray, attack_s: float, release_s: float, hop_s: float) -> np.ndarray:
    """One-pole attack/release. Falling gain (a gust arriving) uses the attack time."""
    if target.size == 0:
        return target
    attack = math.exp(-hop_s / max(attack_s, 1e-4))
    release = math.exp(-hop_s / max(release_s, 1e-4))
    out = np.empty_like(target)
    current = float(target[0])
    for index, sample in enumerate(target):
        value = float(sample)
        coeff = attack if value < current else release
        current = current + (1.0 - coeff) * (value - current)
        out[index] = current
    return out


def _expand_hops(gain: np.ndarray, hop: int, length: int) -> np.ndarray:
    """Hold each hop gain across its samples. The final hop covers the tail."""
    expanded = np.repeat(gain, hop)
    if expanded.size < length:
        expanded = np.pad(expanded, (0, length - expanded.size), mode="edge")
    return expanded[:length]


def _track_floor(state: ConditionerState, low_db: np.ndarray) -> float:
    """Slow floor: drop immediately when a quieter chunk arrives, rise very slowly.

    A chunk that is entirely wind would otherwise treat the gust as the floor and
    refuse to duck it. Carrying the quietest floor forward fixes later chunks.
    The first chunk still depends on having a quiet moment of its own.
    """
    local = float(np.percentile(low_db, 20)) if low_db.size else -80.0
    if not math.isfinite(local):
        local = -80.0
    held = state.low_floor_db
    if held is None or local < held:
        state.low_floor_db = local
    else:
        state.low_floor_db = held + 0.05 * (local - held)
    return float(state.low_floor_db)


def _debounce_voicing(periodicity: np.ndarray) -> np.ndarray:
    """Drop one-frame spikes. A vowel lasts far longer than five 10 ms hops."""
    if periodicity.size < 5:
        return periodicity
    return median_filter(periodicity, size=5)


def apply_gust_duck(
    audio: np.ndarray,
    sr: int,
    preset: ScenePreset,
    state: ConditionerState,
    periodicity: np.ndarray | None = None,
) -> np.ndarray:
    """Duck gust energy below ``gust_split_hz`` while a voice is sounding.

    The split is Linkwitz-Riley, so the unducked high band stays sample-aligned
    with the low band. Voiced hops are capped at ``gust_voiced_cap_db`` and
    therefore keep their fundamental. Pass the periodicity measured on the dry
    chunk: the infrasonic filter rings, and that ring can look like a vowel.
    """
    if preset.gust_split_hz is None or audio.shape[-1] < sr // 5:
        return audio
    low_sos, high_sos = _linkwitz_riley(sr, preset.gust_split_hz)
    low = _apply_sos(audio, low_sos, state, "gust-low")
    high = _apply_sos(audio, high_sos, state, "gust-high")
    mono_low = low.mean(axis=0)
    low_db = _hop_rms_db(mono_low, GUST_HOP)
    floor_db = _track_floor(state, low_db)
    excess = low_db - floor_db - preset.gust_margin_db
    raw_db = np.clip(excess, 0.0, preset.gust_max_db)

    if periodicity is None:
        voicing, _f0 = estimate_voicing(np.ascontiguousarray(audio.mean(axis=0)), sr)
        voicing = _debounce_voicing(voicing)
    else:
        voicing = periodicity
    hops = min(raw_db.shape[0], voicing.shape[0])
    if hops == 0:
        return low + high
    voiced = voicing[:hops] >= VOICING_MIN
    raw_db = raw_db[:hops]
    raw_db = np.where(voiced, np.minimum(raw_db, preset.gust_voiced_cap_db), raw_db)
    target = np.power(10.0, -raw_db / 20.0)
    smoothed = _smooth_gain(target, preset.gust_attack_s, preset.gust_release_s, GUST_HOP / sr)
    gain = _expand_hops(smoothed, GUST_HOP, audio.shape[-1])
    ducked = low * gain
    return ducked + high


def _tone_spectrum(mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Long-window STFT used to tell an idle tone from a vowel."""
    _freqs, _times, spectrum = stft(
        mono,
        fs=sr,
        window="hann",
        nperseg=NOTCH_FRAME,
        noverlap=NOTCH_FRAME - NOTCH_HOP,
        boundary=None,
        padded=False,
    )
    return np.abs(spectrum), spectrum


def _prominence(median: np.ndarray) -> np.ndarray:
    """How far each bin stands above the median of the bins just beside it.

    The four bins on either side, skipping the two closest, are the local
    floor. A broad rumble shelf is as loud as that floor, so its prominence is
    about 0 dB and it is left alone. A narrow idle tone stands proud of it.
    """
    padded = np.pad(median, 8, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 4)
    # Centre of bin i sits at padded[i + 8]. These windows cover offsets -6..-3
    # and +3..+6, leaving the peak's own shoulders out of the floor estimate.
    left = windows[2 : 2 + median.size]
    right = windows[11 : 11 + median.size]
    neighbour = np.median(np.concatenate((left, right), axis=1), axis=1)
    return 20 * np.log10((median + 1e-12) / (neighbour + 1e-12))


def _refine_peak(median: np.ndarray, index: int, bin_hz: float) -> float:
    """Parabolic interpolation, so a tone between FFT bins is not missed."""
    if index <= 0 or index >= median.size - 1:
        return index * bin_hz
    alpha, beta, gamma = median[index - 1], median[index], median[index + 1]
    denom = alpha - 2 * beta + gamma
    if abs(denom) < 1e-12:
        return index * bin_hz
    delta = float(np.clip(0.5 * (alpha - gamma) / denom, -0.5, 0.5))
    return (index + delta) * bin_hz


def find_stable_tones(
    magnitude: np.ndarray,
    sr: int,
    lo_hz: float | None,
    hi_hz: float | None,
    max_count: int,
    forced_hz: tuple[float, ...] = (),
    prominence_db: float = 7.0,
    stability_min: float = 0.72,
) -> list[float]:
    """Frequencies that stay put for the whole fragment.

    A vowel comes and goes, so its median sits far below its loud frames and it
    is rejected. An engine idle or a mains hum is present in almost every frame,
    so the median nearly equals the 90th percentile and the tone is kept.
    """
    if max_count <= 0 or magnitude.shape[1] < 4:
        return []
    median = np.median(magnitude, axis=1)
    upper = np.percentile(magnitude, 90, axis=1)
    stability = median / (upper + 1e-12)
    prominence = _prominence(median)
    bin_hz = sr / NOTCH_FRAME
    chosen: list[tuple[float, float]] = []

    def consider(frequency: float) -> None:
        if not 20.0 < frequency < sr / 2 - 50.0:
            return
        index = int(round(frequency / bin_hz))
        index = max(1, min(magnitude.shape[0] - 2, index))
        if stability[index] < stability_min or prominence[index] < prominence_db:
            return
        if any(abs(frequency - previous) < 18.0 for previous, _score in chosen):
            return
        chosen.append((frequency, float(prominence[index])))

    if lo_hz is not None and hi_hz is not None and hi_hz > lo_hz:
        first = max(1, int(math.ceil(lo_hz / bin_hz)))
        last = min(magnitude.shape[0] - 2, int(math.floor(hi_hz / bin_hz)))
        order = np.argsort(prominence[first : last + 1])[::-1]
        for offset in order:
            if len(chosen) >= max_count:
                break
            index = first + int(offset)
            if stability[index] < stability_min or prominence[index] < prominence_db:
                continue
            if prominence[index] < prominence[index - 1] or prominence[index] < prominence[index + 1]:
                continue
            frequency = _refine_peak(median, index, bin_hz)
            if any(abs(frequency - previous) < 18.0 for previous, _score in chosen):
                continue
            chosen.append((frequency, float(prominence[index])))
    for frequency in forced_hz:
        if len(chosen) >= max_count:
            break
        consider(frequency)
    chosen.sort(key=lambda item: item[1], reverse=True)
    return [frequency for frequency, _score in chosen[:max_count]]


def _istft(spectrum: np.ndarray, sr: int, frame: int, hop: int) -> np.ndarray:
    """Inverse STFT matching ``_tone_spectrum``, the spectral floor and the gate.

    The same arithmetic as ``scipy.signal.istft`` with a periodic Hann window,
    no boundary extension and spectrum scaling, but with the overlap-add done a
    hop-sized block at a time instead of a Python loop over every frame. Each
    output sample still sums its frames in the same order, so the result is
    bit-for-bit the same; the frame size must be a whole number of hops.
    """
    window = get_window("hann", frame)
    frames = sp_fft.irfft(spectrum, axis=-2, n=frame)[:frame, :]
    frames *= window.sum()
    count = frames.shape[1]
    ratio = frame // hop
    restored = np.zeros((count + ratio - 1, hop), dtype=frames.dtype)
    norm = np.zeros((count + ratio - 1, hop), dtype=frames.dtype)
    weighted = frames * window[:, None]
    squared = window * window
    # Frame i lands its block j on output block i + j. Taking j from the top
    # down adds each output block's frames in rising order, as scipy does.
    for block in reversed(range(ratio)):
        part = slice(block * hop, (block + 1) * hop)
        restored[block:block + count] += weighted[part].T
        norm[block:block + count] += squared[part]
    restored = restored.ravel()
    restored /= np.where(norm.ravel() > 1e-10, norm.ravel(), 1.0)
    return restored


def _apply_spectral_gain(
    audio: np.ndarray, sr: int, delta_gain: np.ndarray, frame: int, hop: int
) -> np.ndarray:
    """Add the inverse STFT of ``spectrum * delta_gain`` to the dry signal.

    Bins whose delta is zero are not rewritten, so a stage that leaves a bin
    alone cannot smear it through the inverse transform.

    ``delta_gain`` is laid out on the caller's grid: frames of ``audio`` with no
    boundary padding. The transform here pads ``frame - hop`` zeros in front
    and a full frame behind, and holds the first and last gain columns across
    the extra frames, so every sample of the buffer sits under full window
    overlap. Without that padding the inverse STFT divides by a window sum
    close to zero at the buffer edges, and any gain that is not flat across
    frequency comes out as a spike at the start and end of the buffer.
    """
    length = audio.shape[-1]
    lead = frame - hop
    shift = lead // hop
    cleaned = np.empty(audio.shape, dtype=np.float64)
    for channel in range(audio.shape[0]):
        padded = np.concatenate((
            np.zeros(lead), np.asarray(audio[channel], dtype=np.float64), np.zeros(frame)))
        _frequencies, _times, spectrum = stft(
            padded,
            fs=sr,
            window="hann",
            nperseg=frame,
            noverlap=frame - hop,
            boundary=None,
            padded=False,
        )
        columns = np.clip(np.arange(spectrum.shape[1]) - shift, 0, delta_gain.shape[1] - 1)
        bins = min(spectrum.shape[0], delta_gain.shape[0])
        shaped = np.zeros_like(spectrum)
        shaped[:bins] = spectrum[:bins] * delta_gain[:bins][:, columns]
        restored = _istft(shaped, sr, frame, hop)[lead:lead + length]
        if restored.shape[0] < length:
            restored = np.pad(restored, (0, length - restored.shape[0]))
        cleaned[channel] = audio[channel] + restored
    if not np.isfinite(cleaned).all():
        raise ValueError("Spectral gain produced invalid samples.")
    return cleaned


def suppress_stable_tones(
    audio: np.ndarray, sr: int, preset: ScenePreset, periodicity: np.ndarray | None = None
) -> np.ndarray:
    """Attenuate idle tones and confirmed hum without notching a vowel.

    Tones at or above 70 Hz, which is where a speaking fundamental can sit, are
    turned down only on frames the voicing detector has rejected. A steady
    fundamental is therefore left intact while it is being spoken, and an engine
    idle is still removed in the pauses and under unvoiced noise. Mains hum
    below 70 Hz is turned down on every frame, because it is under the voice.
    """
    hunting = preset.notch_lo_hz is not None and preset.notch_hi_hz is not None
    if not hunting and not preset.hum_hz:
        return audio
    if audio.shape[-1] < NOTCH_FRAME or preset.notch_max_db <= 0 or preset.notch_max_count <= 0:
        return audio
    mono = np.ascontiguousarray(audio.mean(axis=0), dtype=np.float64)
    magnitude, _spectrum = _tone_spectrum(mono, sr)
    tones = find_stable_tones(
        magnitude,
        sr,
        preset.notch_lo_hz if hunting else None,
        preset.notch_hi_hz if hunting else None,
        preset.notch_max_count,
        forced_hz=preset.hum_hz,
    )
    if not tones:
        return audio
    if periodicity is None:
        periodicity, _f0 = estimate_voicing(mono, sr)
        periodicity = _debounce_voicing(periodicity)
    n_bins, n_times = magnitude.shape
    if periodicity.size:
        source = np.arange(periodicity.size) * VOICE_HOP + VOICE_FRAME / 2
        target = np.arange(n_times) * NOTCH_HOP + NOTCH_FRAME / 2
        period_on = np.interp(target, source, periodicity, left=0.0, right=0.0)
    else:
        period_on = np.zeros(n_times)
    unvoiced = period_on < VOICING_MIN
    gain = np.ones((n_bins, n_times), dtype=np.float64)
    depth = 10 ** (-preset.notch_max_db / 20.0)
    bin_hz = sr / NOTCH_FRAME
    for frequency in tones:
        centre = frequency / bin_hz
        always = frequency < 70.0
        for offset in range(-3, 4):
            index = int(round(centre)) + offset
            if not 0 <= index < n_bins:
                continue
            # Full depth on the bin the tone actually occupies and its two
            # neighbours (the Hann main lobe). Further bins taper out so the
            # cut does not become a wide scoop in the vowel range.
            distance = abs(index - centre)
            if distance <= 1.25:
                weight = 1.0
            else:
                weight = max(0.0, 1.0 - (distance - 1.25) / 1.75)
            target_gain = 1.0 - weight * (1.0 - depth)
            if always:
                gain[index] = np.minimum(gain[index], target_gain)
            else:
                gain[index, unvoiced] = np.minimum(gain[index, unvoiced], target_gain)
    if np.all(gain > 0.999):
        return audio
    return _apply_spectral_gain(audio, sr, gain - 1.0, NOTCH_FRAME, NOTCH_HOP)


def _protect_harmonics(
    f0_hz: np.ndarray,
    periodicity: np.ndarray,
    n_bins: int,
    n_times: int,
    sr: int,
    frame: int = FLOOR_FRAME,
    top_hz: float | None = None,
) -> np.ndarray:
    """Mark bins that sit on a voiced harmonic, plus one bin either side.

    ``frame`` is the FFT size of the caller's grid. Without ``top_hz`` the
    first 16 harmonics are marked; with it, every harmonic up to that frequency.
    """
    protect = np.zeros((n_bins, n_times), dtype=bool)
    times = min(n_times, f0_hz.shape[0], periodicity.shape[0])
    if times == 0:
        return protect
    bin_hz = sr / frame
    valid = periodicity[:times] >= VOICING_MIN
    if not np.any(valid):
        return protect
    time_index = np.arange(times)
    usable = np.flatnonzero(valid)
    count = 16 if top_hz is None else int(math.ceil(top_hz / F0_MIN_HZ))
    for harmonic in range(1, count + 1):
        frequency = harmonic * f0_hz[:times][usable]
        keep = usable if top_hz is None else usable[frequency <= top_hz]
        if keep.size == 0:
            break
        centres = np.rint(harmonic * f0_hz[:times][keep] / bin_hz).astype(np.int32)
        for offset in (-1, 0, 1):
            index = np.clip(centres + offset, 0, n_bins - 1)
            protect[index, time_index[keep]] = True
    return protect


def _floor_peak_pull(magnitude: np.ndarray, sr: int, preset: ScenePreset) -> np.ndarray:
    """How far each bin in the scene's band stands above its local median, in dB.

    The local median is taken over 5 bins by 7 frames. The map covers the whole
    grid, zero outside the band, because the smoothing after it spreads the
    band's edges by a bin. Only the band's rows are filtered, plus the two rows
    either side that their median reaches: the median is the costliest step in
    the stage, and on the full grid most of it would go on rows the band then
    discards. Each kept row sees the same 35 values it would on the full grid,
    and a slice that reaches DC or Nyquist ends where the full grid does, so
    its reflected edge is the same too. The map is bit-for-bit the full grid's.
    """
    pull_db = np.zeros_like(magnitude)
    bin_hz = sr / FLOOR_FRAME
    rows = np.arange(magnitude.shape[0])
    band = np.flatnonzero(
        (rows >= preset.floor_lo_hz / bin_hz) & (rows <= preset.floor_hi_hz / bin_hz))
    if band.size == 0:
        return pull_db
    first, last = int(band[0]), int(band[-1]) + 1
    lo, hi = max(0, first - 2), min(magnitude.shape[0], last + 2)
    smoothed = median_filter(magnitude[lo:hi], size=(5, 7), mode="reflect")[first - lo:last - lo]
    excess_db = 20 * np.log10((magnitude[first:last] + 1e-8) / (smoothed + 1e-8))
    pull_db[first:last] = np.clip(excess_db - preset.floor_peak_db, 0.0, preset.floor_max_db)
    return pull_db


def soften_musical_noise(
    audio: np.ndarray,
    sr: int,
    preset: ScenePreset,
    voicing: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Pull isolated spectral peaks in the tyre band down towards the local median.

    Unity gains return the input unchanged, so speech that the detector leaves
    alone is not passed through an inverse STFT at all. Harmonics of a detected
    f0 are marked and left alone, which keeps vowels from being chiselled out.
    ``voicing`` is (debounced periodicity, f0) for this audio, when the caller
    has already measured it; otherwise it is measured here.
    """
    if not preset.uses_floor:
        return audio
    _channels, length = audio.shape
    if length < FLOOR_FRAME * 2:
        return audio
    mid = np.ascontiguousarray(audio.mean(axis=0), dtype=np.float64)
    _freqs, _times, mid_spectrum = stft(
        mid,
        fs=sr,
        window="hann",
        nperseg=FLOOR_FRAME,
        noverlap=FLOOR_FRAME - FLOOR_HOP,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(mid_spectrum)
    pull_db = _floor_peak_pull(magnitude, sr, preset)
    if voicing is None:
        periodicity, f0 = estimate_voicing(mid, sr)
        periodicity = _debounce_voicing(periodicity)
    else:
        periodicity, f0 = voicing
    # estimate_voicing hops by VOICE_HOP; the floor STFT hops by FLOOR_HOP.
    # Resample the f0 track onto the STFT grid with a nearest hold.
    if f0.size:
        source = np.arange(f0.size) * VOICE_HOP + VOICE_FRAME / 2
        target = np.arange(magnitude.shape[1]) * FLOOR_HOP + FLOOR_FRAME / 2
        f0_on_grid = np.interp(target, source, f0, left=f0[0], right=f0[-1])
        period_on_grid = np.interp(target, source, periodicity, left=periodicity[0], right=periodicity[-1])
    else:
        f0_on_grid = np.zeros(magnitude.shape[1])
        period_on_grid = np.zeros(magnitude.shape[1])
    protect = _protect_harmonics(
        f0_on_grid, period_on_grid, magnitude.shape[0], magnitude.shape[1], sr
    )
    pull_db = np.where(protect, 0.0, pull_db)
    gain = np.power(10.0, -pull_db / 20.0)
    gain = uniform_filter1d(gain, size=3, axis=0, mode="nearest")
    gain = uniform_filter1d(gain, size=3, axis=1, mode="nearest")
    gain = np.clip(gain, 10 ** (-preset.floor_max_db / 20.0), 1.0)
    # Ignore sub-audible nibbles so an untouched vowel is returned byte-for-byte.
    gain = np.where(gain > 0.99, 1.0, gain)
    if np.all(gain > 0.999):
        return audio

    return _apply_spectral_gain(audio, sr, gain - 1.0, FLOOR_FRAME, FLOOR_HOP)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous stretches of ``True``, as half-open (first, last) index pairs."""
    if not mask.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _block_power(signal: np.ndarray, hop: int) -> np.ndarray:
    """Mean power of consecutive ``hop``-sample blocks."""
    count = signal.shape[0] // hop
    blocks = signal[: count * hop].reshape(count, hop)
    return np.mean(blocks * blocks, axis=1)


def _window_db(power: np.ndarray, width: int) -> np.ndarray:
    """Level in dB of windows ``width`` blocks long; entry ``k`` starts at block ``k``.

    Direct sums rather than a running total, so a quiet passage after a loud
    one is not lost to rounding. The level never reads below
    ``TRANS_FLOOR_DBFS``: in a band that is all but empty, the flicker of the
    last few bits would otherwise look like a click.
    """
    if power.shape[0] < width:
        return np.zeros(0, dtype=np.float64)
    total = np.convolve(power, np.ones(width), mode="valid")
    return 10.0 * np.log10(total / width + 10.0 ** (TRANS_FLOOR_DBFS / 10.0))


def _steady_around(coarse_db: np.ndarray, width: int, first: int, end: int,
                   reach: int, tolerance_db: float) -> bool:
    """True when the sound either side of blocks ``first:end`` holds steady.

    Only windows that do not overlap the event are judged: ``reach`` blocks of
    them on each side. Each must lie within ``tolerance_db`` of their median,
    so there is no onset, offset or second burst nearby. Too close to either
    end of the buffer to see both sides counts as not steady.
    """
    lo = first - width - reach
    if lo < 0 or end + reach + 1 > coarse_db.shape[0]:
        return False
    around = np.concatenate((coarse_db[lo:first - width + 1], coarse_db[end:end + reach + 1]))
    reference = float(np.median(around))
    return bool(around.max() <= reference + tolerance_db
                and around.min() >= reference - tolerance_db)


def detect_clicks(mono: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Sample ranges holding an isolated click, tick or knock.

    The band above ``CLICK_HP_HZ`` is measured in 1 ms windows and compared with
    its own 100 ms running median. A click stands at least ``CLICK_DB`` above it
    for no more than a few milliseconds, and tops every other peak within
    40 ms by ``CLICK_UNIQUE_DB``: voiced speech puts a burst into this band at
    every glottal closure, but those come in a train of near-equals. The sound
    in 10 ms windows for 40 ms either side must also hold steady. A plosive
    burst fails that, because it comes out of a closure and runs into
    aspiration or a vowel. The range is then narrowed to the samples that
    carry the click, so the repair does not reach into the speech around it.
    """
    if mono.shape[0] < sr // 5:
        return []
    high = _safe_filtfilt(mono[None, :], _sos("high", sr, CLICK_HP_HZ, order=4))[0]
    power = _block_power(high, CLICK_HOP)
    level = _window_db(power, CLICK_FINE)
    coarse = _window_db(power, CLICK_COARSE)
    if coarse.size < 8:
        return []
    size = int(round(CLICK_MEDIAN_S * sr / CLICK_HOP)) | 1
    background = median_filter(level, size=size, mode="nearest")
    excess = level - background
    longest = int(round(CLICK_MAX_S * sr / CLICK_HOP))
    reach = int(round(TRANS_REACH_S * sr / CLICK_HOP))
    nearby = int(round(CLICK_UNIQUE_S * sr / CLICK_HOP))
    events: list[tuple[int, int]] = []
    for first, last in _runs(excess > CLICK_EXTENT_DB):
        peak = float(excess[first:last].max())
        if last - first > longest or peak < CLICK_DB:
            continue
        # One window either side of the run is left out of the comparison:
        # it overlaps the click itself.
        others = np.concatenate((excess[max(0, first - nearby):max(0, first - 1)],
                                 excess[last + 1:last + nearby]))
        if others.size and peak < float(others.max()) + CLICK_UNIQUE_DB:
            continue
        # Fine window k spans blocks k and k + 1, so the run spans blocks first..last.
        if not _steady_around(coarse, CLICK_COARSE, first, last + CLICK_FINE - 1,
                              reach, CLICK_FLAT_DB):
            continue
        lo = first * CLICK_HOP
        hi = min(mono.shape[0], (last + CLICK_FINE - 1) * CLICK_HOP)
        floor_rms = math.sqrt(10.0 ** (float(np.median(background[first:last])) / 10.0))
        loud = np.flatnonzero(np.abs(high[lo:hi]) > CLICK_SAMPLE_RATIO * floor_rms)
        if loud.size == 0:
            continue
        start = max(0, lo + int(loud[0]) - CLICK_MARGIN)
        end = min(mono.shape[0], lo + int(loud[-1]) + 1 + CLICK_MARGIN)
        events.append((start, end))
        if len(events) >= TRANS_MAX_EVENTS:
            break
    return events


def _ar_predictor(pieces: list[np.ndarray], order: int, correction: float) -> np.ndarray | None:
    """Prediction-error filter ``[1, -a1, ..., -ap]`` fitted to the given pieces.

    Each piece is Hann-windowed and its autocorrelation added in, so the gap
    between them never enters the fit. ``correction`` adds that fraction of
    white noise, which caps how deep the model's spectral valleys can go.
    ``None`` means there is nothing to model, which only happens in digital
    silence. Raises ``LinAlgError`` if the model cannot be fitted.
    """
    autocorr = np.zeros(order + 1)
    for piece in pieces:
        if piece.shape[0] <= order:
            continue
        windowed = piece * np.hanning(piece.shape[0])
        spectrum = np.fft.rfft(windowed, n=2 * piece.shape[0])
        autocorr += np.fft.irfft(spectrum.real ** 2 + spectrum.imag ** 2)[: order + 1]
    if autocorr[0] <= 1e-20:
        return None
    autocorr[0] *= 1.0 + correction
    predictor = solve_toeplitz(autocorr[:order], autocorr[1:])
    if not np.isfinite(predictor).all():
        raise LinAlgError("The AR model could not be fitted around this gap.")
    return np.concatenate(([1.0], -predictor))


def _lsar_solve(samples: np.ndarray, start: int, end: int, order: int,
                context: int, correction: float, ridge: float) -> np.ndarray:
    """The LSAR fill for ``samples[start:end]``, without writing it.

    The normal equations for the gap are banded (bandwidth ``order``) and
    positive definite, so a banded Cholesky solve is both fast and stable.
    ``ridge`` adds that fraction of the diagonal, which pulls the fill towards
    silence wherever the model leaves it free to swing.
    """
    gap = end - start
    coeffs = _ar_predictor(
        [samples[max(0, start - context):start], samples[end:end + context]], order, correction)
    if coeffs is None:
        return np.zeros(gap)
    segment = np.array(samples[start - order:end + order], dtype=np.float64)
    segment[order:order + gap] = 0.0
    # Prediction error of the known samples alone, then its correlation with
    # the filter: the right-hand side of the normal equations for the gap.
    known_error = np.convolve(segment, coeffs, mode="valid")
    rhs = np.correlate(known_error, coeffs, mode="valid")
    autocorr = np.correlate(coeffs, coeffs, mode="full")[order:]
    upper = min(order, gap - 1)
    banded = np.zeros((upper + 1, gap))
    for offset in range(upper + 1):
        banded[upper - offset, offset:] = autocorr[offset]
    banded[upper] += ridge * autocorr[0]
    return -solveh_banded(banded, rhs)


def _lsar_fill(samples: np.ndarray, start: int, end: int,
               order: int = CLICK_AR_ORDER, context: int = CLICK_CONTEXT) -> bool:
    """Least-squares autoregressive interpolation of ``samples[start:end]``, in place.

    An AR model is fitted to the audio either side of the gap, and the missing
    samples are the ones that make the model's prediction error across the gap
    as small as possible (Janssen, Veldhuis and Vries, 1986). Unlike a smooth
    curve, the fill carries on the waveform's texture, the harmonics of a vowel
    or the grain of the noise around it.

    In narrow-band noise the model has deep spectral valleys, and anything at
    the gap's edges that the model cannot explain can drive the fill far above
    its surroundings. So a fill louder than the audio either side by more than
    ``CLICK_GUARD_DB`` is replaced by a heavily regularised one, which errs
    towards a brief dip. Returns False, leaving the samples alone, when there
    is too little audio either side or neither fill passes.
    """
    gap = end - start
    if gap <= 0 or start < max(order, CLICK_GUARD_SPAN) + 32:
        return False
    if end + max(order, CLICK_GUARD_SPAN) + 32 > samples.shape[0]:
        return False
    side = max(float(np.sqrt(np.mean(np.square(samples[start - CLICK_GUARD_SPAN:start])))),
               float(np.sqrt(np.mean(np.square(samples[end:end + CLICK_GUARD_SPAN])))))
    if side == 0.0:
        samples[start:end] = 0.0  # a click in digital silence
        return True
    ceiling = side * 10.0 ** (CLICK_GUARD_DB / 20.0)
    for correction, ridge in ((1e-6, 0.0), (1e-3, 1e-2)):
        try:
            fill = _lsar_solve(samples, start, end, order, context, correction, ridge)
        except LinAlgError:
            continue
        if np.isfinite(fill).all() and float(np.sqrt(np.mean(fill * fill))) <= ceiling:
            samples[start:end] = fill
            return True
    return False


def detect_thumps(
    mono: np.ndarray, sr: int, periodicity: np.ndarray
) -> list[tuple[int, np.ndarray]]:
    """Low pops that sit in a pause or under unvoiced sound.

    Returns, for each pop, the first sample of the span to treat and the gain
    for the band below ``THUMP_LP_HZ`` across that span. The gain follows the
    pop's excess over the band's running median, so the band comes back to
    where it was before the pop. A pop within 30 ms of a voiced frame is left
    alone: that band is also where a speaking fundamental lives. Low bursts
    longer than ``THUMP_MAX_S``, or in low-band sound that does not hold steady
    around them, are gusts, and stay with the gust duck.
    """
    if mono.shape[0] < sr // 2 or periodicity.size == 0:
        return []
    low = _safe_filtfilt(mono[None, :], _sos("low", sr, THUMP_LP_HZ, order=4))[0]
    power = _block_power(low, THUMP_HOP)
    level = _window_db(power, THUMP_FINE)
    coarse = _window_db(power, THUMP_COARSE)
    if coarse.size < 8:
        return []
    size = int(round(THUMP_MEDIAN_S * sr / THUMP_HOP)) | 1
    excess = level - median_filter(level, size=size, mode="nearest")
    longest = int(round(THUMP_MAX_S * sr / THUMP_HOP))
    reach = int(round(TRANS_REACH_S * sr / THUMP_HOP))
    guard = THUMP_VOICE_GUARD_S * sr
    voiced = periodicity >= VOICING_MIN
    events: list[tuple[int, np.ndarray]] = []
    for first, last in _runs(excess > THUMP_EXTENT_DB):
        if last - first > longest or excess[first:last].max() < THUMP_DB:
            continue
        if not _steady_around(coarse, THUMP_COARSE, first, last + THUMP_FINE - 1,
                              reach, THUMP_FLAT_DB):
            continue
        # Voicing frame i is centred on i * VOICE_HOP + VOICE_FRAME / 2.
        lo, hi = first * THUMP_HOP, (last + THUMP_FINE - 1) * THUMP_HOP
        v0 = max(0, int(math.ceil((lo - guard - VOICE_FRAME / 2) / VOICE_HOP)))
        v1 = min(voiced.size, int(math.floor((hi + guard - VOICE_FRAME / 2) / VOICE_HOP)) + 1)
        if v1 <= v0 or voiced[v0:v1].any():
            continue
        # Cut by the excess, window by window, softened by one window each side.
        span_lo, span_hi = max(0, first - 2), min(level.size, last + 2)
        cut_db = np.zeros(span_hi - span_lo)
        cut_db[first - span_lo:last - span_lo] = np.clip(excess[first:last], 0.0, THUMP_MAX_DB)
        cut_db = np.maximum(uniform_filter1d(cut_db, size=3, mode="constant"), 0.0)
        centres = (np.arange(span_lo, span_hi) * THUMP_HOP
                   + THUMP_FINE * THUMP_HOP // 2).astype(np.float64)
        start = int(centres[0])
        end = min(mono.shape[0], int(centres[-1]) + 1)
        cut = np.interp(np.arange(start, end), centres, cut_db, left=0.0, right=0.0)
        events.append((start, np.power(10.0, -cut / 20.0)))
        if len(events) >= TRANS_MAX_EVENTS:
            break
    return events


def _duck_low_band(samples: np.ndarray, start: int, gain: np.ndarray, sr: int) -> None:
    """Lower the band under ``THUMP_LP_HZ`` by ``gain`` across one span, in place.

    The low band is a zero-phase low-pass, so the signal minus it is exactly
    the rest of the spectrum, and subtracting ``(1 - gain)`` of it touches
    nothing outside the span or above the band.
    """
    end = start + gain.shape[0]
    pad = sr // 10
    lo, hi = max(0, start - pad), min(samples.shape[0], end + pad)
    low = _safe_filtfilt(samples[None, lo:hi], _sos("low", sr, THUMP_LP_HZ, order=4))[0]
    samples[start:end] -= (1.0 - gain) * low[start - lo:end - lo]


def repair_transients(
    audio: np.ndarray, sr: int, periodicity: np.ndarray
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Repair isolated clicks and low pops on the dry chunk, in every channel.

    ``audio`` must be the caller's own float64 copy: it is repaired in place.
    Detection runs on the mono mix and each repair uses the same indices in
    every channel, so the stereo image holds. Runs before the infrasonic
    filter, because a zero-phase filter smears a click across tens of
    milliseconds. Returns the audio and the repairs, as (kind, start sample).
    """
    mono = np.ascontiguousarray(audio.mean(axis=0), dtype=np.float64)
    clicks = detect_clicks(mono, sr)
    thumps = detect_thumps(mono, sr, periodicity)
    repaired: list[tuple[str, int]] = []
    for start, end in clicks:
        if all(_lsar_fill(audio[channel], start, end) for channel in range(audio.shape[0])):
            repaired.append(("click", start))
    for start, gain in thumps:
        for channel in range(audio.shape[0]):
            _duck_low_band(audio[channel], start, gain, sr)
        repaired.append(("pop", start))
    return audio, repaired


def _noise_floor_track(magnitude: np.ndarray, sr: int) -> np.ndarray:
    """Per-bin noise floor: the minimum of the smoothed magnitude, bias-corrected.

    Smoothing first stops a single dropout from becoming the floor. The sliding
    minimum over ``GATE_TRACK_S`` then finds each bin's quietest moment, which
    is the noise as long as no speech sound holds a bin for that long, and the
    minima of neighbouring bins are averaged, which halves their scatter. The
    minimum of a noisy magnitude sits well below its RMS; ``GATE_BIAS`` puts it
    back, and was measured on Gaussian noise through this exact tracker (3.61).
    """
    window = max(3, int(round(GATE_TRACK_S * sr / GATE_HOP)))
    window = min(window, magnitude.shape[1])
    smoothed = uniform_filter1d(magnitude, size=GATE_SMOOTH_FRAMES, axis=1, mode="nearest")
    floor = minimum_filter1d(smoothed, size=window, axis=1, mode="nearest")
    floor = uniform_filter1d(floor, size=GATE_FLOOR_BINS, axis=0, mode="nearest")
    return np.maximum(floor, 0.0) * GATE_BIAS


def apply_noise_floor_gate(
    audio: np.ndarray,
    sr: int,
    preset: ScenePreset,
    voicing: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Lower the steady residual floor in the scene's band, voiced harmonics kept.

    This is not the peak suppressor in ``soften_musical_noise``: it tracks the
    broadband wash the network leaves behind under its blend limit, and pulls
    it down by up to ``gate_db``. The gain is Wiener-shaped. It is fully closed
    within 4 dB of the floor, and within 0.2 dB of unity 20 dB above it. Bins
    on a voiced harmonic, and one either side, are forced back to unity, so a
    vowel's harmonics come back bit-for-bit. ``voicing`` is (debounced
    periodicity, f0) for this audio, when the caller has already measured it.
    """
    if not preset.uses_gate or audio.shape[-1] < GATE_FRAME * 4:
        return audio
    mono = np.ascontiguousarray(audio.mean(axis=0), dtype=np.float64)
    gain = _gate_gain(mono, sr, preset, voicing)
    if gain is None:
        return audio
    return _apply_spectral_gain(audio, sr, gain - 1.0, GATE_FRAME, GATE_HOP)


def _gate_gain(
    mono: np.ndarray,
    sr: int,
    preset: ScenePreset,
    voicing: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray | None:
    """The floor gate's gain on the GATE_FRAME grid, or None when it is unity.

    Rows stop just above the band: rows past the end of a gain array are left
    untouched by ``_apply_spectral_gain``, so they need not be stored.
    """
    _freqs, _times, mono_spectrum = stft(
        mono,
        fs=sr,
        window="hann",
        nperseg=GATE_FRAME,
        noverlap=GATE_FRAME - GATE_HOP,
        boundary=None,
        padded=False,
    )
    bin_hz = sr / GATE_FRAME
    first = max(1, int(math.floor(preset.gate_lo_hz / bin_hz)))
    last = min(mono_spectrum.shape[0], int(math.ceil(preset.gate_hi_hz / bin_hz)) + 1)
    if last - first < 2:
        return None
    rows = min(mono_spectrum.shape[0], last + 1)
    band = np.abs(mono_spectrum[first:last])
    floor = _noise_floor_track(band, sr)
    # The level is the power over 3 frames by 3 bins. A raw bin's magnitude
    # swings by ±5 dB on steady noise; averaged, the noise rarely reaches the
    # opening threshold, while speech, which fills neighbouring bins, still does.
    # The filter's running sum can dip a hair below zero in digital silence.
    level = np.sqrt(np.maximum(uniform_filter1d(
        uniform_filter1d(band * band, size=3, axis=1, mode="nearest"),
        size=3, axis=0, mode="nearest"), 0.0))
    closeness = GATE_OVERSUB * floor / np.maximum(level, 1e-12)
    wiener = np.clip(1.0 - closeness * closeness, 0.0, 1.0)
    floor_gain = 10.0 ** (-preset.gate_db / 20.0)
    gain = np.ones((rows, band.shape[1]), dtype=np.float64)
    gain[first:last] = floor_gain + (1.0 - floor_gain) * wiener

    if voicing is None:
        periodicity, f0 = estimate_voicing(mono, sr)
        periodicity = _debounce_voicing(periodicity)
    else:
        periodicity, f0 = voicing
    if f0.size:
        source = np.arange(f0.size) * VOICE_HOP + VOICE_FRAME / 2
        target = np.arange(gain.shape[1]) * GATE_HOP + GATE_FRAME / 2
        f0_on_grid = np.interp(target, source, f0, left=f0[0], right=f0[-1])
        period_on_grid = np.interp(
            target, source, periodicity, left=periodicity[0], right=periodicity[-1])
        protect = _protect_harmonics(
            f0_on_grid, period_on_grid, rows, gain.shape[1], sr,
            frame=GATE_FRAME, top_hz=preset.gate_hi_hz)
        gain = np.where(protect, 1.0, gain)

    gain = uniform_filter1d(gain, size=3, axis=0, mode="nearest")
    gain = uniform_filter1d(gain, size=3, axis=1, mode="nearest")
    gain = np.clip(gain, floor_gain, 1.0)
    # Snap near-unity gains, so a region the gate leaves alone comes back
    # byte-for-byte through the difference form rather than an inverse STFT.
    gain = np.where(gain > 0.995, 1.0, gain)
    if np.all(gain > 0.999):
        return None
    return gain


def _band_slice(n_bins: int, sr: int, frame: int, lo_hz: float, hi_hz: float) -> slice:
    """Bins covering ``lo_hz`` to ``hi_hz``. DC is never included."""
    if sr <= 0 or frame <= 0:
        raise ValueError("Sample rate and frame length must be positive.")
    bin_hz = sr / frame
    first = max(1, int(math.floor(lo_hz / bin_hz)))
    last = min(n_bins, int(math.ceil(hi_hz / bin_hz)) + 1)
    if last - first < 2:
        raise ValueError(
            f"Band {lo_hz:g}–{hi_hz:g} Hz covers fewer than two bins at {sr} Hz."
        )
    return slice(first, last)


def _cosine_fade(gain: np.ndarray, rows: slice, edge: int) -> None:
    """Soften the ends of a boosted band back toward unity."""
    width = rows.stop - rows.start
    fade = min(edge, width // 2)
    if fade <= 0:
        return
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, fade, dtype=np.float64))
    band = gain[rows]
    band[:fade] = 1.0 + (band[:fade] - 1.0) * ramp[:, None]
    band[-fade:] = 1.0 + (band[-fade:] - 1.0) * ramp[::-1, None]


def _limit_lift(dry: np.ndarray, lifted: np.ndarray, ceiling: float = DIALOGUE_CEILING) -> np.ndarray:
    """Scale the added part so the chunk peaks at ``ceiling``.

    Where the lift gain was unity, ``lifted`` matches ``dry`` and the mix
    stays there. A chunk that already peaks over the ceiling is returned
    unchanged: the lift does not repair clipping it did not cause.
    """
    peak = float(np.max(np.abs(lifted))) if lifted.size else 0.0
    if peak <= ceiling:
        return lifted
    dry64 = np.asarray(dry, dtype=np.float64)
    if float(np.max(np.abs(dry64))) >= ceiling:
        return dry64
    added = lifted - dry64
    best = dry64
    lo, hi = 0.0, 1.0
    for _ in range(16):
        mid = (lo + hi) * 0.5
        mix = dry64 + added * mid
        if float(np.max(np.abs(mix))) <= ceiling:
            best = mix
            lo = mid
        else:
            hi = mid
    return best


def apply_dialogue_lift(
    audio: np.ndarray,
    sr: int,
    lift_db: float,
    voicing: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Restore presence on voiced frames and on consonants beside a vowel.

    ``lift_db`` is the cap for the 1.8–4 kHz band. Zero or less returns
    ``audio`` unchanged, which is how Manual and the 1.2.1 scenes stay
    bit-identical. ``voicing`` is the debounced periodicity and f0 already
    measured for this chunk. The f0 is accepted so the call matches the
    gate, and it is not used: a consonant has no harmonic to protect.
    """
    if lift_db <= 0.0 or audio.shape[-1] < DIALOGUE_FRAME * 4:
        return audio
    if sr != MODEL_SR or audio.ndim != 2:
        raise ValueError(
            f"Dialogue lift expected {MODEL_SR} Hz audio shaped (channels, frames)."
        )
    if not np.isfinite(audio).all():
        raise ValueError("Dialogue lift was given invalid samples.")

    mono = np.ascontiguousarray(audio.mean(axis=0), dtype=np.float64)
    _freqs, _times, spectrum = stft(
        mono, fs=sr, window="hann", nperseg=DIALOGUE_FRAME,
        noverlap=DIALOGUE_FRAME - DIALOGUE_HOP, boundary=None, padded=False,
    )
    magnitude = np.abs(spectrum)
    n_bins, n_times = magnitude.shape
    if n_times == 0:
        return audio
    body = _band_slice(n_bins, sr, DIALOGUE_FRAME, *DIALOGUE_BODY)
    presence = _band_slice(n_bins, sr, DIALOGUE_FRAME, *DIALOGUE_PRESENCE)
    air = _band_slice(n_bins, sr, DIALOGUE_FRAME, *DIALOGUE_AIR)
    speech = slice(presence.start, air.stop)

    body_power = np.mean(magnitude[body] ** 2, axis=0)
    presence_power = np.mean(magnitude[presence] ** 2, axis=0)
    # dBFS of the samples in each frame. Mean STFT-bin power makes a
    # high-band consonant look like silence, because most bins are empty.
    squared = mono * mono
    cumulative = np.empty(squared.size + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(squared, out=cumulative[1:])
    starts = np.arange(n_times, dtype=np.int64) * DIALOGUE_HOP
    stops = np.minimum(starts + DIALOGUE_FRAME, mono.size)
    frame_ms = (cumulative[stops] - cumulative[starts]) / DIALOGUE_FRAME

    if voicing is None:
        periodicity, _f0 = estimate_voicing(mono, sr)
        periodicity = _debounce_voicing(periodicity)
    else:
        periodicity = voicing[0]
    if periodicity.size:
        source = np.arange(periodicity.size) * VOICE_HOP + VOICE_FRAME / 2
        target = np.arange(n_times) * DIALOGUE_HOP + DIALOGUE_FRAME / 2
        period = np.interp(target, source, periodicity, left=0.0, right=0.0)
    else:
        period = np.zeros(n_times, dtype=np.float64)
    voiced = period >= VOICING_MIN

    level = np.sqrt(np.maximum(presence_power, 0.0))
    smoothed = uniform_filter1d(level, size=3, mode="nearest")
    window = max(3, int(round(GATE_TRACK_S * sr / DIALOGUE_HOP)))
    window = min(window, max(1, smoothed.size))
    floor = np.maximum(minimum_filter1d(smoothed, size=window, mode="nearest"), 0.0)
    floor *= DIALOGUE_FLOOR_BIAS

    band = np.maximum(magnitude[speech], 1e-12)
    flatness = np.exp(np.mean(np.log(band), axis=0)) / np.maximum(np.mean(band, axis=0), 1e-12)
    reach = max(1, int(round(DIALOGUE_VOICE_REACH_S * sr / DIALOGUE_HOP)))
    near_voice = maximum_filter1d(
        voiced.astype(np.float64), size=2 * reach + 1, mode="nearest") > 0.5
    with np.errstate(divide="ignore"):
        frame_db = 10.0 * np.log10(np.maximum(frame_ms, 1e-12))
        margin = 10.0 ** (DIALOGUE_CONSONANT_MARGIN_DB / 20.0)
    audible = frame_db >= DIALOGUE_SILENCE_DBFS
    consonant = (
        ~voiced
        & near_voice
        & (flatness >= DIALOGUE_FLATNESS_MIN)
        & (level >= floor * margin)
        & audible
    )
    active = (voiced & audible) | consonant

    presence_db = 10.0 * np.log10(np.maximum(presence_power, 1e-12))
    body_db = 10.0 * np.log10(np.maximum(body_power, 1e-12))
    body_usable = body_power > np.maximum(presence_power, 1e-12) * 10.0 ** (-30.0 / 10.0)
    deficit = np.clip(DIALOGUE_TARGET_TILT_DB - (presence_db - body_db), 0.0, lift_db)
    deficit = np.where(body_usable, deficit, lift_db)
    deficit = np.where(active, deficit, 0.0)

    gain = np.ones((n_bins, n_times), dtype=np.float64)
    gain[presence] = 10.0 ** (deficit / 20.0)
    gain[air] = 10.0 ** (DIALOGUE_AIR_SHARE * deficit / 20.0)
    _cosine_fade(gain, presence, DIALOGUE_EDGE_BINS)
    _cosine_fade(gain, air, DIALOGUE_EDGE_BINS)
    gain = uniform_filter1d(gain, size=5, axis=1, mode="nearest")
    # uniform_filter1d's running sum can undershoot a non-negative input.
    gain = np.clip(gain, 1.0, 10.0 ** (lift_db / 20.0))
    near_unity = np.abs(gain - 1.0) < (10.0 ** (0.05 / 20.0) - 1.0)
    gain = np.where(near_unity, 1.0, gain)
    if np.all(gain == 1.0):
        return audio

    lifted = _apply_spectral_gain(audio, sr, gain - 1.0, DIALOGUE_FRAME, DIALOGUE_HOP)
    lifted = _limit_lift(audio, lifted)
    if not np.isfinite(lifted).all():
        raise ValueError("Dialogue lift produced invalid samples.")
    return np.ascontiguousarray(lifted, dtype=np.float32)


def apply_consonant_restore(
    dry: np.ndarray,
    processed: np.ndarray,
    sr: int,
    voicing: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Bring voiced presence and air back towards the dry balance. Never cut.

    ``dry`` is the chunk before the network. ``processed`` is that chunk after
    the network, the floor, the gate and the dialogue lift. ``voicing`` is the
    dry chunk's detector output: the network's own output is voiced in pauses
    the dry recording is not. A band is boosted only where the processed tilt
    against the vowel body is darker than the dry tilt, and only up to the dry
    band's own level. Unvoiced frames are left exactly as they are, so wind and
    hiss stay down. A chunk that needs nothing is returned as the same array.
    """
    if sr != MODEL_SR or dry.ndim != 2 or processed.ndim != 2:
        raise ValueError(
            f"Consonant restore expected {MODEL_SR} Hz audio shaped (channels, frames).")
    if dry.shape != processed.shape:
        raise ValueError("Consonant restore expected the dry and processed chunks to share a shape.")
    if processed.shape[-1] < DIALOGUE_FRAME * 4:
        return processed
    if not np.isfinite(dry).all() or not np.isfinite(processed).all():
        raise ValueError("Consonant restore was given invalid samples.")

    dry64 = np.asarray(dry, dtype=np.float64)
    proc64 = np.asarray(processed, dtype=np.float64)
    dry_mono = np.ascontiguousarray(dry64.mean(axis=0), dtype=np.float64)
    proc_mono = np.ascontiguousarray(proc64.mean(axis=0), dtype=np.float64)
    _freqs, _times, dry_spec = stft(
        dry_mono, fs=sr, window="hann", nperseg=DIALOGUE_FRAME,
        noverlap=DIALOGUE_FRAME - DIALOGUE_HOP, boundary=None, padded=False)
    _freqs, _times, proc_spec = stft(
        proc_mono, fs=sr, window="hann", nperseg=DIALOGUE_FRAME,
        noverlap=DIALOGUE_FRAME - DIALOGUE_HOP, boundary=None, padded=False)
    dry_mag = np.abs(dry_spec)
    proc_mag = np.abs(proc_spec)
    n_bins, n_times = proc_mag.shape
    if n_times == 0 or dry_mag.shape != proc_mag.shape:
        return processed

    body = _band_slice(n_bins, sr, DIALOGUE_FRAME, *DIALOGUE_BODY)
    presence = _band_slice(n_bins, sr, DIALOGUE_FRAME, *DIALOGUE_PRESENCE)
    air = _band_slice(n_bins, sr, DIALOGUE_FRAME, *DIALOGUE_AIR)

    def _power(magnitude: np.ndarray, rows: slice) -> np.ndarray:
        return np.mean(magnitude[rows] ** 2, axis=0)

    def _db(power: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(power, 1e-12))

    def _frame_db(mono: np.ndarray) -> np.ndarray:
        squared = mono * mono
        cumulative = np.empty(squared.size + 1, dtype=np.float64)
        cumulative[0] = 0.0
        np.cumsum(squared, out=cumulative[1:])
        starts = np.arange(n_times, dtype=np.int64) * DIALOGUE_HOP
        stops = np.minimum(starts + DIALOGUE_FRAME, mono.size)
        frame_ms = (cumulative[stops] - cumulative[starts]) / DIALOGUE_FRAME
        with np.errstate(divide="ignore"):
            return 10.0 * np.log10(np.maximum(frame_ms, 1e-12))

    if voicing is None:
        periodicity, _f0 = estimate_voicing(proc_mono, sr)
        periodicity = _debounce_voicing(periodicity)
    else:
        periodicity = np.asarray(voicing[0], dtype=np.float64)
    if periodicity.size:
        source = np.arange(periodicity.size) * VOICE_HOP + VOICE_FRAME / 2
        target = np.arange(n_times) * DIALOGUE_HOP + DIALOGUE_FRAME / 2
        period = np.interp(target, source, periodicity, left=0.0, right=0.0)
    else:
        period = np.zeros(n_times, dtype=np.float64)
    voiced = period >= VOICING_MIN
    audible = (_frame_db(dry_mono) >= DIALOGUE_SILENCE_DBFS) & (
        _frame_db(proc_mono) >= DIALOGUE_SILENCE_DBFS)
    margin = 10.0 ** (-RESTORE_BODY_MARGIN_DB / 10.0)

    def _deficit(rows: slice) -> tuple[np.ndarray, np.ndarray]:
        dry_band = _power(dry_mag, rows)
        proc_band = _power(proc_mag, rows)
        dry_body = _power(dry_mag, body)
        proc_body = _power(proc_mag, body)
        usable = (
            (dry_body > np.maximum(dry_band, 1e-12) * margin)
            & (proc_body > np.maximum(proc_band, 1e-12) * margin))
        gap = (_db(dry_band) - _db(dry_body)) - (_db(proc_band) - _db(proc_body))
        absolute = _db(dry_band) - _db(proc_band)
        deficit = np.minimum(gap, absolute) * RESTORE_SHARE
        deficit = np.clip(deficit, 0.0, RESTORE_MAX_DB)
        active = voiced & audible & usable
        deficit = np.where(active, deficit, 0.0)
        return np.where(deficit < RESTORE_SNAP_DB, 0.0, deficit), active

    presence_db, presence_on = _deficit(presence)
    air_db, air_on = _deficit(air)
    gain = np.ones((n_bins, n_times), dtype=np.float64)
    gain[presence] = 10.0 ** (presence_db / 20.0)
    gain[air] = 10.0 ** (air_db / 20.0)
    _cosine_fade(gain, presence, DIALOGUE_EDGE_BINS)
    _cosine_fade(gain, air, DIALOGUE_EDGE_BINS)
    gain = uniform_filter1d(gain, size=5, axis=1, mode="nearest")
    # The running sum can undershoot. This stage never cuts, so stay at unity or above.
    gain = np.clip(gain, 1.0, 10.0 ** (RESTORE_MAX_DB / 20.0))
    # Smoothing must not leak a boost into a pause or an unusable frame.
    gain[presence, ~presence_on] = 1.0
    gain[air, ~air_on] = 1.0
    near_unity = np.abs(gain - 1.0) < (10.0 ** (RESTORE_SNAP_DB / 20.0) - 1.0)
    gain = np.where(near_unity, 1.0, gain)
    if np.all(gain == 1.0):
        return processed

    restored = _apply_spectral_gain(
        processed, sr, gain - 1.0, DIALOGUE_FRAME, DIALOGUE_HOP)
    # _limit_lift returns its dry argument unchanged when that argument is
    # already hot, which would throw this restore away. Limit only a chunk
    # that still has headroom. The dialogue lift keeps the shared limiter.
    peak = float(np.max(np.abs(processed))) if processed.size else 0.0
    if peak < DIALOGUE_CEILING:
        restored = _limit_lift(processed, restored)
    if not np.isfinite(restored).all():
        raise ValueError("Consonant restore produced invalid samples.")
    return np.ascontiguousarray(restored, dtype=np.float32)


def apply_scene(
    audio: np.ndarray, sr: int, preset: ScenePreset, state: ConditionerState | None = None
) -> np.ndarray:
    """Run the pre-model wind and traffic conditioner on one chunk.

    The chunk must include the processing pre-roll. Edge effects of the
    zero-phase filters then fall in audio that the chunk loop discards or
    crossfades, rather than in the speech the listener keeps.
    """
    if not preset.active:
        return audio
    if audio.ndim != 2:
        raise ValueError("Scene conditioning expected audio shaped (channels, frames).")
    if sr != MODEL_SR:
        raise ValueError(
            f"Scene processing is tuned for {MODEL_SR} Hz, but the audio is {sr} Hz."
        )
    if audio.shape[-1] < sr // 5:
        return np.ascontiguousarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("The audio contains invalid samples and cannot be conditioned.")
    if state is None:
        state = ConditionerState()

    # Voicing is measured on the dry chunk. The steep infrasonic filter rings,
    # and that ring can look like a vowel if it is measured afterwards. Clicks
    # do not disturb it (an impulse has no period), and the pop detector needs
    # it to keep away from voiced speech, so it is measured before the repair.
    dry = np.ascontiguousarray(audio.mean(axis=0), dtype=np.float64)
    periodicity, _f0 = estimate_voicing(dry, sr)
    periodicity = _debounce_voicing(periodicity)

    work = np.array(audio, dtype=np.float64, copy=True)
    state.events = []
    if preset.transient_guard:
        work, state.events = repair_transients(work, sr, periodicity)
    if preset.infrasonic_hz is not None:
        work = _safe_filtfilt(work, _sos("high", sr, preset.infrasonic_hz, order=4))
    if preset.gust_split_hz is not None:
        work = apply_gust_duck(work, sr, preset, state, periodicity)

    work = suppress_stable_tones(work, sr, preset, periodicity)
    if not np.isfinite(work).all():
        raise ValueError("Scene conditioning produced invalid samples.")
    return np.ascontiguousarray(work, dtype=np.float32)


def _preset(**overrides) -> ScenePreset:
    base = dict(
        key="custom",
        label="Custom",
        summary="",
        atten_db=None,
        post_filter=False,
        post_filter_beta=0.02,
        infrasonic_hz=None,
        gust_split_hz=None,
        gust_max_db=12.0,
        gust_margin_db=8.0,
        gust_voiced_cap_db=3.0,
        gust_attack_s=0.025,
        gust_release_s=0.40,
        notch_lo_hz=None,
        notch_hi_hz=None,
        notch_max_db=10.0,
        notch_max_count=3,
        hum_hz=(),
        floor_lo_hz=None,
        floor_hi_hz=None,
        floor_peak_db=8.0,
        floor_max_db=6.0,
        transient_guard=False,
        gate_db=0.0,
        gate_lo_hz=None,
        gate_hi_hz=None,
        lift_db=0.0,
    )
    base.update(overrides)
    return ScenePreset(**base)


WIND = _preset(
    key="wind",
    label="Heavy Wind & Gusts",
    summary=(
        "Repairs clicks and pops, strips infrasonic buffeting, and ducks unvoiced gusts under 220 Hz."
    ),
    atten_db=30,
    post_filter=True,
    post_filter_beta=0.02,
    infrasonic_hz=38.0,
    gust_split_hz=220.0,
    gust_max_db=12.0,
    gust_margin_db=8.0,
    gust_voiced_cap_db=3.0,
    gust_attack_s=0.030,
    gust_release_s=0.45,
    floor_lo_hz=2000.0,
    floor_hi_hz=5000.0,
    floor_peak_db=10.0,
    floor_max_db=4.0,
    transient_guard=True,
    gate_db=10.0,
    gate_lo_hz=1200.0,
    gate_hi_hz=6000.0,
)
TRAFFIC = _preset(
    key="traffic",
    label="Highway / Traffic Drone",
    summary="Repairs clicks, tames rumble and engine tones, and gates the tyre-hiss wash.",
    atten_db=24,
    post_filter=True,
    post_filter_beta=0.05,
    infrasonic_hz=32.0,
    gust_split_hz=130.0,
    gust_max_db=8.0,
    gust_margin_db=6.0,
    gust_voiced_cap_db=2.5,
    gust_attack_s=0.040,
    gust_release_s=0.80,
    notch_lo_hz=75.0,
    notch_hi_hz=210.0,
    notch_max_db=10.0,
    notch_max_count=3,
    floor_lo_hz=1200.0,
    floor_hi_hz=4000.0,
    floor_peak_db=8.0,
    floor_max_db=6.0,
    transient_guard=True,
    gate_db=12.0,
    gate_lo_hz=800.0,
    gate_hi_hz=6000.0,
)
STREET = _preset(
    key="street",
    label="Pedestrian Street",
    summary="Light low-end control and a gentle hiss gate, without chewing consonants or nearby voices.",
    atten_db=18,
    post_filter=True,
    post_filter_beta=0.02,
    infrasonic_hz=30.0,
    gust_split_hz=160.0,
    gust_max_db=6.0,
    gust_margin_db=8.0,
    gust_voiced_cap_db=2.5,
    gust_attack_s=0.035,
    gust_release_s=0.50,
    notch_lo_hz=80.0,
    notch_hi_hz=200.0,
    notch_max_db=8.0,
    notch_max_count=2,
    gate_db=8.0,
    gate_lo_hz=1500.0,
    gate_hi_hz=6000.0,
)
CONVERSATION = _preset(
    key="conversation",
    label="Conversation (1–5 m)",
    summary=(
        "For talk a few metres away. Leaves the street down, and lifts "
        "the consonants the other scenes hold back."
    ),
    atten_db=18,
    post_filter=False,
    post_filter_beta=0.02,
    infrasonic_hz=30.0,
    gust_split_hz=110.0,
    gust_max_db=4.0,
    gust_margin_db=8.0,
    gust_voiced_cap_db=1.5,
    gust_attack_s=0.035,
    gust_release_s=0.50,
    notch_lo_hz=80.0,
    notch_hi_hz=200.0,
    notch_max_db=8.0,
    notch_max_count=2,
    gate_db=6.0,
    gate_lo_hz=2200.0,
    gate_hi_hz=7000.0,
    lift_db=LIFT_CLEAR_DB,
)
PODCAST = _preset(
    key="podcast",
    label="Subtle Podcast Clean",
    summary="Keeps the voice natural. A confirmed 50 Hz hum notch only, and modest noise reduction.",
    atten_db=12,
    post_filter=False,
    post_filter_beta=0.02,
    hum_hz=(50.0, 100.0),
    notch_max_db=8.0,
    notch_max_count=2,
)
MANUAL = _preset(
    key="manual",
    label="Manual",
    summary="Uses the controls below, with no extra wind or traffic processing.",
    atten_db=None,
    post_filter=False,
    post_filter_beta=0.02,
)

SCENES: dict[str, ScenePreset] = {
    preset.key: preset for preset in (MANUAL, WIND, TRAFFIC, STREET, CONVERSATION, PODCAST)
}
SCENE_ORDER = ("manual", "wind", "traffic", "street", "conversation", "podcast")
SCENE_BY_LABEL = {preset.label: preset for preset in SCENES.values()}

# Named presets are control snapshots in the scene menu. They are not a DSP path.
PRESET_NAME_LIMIT = 40
PRESET_COUNT_LIMIT = 20


def dialogue_menu_label(key: str) -> str:
    """Menu text for a stored dialogue key (off, gentle, clear)."""
    return {"off": "Off", "gentle": "Gentle", "clear": "Clear"}.get(key, "Off")


def _preset_record(item: object) -> dict[str, object] | None:
    """One stored preset, or None when the entry cannot be applied safely."""
    if not isinstance(item, dict):
        return None
    name = item.get("name")
    if not isinstance(name, str):
        return None
    name = name.strip()
    if (not name or len(name) > PRESET_NAME_LIMIT or name in SCENE_BY_LABEL
            or any(ord(ch) < 32 for ch in name)):
        return None
    scene = item.get("scene")
    if not isinstance(scene, str) or scene not in SCENES:
        return None
    lift = str(item.get("dialogue_lift", "")).lower()
    if lift not in ("off", "gentle", "clear"):
        return None
    index = item.get("atten_index")
    # bool is an int subclass, and it is not a slider step.
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    if index < 0 or index >= len(ATTEN_STEPS):
        return None
    highpass = item.get("highpass")
    if not isinstance(highpass, str) or highpass not in HIGHPASS_CHOICES:
        return None
    post_filter = item.get("post_filter")
    normalise = item.get("normalise")
    if type(post_filter) is not bool or type(normalise) is not bool:
        return None
    return {
        "name": name,
        "scene": scene,
        "dialogue_lift": lift,
        "atten_index": index,
        "highpass": highpass,
        "post_filter": post_filter,
        "normalise": normalise,
    }


def clean_user_presets(raw: object) -> list[dict[str, object]]:
    """Drop broken records, factory-label collisions and duplicate names.

    A missing list becomes empty, so a settings file from 1.5.0 still loads.
    Factory labels are matched with their case intact.
    """
    if not isinstance(raw, list):
        return []
    kept: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in raw:
        record = _preset_record(item)
        if record is None:
            continue
        name = str(record["name"])
        if name in seen:
            continue
        seen.add(name)
        kept.append(record)
        if len(kept) >= PRESET_COUNT_LIMIT:
            break
    return kept


def peak_normalise(audio: np.ndarray, peak_dbfs: float) -> tuple[np.ndarray, float]:
    """Scale so the loudest sample sits at peak_dbfs. Returns (audio, gain_db)."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-6:
        return audio, 0.0
    gain = (10 ** (peak_dbfs / 20)) / peak
    return (audio * gain).astype(np.float32), 20 * math.log10(gain)


# K-weighting for 48 kHz (ITU-R BS.1770-4): the high shelf, then the RLB high-pass.
_K_WEIGHTING = np.array([
    [1.53512485958697, -2.69169618940638, 1.19839281085285,
     1.0, -1.69065929318241, 0.73248077421585],
    [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621],
])
LOUDNESS_STEP_S = 0.1           # blocks are four steps (400 ms) with 75 % overlap
LOUDNESS_ABSOLUTE_GATE = -70.0
LOUDNESS_RELATIVE_GATE = -10.0
LOUDNESS_MATCH_MAX_DB = 12.0


def integrated_loudness(audio: np.ndarray, sr: int = MODEL_SR) -> float:
    """Gated integrated loudness in LUFS (ITU-R BS.1770-4), or -inf.

    Audio is (channels, frames) at 48 kHz, each channel weighted 1.0 as for
    mono and stereo. The K-weighted energy is summed in 100 ms steps, a minute
    at a time with the filter state carried over, so a 45-minute file never
    needs a float64 copy of itself. -inf means there was nothing to measure:
    under 400 ms of audio, or nothing above the -70 LUFS gate.
    """
    if sr != MODEL_SR or audio.ndim != 2:
        return float("-inf")
    step = int(round(LOUDNESS_STEP_S * sr))
    steps = audio.shape[-1] // step
    if steps < 4:
        return float("-inf")
    energy = np.zeros(steps, dtype=np.float64)
    state = np.zeros((_K_WEIGHTING.shape[0], audio.shape[0], 2))
    span = 600 * step
    for first in range(0, steps * step, span):
        last = min(first + span, steps * step)
        weighted, state = sosfilt(
            _K_WEIGHTING, np.asarray(audio[:, first:last], dtype=np.float64), axis=-1, zi=state)
        squares = np.sum(weighted * weighted, axis=0)
        energy[first // step:last // step] = squares.reshape(-1, step).sum(axis=1)
    blocks = (energy[:-3] + energy[1:-2] + energy[2:-1] + energy[3:]) / (4 * step)
    with np.errstate(divide="ignore"):
        loudness = -0.691 + 10.0 * np.log10(blocks)
    keep = loudness > LOUDNESS_ABSOLUTE_GATE
    if not keep.any():
        return float("-inf")
    relative = -0.691 + 10.0 * math.log10(float(np.mean(blocks[keep]))) + LOUDNESS_RELATIVE_GATE
    keep &= loudness > relative
    if not keep.any():
        return float("-inf")
    return float(-0.691 + 10.0 * math.log10(float(np.mean(blocks[keep]))))


def loudness_match_gains(
    original_lufs: float, denoised_lufs: float, denoised_peak: float
) -> tuple[float, float, float] | None:
    """Playback gains (original, denoised, match in dB) that make A and B equally loud.

    The denoised copy moves to the original's integrated loudness, by no more
    than ``LOUDNESS_MATCH_MAX_DB``. If that would push its peak past full
    scale, both copies are lowered together by the overshoot, so they stay
    matched and neither clips. ``None`` means one of them could not be measured.
    """
    if not (math.isfinite(original_lufs) and math.isfinite(denoised_lufs)):
        return None
    match_db = float(np.clip(original_lufs - denoised_lufs,
                             -LOUDNESS_MATCH_MAX_DB, LOUDNESS_MATCH_MAX_DB))
    denoised_gain = 10.0 ** (match_db / 20.0)
    trim = 1.0
    if denoised_peak * denoised_gain > 1.0:
        trim = 1.0 / (denoised_peak * denoised_gain)
    return trim, denoised_gain * trim, match_db


def source_subtype(path: str) -> str | None:
    """The WAV subtype that keeps the source's bit depth, or None if it has none.

    WAV and FLAC carry a real depth. MP3 and OGG decode to floating point with
    no depth to keep, and unusual WAV encodings (8-bit, A-law) are not worth
    keeping; all of those are saved as 24-bit.
    """
    try:
        subtype = sf.info(path).subtype
    except Exception:
        return None
    return subtype if subtype in EXPORT_SUBTYPES else None


def write_wav(path: Path, audio: np.ndarray, sr: int, subtype: str) -> None:
    """Write (channels, frames) audio in [-1, 1] as a WAV of the given subtype.

    16-bit output is rounded here, with TPDF dither of one LSB either way, so
    the quiet tails a denoiser leaves fade into noise rather than distortion.
    libsndfile alone floors every sample with no dither. It is written a block
    at a time, so a long file never needs a second full-size copy. Deeper
    formats go through libsndfile exactly as in 1.1.
    """
    if subtype != "PCM_16":
        sf.write(str(path), audio.T, sr, subtype=subtype)
        return
    rng = np.random.default_rng(DITHER_SEED)
    block = 1 << 20
    with sf.SoundFile(str(path), "w", samplerate=sr, channels=audio.shape[0],
                      subtype="PCM_16", format="WAV") as handle:
        for first in range(0, audio.shape[1], block):
            piece = np.asarray(audio[:, first:first + block].T, dtype=np.float32) * 32768.0
            piece += rng.random(piece.shape, dtype=np.float32)
            piece -= rng.random(piece.shape, dtype=np.float32)
            np.rint(piece, out=piece)
            np.clip(piece, -32768.0, 32767.0, out=piece)
            handle.write(piece.astype(np.int16))


def export_denoised(den48: np.ndarray, opts: dict) -> dict:
    """Write den48 at the source rate. Does not modify den48.

    Same resample, length match and optional peak normalisation the worker
    used when Denoise wrote the file itself. 16-bit dither stays inside
    write_wav, so a repeated save of the same samples matches byte for byte.
    """
    if den48.ndim != 2 or den48.shape[1] == 0:
        raise ValueError("There is no denoised audio to save.")
    if not np.isfinite(den48).all():
        raise ValueError("The denoised audio contains invalid samples.")
    dest = Path(opts["dest"])
    src = Path(opts["src_path"])
    if dest.resolve() == src.resolve():
        raise ValueError("That is the file you opened. Choose a different name.")
    out = resample(den48, MODEL_SR, opts["src_sr"])
    # Resampling can be a sample out either way; match the source length exactly.
    n_src = opts["src_frames"]
    if out.shape[1] >= n_src:
        out = out[:, :n_src]
    else:
        out = np.pad(out, ((0, 0), (0, n_src - out.shape[1])))
    gain_db = None
    if opts.get("normalise"):
        out, gain_db = peak_normalise(out, NORMALISE_PEAK_DBFS)
    clipped = bool(out.size) and float(np.max(np.abs(out))) > 1.0
    out = np.clip(out, -1.0, 1.0)
    subtype = opts.get("subtype", "PCM_24")
    if subtype not in EXPORT_SUBTYPES:
        subtype = "PCM_24"
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".partial.wav")
    try:
        write_wav(partial, out, opts["src_sr"], subtype)
        os.replace(partial, dest)
    except Exception:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return {"path": str(dest), "gain_db": gain_db, "clipped": clipped, "subtype": subtype}


def load_model():
    """Import torch/DeepFilterNet lazily (slow) and load the DFN3 model."""
    import torch
    from df.enhance import init_df

    # log_level "NONE" stops DeepFilterNet attaching console loggers and shelling
    # out to git, and log_file=None stops it writing enhance.log files.
    model, df_state, _ = init_df(log_level="NONE", log_file=None)
    model.eval()
    device = "GPU" if next(model.parameters()).is_cuda else "CPU"
    return model, df_state, device, torch.__version__


def denoise(model, df_state, audio: np.ndarray, atten_lim_db,
            scene: ScenePreset | None = None,
            lift_db: float = 0.0,
            restore: bool = False,
            progress=None, cancel: threading.Event | None = None,
            stats: dict | None = None) -> np.ndarray:
    """Run DeepFilterNet over 48 kHz audio (channels, frames) in overlapping chunks.

    A named scene conditions each extended chunk before the network sees it.
    Manual passes the audio through unchanged, so today's slider and rumble
    settings keep today's results. ``lift_db`` above zero restores consonant
    presence after the network; zero leaves the chunk untouched by that stage.
    ``restore`` compares that result with the dry chunk and puts back voiced
    presence and air that went darker than the dry balance. False, the default,
    does not call that stage. The torch tensor is one chunk, not the file.
    ``stats``, if given, gains "clicks" and "pops": the transients repaired,
    each counted once, in the chunk whose output keeps it.
    """
    import torch
    from df.enhance import enhance

    sr = int(df_state.sr())
    channels, frames = audio.shape
    active = scene is not None and scene.active
    if active and sr != MODEL_SR:
        raise ValueError(
            f"Scene processing is tuned for {MODEL_SR} Hz, but the model reported {sr} Hz.")

    n_chunks = max(1, math.ceil(frames / (CHUNK_SECONDS * sr)))
    chunk = math.ceil(frames / n_chunks)  # balanced chunk sizes, no tiny last chunk
    pre = int(PREROLL_SECONDS * sr)
    xfade = int(CROSSFADE_SECONDS * sr)
    out = np.zeros((channels, frames), dtype=np.float32)
    state = ConditionerState()

    for index in range(n_chunks):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        start = index * chunk
        end = min(frames, start + chunk)
        last = index == n_chunks - 1
        pre_start = max(0, start - pre)
        tail = frames if last else min(frames, end + xfade)

        segment = np.ascontiguousarray(audio[:, pre_start:tail], dtype=np.float32)
        if active:
            state.events = []
            segment = apply_scene(segment, sr, scene, state)
            if stats is not None:
                for kind, at in state.events:
                    if start <= pre_start + at < end:
                        key = "clicks" if kind == "click" else "pops"
                        stats[key] = stats.get(key, 0) + 1
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if hasattr(df_state, "reset"):
            df_state.reset()
        with torch.inference_mode():
            enhanced = enhance(
                model, df_state, torch.from_numpy(segment), atten_lim_db=atten_lim_db)
        piece = np.ascontiguousarray(enhanced.detach().cpu().numpy(), dtype=np.float32)
        wants_lift = lift_db > 0.0
        wants_restore = bool(restore)
        if (active and (scene.uses_floor or scene.uses_gate)) or wants_lift or wants_restore:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            # Voicing of the network's output, measured once for every stage.
            mid = np.ascontiguousarray(piece.mean(axis=0), dtype=np.float64)
            periodicity, f0 = estimate_voicing(mid, sr)
            voicing = (_debounce_voicing(periodicity), f0)
            if active and scene.uses_floor:
                piece = np.ascontiguousarray(
                    soften_musical_noise(piece, sr, scene, voicing), dtype=np.float32)
            if active and scene.uses_gate:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                piece = np.ascontiguousarray(
                    apply_noise_floor_gate(piece, sr, scene, voicing), dtype=np.float32)
            if wants_lift:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                piece = np.ascontiguousarray(
                    apply_dialogue_lift(piece, sr, lift_db, voicing), dtype=np.float32)
            if wants_restore:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                # Match the dry chunk only while the stage is on. The pad below
                # is unchanged when the stage is off, so Manual stays identical.
                target = segment.shape[-1]
                if piece.shape[-1] < target:
                    piece = np.pad(piece, ((0, 0), (0, target - piece.shape[-1])))
                elif piece.shape[-1] > target:
                    piece = piece[:, :target]
                # The network's own output is voiced in pauses the dry recording
                # is not. Follow the dry chunk, or those pauses brighten.
                dry_mid = np.ascontiguousarray(segment.mean(axis=0), dtype=np.float64)
                dry_period, dry_f0 = estimate_voicing(dry_mid, sr)
                piece = np.ascontiguousarray(
                    apply_consonant_restore(
                        segment, piece, sr, (_debounce_voicing(dry_period), dry_f0)),
                    dtype=np.float32)
        expected = tail - pre_start
        if piece.shape[-1] < expected:
            piece = np.pad(piece, ((0, 0), (0, expected - piece.shape[-1])))
        piece = piece[:, start - pre_start:tail - pre_start]

        weight = np.ones(tail - start, dtype=np.float32)
        if index > 0:
            fade = min(xfade, tail - start)
            weight[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        if not last:
            fade = tail - end
            weight[end - start:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        out[:, start:tail] += piece * weight
        if progress is not None:
            progress((index + 1) / n_chunks)
    return out


def waveform_envelope(audio: np.ndarray, bins: int = 1600) -> tuple[np.ndarray, np.ndarray]:
    """Min/max envelope of the mono mix, for drawing."""
    mono = audio.mean(axis=0) if audio.shape[0] > 1 else audio[0]
    bins = max(1, min(bins, len(mono)))
    edges = np.linspace(0, len(mono), bins + 1).astype(np.int64)[:-1]
    return np.minimum.reduceat(mono, edges), np.maximum.reduceat(mono, edges)


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def write_error_log(text: str) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(ERROR_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n{text}\n")
    except OSError:
        pass


def resolve_output_dir(preferred: str | None) -> Path:
    """First folder that exists or can be created: saved choice, D:, then Music."""
    for candidate in (preferred, DEFAULT_OUTPUT_DIR, FALLBACK_OUTPUT_DIR):
        if not candidate:
            continue
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            continue
    return Path.cwd()


def format_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def shorten(text: str, limit: int = 64) -> str:
    return text if len(text) <= limit else "…" + text[-(limit - 1):]


def friendly_error(stage: str, exc: Exception) -> str:
    if isinstance(exc, ModuleNotFoundError):
        name = exc.name or ""
        if name.startswith("torchaudio") or name.startswith("torch."):
            return (f"DeepFilterNet needs torch/torchaudio 2.1.2 ('{name}' is missing in "
                    "this version). Delete the .venv folder and run AudioDeNoiser.bat again.")
        return (f"The Python package '{name}' is missing. "
                "Run AudioDeNoiser.bat to install everything.")
    if isinstance(exc, getattr(sf, "LibsndfileError", ())) or "Format not recognised" in str(exc):
        return ("This file can't be read. Supported formats are WAV, MP3, FLAC and OGG; "
                "convert anything else (M4A, AAC, WMA) to WAV first.")
    if isinstance(exc, PermissionError):
        return "Windows refused access to the output folder. Choose another with Change…"
    if isinstance(exc, MemoryError):
        return "Not enough memory for this file. Split it into shorter parts and try again."
    if stage == "model":
        return (f"The DeepFilterNet model didn't load ({exc}). The first run needs an "
                "internet connection to download it.")
    return f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class CaptionCue:
    """One subtitle, in seconds on the same clock as the 48 kHz buffer."""

    start_s: float
    end_s: float
    text: str


def caption_model_dir(output_dir: Path | None) -> Path:
    """Folder for the subtitle weights. Prefer D:, away from the tight system drive."""
    candidates = [Path(r"D:\Audio\Models")]
    if output_dir:
        candidates.append(Path(output_dir) / "models")
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        return candidate
    raise OSError(
        "Couldn't create a folder for the subtitle model "
        "(tried D:\\Audio\\Models and the output folder)."
    )


def _greedy_words(
    emissions: np.ndarray, labels: tuple[str, ...], offset_s: float
) -> list[CaptionCue]:
    """Greedy CTC decode to lowercase words with start and end times.

    ``emissions`` is (time, classes) for one 16 kHz chunk. Repeated tokens
    collapse. The blank is labels[0]. ``|`` breaks words.
    """
    if emissions.ndim != 2 or emissions.shape[1] != len(labels):
        raise ValueError(
            f"Expected emissions (time, {len(labels)}), got {tuple(emissions.shape)}."
        )
    try:
        boundary = labels.index("|")
    except ValueError as exc:
        raise ValueError("The recogniser labels have no word boundary '|'.") from exc
    blank = 0
    chosen = np.argmax(emissions, axis=1)
    words: list[CaptionCue] = []
    chars: list[str] = []
    start: int | None = None
    previous = blank

    def close(end_frame: int) -> None:
        nonlocal start
        if chars and start is not None:
            words.append(CaptionCue(
                offset_s + start * CTC_STRIDE / ASR_SR,
                offset_s + end_frame * CTC_STRIDE / ASR_SR,
                "".join(chars).lower(),
            ))
        chars.clear()
        start = None

    for frame, token in enumerate(chosen.tolist()):
        if token == previous:
            continue
        previous = token
        if token == blank:
            continue
        if token == boundary:
            close(frame)
            continue
        if start is None:
            start = frame
        letter = labels[token]
        if letter:
            chars.append(letter)
    close(int(chosen.shape[0]))
    return words


def pack_cues(words: list[CaptionCue]) -> list[CaptionCue]:
    """Group words into readable cues. A gap, 42 characters or 6 s starts a new one."""
    cues: list[CaptionCue] = []
    bucket: list[CaptionCue] = []

    def flush() -> None:
        if not bucket:
            return
        text = " ".join(word.text for word in bucket)
        start_s = bucket[0].start_s
        end_s = min(max(bucket[-1].end_s + CUE_TAIL_S, start_s + CUE_MIN_S), start_s + CUE_MAX_S)
        cues.append(CaptionCue(start_s, end_s, text))
        bucket.clear()

    for word in words:
        if not word.text:
            continue
        gap = bool(bucket) and (word.start_s - bucket[-1].end_s) > CUE_GAP_S
        too_long = bool(bucket) and (word.end_s - bucket[0].start_s) > CUE_MAX_S
        width = len(" ".join(item.text for item in bucket))
        too_wide = bool(bucket) and (width + 1 + len(word.text)) > CUE_MAX_CHARS
        if gap or too_long or too_wide:
            flush()
        bucket.append(word)
    flush()
    fitted: list[CaptionCue] = []
    for index, cue in enumerate(cues):
        end_s = cue.end_s
        if index + 1 < len(cues):
            end_s = min(end_s, cues[index + 1].start_s)
        if end_s > cue.start_s:
            fitted.append(CaptionCue(cue.start_s, end_s, cue.text))
    return fitted


def _srt_time(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0.0:
        seconds = 0.0
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000.0))
    if millis == 1000:
        whole += 1
        millis = 0
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _replace_text(path: Path, text: str) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    try:
        partial.write_text(text, encoding="utf-8")
        os.replace(partial, path)
    except OSError:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_transcript(path: Path, cues: list[CaptionCue]) -> Path:
    """Write a .srt beside ``path`` and a .txt of the same words. Returns the .srt path."""
    srt_path = path.with_suffix(".srt")
    txt_path = path.with_suffix(".txt")
    blocks = [
        f"{index}\n{_srt_time(cue.start_s)} --> {_srt_time(cue.end_s)}\n{cue.text}\n"
        for index, cue in enumerate(cues, start=1)
    ]
    _replace_text(srt_path, "\n".join(blocks) + ("\n" if blocks else ""))
    paragraph = " ".join(cue.text for cue in cues).strip()
    _replace_text(txt_path, (paragraph + "\n") if paragraph else "")
    return srt_path


def resampled_length(n_frames: int, sr_in: int, sr_out: int) -> int:
    """Frame count ``resample`` would produce. Matches scipy's ceil(n * up / down)."""
    if n_frames <= 0 or sr_in <= 0 or sr_out <= 0:
        raise ValueError("A length needs a positive buffer and sample rate.")
    if sr_in == sr_out:
        return int(n_frames)
    g = gcd(int(sr_in), int(sr_out))
    up = int(sr_out) // g
    down = int(sr_in) // g
    return int(np.ceil(int(n_frames) * up / down))


def selection_bounds(
    origin: float, current: float, n_frames: int, sr: int,
) -> tuple[int, int] | None:
    """Sample range for a Shift-drag, or None when it is shorter than 50 ms."""
    if n_frames <= 0 or sr <= 0:
        raise ValueError("A selection needs a loaded buffer.")
    start_f = min(origin, current)
    end_f = max(origin, current)
    start = int(round(start_f * n_frames))
    end = int(round(end_f * n_frames))
    start = max(0, min(n_frames - 1, start))
    end = max(start + 1, min(n_frames, end))
    if (end - start) / sr < SELECTION_MIN_S:
        return None
    return start, end


def cut_session(audio: np.ndarray, start: int, end: int, sr: int) -> np.ndarray:
    """Drop [start, end) and fade the join over 2 ms. Length shrinks by exactly the cut.

    The array on disk is not involved. Both kept edges are faded so the splice
    does not click. The fades do not overlap, so later cue times stay exact.
    """
    if audio.ndim != 2:
        raise ValueError("Session audio is shaped (channels, frames).")
    n = audio.shape[1]
    if not (0 <= start < end <= n):
        raise ValueError(f"Cut {start}:{end} is outside 0:{n}.")
    if (n - (end - start)) / sr < SELECTION_MIN_KEEP_S:
        raise ValueError("Cutting that would leave under a quarter of a second.")
    if not np.isfinite(audio).all():
        raise ValueError("The session contains invalid samples.")
    left = np.array(audio[:, :start], dtype=np.float32, copy=True)
    right = np.array(audio[:, end:], dtype=np.float32, copy=True)
    fade = min(int(round(JOIN_FADE_S * sr)), left.shape[1], right.shape[1])
    if fade > 1:
        ramp = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        left[:, -fade:] *= ramp
        right[:, :fade] *= ramp[::-1]
    return np.ascontiguousarray(np.concatenate([left, right], axis=1), dtype=np.float32)


def shift_cues(cues: list[CaptionCue], start_s: float, end_s: float) -> list[CaptionCue]:
    """Keep lines outside a cut and move later lines earlier. Overlapping lines are dropped."""
    if end_s <= start_s:
        raise ValueError("A cut has to have a positive length.")
    span = end_s - start_s
    kept: list[CaptionCue] = []
    for cue in cues:
        if cue.end_s <= start_s:
            kept.append(cue)
        elif cue.start_s >= end_s:
            kept.append(CaptionCue(cue.start_s - span, cue.end_s - span, cue.text))
    return kept


def fill_looped(
    dest: np.ndarray, buf: np.ndarray, pos: int, loop: tuple[int, int] | None,
) -> tuple[int, int, bool]:
    """Fill ``dest`` along axis 0 from ``buf``. Returns samples written, new position, still playing.

    With no loop, playback runs to the end of ``buf`` and then stops. With a
    loop, a read that would pass the end wraps to the loop start and playback
    stays running. ``dest`` and ``buf`` are time-major, as the player stores them.
    """
    frames = int(dest.shape[0])
    n = int(buf.shape[0])
    if frames <= 0 or n <= 0:
        return 0, max(0, min(int(pos), n)), False
    if loop is None:
        if pos < 0 or pos >= n:
            return 0, min(max(int(pos), 0), n), False
        count = min(frames, n - int(pos))
        dest[:count] = buf[int(pos):int(pos) + count]
        new_pos = int(pos) + count
        return count, new_pos, new_pos < n
    start, end = int(loop[0]), int(loop[1])
    end = min(end, n)
    start = max(0, min(start, end - 1))
    if end <= start:
        return 0, int(pos), False
    written = 0
    position = int(pos)
    while written < frames:
        if position < start or position >= end:
            position = start
        count = min(frames - written, end - position)
        if count <= 0:
            return written, position, False
        dest[written:written + count] = buf[position:position + count]
        position += count
        written += count
    return written, position, True


def selection_output_path(folder: Path, stem: str) -> Path:
    """First free '<stem>_selection.wav', then _selection_2, and so on."""
    candidate = folder / f"{stem}_selection.wav"
    index = 2
    while candidate.exists():
        candidate = folder / f"{stem}_selection_{index}.wav"
        index += 1
        if index > 10000:
            raise OSError(f"No free selection name beside {candidate.parent}.")
    return candidate


def save_selection(
    audio48: np.ndarray, start: int, end: int, source_sr: int,
    subtype: str, dest: Path,
) -> Path:
    """Write one range at the source rate. No denoise and no normalisation."""
    if audio48.ndim != 2 or not (0 <= start < end <= audio48.shape[1]):
        raise ValueError("The selection is outside the session.")
    if source_sr <= 0:
        raise ValueError("The source sample rate is missing.")
    piece = np.ascontiguousarray(audio48[:, start:end], dtype=np.float32)
    if not np.isfinite(piece).all():
        raise ValueError("The selection contains invalid samples.")
    if source_sr != MODEL_SR:
        piece = resample(piece, MODEL_SR, source_sr)
    if subtype not in EXPORT_SUBTYPES:
        subtype = "PCM_24"
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".partial.wav")
    try:
        write_wav(partial, piece, source_sr, subtype)
        os.replace(partial, dest)
    except Exception:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return dest


# --------------------------------------------------------------------------- #
#  Playback: in-memory buffers, instant A/B switching at the same position
# --------------------------------------------------------------------------- #
class Player:
    """Streams numpy buffers through sounddevice.

    Both buffers are 48 kHz stereo and the same length, so switching between
    original and denoised keeps the exact playback position. Nothing is read
    from disk, so the output file is never locked while you listen. Each buffer
    has a playback gain for loudness-matched A/B; the buffers themselves are
    never rescaled, and a gain of exactly 1.0 is not applied at all.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffers: dict[str, np.ndarray | None] = {"original": None, "denoised": None}
        self._gains = {"original": 1.0, "denoised": 1.0}
        self._source = "original"
        self._pos = 0
        self._playing = False
        self._stream = None
        self._loop: tuple[int, int] | None = None

    @staticmethod
    def _to_stereo(audio: np.ndarray) -> np.ndarray:
        stereo = np.repeat(audio, 2, axis=0) if audio.shape[0] == 1 else audio[:2]
        return np.ascontiguousarray(stereo.T, dtype=np.float32)

    # State ------------------------------------------------------------------
    @property
    def source(self) -> str:
        return self._source

    @property
    def length(self) -> int:
        buf = self._buffers["original"]
        return 0 if buf is None else len(buf)

    @property
    def position(self) -> int:
        return self._pos

    @property
    def is_playing(self) -> bool:
        return self._playing

    def has(self, key: str) -> bool:
        return self._buffers.get(key) is not None

    # Buffers ----------------------------------------------------------------
    def load_original(self, audio: np.ndarray) -> None:
        self.stop()
        stereo = self._to_stereo(audio)
        with self._lock:
            self._buffers = {"original": stereo, "denoised": None}
            self._gains = {"original": 1.0, "denoised": 1.0}
            self._source = "original"
            self._pos = 0
            self._loop = None

    def set_denoised(self, audio: np.ndarray) -> None:
        stereo = self._to_stereo(audio)
        with self._lock:
            self._buffers["denoised"] = stereo

    def set_gains(self, original: float, denoised: float) -> None:
        with self._lock:
            self._gains = {"original": float(original), "denoised": float(denoised)}

    def set_source(self, key: str) -> None:
        with self._lock:
            if self._buffers.get(key) is not None:
                self._source = key

    def seek(self, fraction: float) -> None:
        with self._lock:
            self._pos = int(max(0.0, min(1.0, fraction)) * self.length)

    def set_loop(self, bounds: tuple[int, int] | None) -> None:
        """Loop [start, end) at 48 kHz, or clear it. Play outside the range jumps to the start."""
        with self._lock:
            self._loop = bounds
            if bounds is None:
                return
            buf = self._buffers.get(self._source)
            n = 0 if buf is None else len(buf)
            start, end = bounds
            end = min(int(end), n)
            start = max(0, min(int(start), max(end - 1, 0)))
            if n and (self._pos < start or self._pos >= end):
                self._pos = start

    # Transport --------------------------------------------------------------
    def play(self) -> None:
        if sd is None:
            raise RuntimeError(f"audio output unavailable ({SD_IMPORT_ERROR})")
        with self._lock:
            buf = self._buffers.get(self._source)
            if buf is None:
                return
            if self._loop is not None:
                start, end = self._loop
                end = min(int(end), len(buf))
                start = max(0, min(int(start), max(end - 1, 0)))
                if self._pos < start or self._pos >= end:
                    self._pos = start
            elif self._pos >= len(buf):
                self._pos = 0
            self._playing = True
        if self._stream is None:
            try:
                self._stream = sd.OutputStream(samplerate=MODEL_SR, channels=2,
                                               dtype="float32", callback=self._callback)
                self._stream.start()
            except Exception:
                self._playing = False
                self._stream = None
                raise

    def pause(self) -> None:
        with self._lock:
            self._playing = False
        self._close_stream()

    def stop(self) -> None:
        self.pause()
        with self._lock:
            self._pos = 0

    def tick(self) -> None:
        """Called from the UI thread; releases the device once playback ends."""
        if self._stream is not None and not self._playing:
            self._close_stream()

    def _close_stream(self) -> None:
        # Never hold self._lock here: abort() waits for the callback, which takes it.
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.abort()
                stream.close()
            except Exception:
                pass

    def _callback(self, outdata, frames, time_info, status) -> None:
        with self._lock:
            buf = self._buffers.get(self._source)
            if not self._playing or buf is None:
                outdata.fill(0)
                return
            written, self._pos, self._playing = fill_looped(
                outdata, buf, self._pos, self._loop)
            gain = self._gains[self._source]
            if gain != 1.0 and written:
                outdata[:written] *= gain
            if written < frames:
                outdata[written:] = 0


# --------------------------------------------------------------------------- #
#  Waveform view
# --------------------------------------------------------------------------- #
class WaveformView(tk.Canvas):
    """Original in concrete grey, denoised in amber on top.

    A plain click or drag seeks and leaves any selection where it is.
    Shift-drag replaces the selection. The rectangle is stippled amber,
    drawn behind the white playhead.
    """

    def __init__(self, master, on_seek, on_select=None, **kwargs):
        super().__init__(master, bg=C_PANEL, highlightthickness=0, bd=0, **kwargs)
        self._on_seek = on_seek
        self._on_select = on_select
        self._orig = None
        self._den = None
        self._playhead = None
        self._sel: tuple[float, float] | None = None
        self._sel_item = None
        self._fraction = 0.0
        self._shift_drag = False
        self._origin = 0.0
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", self._drag)

    def set_data(self, original, denoised=None) -> None:
        self._orig, self._den = original, denoised
        self.configure(cursor="hand2" if original is not None else "")
        self.redraw()

    def set_playhead(self, fraction: float) -> None:
        if abs(fraction - self._fraction) < 1e-4 and self._playhead is not None:
            return
        self._fraction = fraction
        if self._playhead is not None:
            x = fraction * self.winfo_width()
            self.coords(self._playhead, x, 0, x, self.winfo_height())

    def set_selection(self, start: float | None, end: float | None) -> None:
        """Mark a range as fractions of the width, or clear it with None."""
        if start is None or end is None:
            self._sel = None
        else:
            self._sel = (min(float(start), float(end)), max(float(start), float(end)))
        self._paint_selection()

    def _seek_event(self, event) -> None:
        if self._orig is not None and self.winfo_width() > 0:
            self._on_seek(event.x / self.winfo_width())

    def _press(self, event) -> None:
        self._shift_drag = bool(event.state & 0x0001)
        width = self.winfo_width()
        self._origin = 0.0 if width <= 0 else event.x / width
        if not self._shift_drag:
            self._seek_event(event)

    def _drag(self, event) -> None:
        if self._shift_drag:
            width = self.winfo_width()
            if self._orig is not None and self._on_select is not None and width > 0:
                current = min(1.0, max(0.0, event.x / width))
                self._on_select(self._origin, current)
            return
        self._seek_event(event)

    def redraw(self) -> None:
        self.delete("all")
        self._playhead = None
        self._sel_item = None
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or h < 10:
            return
        mid = h / 2
        if self._orig is None:
            self.create_text(w / 2, mid, fill=C_MUTED, font=(FONT_FAMILY, 11),
                             text="Open a WAV, MP3, FLAC or OGG file to see its waveform")
            return
        self.create_line(0, mid, w, mid, fill=C_LINE)
        peak = max(float(np.max(np.abs(env))) for pair in (self._orig, self._den)
                   if pair is not None for env in pair)
        scale = (mid - 8) / max(peak, 1e-6)
        self._polygon(self._orig, C_CONCRETE, w, mid, scale)
        if self._den is not None:
            self._polygon(self._den, C_AMBER, w, mid, scale)
            self.create_text(10, 10, anchor="nw", text="Original", fill=C_MUTED,
                             font=(FONT_FAMILY, 9))
            self.create_text(70, 10, anchor="nw", text="Denoised", fill=C_AMBER,
                             font=(FONT_FAMILY, 9))
        self._sel_item = None
        self._draw_selection_item(w, h)
        x = self._fraction * w
        self._playhead = self.create_line(x, 0, x, h, fill=C_TEXT, width=2)

    def _draw_selection_item(self, w: int, h: int) -> None:
        if self._sel is None:
            return
        x0 = self._sel[0] * w
        x1 = self._sel[1] * w
        self._sel_item = self.create_rectangle(
            x0, 0, x1, h, outline="", fill=C_AMBER, stipple="gray50")

    def _paint_selection(self) -> None:
        """Replace the selection rectangle without redrawing the waveform."""
        if self._sel_item is not None:
            self.delete(self._sel_item)
            self._sel_item = None
        w, h = self.winfo_width(), self.winfo_height()
        if self._sel is None or self._orig is None or w < 10 or h < 10:
            return
        self._draw_selection_item(w, h)
        if self._playhead is not None:
            self.tag_raise(self._playhead)

    def _polygon(self, env, colour, w, mid, scale) -> None:
        lo, hi = env
        xs = np.linspace(0, w, len(lo))
        top = np.column_stack([xs, mid - hi * scale - 0.5])
        bottom = np.column_stack([xs[::-1], mid - lo[::-1] * scale + 0.5])
        points = np.concatenate([top, bottom]).ravel().tolist()
        self.create_polygon(points, fill=colour, outline="")


# --------------------------------------------------------------------------- #
#  Main window
# --------------------------------------------------------------------------- #
class AudioDeNoiserApp(ctk.CTk):
    POLL_MS = 50

    def __init__(self, initial_file: str | None = None) -> None:
        super().__init__()
        self.title(f"{APP_NAME} {APP_VERSION}")
        self.geometry("860x1054")
        # Measured with Save preset… on the scene row: the waveform is 129 px at
        # 1001 and 130 px at 1002. The default 1020 still shows the log.
        self.minsize(760, 1036)
        self.configure(fg_color=C_BG)

        self.settings = load_settings()
        saved_scene = self.settings.get("scene", "manual")
        self._applied_scene = saved_scene if saved_scene in SCENES else "manual"
        self._user_presets = clean_user_presets(self.settings.get("user_presets"))
        self._applying_preset = False
        wanted = self.settings.get("selected_preset", "")
        self._startup_preset = ""
        if isinstance(wanted, str) and any(item["name"] == wanted for item in self._user_presets):
            self._startup_preset = wanted
        self.output_dir = resolve_output_dir(self.settings.get("output_dir"))
        self.events: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.player = Player()

        self.model = None
        self.df_state = None
        self.source_path: str | None = None
        self.source_sr: int | None = None
        self.source_frames = 0
        self.audio48: np.ndarray | None = None
        self.denoised48: np.ndarray | None = None
        self.orig_env = None
        self.output_path: str | None = None
        self.source_subtype: str | None = None  # None: no bit depth to keep, save 24-bit
        self.original_lufs = float("-inf")
        self.denoised_lufs = float("-inf")
        self.denoised_peak = 0.0
        self.busy: str | None = None  # None, "load", "denoise", "captions", "selection" or "save"
        self._export_dirty = False
        self._after_save: str | None = None  # "close", "open" or "denoise" once a save finishes
        self._pending_open: str | None = None
        self._discard_export = False
        self._last_time_text = ""
        self._pending_file = initial_file
        self._selection: tuple[int, int] | None = None
        self._loop_on = False
        self._undo: dict | None = None
        self._cues: list[CaptionCue] = []
        self._cue_text = ""
        self._caption_warned = False
        self._caption_model = None
        self._caption_labels: tuple[str, ...] | None = None

        self._build_ui()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.log(f"Output folder: {self.output_dir}")
        if sd is None:
            self.log(f"Playback is unavailable: {SD_IMPORT_ERROR}")
        self._run_job("model", self._job_load_model)
        if initial_file:
            self.after(200, lambda: self.open_file(initial_file))
        self._refresh_controls()
        self.after(self.POLL_MS, self._poll)

    # ------------------------------------------------------------------ UI --
    def _font(self, size: int, weight: str = "normal") -> ctk.CTkFont:
        return ctk.CTkFont(family=FONT_FAMILY, size=size, weight=weight)

    def _secondary_button(self, master, text, command, width=110) -> ctk.CTkButton:
        return ctk.CTkButton(master, text=text, command=command, width=width, height=34,
                             fg_color=C_LINE, hover_color=C_LINE_HOVER, text_color=C_TEXT,
                             text_color_disabled=C_MUTED, font=self._font(13))

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1, minsize=130)  # the waveform takes spare height
        pad = {"padx": 22}

        # Header -----------------------------------------------------------
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(14, 8), **pad)
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text=APP_NAME, font=self._font(24, "bold"),
                     text_color=C_TEXT).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(header, text="Clean up speech recorded in traffic, wind and city noise",
                     font=self._font(13), text_color=C_MUTED).grid(row=1, column=0, sticky="w")
        self.status_label = ctk.CTkLabel(header, text="Loading model…", font=self._font(12),
                                         text_color=C_MUTED)
        self.status_label.grid(row=0, column=1, rowspan=2, sticky="e")

        # File bar ---------------------------------------------------------
        file_bar = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=10)
        file_bar.grid(row=1, column=0, sticky="ew", pady=(0, 10), **pad)
        file_bar.grid_columnconfigure(0, weight=1)
        self.file_label = ctk.CTkLabel(file_bar, text="No file open", anchor="w",
                                       font=self._font(14, "bold"), text_color=C_TEXT)
        self.file_label.grid(row=0, column=0, sticky="ew", padx=(14, 8), pady=(10, 0))
        self.file_info = ctk.CTkLabel(file_bar, text="WAV, MP3, FLAC or OGG", anchor="w",
                                      font=self._font(12), text_color=C_MUTED)
        self.file_info.grid(row=1, column=0, sticky="ew", padx=(14, 8), pady=(0, 10))
        self.open_btn = self._secondary_button(file_bar, "Open audio…", self.open_file, 130)
        self.open_btn.grid(row=0, column=1, rowspan=2, padx=12)

        # Waveform ---------------------------------------------------------
        wave_frame = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=10, height=170)
        wave_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 8), **pad)
        wave_frame.grid_propagate(False)
        wave_frame.grid_columnconfigure(0, weight=1)
        wave_frame.grid_rowconfigure(0, weight=1)
        self.waveform = WaveformView(wave_frame, on_seek=self._seek,
                                     on_select=self._on_waveform_select)
        self.waveform.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Subtitles --------------------------------------------------------
        sub_bar = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=10, height=48)
        sub_bar.grid(row=3, column=0, sticky="ew", pady=(0, 8), **pad)
        sub_bar.grid_propagate(False)
        sub_bar.grid_columnconfigure(0, weight=1)
        self.subtitle_label = ctk.CTkLabel(
            sub_bar, text="", anchor="w", justify="left", font=self._font(15),
            text_color=C_AMBER, wraplength=620)
        self.subtitle_label.grid(row=0, column=0, sticky="ew", padx=(14, 8), pady=6)
        self.save_text_btn = self._secondary_button(sub_bar, "Save text", self.save_transcript, 100)
        self.save_text_btn.grid(row=0, column=1, padx=(0, 12), pady=6)

        # Selection --------------------------------------------------------
        tools = ctk.CTkFrame(self, fg_color="transparent")
        tools.grid(row=4, column=0, sticky="ew", pady=(0, 8), **pad)
        tools.grid_columnconfigure(5, weight=1)
        self.loop_btn = self._secondary_button(tools, "Loop", self.toggle_loop, 72)
        self.loop_btn.grid(row=0, column=0, padx=(0, 6))
        self.cut_btn = self._secondary_button(tools, "Cut", self.cut_selection, 64)
        self.cut_btn.grid(row=0, column=1, padx=(0, 6))
        self.undo_btn = self._secondary_button(tools, "Undo", self.undo_cut, 72)
        self.undo_btn.grid(row=0, column=2, padx=(0, 6))
        self.save_sel_btn = self._secondary_button(tools, "Save selection", self.save_range, 128)
        self.save_sel_btn.grid(row=0, column=3, padx=(0, 6))
        self.read_btn = self._secondary_button(tools, "Read speech", self.read_speech, 118)
        self.read_btn.grid(row=0, column=4, padx=(0, 6))
        self.range_label = ctk.CTkLabel(tools, text="", anchor="e", font=self._font(13),
                                        text_color=C_MUTED)
        self.range_label.grid(row=0, column=5, sticky="e")

        # Transport --------------------------------------------------------
        transport = ctk.CTkFrame(self, fg_color="transparent")
        transport.grid(row=5, column=0, sticky="ew", pady=(0, 10), **pad)
        transport.grid_columnconfigure(5, weight=1)
        self.play_btn = self._secondary_button(transport, "Play", self.toggle_play, 90)
        self.play_btn.grid(row=0, column=0, padx=(0, 6))
        self.stop_btn = self._secondary_button(transport, "Stop", self.stop_audio, 80)
        self.stop_btn.grid(row=0, column=1, padx=(0, 18))
        self.orig_btn = self._secondary_button(transport, "A  Original",
                                               lambda: self.set_source("original"), 120)
        self.orig_btn.grid(row=0, column=2, padx=(0, 6))
        self.den_btn = self._secondary_button(transport, "B  Denoised",
                                              lambda: self.set_source("denoised"), 120)
        self.den_btn.grid(row=0, column=3)
        # Playback only: it sets the A and B gains, never the saved file.
        self.loudness_var = ctk.BooleanVar(value=bool(self.settings.get("loudness_match", True)))
        self.loudness_switch = ctk.CTkSwitch(transport, text="Match A/B loudness",
                                             variable=self.loudness_var, font=self._font(13),
                                             text_color=C_TEXT, progress_color=C_AMBER,
                                             command=self._on_loudness_toggle)
        self.loudness_switch.grid(row=0, column=4, sticky="w", padx=(18, 0))
        self.time_label = ctk.CTkLabel(transport, text="0:00 / 0:00", font=self._font(13),
                                       text_color=C_MUTED)
        self.time_label.grid(row=0, column=5, sticky="e")

        # Settings ---------------------------------------------------------
        panel = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=10)
        panel.grid(row=6, column=0, sticky="ew", pady=(0, 10), **pad)
        panel.grid_columnconfigure(1, weight=1)
        label_opts = {"font": self._font(13), "text_color": C_TEXT, "anchor": "w"}
        hint_opts = {"font": self._font(12), "text_color": C_MUTED, "anchor": "w"}
        row_pad = {"padx": 14, "pady": (8, 0)}

        opening = self._startup_preset or SCENES[self._applied_scene].label
        self.scene_var = ctk.StringVar(value=opening)
        ctk.CTkLabel(panel, text="Scene", **label_opts).grid(
            row=0, column=0, sticky="w", **row_pad)
        scene_row = ctk.CTkFrame(panel, fg_color="transparent")
        scene_row.grid(row=0, column=1, columnspan=2, sticky="ew",
                       padx=(0, 14), pady=(8, 0))
        scene_row.grid_columnconfigure(1, weight=1)
        # Fixed width: a 40-character name must not push Delete off the row.
        self.scene_menu = ctk.CTkOptionMenu(
            scene_row, values=self._scene_labels(), variable=self.scene_var,
            width=240, dynamic_resizing=False, command=self._on_scene_menu,
            fg_color=C_LINE, button_color=C_LINE, button_hover_color=C_LINE_HOVER,
            text_color=C_TEXT, font=self._font(13))
        self.scene_menu.grid(row=0, column=0, sticky="w")
        self.save_preset_btn = self._secondary_button(
            scene_row, "Save preset…", self.save_preset, 128)
        self.save_preset_btn.grid(row=0, column=2, padx=(12, 6))
        self.delete_preset_btn = self._secondary_button(
            scene_row, "Delete", self.delete_preset, 80)
        self.delete_preset_btn.grid(row=0, column=3, sticky="e")
        self.scene_hint = ctk.CTkLabel(
            panel, text=SCENES[self._applied_scene].summary, wraplength=640, **hint_opts)
        self.scene_hint.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 14))

        ctk.CTkLabel(panel, text="Dialogue", **label_opts).grid(
            row=2, column=0, sticky="w", **row_pad)
        self.lift_var = ctk.StringVar(value=self._saved_lift_label())
        self.lift_menu = ctk.CTkOptionMenu(
            panel, values=list(LIFT_CHOICES), variable=self.lift_var, width=120,
            command=self._on_lift_change, fg_color=C_LINE, button_color=C_LINE,
            button_hover_color=C_LINE_HOVER, text_color=C_TEXT, font=self._font(13))
        self.lift_menu.grid(row=2, column=1, sticky="w", pady=(8, 0))
        ctk.CTkLabel(panel, text="Lifts consonants a few metres away. Off keeps today's sound.",
                     **hint_opts).grid(row=2, column=2, sticky="w", padx=(0, 14), pady=(8, 0))

        ctk.CTkLabel(panel, text="Noise reduction", **label_opts).grid(row=3, column=0, sticky="w", **row_pad)
        self.atten_index = int(self.settings.get("atten_index", len(ATTEN_STEPS) - 1))
        self.atten_index = max(0, min(len(ATTEN_STEPS) - 1, self.atten_index))
        self.atten_slider = ctk.CTkSlider(panel, from_=0, to=len(ATTEN_STEPS) - 1,
                                          number_of_steps=len(ATTEN_STEPS) - 1,
                                          command=self._on_atten_change, progress_color=C_AMBER,
                                          button_color=C_AMBER, button_hover_color=C_AMBER_HOVER)
        self.atten_slider.set(self.atten_index)
        self.atten_slider.grid(row=3, column=1, sticky="ew", padx=(0, 10), pady=(12, 0))
        self.atten_value = ctk.CTkLabel(panel, text="", width=90, **label_opts)
        self.atten_value.grid(row=3, column=2, sticky="w", padx=(0, 14), pady=(8, 0))
        ctk.CTkLabel(panel, text="Lower limits leave a little background in and sound more natural",
                     **hint_opts).grid(row=4, column=1, columnspan=2, sticky="w")

        ctk.CTkLabel(panel, text="Rumble filter", **label_opts).grid(row=5, column=0, sticky="w", **row_pad)
        self.highpass_var = ctk.StringVar(value=self.settings.get("highpass", "80 Hz"))
        if self.highpass_var.get() not in HIGHPASS_CHOICES:
            self.highpass_var.set("80 Hz")
        hp_row = ctk.CTkFrame(panel, fg_color="transparent")
        hp_row.grid(row=5, column=1, columnspan=2, sticky="w", pady=(8, 0))
        self.highpass_menu = ctk.CTkOptionMenu(hp_row, values=list(HIGHPASS_CHOICES),
                                               variable=self.highpass_var, width=110,
                                               command=lambda _v: self._save_settings(),
                                               fg_color=C_LINE, button_color=C_LINE,
                                               button_hover_color=C_LINE_HOVER, text_color=C_TEXT,
                                               font=self._font(13))
        self.highpass_menu.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(hp_row, text="Cuts wind and traffic rumble below this pitch",
                     **hint_opts).grid(row=0, column=1, sticky="w", padx=(12, 0))

        switches = ctk.CTkFrame(panel, fg_color="transparent")
        switches.grid(row=6, column=0, columnspan=3, sticky="w", padx=14, pady=(12, 0))
        self.postfilter_var = ctk.BooleanVar(value=bool(self.settings.get("post_filter", False)))
        self.postfilter_switch = ctk.CTkSwitch(switches, text="Post-filter for very noisy passages",
                                               variable=self.postfilter_var, font=self._font(13),
                                               text_color=C_TEXT, progress_color=C_AMBER,
                                               command=self._save_settings)
        self.postfilter_switch.grid(row=0, column=0, sticky="w", padx=(0, 28))

        self.normalise_var = ctk.BooleanVar(value=bool(self.settings.get("normalise", True)))
        self.normalise_switch = ctk.CTkSwitch(switches, text="Normalise saved file to −1 dBFS",
                                              variable=self.normalise_var, font=self._font(13),
                                              text_color=C_TEXT, progress_color=C_AMBER,
                                              command=self._on_normalise_toggle)
        self.normalise_switch.grid(row=0, column=1, sticky="w")
        self.subtitles_var = ctk.BooleanVar(value=bool(self.settings.get("subtitles", False)))
        self.subtitles_switch = ctk.CTkSwitch(
            switches, text="English transcript after denoise", variable=self.subtitles_var,
            font=self._font(13), text_color=C_TEXT, progress_color=C_AMBER,
            command=self._save_settings)
        self.subtitles_switch.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.restore_var = ctk.BooleanVar(value=bool(self.settings.get("consonant_restore", False)))
        self.restore_switch = ctk.CTkSwitch(
            switches, text="Restore consonants", variable=self.restore_var,
            font=self._font(13), text_color=C_TEXT, progress_color=C_AMBER,
            command=self._save_settings)
        self.restore_switch.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))

        out_row = ctk.CTkFrame(panel, fg_color="transparent")
        out_row.grid(row=7, column=0, columnspan=3, sticky="ew", padx=14, pady=(10, 12))
        out_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(out_row, text="Save to", **label_opts).grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.out_label = ctk.CTkLabel(out_row, text="", **hint_opts)
        self.out_label.grid(row=0, column=1, sticky="ew")
        self.change_btn = self._secondary_button(out_row, "Change…", self.choose_output_dir, 90)
        self.change_btn.grid(row=0, column=2, padx=(8, 6))
        self.folder_btn = self._secondary_button(out_row, "Open folder", self.open_output_dir, 110)
        self.folder_btn.grid(row=0, column=3)

        # Action row -------------------------------------------------------
        action = ctk.CTkFrame(self, fg_color="transparent")
        action.grid(row=7, column=0, sticky="ew", pady=(0, 8), **pad)
        action.grid_columnconfigure(0, weight=1)
        self.denoise_btn = ctk.CTkButton(action, text="Denoise", command=self.start_denoise,
                                         height=44, font=self._font(16, "bold"),
                                         fg_color=C_AMBER, hover_color=C_AMBER_HOVER,
                                         text_color=C_ON_AMBER, text_color_disabled="#6B5F4D")
        self.denoise_btn.grid(row=0, column=0, sticky="ew")
        self.save_wav_btn = ctk.CTkButton(action, text="Save denoised…", command=self.save_denoised,
                                          width=150, height=44, font=self._font(14),
                                          fg_color=C_LINE, hover_color=C_LINE_HOVER,
                                          text_color=C_TEXT, text_color_disabled=C_MUTED)
        self.save_wav_btn.grid(row=0, column=1, padx=(8, 0))
        self.cancel_btn = ctk.CTkButton(action, text="Cancel", command=self.cancel_denoise,
                                        width=100, height=44, font=self._font(14),
                                        fg_color=C_DANGER, hover_color=C_DANGER_HOVER,
                                        text_color=C_TEXT, text_color_disabled=C_MUTED)
        self.cancel_btn.grid(row=0, column=2, padx=(8, 0))
        progress_row = ctk.CTkFrame(action, fg_color="transparent")
        progress_row.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        progress_row.grid_columnconfigure(0, weight=1)
        self.progress = ctk.CTkProgressBar(progress_row, mode="determinate", height=8,
                                           progress_color=C_AMBER, fg_color=C_LINE)
        self.progress.set(0)
        self.progress.grid(row=0, column=0, sticky="ew")
        self.progress_label = ctk.CTkLabel(progress_row, text="", width=48, font=self._font(12),
                                           text_color=C_MUTED)
        self.progress_label.grid(row=0, column=1, padx=(8, 0))

        # Log --------------------------------------------------------------
        self.log_box = ctk.CTkTextbox(self, height=92, fg_color=C_PANEL, text_color=C_TEXT,
                                      font=self._font(12), corner_radius=10, wrap="word")
        self.log_box.grid(row=8, column=0, sticky="ew", pady=(0, 16), **pad)
        self.log_box.configure(state="disabled")

        self._on_atten_change(self.atten_index)
        self._show_output_dir()

    def _bind_keys(self) -> None:
        self.bind("<space>", lambda _e: self.toggle_play())
        self.bind("<Escape>", lambda _e: self.clear_selection())
        self.bind("<Control-o>", lambda _e: self.open_file())
        self.bind("<Control-d>", lambda _e: self.start_denoise())
        for key, source in (("a", "original"), ("A", "original"), ("b", "denoised"), ("B", "denoised")):
            self.bind(f"<KeyPress-{key}>", lambda _e, s=source: self.set_source(s))

    # ------------------------------------------------------------ helpers --
    def log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{time.strftime('%H:%M:%S')}  {message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _scene_key(self) -> str:
        """Factory scene key. A saved preset's menu label is not a factory scene."""
        if not hasattr(self, "scene_var"):
            return "manual"
        preset = SCENE_BY_LABEL.get(self.scene_var.get())
        if preset is not None:
            return preset.key
        # The menu is showing a user preset. The scene underneath is _applied_scene.
        # Returning "manual" here would persist the wrong scene on the next save.
        applied = self._applied_scene
        return applied if applied in SCENES else "manual"

    def _scene_labels(self) -> list[str]:
        labels = [SCENES[key].label for key in SCENE_ORDER]
        labels.extend(str(item["name"]) for item in self._user_presets)
        return labels

    def _rebuild_scene_menu(self) -> None:
        self.scene_menu.configure(values=self._scene_labels())

    def _current_preset_name(self) -> str:
        if not hasattr(self, "scene_var"):
            return ""
        label = self.scene_var.get()
        for item in self._user_presets:
            if item["name"] == label:
                return label
        return ""

    def _find_preset(self, name: str) -> dict[str, object] | None:
        for item in self._user_presets:
            if item["name"] == name:
                return item
        return None

    def _remember_manual(self) -> None:
        """Keep the Manual slider, rumble choice and post-filter for the return trip."""
        self.settings["manual_highpass"] = self.highpass_var.get()
        self.settings["manual_atten_index"] = self.atten_index
        self.settings["manual_post_filter"] = bool(self.postfilter_var.get())

    def _restore_manual(self) -> None:
        highpass = self.settings.get("manual_highpass", "80 Hz")
        if highpass not in HIGHPASS_CHOICES:
            highpass = "80 Hz"
        self.highpass_var.set(highpass)
        index = int(self.settings.get("manual_atten_index", len(ATTEN_STEPS) - 1))
        self.atten_index = max(0, min(len(ATTEN_STEPS) - 1, index))
        self.atten_slider.set(self.atten_index)
        self.postfilter_var.set(bool(self.settings.get("manual_post_filter", False)))
        self._on_atten_change(self.atten_index)

    def _on_scene_change(self, label: str) -> None:
        preset = SCENE_BY_LABEL.get(label, MANUAL)
        previous = SCENES.get(self._applied_scene, MANUAL)
        if preset.active and not previous.active:
            self._remember_manual()
        # Set this before the slider moves. The slider callback saves settings, and
        # the save must already know the scene is active or it will overwrite the
        # Manual snapshot with the preset values.
        self._applied_scene = preset.key
        self.lift_var.set(lift_label(preset.lift_db))
        if preset.active:
            if preset.atten_db in ATTEN_STEPS:
                self.atten_index = ATTEN_STEPS.index(preset.atten_db)
                self.atten_slider.set(self.atten_index)
            self.postfilter_var.set(preset.post_filter)
            self.highpass_var.set("Off")
        else:
            self._restore_manual()
        self.scene_hint.configure(text=preset.summary)
        self._save_settings()
        self._refresh_controls()

    def _on_scene_menu(self, label: str) -> None:
        """Factory scenes keep their old behaviour. A saved name loads that snapshot."""
        if self._applying_preset:
            return
        # The dropdown sets the variable before this runs. Set it here as well,
        # so a save during the change cannot still be looking at the previous name.
        if self.scene_var.get() != label:
            self.scene_var.set(label)
        record = self._find_preset(label)
        if record is not None:
            self._apply_user_preset(record)
            return
        if label not in SCENE_BY_LABEL:
            return
        self._on_scene_change(label)

    def _preset_name_error(self, name: str) -> str | None:
        """None when the name can be saved. 'replace' when that name already exists."""
        name = name.strip()
        if not name or any(ord(ch) < 32 for ch in name):
            return "Type a name for the preset."
        if len(name) > PRESET_NAME_LIMIT:
            return f"Use at most {PRESET_NAME_LIMIT} characters."
        if name in SCENE_BY_LABEL:
            return "That name is a built-in scene. Choose another."
        if self._find_preset(name) is not None:
            return "replace"
        if len(self._user_presets) >= PRESET_COUNT_LIMIT:
            return f"The list holds {PRESET_COUNT_LIMIT} presets. Delete one first."
        return None

    def _capture_preset(self, name: str) -> dict[str, object]:
        scene = self._applied_scene if self._applied_scene in SCENES else "manual"
        lift = self.lift_var.get().strip().lower()
        if lift not in ("off", "gentle", "clear"):
            lift = "off"
        highpass = self.highpass_var.get()
        if highpass not in HIGHPASS_CHOICES:
            highpass = "Off" if SCENES[scene].active else "80 Hz"
        return {
            "name": name,
            "scene": scene,
            "dialogue_lift": lift,
            "atten_index": self.atten_index,
            "highpass": highpass,
            "post_filter": bool(self.postfilter_var.get()),
            "normalise": bool(self.normalise_var.get()),
        }

    def _commit_preset(self, name: str, *, replace: bool = False) -> str | None:
        """Store the live controls under name. None means saved. A string is the refusal."""
        name = name.strip()
        error = self._preset_name_error(name)
        if error == "replace":
            if not replace:
                return "replace"
        elif error:
            return error
        record = self._capture_preset(name)
        self._user_presets = [item for item in self._user_presets if item["name"] != name]
        self._user_presets.append(record)
        self._applying_preset = True
        try:
            self._rebuild_scene_menu()
            self.scene_var.set(name)
        finally:
            self._applying_preset = False
        self._save_settings()
        self._refresh_controls()
        self.log(f"Saved preset {name}.")
        return None

    def save_preset(self) -> None:
        if self.busy is not None:
            return
        typed = simpledialog.askstring("Save preset", "Name for these controls:", parent=self)
        if typed is None:
            return
        name = typed.strip()
        error = self._preset_name_error(name)
        if error == "replace":
            if not messagebox.askyesno(
                    "Save preset", f"Replace {name} with the controls on screen?", parent=self):
                return
            error = self._commit_preset(name, replace=True)
        elif error is None:
            error = self._commit_preset(name)
        if error:
            messagebox.showerror("Save preset", error, parent=self)

    def _apply_user_preset(self, record: dict[str, object]) -> None:
        """Apply the factory scene, then the stored controls, then show the name.

        The scene's own defaults run first, which is what a click on that scene
        does. The stored step, dialogue, post-filter and normalise are written
        back afterwards so a saved tweak is not immediately replaced. Rumble is
        restored only for Manual. A named scene still forces it off.
        """
        self._applying_preset = True
        try:
            normalise_before = bool(self.normalise_var.get())
            scene_key = str(record["scene"])
            if scene_key not in SCENES:
                scene_key = "manual"
            self.scene_var.set(SCENES[scene_key].label)
            self._on_scene_change(SCENES[scene_key].label)
            index = int(record["atten_index"])
            index = max(0, min(len(ATTEN_STEPS) - 1, index))
            self.atten_index = index
            self.atten_slider.set(index)
            self.lift_var.set(dialogue_menu_label(str(record["dialogue_lift"])))
            self.postfilter_var.set(bool(record["post_filter"]))
            self.normalise_var.set(bool(record["normalise"]))
            if scene_key == "manual":
                highpass = str(record["highpass"])
                self.highpass_var.set(highpass if highpass in HIGHPASS_CHOICES else "80 Hz")
            else:
                self.highpass_var.set("Off")
            self.scene_var.set(str(record["name"]))
            self._on_atten_change(index)
            if self.denoised48 is not None and bool(self.normalise_var.get()) != normalise_before:
                self._export_dirty = True
            self._save_settings()
            self._refresh_controls()
        finally:
            self._applying_preset = False
        limit = ATTEN_STEPS[self.atten_index]
        limit_text = "maximum" if limit is None else f"up to {limit} dB"
        self.log(
            f"Loaded preset {record['name']}: {SCENES[self._applied_scene].label}, "
            f"dialogue {self.lift_var.get().lower()}, {limit_text}.")

    def delete_preset(self) -> None:
        if self.busy is not None:
            return
        name = self.scene_var.get()
        if self._find_preset(name) is None:
            self.log("Factory scenes stay in the list.")
            return
        if not messagebox.askyesno(
                "Delete preset",
                f"Delete {name} from the scene list? The controls on screen stay as they are.",
                parent=self):
            return
        self._remove_preset(name)

    def _remove_preset(self, name: str) -> None:
        """Drop a saved name. The live controls stay, and the menu shows the factory scene."""
        self._user_presets = [item for item in self._user_presets if item["name"] != name]
        scene_key = self._applied_scene if self._applied_scene in SCENES else "manual"
        self._applying_preset = True
        try:
            # Set the factory label before the menu values change, so the command
            # cannot treat a missing name as Manual and replace the live controls.
            self.scene_var.set(SCENES[scene_key].label)
            self._rebuild_scene_menu()
        finally:
            self._applying_preset = False
        self._save_settings()
        self._refresh_controls()
        self.log(f"Deleted preset {name}. The controls are unchanged.")

    def _save_settings(self) -> None:
        scene_key = self._scene_key() if hasattr(self, "scene_var") else "manual"
        data = {
            "output_dir": str(self.output_dir),
            "atten_index": self.atten_index,
            "highpass": self.highpass_var.get(),
            "post_filter": bool(self.postfilter_var.get()),
            "normalise": bool(self.normalise_var.get()),
            "loudness_match": bool(self.loudness_var.get()),
            "scene": scene_key,
        }
        if hasattr(self, "lift_var"):
            data["dialogue_lift"] = self.lift_var.get().lower()
        if hasattr(self, "subtitles_var"):
            data["subtitles"] = bool(self.subtitles_var.get())
        if hasattr(self, "restore_var"):
            data["consonant_restore"] = bool(self.restore_var.get())
        if not SCENES[scene_key].active:
            data["manual_highpass"] = self.highpass_var.get()
            data["manual_atten_index"] = self.atten_index
            data["manual_post_filter"] = bool(self.postfilter_var.get())
        else:
            for key in ("manual_highpass", "manual_atten_index", "manual_post_filter"):
                if key in self.settings:
                    data[key] = self.settings[key]
        presets = getattr(self, "_user_presets", None)
        if presets is None:
            presets = clean_user_presets(self.settings.get("user_presets"))
        data["user_presets"] = [dict(item) for item in presets]
        data["selected_preset"] = (
            self._current_preset_name() if hasattr(self, "scene_var") else "")
        self.settings.update(data)
        save_settings(self.settings)

    def _on_normalise_toggle(self) -> None:
        """The switch is read when the file is written, so a change makes that file stale."""
        self._save_settings()
        if self.denoised48 is not None:
            self._export_dirty = True

    def _on_loudness_toggle(self) -> None:
        self._save_settings()
        if self.player.has("denoised"):
            self._apply_playback_gains(announce=True)

    def _apply_playback_gains(self, announce: bool = False) -> None:
        """Set the A and B playback gains from the loudness switch. Files are untouched."""
        matched = None
        if self.loudness_var.get() and self.player.has("denoised"):
            matched = loudness_match_gains(
                self.original_lufs, self.denoised_lufs, self.denoised_peak)
        if matched is None:
            self.player.set_gains(1.0, 1.0)
        else:
            self.player.set_gains(matched[0], matched[1])
        if not announce:
            return

        def lufs(value: float) -> str:
            return f"{value:.1f} LUFS" if math.isfinite(value) else "too quiet to measure"

        levels = f"original {lufs(self.original_lufs)}, denoised {lufs(self.denoised_lufs)}"
        if not self.loudness_var.get():
            self.log(f"A/B plays each version at its own level ({levels}).")
        elif matched is None:
            self.log("A/B loudness can't be matched: one version is too short or too "
                     "quiet to measure, so both play at their own level.")
        else:
            original_gain, _denoised_gain, match_db = matched
            text = f"A/B loudness matched ({levels}): B plays {match_db:+.1f} dB."
            if original_gain < 1.0:
                text += (f" Both play {20 * math.log10(original_gain):.1f} dB lower so B "
                         "stays below full scale.")
            self.log(text)

    def _on_atten_change(self, value) -> None:
        self.atten_index = int(round(float(value)))
        limit = ATTEN_STEPS[self.atten_index]
        self.atten_value.configure(text="Maximum" if limit is None else f"Up to {limit} dB")
        if hasattr(self, "highpass_var"):
            self._save_settings()

    def _show_output_dir(self) -> None:
        self.out_label.configure(text=shorten(str(self.output_dir), 56))

    def _refresh_controls(self) -> None:
        idle = self.busy is None
        has_file = self.audio48 is not None
        on = lambda flag: "normal" if flag else "disabled"
        self.open_btn.configure(state=on(idle))
        self.denoise_btn.configure(state=on(idle and has_file and self.model is not None))
        self.save_wav_btn.configure(state=on(idle and self.denoised48 is not None))
        cancellable = self.busy in ("denoise", "captions")
        self.cancel_btn.configure(state=on(cancellable),
                                  fg_color=C_DANGER if cancellable else C_LINE)
        scene_active = SCENES.get(self._applied_scene, MANUAL).active
        self.scene_menu.configure(state=on(idle))
        self.save_preset_btn.configure(state=on(idle))
        self.delete_preset_btn.configure(state=on(idle and self._current_preset_name() != ""))
        self.lift_menu.configure(state=on(idle))
        self.highpass_menu.configure(state=on(idle and not scene_active))
        for widget in (self.atten_slider, self.postfilter_switch,
                       self.normalise_switch, self.subtitles_switch,
                       self.restore_switch, self.change_btn):
            widget.configure(state=on(idle))
        self.save_text_btn.configure(state=on(bool(self._cues) and self.transcript_path() is not None))
        has_sel = self._selection is not None
        self.loop_btn.configure(state=on(has_sel))
        self.cut_btn.configure(state=on(idle and has_sel))
        self.undo_btn.configure(state=on(idle and self._undo is not None))
        self.save_sel_btn.configure(state=on(idle and has_sel))
        self.read_btn.configure(state=on(idle and has_file and self.model is not None))
        self._style_loop_button()
        can_play = has_file and sd is not None
        self.play_btn.configure(state=on(can_play))
        self.stop_btn.configure(state=on(can_play))
        self.orig_btn.configure(state=on(has_file))
        self.den_btn.configure(state=on(self.player.has("denoised")))
        self._style_source_buttons()

    def _style_source_buttons(self) -> None:
        for button, key in ((self.orig_btn, "original"), (self.den_btn, "denoised")):
            selected = self.player.source == key and self.player.has(key)
            button.configure(fg_color=C_AMBER if selected else C_LINE,
                             hover_color=C_AMBER_HOVER if selected else C_LINE_HOVER,
                             text_color=C_ON_AMBER if selected else C_TEXT)

    def _style_loop_button(self) -> None:
        active = self._loop_on and self._selection is not None
        self.loop_btn.configure(fg_color=C_AMBER if active else C_LINE,
                                hover_color=C_AMBER_HOVER if active else C_LINE_HOVER,
                                text_color=C_ON_AMBER if active else C_TEXT)

    def transcript_path(self) -> Path | None:
        """Where Save text writes. Beside the denoised WAV, or else the output folder.

        The ``.wav`` suffix only chooses the stem. ``write_transcript`` swaps it
        for ``.srt`` and ``.txt`` and does not create that wav.
        """
        if self.output_path:
            return Path(self.output_path)
        if self.source_path is None:
            return None
        return self.output_dir / f"{Path(self.source_path).stem}_transcript.wav"

    def _show_range(self) -> None:
        if self._selection is None or self.audio48 is None:
            self.range_label.configure(text="")
            self.waveform.set_selection(None, None)
            return
        start, end = self._selection
        n = max(1, self.audio48.shape[1])
        self.range_label.configure(
            text=f"{format_time(start / MODEL_SR)} – {format_time(end / MODEL_SR)}")
        self.waveform.set_selection(start / n, end / n)

    def _on_waveform_select(self, origin: float, current: float) -> None:
        if self.audio48 is None:
            return
        bounds = selection_bounds(origin, current, self.audio48.shape[1], MODEL_SR)
        if bounds is None:
            return
        had = self._selection is not None
        self._selection = bounds
        if self._loop_on:
            self.player.set_loop(bounds)
        self._show_range()
        if not had:
            self._refresh_controls()

    def clear_selection(self) -> None:
        if self._selection is None:
            return
        self._selection = None
        self._loop_on = False
        self.player.set_loop(None)
        self._show_range()
        self._refresh_controls()

    def toggle_loop(self) -> None:
        if self._selection is None:
            return
        self._loop_on = not self._loop_on
        self.player.set_loop(self._selection if self._loop_on else None)
        if self._loop_on:
            start, end = self._selection
            self.log(f"Looping {format_time(start / MODEL_SR)}–"
                     f"{format_time(end / MODEL_SR)} until Stop.")
        else:
            self.log("Loop off. The selection is still marked.")
        self._refresh_controls()

    def _heard(self) -> tuple[np.ndarray | None, str]:
        """The buffer Play is using, and a short name for the log."""
        if self.player.source == "denoised" and self.denoised48 is not None:
            return self.denoised48, "the denoised audio"
        return self.audio48, "the original"

    def _playhead_after_cut(self, playhead: int, start: int, end: int, n_new: int) -> int:
        """Keep the playhead on the same kept sample. Inside the cut, it sits on the join."""
        if playhead >= end:
            playhead = playhead - (end - start)
        else:
            playhead = min(playhead, start)
        if n_new <= 0:
            return 0
        return max(0, min(playhead, n_new - 1))

    def _remeasure_loudness(self) -> None:
        if self.audio48 is None:
            return
        self.original_lufs = integrated_loudness(self.audio48, MODEL_SR)
        if self.denoised48 is not None and self.denoised48.size:
            self.denoised_lufs = integrated_loudness(self.denoised48, MODEL_SR)
            self.denoised_peak = float(np.max(np.abs(self.denoised48)))
        else:
            self.denoised_lufs = float("-inf")
            self.denoised_peak = 0.0

    def _reload_session(self, playhead: int, source: str) -> None:
        """Point the player and the waveform at the current session arrays."""
        if self.audio48 is None:
            return
        self.player.load_original(self.audio48)
        if self.denoised48 is not None:
            self.player.set_denoised(self.denoised48)
            if source == "denoised":
                self.player.set_source("denoised")
        n = self.audio48.shape[1]
        playhead = max(0, min(playhead, max(n - 1, 0)))
        if n:
            self.player.seek(playhead / n)
        if self._loop_on and self._selection is not None:
            self.player.set_loop(self._selection)
        else:
            self._loop_on = False
            self.player.set_loop(None)
        if n:
            playhead = self.player.position
        self.orig_env = waveform_envelope(self.audio48)
        den_env = waveform_envelope(self.denoised48) if self.denoised48 is not None else None
        self._show_range()
        self.waveform.set_data(self.orig_env, den_env)
        self.waveform.set_playhead(0.0 if n == 0 else playhead / n)
        self._apply_playback_gains(announce=False)

    def cut_selection(self) -> None:
        """Remove the marked range from this session. The file on disk is not opened."""
        if self.busy or self.audio48 is None or self._selection is None:
            return
        start, end = self._selection
        n = self.audio48.shape[1]
        if self.denoised48 is not None and self.denoised48.shape[1] != n:
            self.log("The denoised audio does not match this session, so the cut was left as it is.")
            return
        try:
            new_audio = cut_session(self.audio48, start, end, MODEL_SR)
            new_den = None
            if self.denoised48 is not None:
                new_den = cut_session(self.denoised48, start, end, MODEL_SR)
        except ValueError as exc:
            self.log(str(exc))
            return
        playhead = self.player.position
        source = self.player.source
        self._undo = {
            "audio48": self.audio48,
            "denoised48": self.denoised48,
            "cues": list(self._cues),
            "playhead": playhead,
            "selection": self._selection,
            "source_frames": self.source_frames,
            "original_lufs": self.original_lufs,
            "denoised_lufs": self.denoised_lufs,
            "denoised_peak": self.denoised_peak,
            "output_path": self.output_path,
            "source": source,
            "loop_on": self._loop_on,
            "export_dirty": self._export_dirty,
        }
        self.audio48 = new_audio
        self.denoised48 = new_den
        self._cues = shift_cues(self._cues, start / MODEL_SR, end / MODEL_SR)
        self._cue_text = ""
        self.subtitle_label.configure(text="")
        self.output_path = None
        if new_den is not None:
            self._export_dirty = True
        if self.source_sr:
            self.source_frames = resampled_length(new_audio.shape[1], MODEL_SR, self.source_sr)
        self._selection = None
        self._loop_on = False
        new_pos = self._playhead_after_cut(playhead, start, end, new_audio.shape[1])
        self._remeasure_loudness()
        self._reload_session(new_pos, source)
        span_s = (end - start) / MODEL_SR
        self.log(f"Removed {format_time(start / MODEL_SR)}–{format_time(end / MODEL_SR)} "
                 f"from this session ({span_s:.0f} s). The file on disk is unchanged.")
        self._refresh_controls()

    def undo_cut(self) -> None:
        if self.busy or self._undo is None:
            return
        snap = self._undo
        self._undo = None
        self.audio48 = snap["audio48"]
        self.denoised48 = snap["denoised48"]
        self._cues = list(snap["cues"])
        self._cue_text = ""
        self.subtitle_label.configure(text="")
        self._selection = snap["selection"]
        self.source_frames = snap["source_frames"]
        self.original_lufs = snap["original_lufs"]
        self.denoised_lufs = snap["denoised_lufs"]
        self.denoised_peak = snap["denoised_peak"]
        self.output_path = snap["output_path"]
        self._export_dirty = bool(snap.get("export_dirty", False))
        self._loop_on = bool(snap.get("loop_on")) and self._selection is not None
        self._reload_session(int(snap["playhead"]), str(snap.get("source", "original")))
        self.log("Restored the removed section.")
        self._refresh_controls()

    def save_range(self) -> None:
        """Write the marked range of the buffer being heard. No denoise and no normalisation."""
        if self.busy or self._selection is None or self.source_path is None or not self.source_sr:
            return
        audio, which = self._heard()
        if audio is None:
            return
        start, end = self._selection
        if end > audio.shape[1]:
            self.log("The selection is outside this session, so nothing was written.")
            return
        try:
            dest = selection_output_path(self.output_dir, Path(self.source_path).stem)
        except OSError as exc:
            self.log(f"Couldn't save the selection: {exc}")
            return
        subtype = self.source_subtype or "PCM_24"
        seconds = (end - start) / MODEL_SR
        self.progress.set(0)
        self.progress_label.configure(text="")
        self._run_job("selection", self._job_save_selection, audio, start, end,
                      self.source_sr, subtype, dest, which, seconds)

    def _job_save_selection(self, audio, start, end, source_sr, subtype, dest, which, seconds) -> None:
        used = subtype if subtype in EXPORT_SUBTYPES else "PCM_24"
        path = save_selection(audio, start, end, source_sr, used, dest)
        self.events.put(("selection_saved", {
            "path": str(path), "which": which, "seconds": seconds, "subtype": used}))

    def read_speech(self) -> None:
        """Transcribe the buffer being heard. Play does not start this."""
        if self.busy or self.audio48 is None or self.model is None:
            return
        audio, which = self._heard()
        if audio is None:
            return
        self.log(f"English transcript of {which}. A read-speech model: "
                 "names and street noise will be misspelt.")
        self._caption_warned = True
        self.cancel_event.clear()
        self.progress.set(0)
        self.progress_label.configure(text="0%")
        target = self.transcript_path()
        self._run_job("captions", self._job_captions, audio, str(target or ""))

    def _update_transport(self) -> None:
        playing = self.player.is_playing
        label = "Pause" if playing else "Play"
        if self.play_btn.cget("text") != label:
            self.play_btn.configure(text=label)
        n = self.player.length
        if n:
            pos = self.player.position
            text = f"{format_time(pos / MODEL_SR)} / {format_time(n / MODEL_SR)}"
            if text != self._last_time_text:
                self.time_label.configure(text=text)
                self._last_time_text = text
            self.waveform.set_playhead(pos / n)
            self._show_subtitle(pos / MODEL_SR)

    def _saved_lift_label(self) -> str:
        """Dialogue menu value: the saved choice, or the scene's own default."""
        fallback = lift_label(SCENES.get(self._applied_scene, MANUAL).lift_db)
        saved = str(self.settings.get("dialogue_lift", "")).lower()
        return {"off": "Off", "gentle": "Gentle", "clear": "Clear"}.get(saved, fallback)

    def _on_lift_change(self, _value: str) -> None:
        self._save_settings()

    def _show_subtitle(self, seconds: float) -> None:
        text = ""
        for cue in self._cues:
            if cue.start_s <= seconds < cue.end_s:
                text = cue.text
                break
        if text == self._cue_text:
            return
        self._cue_text = text
        self.subtitle_label.configure(text=text)

    def save_transcript(self) -> None:
        target = self.transcript_path()
        if not self._cues or target is None:
            return
        try:
            srt_path = write_transcript(target, self._cues)
        except OSError as exc:
            self.log(f"Couldn't save the transcript: {exc}")
            return
        self.log(f"Saved {srt_path} and {srt_path.with_suffix('.txt')}.")

    # ------------------------------------------------------------ actions --
    def open_file(self, path: str | None = None) -> None:
        if self.busy:
            return
        if self._export_dirty and self.denoised48 is not None:
            self._pending_open = path
            self._confirm_discard("open")
            return
        self._begin_open(path)

    def _begin_open(self, path: str | None) -> None:
        if not path:
            path = filedialog.askopenfilename(
                title="Open audio",
                initialdir=self.settings.get("last_dir") or str(Path.home()),
                filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg"), ("All files", "*.*")])
        if not path:
            self._discard_export = False
            return
        if not os.path.isfile(path):
            self.log(f"File not found: {path}")
            self._discard_export = False
            return
        if self._discard_export:
            self._export_dirty = False
            self._discard_export = False
        self.settings["last_dir"] = str(Path(path).parent)
        self._save_settings()
        self.player.stop()
        self.log(f"Opening {Path(path).name}…")
        self._run_job("load", self._job_load_file, path)

    def start_denoise(self) -> None:
        if self.busy or self.audio48 is None or self.model is None:
            return
        if self._export_dirty and self.denoised48 is not None:
            self._confirm_discard("denoise")
            return
        self._begin_denoise()

    def _begin_denoise(self) -> None:
        if self._discard_export:
            self._export_dirty = False
            self._discard_export = False
        if self.busy or self.audio48 is None or self.model is None:
            return
        self.player.pause()
        opts = {
            "src_path": self.source_path,
            "src_sr": self.source_sr,
            "src_frames": self.source_frames,
            "out_dir": str(self.output_dir),
            "atten": ATTEN_STEPS[self.atten_index],
            "highpass": HIGHPASS_CHOICES[self.highpass_var.get()],
            "post_filter": bool(self.postfilter_var.get()),
            "normalise": bool(self.normalise_var.get()),
            "scene": self._applied_scene,
            "subtype": self.source_subtype or "PCM_24",
            "lift_db": LIFT_CHOICES.get(self.lift_var.get(), 0.0),
            "consonant_restore": bool(self.restore_var.get()),
            "subtitles": bool(self.subtitles_var.get()),
            "write_file": False,
        }
        self.cancel_event.clear()
        self.progress.set(0)
        self.progress_label.configure(text="0%")
        self._run_job("denoise", self._job_denoise, self.audio48, self.model, self.df_state, opts)

    def save_denoised(self, then: str | None = None) -> None:
        """Ask for a path and write the denoised session. then is close, open or denoise."""
        if self.busy or self.denoised48 is None or self.source_path is None or not self.source_sr:
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.log(f"Couldn't use the Save to folder: {exc}")
            return
        initial = f"{Path(self.source_path).stem}_denoised.wav"
        chosen = filedialog.asksaveasfilename(
            title="Save denoised audio",
            initialdir=str(self.output_dir),
            initialfile=initial,
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")])
        if not chosen:
            self._after_save = None
            return
        if Path(chosen).resolve() == Path(self.source_path).resolve():
            self.log("That is the file you opened. Choose a different name.")
            self._after_save = None
            return
        self._after_save = then
        opts = {
            "dest": chosen,
            "src_path": self.source_path,
            "src_sr": self.source_sr,
            "src_frames": self.source_frames,
            "normalise": bool(self.normalise_var.get()),
            "subtype": self.source_subtype or "PCM_24",
        }
        self.progress.set(0)
        self.progress_label.configure(text="")
        self._run_job("save", self._job_save, self.denoised48, opts)

    def _job_save(self, den48: np.ndarray, opts: dict) -> None:
        saved = export_denoised(den48, opts)
        self.events.put(("save_done", saved))

    def _confirm_discard(self, then: str) -> None:
        """Save, drop, or keep an unsaved denoise before close, open, or another denoise."""
        if not self._export_dirty or self.denoised48 is None:
            self._follow(then)
            return
        answer = messagebox.askyesnocancel(
            "Audio DeNoiser",
            "The denoised audio has not been saved. Save it before continuing?",
            parent=self)
        if answer is None:
            self._pending_open = None
            return
        if answer:
            self.save_denoised(then=then)
            return
        # Dropping the pass waits until the next action really starts. Cancelling
        # the open dialog after No must still ask on the next close.
        self._discard_export = True
        self._follow(then)

    def _follow(self, then: str) -> None:
        if then == "close":
            self._finish_close()
        elif then == "open":
            path = self._pending_open
            self._pending_open = None
            self._begin_open(path)
        elif then == "denoise":
            self._begin_denoise()

    def _finish_close(self) -> None:
        self.cancel_event.set()
        self.player.stop()
        self._save_settings()
        self.destroy()

    def cancel_denoise(self) -> None:
        if self.busy in ("denoise", "captions"):
            self.cancel_event.set()
            self.cancel_btn.configure(state="disabled", fg_color=C_LINE)
            if self.busy == "captions":
                self.log("Cancelling subtitles after the current block…")
            else:
                self.log("Cancelling after the current chunk…")

    def toggle_play(self) -> None:
        if self.audio48 is None or sd is None:
            return
        if self.player.is_playing:
            self.player.pause()
            return
        try:
            self.player.play()
        except Exception as exc:
            self.log(f"Playback failed: {exc}")

    def stop_audio(self) -> None:
        self.player.stop()
        self.waveform.set_playhead(0.0)

    def set_source(self, key: str) -> None:
        if self.player.has(key):
            self.player.set_source(key)
            self._style_source_buttons()

    def _seek(self, fraction: float) -> None:
        self.player.seek(fraction)
        self.waveform.set_playhead(fraction)

    def choose_output_dir(self) -> None:
        chosen = filedialog.askdirectory(title="Choose output folder",
                                         initialdir=str(self.output_dir))
        if chosen:
            self.output_dir = resolve_output_dir(chosen)
            self._show_output_dir()
            self._save_settings()
            self.log(f"Output folder: {self.output_dir}")

    def open_output_dir(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(os, "startfile"):
                os.startfile(str(self.output_dir))
            else:
                subprocess.Popen(["xdg-open", str(self.output_dir)])
        except Exception as exc:
            self.log(f"Couldn't open the folder: {exc}")

    def _on_close(self) -> None:
        if self.busy == "save":
            self._after_save = "close"
            self.log("The file is still being saved. The window will close when that finishes.")
            return
        if self.busy:
            self.cancel_event.set()
            self.player.stop()
            self._save_settings()
            self.destroy()
            return
        self._confirm_discard("close")

    # ------------------------------------------------------------- worker --
    def _run_job(self, stage: str, job, *args) -> None:
        """Run job(*args) on a daemon thread. Jobs talk to the UI only via self.events."""
        if stage != "model":
            self.busy = stage
            self._refresh_controls()

        def runner():
            try:
                job(*args)
            except Cancelled:
                self.events.put(("cancelled", None))
            except Exception as exc:
                write_error_log(traceback.format_exc())
                self.events.put(("error", (stage, friendly_error(stage, exc))))

        threading.Thread(target=runner, daemon=True).start()

    def _job_load_model(self) -> None:
        self.events.put(("log", "Loading DeepFilterNet 3. The first run downloads the model "
                                "(a few MB), so it needs an internet connection."))
        self.events.put(("model_ready", load_model()))

    def _job_load_file(self, path: str) -> None:
        audio, sr = load_audio(path)
        duration = audio.shape[1] / sr
        if duration < 0.25:
            raise ValueError("The file is too short to process (under a quarter of a second).")
        audio48 = resample(audio, sr, MODEL_SR)
        self.events.put(("file_loaded", {
            "path": path, "sr": sr, "channels": audio.shape[0], "duration": duration,
            "frames": audio.shape[1], "subtype": source_subtype(path),
            "lufs": integrated_loudness(audio48, MODEL_SR),
            "audio48": audio48, "env": waveform_envelope(audio48)}))

    def _job_denoise(self, audio48, model, df_state, opts) -> None:
        put = self.events.put
        preset = SCENES.get(opts.get("scene", "manual"), MANUAL)
        work = audio48
        if not preset.active and opts["highpass"]:
            work = highpass(work, MODEL_SR, opts["highpass"])
        previous_filter = bool(getattr(model, "post_filter", False))
        previous_beta = getattr(model, "post_filter_beta", None)
        den48 = None
        elapsed = 0.0
        stats: dict[str, int] = {}
        lift_db = float(opts.get("lift_db", 0.0) or 0.0)
        restore = bool(opts.get("consonant_restore", False))
        try:
            model.post_filter = opts["post_filter"]
            if preset.active and previous_beta is not None:
                model.post_filter_beta = preset.post_filter_beta
            limit = opts["atten"]
            detail = "Denoising: " + preset.label
            detail += ", noise reduction " + ("maximum" if limit is None else f"up to {limit} dB")
            if not preset.active and opts["highpass"]:
                detail += f", rumble filter {opts['highpass']} Hz"
            if opts["post_filter"]:
                detail += ", post-filter on"
            if lift_db >= LIFT_CLEAR_DB:
                detail += ", dialogue clear"
            elif lift_db > 0.0:
                detail += ", dialogue gentle"
            if restore:
                detail += ", consonant restore on"
            put(("log", detail))
            started = time.perf_counter()
            den48 = denoise(
                model, df_state, work, limit, scene=preset if preset.active else None,
                lift_db=lift_db, restore=restore,
                progress=lambda fraction: put(("progress", fraction)),
                cancel=self.cancel_event, stats=stats)
            elapsed = time.perf_counter() - started
        finally:
            model.post_filter = previous_filter
            if previous_beta is not None:
                model.post_filter_beta = previous_beta
        if den48 is None:
            return
        # Measured at 48 kHz, as the original was, for the loudness-matched A/B.
        lufs = integrated_loudness(den48, MODEL_SR)
        peak = float(np.max(np.abs(den48))) if den48.size else 0.0

        subtype = opts.get("subtype", "PCM_24")
        if subtype not in EXPORT_SUBTYPES:
            subtype = "PCM_24"
        if opts.get("write_file", True):
            # Parity scripts omit write_file, so they still get the file 1.4 wrote.
            saved = export_denoised(den48, {
                "dest": str(Path(opts["out_dir"]) / f"{Path(opts['src_path']).stem}_denoised.wav"),
                "src_path": opts["src_path"],
                "src_sr": opts["src_sr"],
                "src_frames": opts["src_frames"],
                "normalise": opts["normalise"],
                "subtype": subtype,
            })
        else:
            saved = {"path": None, "gain_db": None, "clipped": False, "subtype": subtype}

        put(("denoise_done", {"path": saved["path"], "den48": den48,
                              "env": waveform_envelope(den48), "elapsed": elapsed,
                              "gain_db": saved["gain_db"], "clipped": saved["clipped"],
                              "lufs": lufs, "peak": peak, "subtype": saved["subtype"],
                              "guard": preset.transient_guard, "stats": stats,
                              "subtitles": bool(opts.get("subtitles", False))}))

    def _job_captions(self, den48: np.ndarray, wav_path: str) -> None:
        """Recognise one buffer in 20 s blocks. ``wav_path`` only names the transcript."""
        import torch
        from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H

        put = self.events.put
        if den48 is None or den48.size == 0 or den48.shape[-1] < MODEL_SR // 4:
            put(("captions_done", {"cues": [], "wav_path": wav_path}))
            return
        try:
            if self._caption_model is None or self._caption_labels is None:
                model_dir = caption_model_dir(self.output_dir)
                torch.hub.set_dir(str(model_dir))
                put(("log", f"Loading the subtitle model. The first run downloads it to {model_dir}."))
                bundle = WAV2VEC2_ASR_BASE_960H
                model = bundle.get_model()
                model.eval()
                self._caption_model = model
                self._caption_labels = tuple(bundle.get_labels())
            model = self._caption_model
            labels = self._caption_labels
        except Exception as exc:
            put(("captions_failed", str(exc)))
            return

        mono = den48.mean(axis=0) if den48.shape[0] > 1 else den48[0]
        mono = np.ascontiguousarray(mono, dtype=np.float32)
        audio16 = resample(mono[None, :], MODEL_SR, ASR_SR)[0]
        chunk = int(ASR_CHUNK_S * ASR_SR)
        overlap = int(ASR_OVERLAP_S * ASR_SR)
        step = max(1, chunk - overlap)
        words: list[CaptionCue] = []
        covered = 0.0
        pos = 0
        total = max(1, int(audio16.shape[0]))
        while pos < audio16.shape[0]:
            if self.cancel_event.is_set():
                put(("captions_cancelled", pack_cues(words)))
                return
            piece = np.ascontiguousarray(audio16[pos:pos + chunk], dtype=np.float32)
            if piece.shape[0] < CTC_STRIDE * 2:
                break
            offset_s = pos / ASR_SR
            is_last = pos + chunk >= audio16.shape[0]
            limit = math.inf if is_last else (pos + step) / ASR_SR
            try:
                with torch.inference_mode():
                    emissions, _length = model(torch.from_numpy(piece).unsqueeze(0))
                fresh = _greedy_words(emissions[0].detach().cpu().numpy(), labels, offset_s)
            except Exception as exc:
                put(("captions_failed", str(exc)))
                return
            words.extend(word for word in fresh if covered <= word.start_s < limit)
            covered = limit
            pos += step
            put(("captions_progress", (min(pos, total) / total, pack_cues(words))))
        put(("captions_done", {"cues": pack_cues(words), "wav_path": wav_path}))

    # ------------------------------------------------------- event pump --
    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                try:
                    getattr(self, f"_on_{kind}")(payload)
                except Exception:
                    write_error_log(traceback.format_exc())
                    self.log(f"Internal error handling '{kind}' (details in {ERROR_LOG})")
        except queue.Empty:
            pass
        finally:
            # A save can close the window from this same turn. Do not touch it afterwards.
            if self.winfo_exists():
                self.player.tick()
                self._update_transport()
                self.after(self.POLL_MS, self._poll)

    def _on_log(self, message: str) -> None:
        self.log(message)

    def _on_model_ready(self, payload) -> None:
        self.model, self.df_state, device, torch_version = payload
        self.status_label.configure(text=f"Model ready on {device}", text_color=C_AMBER)
        self.log(f"DeepFilterNet 3 ready ({device}, torch {torch_version}).")
        self._refresh_controls()

    def _on_file_loaded(self, info: dict) -> None:
        self.busy = None
        self.source_path, self.source_sr = info["path"], info["sr"]
        self.source_frames = info["frames"]
        self.audio48, self.orig_env = info["audio48"], info["env"]
        self.denoised48 = None
        self._selection = None
        self._loop_on = False
        self._undo = None
        self.output_path = None
        self._export_dirty = False
        self._after_save = None
        self._pending_open = None
        self._discard_export = False
        self.source_subtype = info["subtype"]
        self.original_lufs = info["lufs"]
        self.denoised_lufs = float("-inf")
        self.denoised_peak = 0.0
        self.player.load_original(self.audio48)
        self.waveform.set_selection(None, None)
        self.waveform.set_data(self.orig_env, None)
        self.waveform.set_playhead(0.0)
        self.range_label.configure(text="")
        self._cues = []
        self._cue_text = ""
        self.subtitle_label.configure(text="")
        channels = {1: "mono", 2: "stereo"}.get(info["channels"], f"{info['channels']} channels")
        details = f"{info['sr'] / 1000:g} kHz {channels}"
        if self.source_subtype is not None:
            details += f", {DEPTH_LABELS[self.source_subtype]}"
        details += f", {format_time(info['duration'])}"
        self.file_label.configure(text=shorten(Path(info["path"]).name, 70))
        self.file_info.configure(text=details)
        self.progress.set(0)
        self.progress_label.configure(text="")
        self.log(f"Opened {Path(info['path']).name} ({details}).")
        if math.isfinite(self.original_lufs):
            self.log(f"Integrated loudness of the original: {self.original_lufs:.1f} LUFS.")
        if info["duration"] > LONG_FILE_MINUTES * 60:
            self.log(f"This recording is over {LONG_FILE_MINUTES} minutes and will use a lot "
                     "of memory. Close other programs if the laptop starts to struggle.")
        self._refresh_controls()

    def _on_progress(self, fraction: float) -> None:
        self.progress.set(fraction)
        self.progress_label.configure(text=f"{int(fraction * 100)}%")

    def _on_denoise_done(self, info: dict) -> None:
        self.busy = None
        self.output_path = info["path"]
        self._export_dirty = not info["path"]
        self.denoised48 = info["den48"]
        self.denoised_lufs = info["lufs"]
        self.denoised_peak = info["peak"]
        self.player.set_denoised(info["den48"])
        self.player.set_source("denoised")
        self.waveform.set_data(self.orig_env, info["env"])
        duration = self.audio48.shape[1] / MODEL_SR
        speed = duration / max(info["elapsed"], 1e-6)
        self.log(f"Denoised {format_time(duration)} of audio in {info['elapsed']:.1f} s "
                 f"({speed:.0f}× real time).")
        if info["guard"]:
            clicks, pops = info["stats"].get("clicks", 0), info["stats"].get("pops", 0)
            if clicks or pops:
                self.log(f"Transient guard repaired {clicks} click{'s' if clicks != 1 else ''} "
                         f"and {pops} pop{'s' if pops != 1 else ''}.")
            else:
                self.log("Transient guard found no isolated clicks or pops to repair.")
        if info["path"]:
            if info["gain_db"] is not None:
                self.log(f"Normalised to {NORMALISE_PEAK_DBFS:g} dBFS ({info['gain_db']:+.1f} dB).")
            if info["clipped"]:
                self.log("Some peaks went over full scale and were clipped. "
                         "Turn on normalising to avoid this.")
            self.log(f"Saved {self.output_path} ({DEPTH_LABELS[info['subtype']]} WAV).")
        else:
            self.log("Denoise finished. Press Save denoised to write the WAV.")
        self._apply_playback_gains(announce=True)
        self.log("Press B to hear the denoised version, A for the original.")
        self._cues = []
        self._show_subtitle(0.0)
        if info.get("subtitles"):
            self.log("English transcript of the denoised audio. A read-speech model: "
                     "names and street noise will be misspelt.")
            self._caption_warned = True
            self._run_job("captions", self._job_captions, info["den48"], info["path"] or "")
        else:
            self._refresh_controls()

    def _on_save_done(self, info: dict) -> None:
        self.busy = None
        self.output_path = info["path"]
        self._export_dirty = False
        self.progress.set(1.0)
        self.progress_label.configure(text="")
        depth = DEPTH_LABELS.get(info.get("subtype"), "WAV")
        self.log(f"Saved {info['path']} ({depth} WAV).")
        if info.get("gain_db") is not None:
            self.log(f"Normalised to {NORMALISE_PEAK_DBFS:g} dBFS ({info['gain_db']:+.1f} dB).")
        if info.get("clipped"):
            self.log("Some peaks went over full scale and were clipped. "
                     "Turn on normalising to avoid this.")
        follow = self._after_save
        self._after_save = None
        self._refresh_controls()
        if follow:
            self._follow(follow)

    def _on_cancelled(self, _payload) -> None:
        self.busy = None
        self.progress.set(0)
        self.progress_label.configure(text="")
        self.log("Cancelled. Nothing was saved.")
        self._refresh_controls()

    def _on_error(self, payload) -> None:
        stage, message = payload
        if stage == "model":
            self.status_label.configure(text="Model failed to load", text_color=C_DANGER)
        else:
            self.busy = None
            self.progress.set(0)
            self.progress_label.configure(text="")
        if stage == "save":
            # The pass is still unsaved, and a close or open waiting on this write stops.
            self._after_save = None
        self.log(f"Error: {message}")
        self._refresh_controls()

    def _on_captions_progress(self, payload) -> None:
        fraction, cues = payload
        self._cues = cues
        self.progress.set(fraction)
        self.progress_label.configure(text=f"{int(fraction * 100)}%")
        if self.player.length:
            self._show_subtitle(self.player.position / MODEL_SR)
        self._refresh_controls()

    def _on_captions_done(self, info: dict) -> None:
        self.busy = None
        self._cues = info.get("cues") or []
        self.progress.set(1.0 if self._cues else 0.0)
        self.progress_label.configure(text="100%" if self._cues else "")
        count = len(self._cues)
        if count == 0:
            self.log("No speech recognised for subtitles.")
        else:
            noun = "line" if count == 1 else "lines"
            self.log(f"Subtitles ready: {count} {noun}.")
            if not self._caption_warned:
                self.log("Transcript is a best effort from a read-speech model. "
                         "Names and street noise will be misspelt.")
                self._caption_warned = True
        if self.player.length:
            self._show_subtitle(self.player.position / MODEL_SR)
        self._refresh_controls()

    def _on_captions_cancelled(self, cues) -> None:
        self.busy = None
        self._cues = cues or []
        self.progress.set(0)
        self.progress_label.configure(text="")
        if self._cues:
            self.log("Transcript stopped. Save text will write what was recognised so far.")
        else:
            self.log("Transcript stopped.")
        self._refresh_controls()

    def _on_selection_saved(self, info: dict) -> None:
        self.busy = None
        self.progress.set(1.0)
        self.progress_label.configure(text="")
        depth = DEPTH_LABELS.get(info.get("subtype"), "WAV")
        self.log(f"Saved {info['path']} ({format_time(info['seconds'])} of {info['which']}, "
                 f"{depth} WAV).")
        self._refresh_controls()

    def _on_captions_failed(self, message: str) -> None:
        self.busy = None
        self.progress.set(0)
        self.progress_label.configure(text="")
        self.log(f"Subtitles skipped: {message}")
        self._refresh_controls()


def main() -> None:
    ctk.set_appearance_mode("dark")
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    app = AudioDeNoiserApp(initial_file=initial)
    app.mainloop()


if __name__ == "__main__":
    main()
