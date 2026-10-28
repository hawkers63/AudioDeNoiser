#!/usr/bin/env python3
"""
Audio DeNoiser 1.1
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
Keys     : Space play/pause, A original, B denoised, Ctrl+O open, Ctrl+D denoise

How the processing chain works
------------------------------
1. Decode WAV/MP3/FLAC/OGG with soundfile and resample to 48 kHz (the model's rate).
2. Manual runs the 1.0 chain: an optional high-pass below 60-120 Hz, then the
   model. A named scene instead conditions each chunk before the model, with an
   infrasonic filter, a gust duck and an idle-tone stage. Heavy Wind and Highway
   also soften isolated spectral peaks before the chunk crossfade.
3. DeepFilterNet 3 in 30-second chunks, each with a 2-second pre-roll so the
   model has settled before its output is used, joined with 0.25 s crossfades.
   Chunking gives a real progress bar, a working Cancel button and flat memory use.
4. Resample back to the source rate, optionally peak-normalise, and save a
   24-bit WAV named <name>_denoised.wav in the output folder.
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
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, istft, resample_poly, sosfilt, sosfilt_zi, sosfiltfilt, stft
import tkinter as tk
from tkinter import filedialog
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
APP_VERSION = "1.1"

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

_SOS_CACHE: dict[tuple, np.ndarray] = {}


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

    @property
    def active(self) -> bool:
        return self.key != "manual"

    @property
    def uses_floor(self) -> bool:
        return self.floor_lo_hz is not None and self.floor_hi_hz is not None


class ConditionerState:
    """Reusable scratch and the slow low-band floor carried across chunks."""

    def __init__(self) -> None:
        self.low_floor_db: float | None = None
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
    """Inverse STFT matching ``_tone_spectrum`` / the spectral floor.

    Scipy warns that the Hann window fails NOLA because its first sample is
    exactly zero. The overlap itself is the standard 50 percent. The difference
    method in the callers never replaces samples whose gain is unity, so that
    endpoint does not touch the speech we meant to leave alone.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="NOLA condition failed")
        _times, restored = istft(
            spectrum,
            fs=sr,
            window="hann",
            nperseg=frame,
            noverlap=frame - hop,
            boundary=False,
            input_onesided=True,
        )
    return np.asarray(restored, dtype=np.float64)


def _apply_spectral_gain(
    audio: np.ndarray, sr: int, delta_gain: np.ndarray, frame: int, hop: int
) -> np.ndarray:
    """Add the inverse STFT of ``spectrum * delta_gain`` to the dry signal.

    Bins whose delta is zero are not rewritten, so a stage that leaves a bin
    alone cannot smear it through the inverse transform.
    """
    length = audio.shape[-1]
    cleaned = np.empty(audio.shape, dtype=np.float64)
    for channel in range(audio.shape[0]):
        _frequencies, _times, spectrum = stft(
            np.ascontiguousarray(audio[channel], dtype=np.float64),
            fs=sr,
            window="hann",
            nperseg=frame,
            noverlap=frame - hop,
            boundary=None,
            padded=False,
        )
        times = min(spectrum.shape[1], delta_gain.shape[1])
        bins = min(spectrum.shape[0], delta_gain.shape[0])
        shaped = np.zeros_like(spectrum)
        shaped[:bins, :times] = spectrum[:bins, :times] * delta_gain[:bins, :times]
        restored = _istft(shaped, sr, frame, hop)
        if restored.shape[0] < length:
            restored = np.pad(restored, (0, length - restored.shape[0]))
        cleaned[channel] = audio[channel] + restored[:length]
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
    f0_hz: np.ndarray, periodicity: np.ndarray, n_bins: int, n_times: int, sr: int
) -> np.ndarray:
    """Mark bins that sit on a voiced harmonic, plus one bin either side."""
    protect = np.zeros((n_bins, n_times), dtype=bool)
    times = min(n_times, f0_hz.shape[0], periodicity.shape[0])
    if times == 0:
        return protect
    bin_hz = sr / FLOOR_FRAME
    valid = periodicity[:times] >= VOICING_MIN
    if not np.any(valid):
        return protect
    time_index = np.arange(times)
    usable = np.flatnonzero(valid)
    for harmonic in range(1, 17):
        centres = np.rint(harmonic * f0_hz[:times] / bin_hz).astype(np.int32)
        for offset in (-1, 0, 1):
            index = np.clip(centres[usable] + offset, 0, n_bins - 1)
            protect[index, time_index[usable]] = True
    return protect


def soften_musical_noise(audio: np.ndarray, sr: int, preset: ScenePreset) -> np.ndarray:
    """Pull isolated spectral peaks in the tyre band down towards the local median.

    Unity gains return the input unchanged, so speech that the detector leaves
    alone is not passed through an inverse STFT at all. Harmonics of a detected
    f0 are marked and left alone, which keeps vowels from being chiselled out.
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
    smoothed = median_filter(magnitude, size=(5, 7))
    excess_db = 20 * np.log10((magnitude + 1e-8) / (smoothed + 1e-8))
    pull_db = np.clip(excess_db - preset.floor_peak_db, 0.0, preset.floor_max_db)
    bin_hz = sr / FLOOR_FRAME
    band = (np.arange(magnitude.shape[0]) >= preset.floor_lo_hz / bin_hz) & (
        np.arange(magnitude.shape[0]) <= preset.floor_hi_hz / bin_hz
    )
    pull_db *= band[:, None]
    periodicity, f0 = estimate_voicing(mid, sr)
    periodicity = _debounce_voicing(periodicity)
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
    # and that ring can look like a vowel if it is measured afterwards.
    dry = np.ascontiguousarray(audio.mean(axis=0), dtype=np.float64)
    periodicity, _f0 = estimate_voicing(dry, sr)
    periodicity = _debounce_voicing(periodicity)

    work = np.array(audio, dtype=np.float64, copy=True)
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
    )
    base.update(overrides)
    return ScenePreset(**base)


WIND = _preset(
    key="wind",
    label="Heavy Wind & Gusts",
    summary=(
        "Strips infrasonic buffeting, then ducks gusts under 220 Hz only while the voice is unvoiced."
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
)
TRAFFIC = _preset(
    key="traffic",
    label="Highway / Traffic Drone",
    summary="Tames road rumble and steady engine tones, and softens the tyre-hiss floor.",
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
)
STREET = _preset(
    key="street",
    label="Pedestrian Street",
    summary="Light low-end control for street talk, without chewing consonants or nearby voices.",
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
    preset.key: preset for preset in (MANUAL, WIND, TRAFFIC, STREET, PODCAST)
}
SCENE_ORDER = ("manual", "wind", "traffic", "street", "podcast")
SCENE_BY_LABEL = {preset.label: preset for preset in SCENES.values()}


def peak_normalise(audio: np.ndarray, peak_dbfs: float) -> tuple[np.ndarray, float]:
    """Scale so the loudest sample sits at peak_dbfs. Returns (audio, gain_db)."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-6:
        return audio, 0.0
    gain = (10 ** (peak_dbfs / 20)) / peak
    return (audio * gain).astype(np.float32), 20 * math.log10(gain)


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
            progress=None, cancel: threading.Event | None = None) -> np.ndarray:
    """Run DeepFilterNet over 48 kHz audio (channels, frames) in overlapping chunks.

    A named scene conditions each extended chunk before the network sees it.
    Manual passes the audio through unchanged, so today's slider and rumble
    settings keep today's results. The torch tensor is one chunk, not the file.
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
            segment = apply_scene(segment, sr, scene, state)
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if hasattr(df_state, "reset"):
            df_state.reset()
        with torch.inference_mode():
            enhanced = enhance(
                model, df_state, torch.from_numpy(segment), atten_lim_db=atten_lim_db)
        piece = np.ascontiguousarray(enhanced.detach().cpu().numpy(), dtype=np.float32)
        if active and scene.uses_floor:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            piece = np.ascontiguousarray(soften_musical_noise(piece, sr, scene), dtype=np.float32)
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


# --------------------------------------------------------------------------- #
#  Playback: in-memory buffers, instant A/B switching at the same position
# --------------------------------------------------------------------------- #
class Player:
    """Streams numpy buffers through sounddevice.

    Both buffers are 48 kHz stereo and the same length, so switching between
    original and denoised keeps the exact playback position. Nothing is read
    from disk, so the output file is never locked while you listen.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffers: dict[str, np.ndarray | None] = {"original": None, "denoised": None}
        self._source = "original"
        self._pos = 0
        self._playing = False
        self._stream = None

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
            self._source = "original"
            self._pos = 0

    def set_denoised(self, audio: np.ndarray) -> None:
        stereo = self._to_stereo(audio)
        with self._lock:
            self._buffers["denoised"] = stereo

    def set_source(self, key: str) -> None:
        with self._lock:
            if self._buffers.get(key) is not None:
                self._source = key

    def seek(self, fraction: float) -> None:
        with self._lock:
            self._pos = int(max(0.0, min(1.0, fraction)) * self.length)

    # Transport --------------------------------------------------------------
    def play(self) -> None:
        if sd is None:
            raise RuntimeError(f"audio output unavailable ({SD_IMPORT_ERROR})")
        with self._lock:
            buf = self._buffers.get(self._source)
            if buf is None:
                return
            if self._pos >= len(buf):
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
            end = min(self._pos + frames, len(buf))
            count = end - self._pos
            outdata[:count] = buf[self._pos:end]
            if count < frames:
                outdata[count:] = 0
            self._pos = end
            if end >= len(buf):
                self._playing = False


# --------------------------------------------------------------------------- #
#  Waveform view
# --------------------------------------------------------------------------- #
class WaveformView(tk.Canvas):
    """Original in concrete grey, denoised in amber on top. Click or drag to seek."""

    def __init__(self, master, on_seek, **kwargs):
        super().__init__(master, bg=C_PANEL, highlightthickness=0, bd=0, **kwargs)
        self._on_seek = on_seek
        self._orig = None
        self._den = None
        self._playhead = None
        self._fraction = 0.0
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Button-1>", self._seek_event)
        self.bind("<B1-Motion>", self._seek_event)

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

    def _seek_event(self, event) -> None:
        if self._orig is not None and self.winfo_width() > 0:
            self._on_seek(event.x / self.winfo_width())

    def redraw(self) -> None:
        self.delete("all")
        self._playhead = None
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
        x = self._fraction * w
        self._playhead = self.create_line(x, 0, x, h, fill=C_TEXT, width=2)

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
        self.geometry("860x820")
        self.minsize(760, 760)
        self.configure(fg_color=C_BG)

        self.settings = load_settings()
        saved_scene = self.settings.get("scene", "manual")
        self._applied_scene = saved_scene if saved_scene in SCENES else "manual"
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
        self.orig_env = None
        self.output_path: str | None = None
        self.busy: str | None = None  # None, "load" or "denoise"
        self._last_time_text = ""
        self._pending_file = initial_file

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
        self.waveform = WaveformView(wave_frame, on_seek=self._seek)
        self.waveform.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Transport --------------------------------------------------------
        transport = ctk.CTkFrame(self, fg_color="transparent")
        transport.grid(row=3, column=0, sticky="ew", pady=(0, 10), **pad)
        transport.grid_columnconfigure(4, weight=1)
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
        self.time_label = ctk.CTkLabel(transport, text="0:00 / 0:00", font=self._font(13),
                                       text_color=C_MUTED)
        self.time_label.grid(row=0, column=4, sticky="e")

        # Settings ---------------------------------------------------------
        panel = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=10)
        panel.grid(row=4, column=0, sticky="ew", pady=(0, 10), **pad)
        panel.grid_columnconfigure(1, weight=1)
        label_opts = {"font": self._font(13), "text_color": C_TEXT, "anchor": "w"}
        hint_opts = {"font": self._font(12), "text_color": C_MUTED, "anchor": "w"}
        row_pad = {"padx": 14, "pady": (8, 0)}

        self.scene_var = ctk.StringVar(value=SCENES[self._applied_scene].label)
        ctk.CTkLabel(panel, text="Scene", **label_opts).grid(
            row=0, column=0, sticky="w", **row_pad)
        self.scene_menu = ctk.CTkOptionMenu(
            panel, values=[SCENES[key].label for key in SCENE_ORDER],
            variable=self.scene_var, width=220, command=self._on_scene_change,
            fg_color=C_LINE, button_color=C_LINE, button_hover_color=C_LINE_HOVER,
            text_color=C_TEXT, font=self._font(13))
        self.scene_menu.grid(row=0, column=1, columnspan=2, sticky="w",
                             padx=(0, 14), pady=(8, 0))
        self.scene_hint = ctk.CTkLabel(
            panel, text=SCENES[self._applied_scene].summary, wraplength=640, **hint_opts)
        self.scene_hint.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 14))

        ctk.CTkLabel(panel, text="Noise reduction", **label_opts).grid(row=2, column=0, sticky="w", **row_pad)
        self.atten_index = int(self.settings.get("atten_index", len(ATTEN_STEPS) - 1))
        self.atten_index = max(0, min(len(ATTEN_STEPS) - 1, self.atten_index))
        self.atten_slider = ctk.CTkSlider(panel, from_=0, to=len(ATTEN_STEPS) - 1,
                                          number_of_steps=len(ATTEN_STEPS) - 1,
                                          command=self._on_atten_change, progress_color=C_AMBER,
                                          button_color=C_AMBER, button_hover_color=C_AMBER_HOVER)
        self.atten_slider.set(self.atten_index)
        self.atten_slider.grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=(12, 0))
        self.atten_value = ctk.CTkLabel(panel, text="", width=90, **label_opts)
        self.atten_value.grid(row=2, column=2, sticky="w", padx=(0, 14), pady=(8, 0))
        ctk.CTkLabel(panel, text="Lower limits leave a little background in and sound more natural",
                     **hint_opts).grid(row=3, column=1, columnspan=2, sticky="w")

        ctk.CTkLabel(panel, text="Rumble filter", **label_opts).grid(row=4, column=0, sticky="w", **row_pad)
        self.highpass_var = ctk.StringVar(value=self.settings.get("highpass", "80 Hz"))
        if self.highpass_var.get() not in HIGHPASS_CHOICES:
            self.highpass_var.set("80 Hz")
        hp_row = ctk.CTkFrame(panel, fg_color="transparent")
        hp_row.grid(row=4, column=1, columnspan=2, sticky="w", pady=(8, 0))
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
        switches.grid(row=5, column=0, columnspan=3, sticky="w", padx=14, pady=(12, 0))
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
                                              command=self._save_settings)
        self.normalise_switch.grid(row=0, column=1, sticky="w")

        out_row = ctk.CTkFrame(panel, fg_color="transparent")
        out_row.grid(row=6, column=0, columnspan=3, sticky="ew", padx=14, pady=(10, 12))
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
        action.grid(row=5, column=0, sticky="ew", pady=(0, 8), **pad)
        action.grid_columnconfigure(0, weight=1)
        self.denoise_btn = ctk.CTkButton(action, text="Denoise", command=self.start_denoise,
                                         height=44, font=self._font(16, "bold"),
                                         fg_color=C_AMBER, hover_color=C_AMBER_HOVER,
                                         text_color=C_ON_AMBER, text_color_disabled="#6B5F4D")
        self.denoise_btn.grid(row=0, column=0, sticky="ew")
        self.cancel_btn = ctk.CTkButton(action, text="Cancel", command=self.cancel_denoise,
                                        width=100, height=44, font=self._font(14),
                                        fg_color=C_DANGER, hover_color=C_DANGER_HOVER,
                                        text_color=C_TEXT, text_color_disabled=C_MUTED)
        self.cancel_btn.grid(row=0, column=1, padx=(8, 0))
        progress_row = ctk.CTkFrame(action, fg_color="transparent")
        progress_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
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
        self.log_box.grid(row=6, column=0, sticky="ew", pady=(0, 16), **pad)
        self.log_box.configure(state="disabled")

        self._on_atten_change(self.atten_index)
        self._show_output_dir()

    def _bind_keys(self) -> None:
        self.bind("<space>", lambda _e: self.toggle_play())
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
        preset = SCENE_BY_LABEL.get(self.scene_var.get())
        return preset.key if preset is not None else "manual"

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

    def _save_settings(self) -> None:
        scene_key = self._scene_key() if hasattr(self, "scene_var") else "manual"
        data = {
            "output_dir": str(self.output_dir),
            "atten_index": self.atten_index,
            "highpass": self.highpass_var.get(),
            "post_filter": bool(self.postfilter_var.get()),
            "normalise": bool(self.normalise_var.get()),
            "scene": scene_key,
        }
        if not SCENES[scene_key].active:
            data["manual_highpass"] = self.highpass_var.get()
            data["manual_atten_index"] = self.atten_index
            data["manual_post_filter"] = bool(self.postfilter_var.get())
        else:
            for key in ("manual_highpass", "manual_atten_index", "manual_post_filter"):
                if key in self.settings:
                    data[key] = self.settings[key]
        self.settings.update(data)
        save_settings(self.settings)

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
        cancellable = self.busy == "denoise"
        self.cancel_btn.configure(state=on(cancellable),
                                  fg_color=C_DANGER if cancellable else C_LINE)
        scene_active = SCENES.get(self._applied_scene, MANUAL).active
        self.scene_menu.configure(state=on(idle))
        self.highpass_menu.configure(state=on(idle and not scene_active))
        for widget in (self.atten_slider, self.postfilter_switch,
                       self.normalise_switch, self.change_btn):
            widget.configure(state=on(idle))
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

    # ------------------------------------------------------------ actions --
    def open_file(self, path: str | None = None) -> None:
        if self.busy:
            return
        if not path:
            path = filedialog.askopenfilename(
                title="Open audio",
                initialdir=self.settings.get("last_dir") or str(Path.home()),
                filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg"), ("All files", "*.*")])
        if not path:
            return
        if not os.path.isfile(path):
            self.log(f"File not found: {path}")
            return
        self.settings["last_dir"] = str(Path(path).parent)
        self._save_settings()
        self.player.stop()
        self.log(f"Opening {Path(path).name}…")
        self._run_job("load", self._job_load_file, path)

    def start_denoise(self) -> None:
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
        }
        self.cancel_event.clear()
        self.progress.set(0)
        self.progress_label.configure(text="0%")
        self._run_job("denoise", self._job_denoise, self.audio48, self.model, self.df_state, opts)

    def cancel_denoise(self) -> None:
        if self.busy == "denoise":
            self.cancel_event.set()
            self.cancel_btn.configure(state="disabled", fg_color=C_LINE)
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
        self.cancel_event.set()
        self.player.stop()
        self._save_settings()
        self.destroy()

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
            "frames": audio.shape[1],
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
            put(("log", detail))
            started = time.perf_counter()
            den48 = denoise(
                model, df_state, work, limit, scene=preset if preset.active else None,
                progress=lambda fraction: put(("progress", fraction)),
                cancel=self.cancel_event)
            elapsed = time.perf_counter() - started
        finally:
            model.post_filter = previous_filter
            if previous_beta is not None:
                model.post_filter_beta = previous_beta
        if den48 is None:
            return

        out = resample(den48, MODEL_SR, opts["src_sr"])
        # Resampling can be a sample out either way; match the source length exactly.
        n_src = opts["src_frames"]
        if out.shape[1] >= n_src:
            out = out[:, :n_src]
        else:
            out = np.pad(out, ((0, 0), (0, n_src - out.shape[1])))
        gain_db = None
        if opts["normalise"]:
            out, gain_db = peak_normalise(out, NORMALISE_PEAK_DBFS)
        clipped = float(np.max(np.abs(out))) > 1.0
        out = np.clip(out, -1.0, 1.0)

        out_dir = Path(opts["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(opts['src_path']).stem}_denoised.wav"
        tmp_path = out_dir / f"{out_path.stem}.partial.wav"
        sf.write(str(tmp_path), out.T, opts["src_sr"], subtype="PCM_24")
        os.replace(tmp_path, out_path)

        put(("denoise_done", {"path": str(out_path), "den48": den48,
                              "env": waveform_envelope(den48), "elapsed": elapsed,
                              "gain_db": gain_db, "clipped": clipped}))

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
        self.output_path = None
        self.player.load_original(self.audio48)
        self.waveform.set_data(self.orig_env, None)
        self.waveform.set_playhead(0.0)
        channels = {1: "mono", 2: "stereo"}.get(info["channels"], f"{info['channels']} channels")
        details = f"{info['sr'] / 1000:g} kHz {channels}, {format_time(info['duration'])}"
        self.file_label.configure(text=shorten(Path(info["path"]).name, 70))
        self.file_info.configure(text=details)
        self.progress.set(0)
        self.progress_label.configure(text="")
        self.log(f"Opened {Path(info['path']).name} ({details}).")
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
        self.player.set_denoised(info["den48"])
        self.player.set_source("denoised")
        self.waveform.set_data(self.orig_env, info["env"])
        duration = self.audio48.shape[1] / MODEL_SR
        speed = duration / max(info["elapsed"], 1e-6)
        self.log(f"Denoised {format_time(duration)} of audio in {info['elapsed']:.1f} s "
                 f"({speed:.0f}× real time).")
        if info["gain_db"] is not None:
            self.log(f"Normalised to {NORMALISE_PEAK_DBFS:g} dBFS ({info['gain_db']:+.1f} dB).")
        if info["clipped"]:
            self.log("Some peaks went over full scale and were clipped. "
                     "Turn on normalising to avoid this.")
        self.log(f"Saved {self.output_path}")
        self.log("Press B to hear the denoised version, A for the original.")
        self._refresh_controls()

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
        self.log(f"Error: {message}")
        self._refresh_controls()


def main() -> None:
    ctk.set_appearance_mode("dark")
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    app = AudioDeNoiserApp(initial_file=initial)
    app.mainloop()


if __name__ == "__main__":
    main()
