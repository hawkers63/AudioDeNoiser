#!/usr/bin/env python3
"""
Audio DeNoiser 1.0
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
2. Optional high-pass filter to strip wind and traffic rumble below 60-120 Hz.
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
from scipy.signal import butter, resample_poly, sosfiltfilt
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
APP_VERSION = "1.0"

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
            progress=None, cancel: threading.Event | None = None) -> np.ndarray:
    """Run DeepFilterNet over 48 kHz audio (channels, frames) in overlapping chunks."""
    import torch
    from df.enhance import enhance

    sr = df_state.sr()
    channels, n = audio.shape
    n_chunks = max(1, math.ceil(n / (CHUNK_SECONDS * sr)))
    chunk = math.ceil(n / n_chunks)  # balanced chunk sizes, no tiny last chunk
    pre = int(PREROLL_SECONDS * sr)
    xfade = int(CROSSFADE_SECONDS * sr)

    src = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    out = np.zeros((channels, n), dtype=np.float32)

    for k in range(n_chunks):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        start = k * chunk
        end = min(n, start + chunk)
        last = k == n_chunks - 1
        a = max(0, start - pre)                 # pre-roll: model settles here
        b = n if last else min(n, end + xfade)  # tail: crossfades into next chunk

        if hasattr(df_state, "reset"):
            df_state.reset()
        with torch.inference_mode():
            seg = enhance(model, df_state, src[:, a:b].contiguous(), atten_lim_db=atten_lim_db)
        piece = seg.detach().cpu().numpy()[:, start - a: b - a]

        weight = np.ones(b - start, dtype=np.float32)
        if k > 0:
            m = min(xfade, b - start)
            weight[:m] = np.linspace(0.0, 1.0, m, dtype=np.float32)
        if not last:
            m = b - end
            weight[end - start:] *= np.linspace(1.0, 0.0, m, dtype=np.float32)
        out[:, start:b] += piece * weight

        if progress is not None:
            progress((k + 1) / n_chunks)
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
        self.geometry("860x760")
        self.minsize(760, 700)
        self.configure(fg_color=C_BG)

        self.settings = load_settings()
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

        ctk.CTkLabel(panel, text="Noise reduction", **label_opts).grid(row=0, column=0, sticky="w", **row_pad)
        self.atten_index = int(self.settings.get("atten_index", len(ATTEN_STEPS) - 1))
        self.atten_index = max(0, min(len(ATTEN_STEPS) - 1, self.atten_index))
        self.atten_slider = ctk.CTkSlider(panel, from_=0, to=len(ATTEN_STEPS) - 1,
                                          number_of_steps=len(ATTEN_STEPS) - 1,
                                          command=self._on_atten_change, progress_color=C_AMBER,
                                          button_color=C_AMBER, button_hover_color=C_AMBER_HOVER)
        self.atten_slider.set(self.atten_index)
        self.atten_slider.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(12, 0))
        self.atten_value = ctk.CTkLabel(panel, text="", width=90, **label_opts)
        self.atten_value.grid(row=0, column=2, sticky="w", padx=(0, 14), pady=(8, 0))
        ctk.CTkLabel(panel, text="Lower limits leave a little background in and sound more natural",
                     **hint_opts).grid(row=1, column=1, columnspan=2, sticky="w")

        ctk.CTkLabel(panel, text="Rumble filter", **label_opts).grid(row=2, column=0, sticky="w", **row_pad)
        self.highpass_var = ctk.StringVar(value=self.settings.get("highpass", "80 Hz"))
        if self.highpass_var.get() not in HIGHPASS_CHOICES:
            self.highpass_var.set("80 Hz")
        hp_row = ctk.CTkFrame(panel, fg_color="transparent")
        hp_row.grid(row=2, column=1, columnspan=2, sticky="w", pady=(8, 0))
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
        switches.grid(row=3, column=0, columnspan=3, sticky="w", padx=14, pady=(12, 0))
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
        out_row.grid(row=4, column=0, columnspan=3, sticky="ew", padx=14, pady=(10, 12))
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

    def _save_settings(self) -> None:
        self.settings.update({
            "output_dir": str(self.output_dir),
            "atten_index": self.atten_index,
            "highpass": self.highpass_var.get(),
            "post_filter": bool(self.postfilter_var.get()),
            "normalise": bool(self.normalise_var.get()),
        })
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
        for widget in (self.atten_slider, self.highpass_menu, self.postfilter_switch,
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
        work = audio48
        if opts["highpass"]:
            work = highpass(work, MODEL_SR, opts["highpass"])
        model.post_filter = opts["post_filter"]
        limit = opts["atten"]
        put(("log", "Denoising: noise reduction " + ("maximum" if limit is None else f"up to {limit} dB")
                    + (f", rumble filter {opts['highpass']} Hz" if opts["highpass"] else "")
                    + (", post-filter on" if opts["post_filter"] else "")))

        started = time.perf_counter()
        den48 = denoise(model, df_state, work, limit,
                        progress=lambda f: put(("progress", f)), cancel=self.cancel_event)
        elapsed = time.perf_counter() - started

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
