"""Dependency-light audio IO and STFT.

Reads WAV via the standard-library :mod:`wave` module and computes an rFFT-based
STFT with numpy only. This keeps the SNR pipeline runnable on a bare Python
install (no soundfile / librosa / scipy), which is exactly the environment we
have locally and want to avoid coupling to.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np

__all__ = ["read_wav", "write_wav", "hann_window", "stft", "power_spectrogram"]


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV file to a mono float32 array in [-1, 1] and its sample rate.

    Supports 8/16/24/32-bit integer PCM and 32/64-bit IEEE float. Multi-channel
    audio is downmixed to mono by averaging channels.

    The stdlib ``wave`` module reads PCM only (it raises on IEEE-float / EXTENSIBLE
    headers, e.g. the float32 InfantMarmosetsVox recordings), so those formats fall
    back to a numpy-only RIFF parser — still no soundfile/scipy dependency.
    """
    path = str(path)
    try:
        with wave.open(path, "rb") as w:
            n_channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            sr = w.getframerate()
            n_frames = w.getnframes()
            raw = w.readframes(n_frames)
    except (wave.Error, EOFError):
        return _read_wav_numpy(path)

    if sampwidth == 1:  # unsigned 8-bit
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:  # signed 16-bit
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:  # signed 24-bit packed
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = (a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16))
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float32) / 8388608.0
    elif sampwidth == 4:  # signed 32-bit int
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes ({path})")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sr


def _read_wav_numpy(path: str) -> tuple[np.ndarray, int]:
    """Numpy-only RIFF/WAVE reader for headers the stdlib ``wave`` module rejects.

    Handles IEEE float (format tag 3, 32/64-bit) and WAVE_FORMAT_EXTENSIBLE
    (0xFFFE, whose real tag is the first 2 bytes of the SubFormat GUID), plus PCM
    as a complete fallback. Returns mono float32 in [-1, 1] and the sample rate.
    """
    with open(path, "rb") as fh:
        buf = fh.read()
    if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {path}")

    fmt_tag = n_channels = sr = bits = None
    data = None
    pos, n = 12, len(buf)
    while pos + 8 <= n:  # walk the chunk list to find 'fmt ' and 'data'
        cid = buf[pos : pos + 4]
        csize = int.from_bytes(buf[pos + 4 : pos + 8], "little")
        body = buf[pos + 8 : pos + 8 + csize]
        if cid == b"fmt " and len(body) >= 16:
            fmt_tag, n_channels, sr, _byte_rate, _block_align, bits = struct.unpack_from(
                "<HHLLHH", body, 0
            )
            if fmt_tag == 0xFFFE and len(body) >= 26:  # extensible: unwrap SubFormat
                fmt_tag = struct.unpack_from("<H", body, 24)[0]
        elif cid == b"data":
            data = body
        pos += 8 + csize + (csize & 1)  # chunks are word-aligned (pad byte if odd)

    if fmt_tag is None or data is None:
        raise ValueError(f"WAV missing fmt/data chunk: {path}")

    if fmt_tag == 3:  # IEEE float
        if bits == 32:
            samples = np.frombuffer(data, dtype="<f4")
        elif bits == 64:
            samples = np.frombuffer(data, dtype="<f8")
        else:
            raise ValueError(f"Unsupported float width: {bits}-bit ({path})")
    elif fmt_tag == 1:  # integer PCM
        if bits == 8:
            samples = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif bits == 16:
            samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        elif bits == 24:
            a = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            ints = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
            ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
            samples = ints.astype(np.float32) / 8388608.0
        elif bits == 32:
            samples = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported PCM width: {bits}-bit ({path})")
    else:
        raise ValueError(f"Unsupported WAV format tag: {fmt_tag} ({path})")

    samples = samples.astype(np.float32, copy=False)
    if n_channels and n_channels > 1:
        # trim any partial trailing frame so the reshape is exact, then downmix
        usable = (len(samples) // n_channels) * n_channels
        samples = samples[:usable].reshape(-1, n_channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), int(sr)


def read_wav_segment(path: str | Path, start: int, length: int) -> tuple[np.ndarray, int]:
    """Read only ``length`` frames starting at frame ``start`` (mono float32).

    Uses ``wave.setpos`` to seek, so grabbing a 0.5 s slice out of a 60 s file
    reads ~1% of the bytes — essential when the source lives on a slow FUSE mount.
    """
    path = str(path)
    with wave.open(path, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        total = w.getnframes()
        start = max(0, min(start, max(0, total - length)))
        w.setpos(start)
        raw = w.readframes(length)
    if sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        # Fall back to the full reader for uncommon widths (e.g. 24-bit).
        y, sr = read_wav(path)
        seg = y[start : start + length]
        return seg, sr
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sr


def write_wav(path: str | Path, y: np.ndarray, sr: int) -> None:
    """Write a mono float array in [-1, 1] to a 16-bit PCM WAV (stdlib only)."""
    i16 = (np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(i16.tobytes())


def wav_num_frames(path: str | Path) -> int:
    """Frame count from the WAV header (metadata only — no sample read)."""
    with wave.open(str(path), "rb") as w:
        return w.getnframes()


def wav_meta(path: str | Path) -> tuple[int, int]:
    """(sample_rate, n_frames) from the WAV header — metadata only, no sample read."""
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getnframes()


def resample_linear(y: np.ndarray, target_len: int) -> np.ndarray:
    """Resample ``y`` to ``target_len`` samples by linear interpolation.

    Crude but dependency-free (numpy only). Adequate for rate-matching a broadband
    noise segment to a call's sample rate before mixing — we never need audio-grade
    resampling of the calls themselves here.
    """
    if len(y) == target_len or len(y) < 2:
        return y[:target_len] if len(y) >= target_len else np.pad(y, (0, target_len - len(y)))
    src = np.linspace(0.0, 1.0, len(y))
    dst = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst, src, y).astype(np.float32)


def hann_window(n: int) -> np.ndarray:
    """Periodic Hann window (matches librosa/scipy 'hann' with sym=False)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def _frame(y: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Frame a signal into overlapping windows -> shape (n_frames, n_fft).

    Pads with reflection so that a short clip still yields several frames.
    """
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), mode="reflect") if len(y) > 1 else np.pad(
            y, (0, n_fft - len(y))
        )
    n_frames = 1 + (len(y) - n_fft) // hop
    n_frames = max(n_frames, 1)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return y[idx]


def stft(y: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Short-time Fourier transform -> complex array of shape (F, T), F = n_fft//2 + 1."""
    frames = _frame(np.asarray(y, dtype=np.float64), n_fft, hop)
    frames = frames * hann_window(n_fft)[None, :]
    return np.fft.rfft(frames, axis=1).T


def power_spectrogram(y: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Power spectrogram |STFT|**2 -> shape (F, T)."""
    return np.abs(stft(y, n_fft, hop)) ** 2
