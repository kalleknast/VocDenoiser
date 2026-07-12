"""Downsample a directory of WAVs to a target sample rate (anti-aliased).

The archive is 96 kHz, but the model works at 44.1 kHz (``config.Config.sr``) —
phee energy is essentially all below ~22 kHz. This tool makes a smaller 44.1 kHz
copy of a call set *once*, so training reads it directly instead of resampling
every clip every epoch, and the on-disk / upload footprint drops ~2.2x.

    python -m vocdenoiser.datasets.resample --src data/Vocalizations \
        --dst data/Vocalizations_44k --target-sr 44100 --workers 8

Uses anti-aliased polyphase resampling (``scipy.signal.resample_poly``, else
``torchaudio``). It deliberately does NOT fall back to the aliasing linear
interpolator in :mod:`vocdenoiser.audio` — that is fine for rate-matching a
noise bed, but not for downsampling the calls themselves. Output is mono 16-bit
PCM, mirroring the source directory tree.
"""

from __future__ import annotations

import argparse
from math import gcd
from pathlib import Path

import numpy as np

from vocdenoiser.audio import read_wav, write_wav


def _resampler():
    """Return an anti-aliased ``(sig, sr, target_sr) -> sig`` callable, or raise."""
    try:
        from scipy.signal import resample_poly

        def _rs(sig: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
            g = gcd(int(sr), int(target_sr))
            return resample_poly(sig, target_sr // g, sr // g).astype(np.float32)

        return _rs
    except Exception:  # noqa: BLE001 - scipy absent; try torchaudio next
        pass
    try:
        import torch
        import torchaudio.functional as AF

        def _rs(sig: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
            out = AF.resample(torch.from_numpy(np.ascontiguousarray(sig)), sr, target_sr)
            return out.numpy().astype(np.float32)

        return _rs
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "Anti-aliased resampling needs scipy or torchaudio, neither importable. "
            "Install one:  pip install scipy   (or the project's .[ml] / .[resample] extra)."
        ) from e


def _convert_one(
    src_path: Path, src_root: Path, dst_root: Path, target_sr: int, resample
) -> None:
    out_path = (dst_root / src_path.relative_to(src_root)).with_suffix(".wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y, sr = read_wav(src_path)
    if sr != target_sr:
        y = resample(y, sr, target_sr)
    write_wav(out_path, y, target_sr)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Anti-aliased WAV directory downsampler.")
    p.add_argument("--src", required=True, type=Path, help="source WAV directory")
    p.add_argument("--dst", required=True, type=Path, help="output directory (tree mirrored)")
    p.add_argument("--target-sr", type=int, default=44_100, help="output sample rate (default 44100)")
    p.add_argument("--glob", default="**/*", help="recursive glob under --src (default '**/*')")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    src_root, dst_root = args.src.expanduser(), args.dst.expanduser()
    files = [
        f
        for f in sorted(src_root.glob(args.glob))
        if f.is_file() and f.suffix.lower() == ".wav"
    ]
    if not files:
        raise SystemExit(f"No .wav files under {src_root} matching {args.glob!r}")
    resample = _resampler()

    from concurrent.futures import ThreadPoolExecutor

    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_convert_one, f, src_root, dst_root, args.target_sr, resample) for f in files]
        for i, (f, fut) in enumerate(zip(files, futs), 1):
            try:
                fut.result()
                n_ok += 1
            except Exception as e:  # noqa: BLE001 - one bad file must not abort the batch
                n_err += 1
                print(f"  skip {f.name}: {e}")
            if i % 1000 == 0 or i == len(files):
                print(f"  {i}/{len(files)} ({n_ok} ok, {n_err} failed)")
    print(f"Wrote {n_ok} WAVs at {args.target_sr} Hz -> {dst_root}  ({n_err} failed)")


if __name__ == "__main__":
    main()
