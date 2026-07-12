"""Prepare MarmAudio call-type labels for the VocDenoiser pipeline.

MarmAudio (Lamothe, Obliger-Debouche et al., Sci Data 2025; Zenodo record
15017207, CC-BY-4.0) ships ~871k pre-segmented FLAC vocalization clips recorded
at 96 kHz plus an ``Annotations.tsv``. About 215k clips carry a real call-type
label (Twitter, Tsik, Phee, Trill, Infant, Seep); the remaining ~656k are the
generic ``Vocalization`` (detected, not typed). **MarmAudio has no per-call
caller identity** — the housing room holds ~20 animals and identity is not
tracked, so it labels call *type* only.

This writes an ``id,identity`` CSV (``identity`` = call type) over the
type-labelled clips, for either ``vocdenoiser.denoise.eval --labels-csv``
(call-type separability of the latents) or as a large 96 kHz labelled corpus to
re-check the SNR metric's call-agnosticism at scale (``snr validate-types``).

The clips already exist as individual files, so the default run only writes the
label CSV, keyed to each clip's stem (e.g. ``Vocalization_862577``). Because the
clips are FLAC and eval globs ``*.wav``, pass ``--extract-dir`` to also decode the
labelled subset to mono WAV at ``--target-sr`` (needs ``soundfile``).

``Annotations.tsv`` columns (tab-separated; first column is an unnamed index)::

    <idx>  parent_name  file_name  label  duration  year  month  day  hour  second  millisecond

Usage::

    # label CSV only (fast, no audio needed):
    python -m vocdenoiser.datasets.marmaudio \
        --annotations data/labelled/MarmAudio/Annotations.tsv

    # also decode the labelled FLAC clips to WAV at the model's rate for eval
    # (MarmAudio is native 96 kHz; --target-sr 44100 downsamples to match the model):
    python -m vocdenoiser.datasets.marmaudio --extract \
        --flac-dir data/labelled/MarmAudio/Vocalizations \
        --extract-dir data/labelled/MarmAudio/clips --target-sr 44100
"""

from __future__ import annotations

import argparse
import csv
import wave
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np

from vocdenoiser.datasets.quality import (
    QUALITY_COLS,
    clip_quality,
    quality_fail_reasons,
    summarize,
)

# Non-call-type labels in Annotations.tsv: 'Vocalization' = detected but untyped.
DEFAULT_EXCLUDE = frozenset({"Vocalization", "Noise"})

DEFAULT_ROOT = "data/labelled/MarmAudio"


@dataclass
class LabeledClip:
    """One type-labelled MarmAudio clip."""

    uid: str          # clip stem (also the source FLAC / output WAV stem)
    file_name: str    # original clip filename, e.g. Vocalization_862577.flac
    label: str        # call type
    parent_name: str  # source recording, e.g. 2020_10_4


def parse_annotations(
    tsv_path: str | Path,
    exclude_labels: frozenset[str] = DEFAULT_EXCLUDE,
) -> list[LabeledClip]:
    """Read ``Annotations.tsv`` into type-labelled clip records.

    Pure (no audio IO). Rows whose ``label`` is in ``exclude_labels`` (default the
    generic ``Vocalization`` / ``Noise``) are dropped, leaving the ~215k typed
    calls.
    """
    required = {"parent_name", "file_name", "label"}
    out: list[LabeledClip] = []
    with open(tsv_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{tsv_path} is missing columns {sorted(missing)}; "
                f"got {reader.fieldnames}. Is this the MarmAudio Annotations.tsv?"
            )
        for row in reader:
            label = (row["label"] or "").strip()
            if not label or label in exclude_labels:
                continue
            out.append(
                LabeledClip(
                    uid=Path(row["file_name"]).stem,
                    file_name=row["file_name"],
                    label=label,
                    parent_name=row["parent_name"],
                )
            )
    return out


def _fmt_quality(q: dict, col: str) -> str:
    val = q.get(col, "")
    return f"{val:.4f}" if isinstance(val, float) else str(val)


def write_label_csv(
    clips: list[LabeledClip],
    out_csv: str | Path,
    qualities: dict[str, dict] | None = None,
) -> None:
    """Write the ``id,identity(,quality…)`` CSV. ``identity`` = call type (no caller id exists).

    If ``qualities`` (``uid -> quality dict``) is given, per-clip quality columns
    are appended (only available when the audio was decoded via ``--extract``).
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    base = ["id", "identity", "call_type", "source_file", "source_recording"]
    qcols = QUALITY_COLS if qualities else []
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(base + qcols)
        for c in clips:
            row = [c.uid, c.label, c.label, c.file_name, c.parent_name]
            if qualities:
                q = qualities.get(c.uid, {})
                row += [_fmt_quality(q, col) for col in qcols]
            writer.writerow(row)


def _resample(sig: np.ndarray, sr: int, target_sr: int, _warned: list[bool]) -> np.ndarray:
    if target_sr <= 0 or sr == target_sr:
        return sig
    try:
        from scipy.signal import resample_poly

        g = gcd(int(sr), int(target_sr))
        return resample_poly(sig, target_sr // g, sr // g).astype(np.float32)
    except ImportError:
        if not _warned:
            print("scipy not installed — falling back to low-quality linear resampling.")
            _warned.append(True)
        from vocdenoiser.audio import resample_linear

        return resample_linear(sig, int(round(len(sig) * target_sr / sr)))


def _write_wav(path: Path, sig: np.ndarray, sr: int) -> None:
    i16 = (np.clip(np.asarray(sig, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(i16.tobytes())


def extract_clips(
    clips: list[LabeledClip],
    flac_dir: str | Path,
    out_dir: str | Path,
    target_sr: int = 44_100,
    quality: bool = True,
    out_csv: str | Path | None = None,
    min_snr: float | None = None,
    min_active_frac: float | None = None,
    max_clip_frac: float | None = None,
    min_peak_dbfs: float | None = None,
    min_dur: float | None = None,
    max_segments: int | None = None,
) -> list[LabeledClip]:
    """Decode the labelled FLAC clips to mono WAV at ``target_sr`` in ``out_dir``.

    Needs ``soundfile`` (FLAC support). Missing source files are skipped. With
    ``quality`` on, each decoded clip is scored and (if any ``min_*``/``max_*``
    threshold is set) low-quality clips are dropped. When ``out_csv`` is given, an
    ``id,identity`` CSV with quality columns is written for the kept clips.
    Returns the list of kept clips.
    """
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "--extract needs the 'soundfile' package to read FLAC "
            "(pip install soundfile)."
        ) from exc

    flac_dir = Path(flac_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = dict(min_snr=min_snr, min_active_frac=min_active_frac,
                      max_clip_frac=max_clip_frac, min_peak_dbfs=min_peak_dbfs,
                      min_dur=min_dur, max_segments=max_segments)
    filtering = any(v is not None for v in thresholds.values())
    score = quality or filtering

    warned: list[bool] = []
    kept: list[LabeledClip] = []
    quals: dict[str, dict] = {}
    scored: list[dict] = []
    drop_reasons: dict[str, int] = {}
    n_missing = 0
    n_dropped = 0
    for c in clips:
        src = flac_dir / c.file_name
        if not src.exists():
            n_missing += 1
            continue
        sig, sr = sf.read(str(src), always_2d=True)
        sig = sig[:, 0]  # downmix to mono (first channel)
        out_sr = target_sr if target_sr > 0 else sr
        sig = _resample(sig, sr, out_sr, warned)
        q = clip_quality(sig, out_sr) if score else None
        if filtering:
            reasons = quality_fail_reasons(q, **thresholds)
            if reasons:
                n_dropped += 1
                for r in reasons:
                    drop_reasons[r] = drop_reasons.get(r, 0) + 1
                continue
        _write_wav(out_dir / f"{c.uid}.wav", sig, out_sr)
        kept.append(c)
        if q is not None:
            quals[c.uid] = q
            scored.append(q)

    if out_csv is not None:
        write_label_csv(kept, out_csv, quals or None)
    print(f"Decoded {len(kept)}/{len(clips)} FLAC clips -> {out_dir}")
    if n_missing:
        print(f"  NOTE: {n_missing} FLAC files absent (partial download) — skipped.")
    if filtering:
        reason_str = ", ".join(f"{k}={v}" for k, v in sorted(drop_reasons.items())) or "none"
        print(f"  quality filter dropped {n_dropped} clips ({reason_str}).")
    if scored:
        print(summarize(scored))
    return kept


def _label_histogram(clips: list[LabeledClip]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for c in clips:
        hist[c.label] = hist.get(c.label, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Prepare MarmAudio call-type labels (+ optional WAV clips).")
    p.add_argument("--annotations", default=f"{DEFAULT_ROOT}/Annotations.tsv",
                   help="path to MarmAudio Annotations.tsv")
    p.add_argument("--out-csv", default=f"{DEFAULT_ROOT}/marmaudio_labels.csv",
                   help="output id,identity(call-type) CSV")
    p.add_argument("--keep-generic", action="store_true",
                   help="also keep the generic 'Vocalization' (untyped) rows")
    p.add_argument("--extract", action="store_true",
                   help="also decode the labelled FLAC clips to WAV (needs soundfile)")
    p.add_argument("--flac-dir", default=f"{DEFAULT_ROOT}/Vocalizations",
                   help="dir holding the FLAC clips (for --extract)")
    p.add_argument("--extract-dir", default=f"{DEFAULT_ROOT}/clips",
                   help="output dir for decoded WAV clips (for --extract)")
    p.add_argument("--target-sr", type=int, default=44_100,
                   help="resample decoded clips to this rate (match model effective_sr; "
                        "MarmAudio is native 96 kHz); 0 keeps native (for --extract)")
    # Call-quality (only with --extract, since it needs the decoded audio):
    p.add_argument("--no-quality", dest="quality", action="store_false",
                   help="skip per-clip quality scoring during --extract")
    p.add_argument("--min-snr", type=float, default=None, help="drop decoded clips with snr_db below this")
    p.add_argument("--min-active-frac", type=float, default=None,
                   help="drop decoded clips with active_frac below this")
    p.add_argument("--max-clip-frac", type=float, default=None,
                   help="drop decoded clips with more than this fraction of samples clipped")
    p.add_argument("--min-peak-dbfs", type=float, default=None,
                   help="drop decoded clips whose peak level is below this dBFS")
    p.add_argument("--min-dur", type=float, default=None, help="drop decoded clips shorter than this many seconds")
    p.add_argument("--max-segments", type=int, default=None,
                   help="drop decoded clips with more than this many active segments")
    args = p.parse_args(argv)

    exclude = frozenset({"Noise"}) if args.keep_generic else DEFAULT_EXCLUDE
    clips = parse_annotations(args.annotations, exclude_labels=exclude)
    write_label_csv(clips, args.out_csv)
    hist = _label_histogram(clips)
    print(f"Wrote {len(clips)} labelled clips -> {args.out_csv}")
    print("  call-type distribution: " + ", ".join(f"{k}={v}" for k, v in hist.items()))

    if args.extract:
        # The quality-scored, eval-ready CSV for the decoded subset lives with the clips.
        extracted_csv = Path(args.extract_dir) / "marmaudio_labels.csv"
        extract_clips(clips, args.flac_dir, args.extract_dir, target_sr=args.target_sr,
                      quality=args.quality, out_csv=extracted_csv,
                      min_snr=args.min_snr, min_active_frac=args.min_active_frac,
                      max_clip_frac=args.max_clip_frac, min_peak_dbfs=args.min_peak_dbfs,
                      min_dur=args.min_dur, max_segments=args.max_segments)


if __name__ == "__main__":
    main()
