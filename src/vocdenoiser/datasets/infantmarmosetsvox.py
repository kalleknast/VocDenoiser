"""Prepare the InfantMarmosetsVox dataset for the VocDenoiser identity eval.

InfantMarmosetsVox (Sarkar & Magimai-Doss, Idiap; Zenodo record 10130104,
CC-BY-4.0; originally recorded by Zhang et al. 2018) ships 350 ten-minute
44.1 kHz recordings plus a ``labels.csv`` annotating every vocalization with
start/end time, call-type (0-12) and caller identity (0-9). This module cuts each
vocalization segment into its own WAV clip (resampled to the model's rate) and
writes an ``id,identity`` CSV consumable by
``vocdenoiser.denoise.eval --labels-csv``.

It is an **external benchmark** for the "does compression preserve caller
identity?" proxy: a different colony (infant twins) at a different native rate
than our own ``data/Vocalizations``, so read the RandomForest accuracy as a
cross-dataset check, not a number comparable to an in-domain split.

**Sample-rate caveat.** IMV is captured at 44.1 kHz (content <= ~22 kHz). Our
β-VAE is trained at 96 kHz and its mel geometry is fixed to ``effective_sr``, so
clips must be emitted at that rate (``--target-sr 96000``, the default) to be
drop-in for eval. Upsampling leaves the 22-48 kHz band empty — inherent domain
shift, not a bug; most phee energy sits below 22 kHz so call structure survives.

Expected layout under ``--imv-root`` (extract the five twin tarballs into it)::

    InfantMarmosetsVox/
    ├── labels.csv
    └── data/
        ├── twin_1/<date>_Twin1_marmoset1.wav
        ├── twin_2/...
        └── twin_5/...

Usage::

    # one-shot: download the ~21 GB audio from Zenodo, then cut clips + write CSV
    python -m vocdenoiser.datasets.infantmarmosetsvox --download --target-sr 96000

    # or, if the audio is already extracted under <imv-root>/data/twin_*/
    python -m vocdenoiser.datasets.infantmarmosetsvox \
        --imv-root data/labelled/InfantMarmosetsVox \
        --target-sr 96000

Then run the identity proxy::

    python -m vocdenoiser.denoise.eval \
        --data-root data/labelled/InfantMarmosetsVox/clips \
        --ckpt checkpoints/last.ckpt \
        --labels-csv data/labelled/InfantMarmosetsVox/imv_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import wave
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np

from vocdenoiser.audio import read_wav

# labels.csv calltype index -> name (from the dataset README). 11/12 are dropped.
CALLTYPE_NAMES = {
    0: "Peep(Pre-Phee)", 1: "Phee", 2: "Twitter", 3: "Trill", 4: "Trillphee",
    5: "TsikTse", 6: "Egg", 7: "Pheecry(cry)", 8: "TrllTwitter", 9: "Pheetwitter",
    10: "Peep", 11: "Silence", 12: "Noise",
}
DROP_CALLTYPES = {11, 12}  # Silence, Noise

DEFAULT_ROOT = "data/labelled/InfantMarmosetsVox"

# Zenodo record 10130104 (CC-BY-4.0). The five twin tarballs total ~21 GB.
IMV_ZENODO_BASE = "https://zenodo.org/records/10130104/files"
ALL_TWINS = (1, 2, 3, 4, 5)


@dataclass
class Vocalization:
    """One annotated vocalization segment resolved to a source WAV + time span."""

    uid: str          # globally-unique clip id (also the output WAV stem)
    source_wav: Path  # the 10-minute recording to cut from
    start: float      # seconds
    end: float        # seconds
    calltype: int
    calltype_name: str
    caller: int       # global caller identity 0-9


def _source_wav(imv_root: Path, filename: str) -> Path:
    """Resolve a labels.csv ``filename`` to its 10-minute WAV path.

    ``filename`` is e.g. ``20160907_Twin1_marmoset1`` (no extension); the twin id
    is the last char of the second underscore token, matching the reference
    ``infantmarmosetsvox.py`` (``data/twin_<T>/<filename>.wav``).
    """
    twin_id = filename.split("_")[1][-1]
    return imv_root / "data" / f"twin_{twin_id}" / f"{filename}.wav"


def parse_labels(labels_csv: str | Path, imv_root: str | Path, id_prefix: str = "imv") -> list[Vocalization]:
    """Read ``labels.csv`` into vocalization records (Silence/Noise dropped).

    Pure: does no audio IO, so it is testable without the 21 GB audio. ids are
    assigned in CSV order (stable) as ``<id_prefix>_<i:05d>``.
    """
    imv_root = Path(imv_root)
    required = {"filename", "start", "end", "calltype", "caller"}
    out: list[Vocalization] = []
    with open(labels_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{labels_csv} is missing columns {sorted(missing)}; "
                f"got {reader.fieldnames}. Is this the InfantMarmosetsVox labels.csv?"
            )
        i = 0
        for row in reader:
            calltype = int(row["calltype"])
            if calltype in DROP_CALLTYPES:
                continue
            out.append(
                Vocalization(
                    uid=f"{id_prefix}_{i:05d}",
                    source_wav=_source_wav(imv_root, row["filename"]),
                    start=float(row["start"]),
                    end=float(row["end"]),
                    calltype=calltype,
                    calltype_name=CALLTYPE_NAMES.get(calltype, str(calltype)),
                    caller=int(row["caller"]),
                )
            )
            i += 1
    return out


def _resample(sig: np.ndarray, sr: int, target_sr: int, _warned: list[bool]) -> np.ndarray:
    """Resample ``sig`` sr -> target_sr. Polyphase (scipy) if available, else linear."""
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


def _write_wav(path: Path, sig: np.ndarray, sr: int, peak_normalize: bool) -> None:
    """Write a mono 16-bit PCM WAV."""
    sig = np.asarray(sig, dtype=np.float32)
    if peak_normalize:
        peak = float(np.max(np.abs(sig))) or 1.0
        sig = sig / peak * 0.98
    i16 = (np.clip(sig, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(i16.tobytes())


def write_label_csv(vocs: list[Vocalization], out_csv: str | Path, sr: int) -> None:
    """Write the ``id,identity(,...)`` CSV. ``identity`` = caller for the RF proxy."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "identity", "caller", "calltype", "calltype_name",
                         "source_file", "start", "end", "sr"])
        for v in vocs:
            writer.writerow([v.uid, v.caller, v.caller, v.calltype, v.calltype_name,
                             v.source_wav.name, f"{v.start:.4f}", f"{v.end:.4f}", sr])


def prepare(
    imv_root: str | Path,
    out_dir: str | Path,
    out_csv: str | Path,
    target_sr: int = 96_000,
    peak_normalize: bool = False,
    limit: int | None = None,
    id_prefix: str = "imv",
) -> dict:
    """Cut every vocalization into a clip and write the label CSV.

    Groups segments by source recording so each 10-minute WAV is read once.
    Segments whose source WAV is absent are skipped and counted (so a partial
    download still yields a usable subset).
    """
    imv_root = Path(imv_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocs = parse_labels(imv_root / "labels.csv", imv_root, id_prefix=id_prefix)
    if limit is not None:
        vocs = vocs[:limit]

    by_file: dict[Path, list[Vocalization]] = {}
    for v in vocs:
        by_file.setdefault(v.source_wav, []).append(v)

    written: list[Vocalization] = []
    n_missing_files = 0
    n_empty = 0
    warned: list[bool] = []
    for source_wav, segs in sorted(by_file.items(), key=lambda kv: str(kv[0])):
        if not source_wav.exists():
            n_missing_files += 1
            continue
        full, sr = read_wav(source_wav)
        out_sr = target_sr if target_sr > 0 else sr
        for v in segs:
            a = max(0, int(round(v.start * sr)))
            b = min(len(full), int(round(v.end * sr)))
            if b <= a:
                n_empty += 1
                continue
            clip = _resample(full[a:b], sr, out_sr, warned)
            _write_wav(out_dir / f"{v.uid}.wav", clip, out_sr, peak_normalize)
            written.append(v)

    out_sr = target_sr if target_sr > 0 else 44_100
    write_label_csv(written, out_csv, out_sr)

    n_callers = len({v.caller for v in written})
    summary = {
        "n_vocalizations": len(vocs),
        "n_written": len(written),
        "n_missing_source_files": n_missing_files,
        "n_empty_segments": n_empty,
        "n_callers": n_callers,
        "target_sr": out_sr,
        "clips_dir": str(out_dir),
        "labels_csv": str(out_csv),
    }
    print(
        f"Wrote {len(written)}/{len(vocs)} clips from {len(by_file) - n_missing_files} "
        f"recordings ({n_callers} callers) at {out_sr} Hz -> {out_dir}\n"
        f"Labels ({len(written)} rows) -> {out_csv}"
    )
    if n_missing_files:
        print(f"  NOTE: {n_missing_files} source recordings absent (partial download) — segments skipped.")
    return summary


def _fetch(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``, resuming with curl when a partial exists."""
    import shutil
    import subprocess

    if shutil.which("curl"):
        cmd = ["curl", "-L", "--fail", "--retry", "3"]
        if dest.exists() and dest.stat().st_size > 0:
            cmd += ["-C", "-"]  # resume a partial download
        cmd += ["-o", str(dest), url]
        print(f"  downloading {dest.name} (curl)…")
        subprocess.run(cmd, check=True)
        return
    # stdlib fallback (no resume): skip if something is already there.
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {dest.name} already present — skipping (urllib fallback cannot resume/verify).")
        return
    import urllib.request

    print(f"  downloading {dest.name} (urllib)…")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as fh:  # noqa: S310 (trusted Zenodo URL)
        shutil.copyfileobj(r, fh, 1 << 20)


def _extract_tarball(tar_path: Path, dest_root: Path) -> None:
    """Extract a .tar.gz into ``dest_root`` (path-traversal-safe on Python >= 3.12)."""
    import tarfile

    print(f"  extracting {tar_path.name}…")
    with tarfile.open(tar_path, "r:gz") as tf:
        try:
            tf.extractall(dest_root, filter="data")  # Python 3.12+
        except TypeError:
            tf.extractall(dest_root)  # older Python: no data filter


def _normalize_layout(imv_root: Path) -> int:
    """Move any extracted ``twin_*`` dirs to ``<imv_root>/data/`` and surface labels.csv.

    Makes the loader robust to whatever top-level prefix the archives use
    (``InfantMarmosetsVox/data/twin_1``, ``data/twin_1``, or ``twin_1``). Returns
    the number of twin directories relocated.
    """
    import shutil

    data_dir = imv_root / "data"
    data_dir.mkdir(exist_ok=True)
    moved = 0
    for p in list(imv_root.rglob("twin_*")):
        if not p.is_dir() or p.parent == data_dir or "_downloads" in p.parts:
            continue
        if not any(p.rglob("*.wav")):
            continue
        target = data_dir / p.name
        if target.exists():
            continue
        shutil.move(str(p), str(target))
        moved += 1
    if not (imv_root / "labels.csv").exists():
        found = next((p for p in imv_root.rglob("labels.csv") if "_downloads" not in p.parts), None)
        if found:
            shutil.copy(str(found), str(imv_root / "labels.csv"))
    return moved


def download_imv(
    imv_root: str | Path,
    twins: tuple[int, ...] = ALL_TWINS,
    keep_archives: bool = False,
) -> None:
    """Fetch + extract the InfantMarmosetsVox twin tarballs from Zenodo (~21 GB).

    Each tarball is downloaded to ``<imv_root>/_downloads`` (curl resumes a partial
    file), extracted, then deleted unless ``keep_archives``. Finally the layout is
    normalized so audio lands at ``<imv_root>/data/twin_*/``.
    """
    imv_root = Path(imv_root)
    dl_dir = imv_root / "_downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    for t in twins:
        name = f"InfantMarmosetsVox_twin_{t}.tar.gz"
        dest = dl_dir / name
        _fetch(f"{IMV_ZENODO_BASE}/{name}?download=1", dest)
        _extract_tarball(dest, imv_root)
        if not keep_archives:
            dest.unlink(missing_ok=True)
    moved = _normalize_layout(imv_root)
    n_recordings = len(list((imv_root / "data").rglob("*.wav")))
    print(f"Download complete: {moved} twin dirs normalized, {n_recordings} recordings under {imv_root/'data'}.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Prepare InfantMarmosetsVox clips + id,identity CSV.")
    p.add_argument("--imv-root", default=DEFAULT_ROOT,
                   help=f"extracted dataset root (labels.csv + data/twin_*/) [default {DEFAULT_ROOT}]")
    p.add_argument("--out-dir", default=None,
                   help="output clips dir [default <imv-root>/clips]")
    p.add_argument("--out-csv", default=None,
                   help="output labels CSV [default <imv-root>/imv_labels.csv]")
    p.add_argument("--target-sr", type=int, default=96_000,
                   help="resample clips to this rate; 0 keeps native 44.1 kHz. "
                        "Set to your model's effective_sr (default 96000).")
    p.add_argument("--peak-normalize", action="store_true",
                   help="peak-normalize each clip (matches the reference Dataset)")
    p.add_argument("--limit", type=int, default=None, help="cap number of vocalizations (quick subset)")
    p.add_argument("--id-prefix", default="imv", help="clip id prefix")
    p.add_argument("--download", action="store_true",
                   help="fetch + extract the twin audio tarballs from Zenodo (~21 GB) first")
    p.add_argument("--twins", type=int, nargs="+", default=list(ALL_TWINS),
                   help="which twin tarballs to download (with --download)")
    p.add_argument("--keep-archives", action="store_true",
                   help="keep the downloaded .tar.gz files after extraction")
    p.add_argument("--download-only", action="store_true",
                   help="download + extract, then stop (skip clip preparation)")
    args = p.parse_args(argv)

    imv_root = Path(args.imv_root)
    if args.download or args.download_only:
        download_imv(imv_root, twins=tuple(args.twins), keep_archives=args.keep_archives)
    if args.download_only:
        return

    out_dir = args.out_dir or imv_root / "clips"
    out_csv = args.out_csv or imv_root / "imv_labels.csv"
    prepare(imv_root, out_dir, out_csv, target_sr=args.target_sr,
            peak_normalize=args.peak_normalize, limit=args.limit, id_prefix=args.id_prefix)


if __name__ == "__main__":
    main()
