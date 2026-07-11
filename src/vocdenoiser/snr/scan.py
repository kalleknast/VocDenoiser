"""Scan a folder of clips and write a per-clip feature/score table (CSV)."""

from __future__ import annotations

import csv
import os
import time
from multiprocessing import Pool
from pathlib import Path

from vocdenoiser.snr.metric import DEFAULT_PARAMS, SNRParams, clip_features_from_path

FIELDS = [
    "filename",
    "snr_db",
    "snr_broadband_db",
    "snr_temporal_db",
    "active_frac",
    "one_minus_entropy",
    "n_segments",
    "dom_freq_hz",
    "bandwidth_hz",
    "flatness",
    "duration_s",
    "n_frames",
    "sr",
    "error",
]

# Module-level params so the worker can be a top-level (picklable) function.
_PARAMS = DEFAULT_PARAMS


def _init_worker(params: SNRParams) -> None:
    global _PARAMS
    _PARAMS = params


def _process(path: str) -> dict:
    feats = clip_features_from_path(path, _PARAMS)
    feats["filename"] = os.path.basename(path)
    return feats


def find_clips(folder: str | Path) -> list[str]:
    folder = Path(folder)
    return sorted(str(p) for p in folder.rglob("*") if p.suffix.lower() == ".wav")


def scan_folder(
    folder: str | Path,
    out_csv: str | Path,
    params: SNRParams = DEFAULT_PARAMS,
    workers: int | None = None,
    limit: int | None = None,
) -> str:
    """Compute SNR features for every WAV under ``folder`` -> ``out_csv``.

    Returns the output path. Rows stream to disk as they complete so a run over
    50k files uses flat memory and is resumable-friendly (re-run with a new csv).
    """
    paths = find_clips(folder)
    if limit:
        paths = paths[:limit]
    total = len(paths)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    done = 0
    with open(out_csv, "w", newline="") as fh, Pool(
        processes=workers, initializer=_init_worker, initargs=(params,)
    ) as pool:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for feats in pool.imap_unordered(_process, paths, chunksize=64):
            writer.writerow(feats)
            done += 1
            if done % 2000 == 0 or done == total:
                rate = done / max(time.time() - t0, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                print(
                    f"  {done}/{total}  ({rate:.0f} clips/s, ETA {eta:.0f}s)", flush=True
                )
    print(f"Wrote {out_csv} ({total} rows in {time.time() - t0:.1f}s)")
    return str(out_csv)
