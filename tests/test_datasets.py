"""Tests for the external-dataset loaders (numpy-only; no torch needed)."""

import csv
import wave
from pathlib import Path

import numpy as np
import pytest

from vocdenoiser.datasets import infantmarmosetsvox as imv
from vocdenoiser.datasets import marmaudio as ma
from vocdenoiser.denoise.eval import _load_label_map


def _write_wav(path, y, sr):
    i16 = (np.clip(y, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(i16.tobytes())


def _wav_meta(path):
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getnframes()


# --------------------------- InfantMarmosetsVox ---------------------------

IMV_ROWS = [
    # filename, start, end, duration, calltype, caller
    ("20160907_Twin1_marmoset1", 0.0, 0.5, 0.5, 1, 0),   # Phee
    ("20160907_Twin1_marmoset1", 0.5, 0.6, 0.1, 11, 0),  # Silence -> dropped
    ("20160907_Twin1_marmoset1", 1.0, 1.8, 0.8, 2, 0),   # Twitter
    ("20160907_Twin1_marmoset1", 2.0, 2.4, 0.4, 12, 0),  # Noise -> dropped
    ("20160907_Twin1_marmoset1", 2.4, 2.9, 0.5, 3, 0),   # Trill
    ("20160908_Twin2_marmoset2", 0.2, 0.7, 0.5, 1, 3),   # Phee, caller 3, file absent
]


def _make_imv_root(tmp_path: Path, sr=44100) -> Path:
    root = tmp_path / "InfantMarmosetsVox"
    (root / "data" / "twin_1").mkdir(parents=True)
    with open(root / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "start", "end", "duration", "calltype", "caller"])
        w.writerows(IMV_ROWS)
    # Only twin_1's recording exists (3 s); twin_2's is intentionally absent.
    y = 0.5 * np.sin(2 * np.pi * 8000 * np.arange(3 * sr) / sr)
    _write_wav(root / "data" / "twin_1" / "20160907_Twin1_marmoset1.wav", y, sr)
    return root


def test_imv_parse_drops_silence_and_noise(tmp_path: Path):
    root = _make_imv_root(tmp_path)
    vocs = imv.parse_labels(root / "labels.csv", root)
    assert len(vocs) == 4  # 6 rows minus Silence(11) and Noise(12)
    assert [v.calltype for v in vocs] == [1, 2, 3, 1]
    assert [v.caller for v in vocs] == [0, 0, 0, 3]
    assert len({v.uid for v in vocs}) == 4  # unique ids
    assert vocs[0].uid == "imv_00000"
    assert vocs[0].calltype_name == "Phee"


def test_imv_source_wav_path(tmp_path: Path):
    root = _make_imv_root(tmp_path)
    vocs = imv.parse_labels(root / "labels.csv", root)
    assert vocs[0].source_wav == root / "data" / "twin_1" / "20160907_Twin1_marmoset1.wav"
    assert vocs[-1].source_wav == root / "data" / "twin_2" / "20160908_Twin2_marmoset2.wav"


def test_imv_parse_errors_on_bad_header(tmp_path: Path):
    bad = tmp_path / "labels.csv"
    with open(bad, "w", newline="") as fh:
        csv.writer(fh).writerow(["file", "t0", "t1"])
    with pytest.raises(ValueError, match="missing columns"):
        imv.parse_labels(bad, tmp_path)


def test_imv_prepare_cuts_clips_and_writes_csv(tmp_path: Path):
    root = _make_imv_root(tmp_path)
    out_dir = tmp_path / "clips"
    out_csv = tmp_path / "imv_labels.csv"
    target_sr = 22050
    summary = imv.prepare(root, out_dir, out_csv, target_sr=target_sr)

    assert summary["n_vocalizations"] == 4
    assert summary["n_written"] == 3          # 3 from twin_1
    assert summary["n_missing_source_files"] == 1  # twin_2 recording absent
    assert summary["n_callers"] == 1          # only caller 0 was written

    clips = sorted(out_dir.glob("*.wav"))
    assert len(clips) == 3
    # Phee clip spans 0.0-0.5 s -> ~0.5 s at the target rate.
    sr, n = _wav_meta(out_dir / "imv_00000.wav")
    assert sr == target_sr
    assert abs(n - int(0.5 * target_sr)) <= 2

    # The CSV is consumable by eval's --labels-csv loader.
    lmap = _load_label_map(str(out_csv), "id", "identity")
    assert lmap == {"imv_00000": "0", "imv_00001": "0", "imv_00002": "0"}


def test_imv_prepare_native_rate(tmp_path: Path):
    root = _make_imv_root(tmp_path, sr=44100)
    out_dir = tmp_path / "clips"
    summary = imv.prepare(root, out_dir, tmp_path / "l.csv", target_sr=0)  # keep native
    assert summary["target_sr"] == 44100
    sr, _ = _wav_meta(out_dir / "imv_00000.wav")
    assert sr == 44100


# ------------------------------- MarmAudio -------------------------------

MA_ROWS = [
    # idx, parent_name, file_name, label, duration, ...
    ("0", "2020_10_4", "Twitter_1.flac", "Twitter", "0.75"),
    ("1", "2020_10_4", "Phee_2.flac", "Phee", "0.75"),
    ("2", "2020_10_4", "Vocalization_3.flac", "Vocalization", "0.75"),  # untyped -> dropped
    ("3", "2020_10_4", "Noise_4.flac", "Noise", "0.75"),                # -> dropped
    ("4", "2020_10_5", "Tsik_5.flac", "Tsik", "1.05"),
]


def _make_ma_tsv(tmp_path: Path) -> Path:
    tsv = tmp_path / "Annotations.tsv"
    with open(tsv, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["", "parent_name", "file_name", "label", "duration"])
        w.writerows(MA_ROWS)
    return tsv


def test_marmaudio_parse_filters_generic(tmp_path: Path):
    tsv = _make_ma_tsv(tmp_path)
    clips = ma.parse_annotations(tsv)  # default excludes Vocalization + Noise
    assert [c.label for c in clips] == ["Twitter", "Phee", "Tsik"]
    assert clips[0].uid == "Twitter_1"  # stem of file_name
    assert clips[0].file_name == "Twitter_1.flac"


def test_marmaudio_keep_generic(tmp_path: Path):
    tsv = _make_ma_tsv(tmp_path)
    clips = ma.parse_annotations(tsv, exclude_labels=frozenset({"Noise"}))
    labels = [c.label for c in clips]
    assert "Vocalization" in labels and "Noise" not in labels
    assert len(clips) == 4


def test_marmaudio_csv_plugs_into_eval_loader(tmp_path: Path):
    tsv = _make_ma_tsv(tmp_path)
    out_csv = tmp_path / "ma_labels.csv"
    clips = ma.parse_annotations(tsv)
    ma.write_label_csv(clips, out_csv)
    lmap = _load_label_map(str(out_csv), "id", "identity")
    assert lmap == {"Twitter_1": "Twitter", "Phee_2": "Phee", "Tsik_5": "Tsik"}


def test_marmaudio_parse_errors_on_bad_header(tmp_path: Path):
    bad = tmp_path / "Annotations.tsv"
    with open(bad, "w", newline="") as fh:
        csv.writer(fh, delimiter="\t").writerow(["", "wrong", "cols"])
    with pytest.raises(ValueError, match="missing columns"):
        ma.parse_annotations(bad)
