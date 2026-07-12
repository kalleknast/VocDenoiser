# VocDenoiser

Denoising **and** compression of common-marmoset (*Callithrix jacchus*) vocalizations —
targeting isolated **phee** calls. Three stages:

1. **SNR filtering** (`vocdenoiser.snr`) — a call-agnostic, numpy-only pipeline that scores
   clips by the **mean per-bin SNR over active bins** (an intensive, band-free measure that
   does not reward bandwidth, so it is not biased toward tonal phee calls) and selects a clean
   training subset. Runs on a bare Python install (no scientific stack required).
2. **β-VAE denoiser** (`vocdenoiser.denoise`) — a 2D convolutional β-Variational Autoencoder
   that maps a **noisy** phee spectrogram to the **clean** one and compresses it to a
   **16-dim** latent for downstream clustering / identity analysis. GPU/Colab territory
   (optional `ml` extra).
3. **Architecture search** (`vocdenoiser.search`) — an autoresearch-style loop that searches
   denoiser architectures + hyperparameters under a fixed compute budget. GPU/Colab (real
   runs); the loop mechanics run on CPU via a mock harness.

See [`SPECS.md`](SPECS.md) for the model/loss/augmentation design and
[`.claude/skills/noise-recipe/SKILL.md`](.claude/skills/noise-recipe/SKILL.md) for the exact
colony-noise recipe.

## Install

The core SNR pipeline depends on numpy only. The denoiser adds the heavy ML stack via the
`ml` extra. `uv` is *not* required — plain `pip` + a venv works (Python ≥ 3.10; use 3.13 for
current torch wheels):

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,dev]"      # denoiser + torch/lightning + dev tooling
# or, SNR pipeline only (numpy-only):
pip install -e .
```

## Data

All audio ships in the repo under `data/` (gitignored — local only):

- `data/Vocalizations/` — **~50,000 isolated phee calls** (96 kHz mono 16-bit PCM, ~0.5 s each),
  named by bare numeric call ID (`100000.wav`). The clean-call training set and default
  `data_root`.
- `data/Noise/` — 30 one-minute colony-noise clips (real-noise augmentation / SNR validation).
- `data/Cigarra/` — 136 cicada-noise clips (second noise source for SNR `validate`).

`resample_sr` in the config resamples before the STFT if you want a rate below 96 kHz.

The clean-call root is **never hardcoded** — it resolves as `--data-root` >
`$VOCDENOISER_DATA_ROOT` > `data/Vocalizations` (default), so the same commands run on Colab and
locally. The 50k calls are the isolated-call *pool*; run the SNR filtering below first to train
only on the selected clean subset.

**Copy the calls to local disk before training.** `data/` sits on a slow pCloud FUSE mount and
the loader reads WAVs per item — don't stream them off the mount in the training loop.

## SNR filtering — select the clean training set

Score every clip, inspect the distribution, choose a cutoff, then select. The metric is
**call-agnostic** (see `reports/snr_validation.md` for the validation on this dataset):

```bash
# 1. score every clip -> CSV (numpy-only; parallel; ~200 clips/s over pCloud, faster local)
python -m vocdenoiser.cli snr scan data/Vocalizations --out artifacts/snr_scores.csv --workers 16

# 2. distribution report: histogram, GMM/Otsu thresholds, kept-count curve, per-decile samples
python -m vocdenoiser.cli snr report artifacts/snr_scores.csv --out-dir reports

# 3. (optional) prove the score is not biased toward any call type on THIS data
python -m vocdenoiser.cli snr validate artifacts/snr_scores.csv --src-dir data/Vocalizations \
    --noise-dir data/Noise --noise-dir data/Cigarra --out-dir reports

# 4. apply a threshold (or --keep-percentile N) -> clean manifest (+ optional --link-dir).
#    --broadband-floor adds a secondary snr_broadband_db cutoff to drop hissy narrowband clips.
python -m vocdenoiser.cli snr select artifacts/snr_scores.csv --src-dir data/Vocalizations \
    --out artifacts/clean_manifest.csv --threshold 19.82 --broadband-floor 15 --link-dir clean_subset
```

**How the score works.** Per frequency bin, the noise floor is a temporal low-percentile and
the signal a high-percentile; the score is the *mean of the per-bin SNR over the active bins*.
Because it is intensive (a per-bin average, not an energy sum) it does not reward broadband
calls, and because the floor is per-bin it makes no assumption about which band a call occupies
— so trills / twitters / eks are scored on equal footing with tonal phees. `snr_broadband_db`
is a secondary, broadband-sensitive score (bandwidth-biased); pass `--broadband-floor` to
`select` to drop hissy narrowband clips the primary cutoff lets through. `n_segments>1` flags
likely co-occurring sources (target call + an overlapping bird / word / transient), which
`select` drops by default.

**Validation.** `snr validate` runs the definitive check: it mixes real background noise
(`data/Noise` + `data/Cigarra`) into clean calls at a known SNR sweep and confirms the score
(a) rises monotonically with injected SNR and (b) tracks it *the same way* across low/high
dominant-frequency and narrow/broad-band strata (small cross-stratum spread = call-agnostic),
plus a morphology-correlation guard measured on already-clean clips.

## Train

```bash
export VOCDENOISER_DATA_ROOT=/local/disk/clean_phee_calls   # or pass --data-root
python -m vocdenoiser.denoise.train \
    --data-root "$VOCDENOISER_DATA_ROOT" \
    --max-epochs 100 --batch-size 32 --beta 4.0
```

Every `Config` field (`config.py`) is a CLI flag (`--n-mels`, `--hop`, `--lr`, `--beta`, …).
Checkpoints are written to `--ckpt-dir` (default `checkpoints/`), best-3 by `val_loss` plus
`last.ckpt`. Training synthesises noisy→clean pairs on the fly per the colony-noise recipe.

## Evaluate (SPECS.md metrics)

```bash
python -m vocdenoiser.denoise.eval --ckpt checkpoints/last.ckpt \
    --labels-csv individuals.csv        # columns: id,identity  (id matches the WAV stem)
```

Produces a **UMAP** scatter of the 16 latents (`umap_latents.png`), saves the raw latents
(`latents.npy`), and reports a cross-validated **RandomForest** identity accuracy — the
"is individual identity preserved under compression?" proxy.

`data/Vocalizations` filenames are bare call IDs with no identity in the path, so the RF proxy
needs `--labels-csv` (an ID→individual mapping). Without labels the UMAP still renders and the
RF step is skipped. If your calls *do* encode identity in the path, use
`--label-from parent|stem|prefix` instead.

### External benchmark datasets (`vocdenoiser.datasets`)

Two public, CC-BY-4.0 datasets can supply the labels the eval needs. Each loader cuts/decodes
per-call WAV clips (resampled to `--target-sr`, default 96 kHz to match the model) and writes an
`id,identity` CSV for `--labels-csv`:

```bash
# InfantMarmosetsVox — the only open set with per-call CALLER IDENTITY (10 individuals).
# --download fetches+extracts the ~21 GB audio from Zenodo, then cuts labels.csv
# segments from the 350 ten-minute recordings; identity = caller.
python -m vocdenoiser.datasets.infantmarmosetsvox --download --target-sr 96000

# MarmAudio — 96 kHz, ~215k clips labelled by CALL TYPE only (no caller identity).
python -m vocdenoiser.datasets.marmaudio [--extract --target-sr 96000]
```

Each prepared clip is **quality-scored** (call-agnostic SNR + level + clipping) and the metrics
are written into the label CSV; external corpora vary in recording quality, so pass thresholds to
drop the bad clips before they bias the benchmark:

```bash
# drop noisy (< 6 dB), near-silent (peak < -40 dBFS), or over-recorded (>1% clipped) clips:
python -m vocdenoiser.datasets.infantmarmosetsvox --min-snr 6 --min-peak-dbfs -40 --max-clip-frac 0.01
```

The run prints a quality distribution + concern counts (`--no-quality` skips scoring). See
`data/labelled/README.md` for provenance, licenses, and download steps. InfantMarmosetsVox is a
*cross-dataset* benchmark (different colony, native 44.1 kHz), so read its RF accuracy as an
external check, not an in-domain number.

## Architecture search (autoresearch-style)

Inspired by Karpathy's [`autoresearch`](https://github.com/karpathy/autoresearch): a greedy
LLM/evolutionary hill-climb where a **fixed compute budget** is the equalizer, one scalar
metric ranks candidates, an append-only **JSONL ledger** is the resumable state, and a
**frozen harness** (data + budget + metric) is separated from the **editable candidate**
(`search/space.py`) and the human-tuned **policy** (`search/program.md`). It adapts the toy to
a noisier audio objective: a **typed candidate grammar** instead of free-file edits, a
**noise-aware accept** rule (metrics are seed-averaged; a challenger must beat the incumbent by
more than the seed-noise band), and a **top-K frontier with crossover**.

```bash
# Dry-run the loop with NO GPU (synthetic fitness landscape) — see the mechanics + tests:
python -m vocdenoiser.cli search run --harness mock --iters 40 --ledger artifacts/mock.jsonl
python -m vocdenoiser.cli search report --ledger artifacts/mock.jsonl

# Real search (GPU): trains each candidate for --max-steps, scores by spectrogram SI-SDR:
python -m vocdenoiser.cli search run --harness torch --data-root "$VOCDENOISER_DATA_ROOT" \
    --iters 60 --max-steps 400 --seeds 0 1 --ledger artifacts/search_ledger.jsonl
```

The metric is **spectrogram-domain SI-SDR** on a pinned validation set; the spectrogram
geometry is frozen (not searchable) so every candidate is scored in the same domain. Plug an
LLM proposer in by passing an `llm_fn(prompt)->json` to `search.propose.Proposer` (it reads
the frontier + `program.md` and proposes the next candidate; falls back to mutation/crossover).

## Develop

```bash
ruff format . && ruff check --fix .
pytest -q          # SNR + search tests run numpy-only; model tests need the ml extra
```

## Notes / deviations from SPECS.md

- **Bioacoustic perturbations (pitch ±5 %, time-stretch ±10 %) use torchaudio's phase
  vocoder**, not librosa/scipy — keeping the dependency surface minimal (torchaudio is
  already required for the mel transform).
- The colony-noise percentages (40 % pink / 50 % babble / 10 % transients) are implemented as
  **energy-mixing weights of an RMS-normalised composite noise bed** (documented in
  `augment.py`); with no prior `augment.py` to match, this is the chosen interpretation of the
  recipe. The constants themselves are unchanged from SPECS.md.
- Spectrograms are fixed at `n_mels × n_frames` (default **128 × 256**), both required
  divisible by 16 so the 4-layer stride-2 encoder/decoder reconstruct the exact input shape.
