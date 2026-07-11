---
name: scaffold-project
description: Bootstrap the VocDenoiser repo structure per SPECS.md — uv/pyproject, ruff config, a src/vocdenoiser package with a PyTorch-Lightning β-VAE skeleton, colony-noise augmentation, and train/eval entry points. User-triggered one-time setup for this greenfield project.
disable-model-invocation: true
---

# Scaffold VocDenoiser

Stand up the initial project skeleton described in `@SPECS.md`. This is a **greenfield** repo
(currently only `SPECS.md` + `data/Noise/`), so create the structure below rather than editing
existing code. Confirm before overwriting anything that already exists.

## Ground rules
- **Env-agnostic**: works on both Colab (A100/L4) and a local Linux box. Never hardcode a Colab
  Drive path or a local path — take the data root from a CLI flag / env var (default sensibly,
  document it).
- Source audio is **96 kHz mono, 16-bit**; the clean phee-call set is **off-repo** (see
  `CLAUDE.local.md`). Data-loader stubs should point at a configurable clean-call root, not at
  `data/Noise/` (that folder is *noise*, used for augmentation only).
- Use **`uv`** for env/deps. If `uv` isn't installed, tell the user (`pip install uv` or the
  official installer) before proceeding.

## Layout to create
```
pyproject.toml          # uv-managed; deps below; ruff config (format + lint, line-length 100)
src/vocdenoiser/
  __init__.py
  config.py             # dataclass config: data_root, sr=96000, n_fft/hop, beta, latent_dim=16, batch, lr
  data/
    dataset.py          # Dataset yielding (noisy_spec, clean_spec) pairs; STFT/log-mel transform
    augment.py          # colony-noise mixing — implement via the /noise-recipe skill's spec
  models/
    beta_vae.py         # LightningModule: 4-layer conv encoder → 1×16 latent → symmetric decoder
  train.py              # Lightning Trainer wiring; reads config; checkpoints
  eval.py               # UMAP on 16 latents + RandomForest identity-classification proxy
tests/
  test_shapes.py        # smoke test: forward pass shapes, latent dim == 16, loss is finite
```

## Dependencies (pyproject)
Python ≥ 3.10; `torch`, `torchaudio`, `pytorch-lightning`, `librosa`, `numpy`, `scipy`,
`umap-learn`, `scikit-learn`, `matplotlib`; dev: `ruff`, `pytest`.

## Model contract (from SPECS.md — keep exact)
- 2D convolutional **β-VAE**; encoder = 4 conv layers → **16-dim** latent (`mu`, `logvar`).
- Decoder = symmetric mirror reconstructing the cleaned spectrogram.
- Loss: `MSE(x̂, x) + β · D_KL(q(z|x) ‖ p(z))`, standard-normal prior. `β` is configurable.
- Training is **supervised noisy→clean**: input noisy phee spectrogram, target clean phee.

## After scaffolding
- Run `uv run ruff format . && uv run ruff check --fix .` and `uv run pytest -q`.
- Report what was created and the exact commands to train (`uv run python -m vocdenoiser.train ...`)
  and evaluate. Remind the user to set the clean-call `data_root` and to copy audio to local disk
  (not the pCloud mount) before training.
