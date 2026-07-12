# Denoiser: Denoising & Compression ($\beta$-VAE)

This document provides technical specifications for building a denoising compression model for common marmoset (Callithrix jacchus) vocalizations, specifically targeting "phee" calls. A model to clean single-source vocalizations and compress them into a latent space for clustering.

## Project Context & Environment

Goal: Develop a robust pipeline to extract low-dimensional, noise-free features for behavioral clustering.

Dataset: 50,000+ isolated marmoset phee calls.

Platform: Google Colab (PyTorch/Lightning).

Hardware Target: NVIDIA A100 or L4 GPU.

## Audio & Spectrogram Representation

* Source audio: 96 kHz, mono, 16-bit PCM (isolated phee calls ~0.5 s; colony-noise clips ~60 s).
* Working sample rate: **44.1 kHz**. Phee-call energy is essentially all below ~22 kHz, so the 96 kHz source is downsampled (anti-aliased) to 44.1 kHz for training and inference. This roughly halves data/IO, gives finer in-band mel resolution for a fixed spectrogram size, and matches the native rate of the InfantMarmosetsVox identity benchmark (removing the upsampling artifact). *Earlier revisions of this spec operated at 96 kHz.*
* Representation: log-mel spectrogram.
  * STFT: `n_fft = 1024`, `hop = 256` (~5.8 ms frames, 75% overlap), Hann window.
  * Mel: `n_mels = 128` over `f_min = 1000 Hz` … `f_max = 22050 Hz` (Nyquist). A non-zero `f_min` keeps the model on the phee band and avoids empty low-frequency filterbanks.
  * Fixed geometry: `n_frames = 256` (crop/pad) → input tensor `(1, 128, 256)`, ≈1.48 s. `n_mels` and `n_frames` are divisible by the encoder's `2⁴` downsample so the decoder reconstructs the exact shape.
  * Power mel → dB → normalized to `[0, 1]` via `(dB − dB_min)/(dB_ref − dB_min)`, clamped.
* All constants live in `denoise/config.py` (nothing hardcoded). The dataset resamples any non-44.1 kHz input on load as a safety net; `datasets/resample.py` pre-builds a 44.1 kHz copy of a set to skip that cost.

## Architecture

* Type: 2D Convolutional $\beta$-Variational Autoencoder.
* Latent Bottleneck: 16 dimensions (Scientifically validated for marmoset vocalization similarity).
* Encoder: 4 convolutional layers reducing the spectrogram to a $1 \times 16$ latent vector.
* Decoder: Symmetrical mirror of the encoder to reconstruct the cleaned spectrogram.

## Training Objective: Disentangled Reconstruction

Input is a noisy phee; target is the clean ground-truth phee.

* Loss Function:

$$L_{VAE} = \text{MSE}(\hat{x}, x) + \beta \cdot D_{KL}(q(z|x) \| p(z))$$

* Denoising Strategy: Supervised Noisy-to-Clean mapping.
* Term scaling (implementation): the reconstruction term is **summed over spectrogram bins** (equivalently, per-bin MSE × `n_bins`) so it shares a scale with the latent-summed KL. Without this, MSE averaged over ~32k bins is dwarfed by $\beta \cdot D_{KL}$, the objective becomes >99% KL, and the posterior collapses instead of learning to denoise. For numerical stability `logvar` is clamped and gradients are clipped, so a single degenerate batch cannot NaN the run.

## Data Augmentation & Noise Simulation

The model utilizes a dynamic mixing pipeline to generate training samples from isolated calls.

### Synthetic Colony Noise "Recipe"

Simulate the lab husbandry environment using a weighted mixture:

* 40% Pink Noise ($1/f$): Models background ventilation/AC with high-frequency coverage.
* 50% Synthetic Babble: Mix 5–10 attenuated, randomly shifted isolated calls from the dataset to simulate conspecific "chatter".
* 10% Transients: Short white-noise bursts (5–20 ms) to simulate cage rattling.

### Real recorded backgrounds (extension beyond this recipe)

Training does **not** rely on synthetic noise alone: the synthetic bed above is blended with a **real recorded colony-noise** segment drawn per-example from `cfg.noise_dirs` (`data/Noise` = ventilation/colony background, `data/Cigarra` = cicada), at weight `cfg.real_noise_weight` (`0` = synthetic only, `1` = real only, default `0.5`). This exposes the model to the actual acoustic backgrounds, not just their synthetic approximation. Implemented in `augment.noise_bed` / `dataset.PheeDenoiseDataset`.

## Mixing Pipeline

* Dynamic SNR: Relative levels between calls or signal-to-noise sampled uniformly from $-5$ dB to $+15$ dB.
* Temporal Offsets: Random shifts ($0$ to $500$ ms) between sources.* Bioacoustic Perturbations:
  * Pitch Shifting: $\pm 5\%$.
  * Time Stretching: $\pm 10\%$.

## Evaluation Metrics

* Latent Separability: Use UMAP on the 16 VAE features to visualize cluster density.
* Identity Classification Proxy: Train a Random Forest on the 16 features to verify that individual identity markers are preserved during compression
