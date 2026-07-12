# Denoiser: Denoising & Compression ($\beta$-VAE)

This document provides technical specifications for building a denoising compression model for common marmoset (Callithrix jacchus) vocalizations, specifically targeting "phee" calls. A model to clean single-source vocalizations and compress them into a latent space for clustering.

## Project Context & Environment

Goal: Develop a robust pipeline to extract low-dimensional, noise-free features for behavioral clustering.

Dataset: 50,000+ isolated marmoset phee calls.

Platform: Google Colab (PyTorch/Lightning).

Hardware Target: NVIDIA A100 or L4 GPU.

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
