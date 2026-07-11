---
name: noise-recipe
description: The exact colony-noise synthesis and mixing recipe from SPECS.md for generating training samples (noisy phee → clean phee). Use whenever writing, reviewing, or debugging data-augmentation / noise-simulation code so the pipeline stays faithful to the spec.
---

# Colony-Noise Synthesis Recipe

Reference for the dynamic mixing pipeline (`@SPECS.md`). Training pairs are synthesized on the
fly: a clean isolated phee call is the **target**; the same call mixed with synthetic colony
noise is the **input**. Keep these numbers exact.

## Noise mixture (weighted composition)
- **40% Pink noise (1/f)** — models ventilation/AC; broad high-frequency coverage.
- **50% Synthetic babble** — mix **5–10** attenuated, randomly time-shifted isolated calls
  drawn from the dataset (conspecific "chatter").
- **10% Transients** — short **white-noise bursts, 5–20 ms** (cage rattling).

Interpret the percentages as the sampling weight for which noise component(s) dominate a given
training example (or the mixing proportion of a composite noise bed) — match whatever the
existing `augment.py` does; don't silently change the ratios.

## Mixing parameters
- **Dynamic SNR**: relative level between call and noise (and between mixed calls) sampled
  **uniformly from −5 dB to +15 dB**.
- **Temporal offsets**: random shifts **0–500 ms** between sources.

## Bioacoustic perturbations (applied to calls)
- **Pitch shift: ±5%**
- **Time stretch: ±10%**

## Audio / spectrogram params
- Source audio is **96 kHz, mono, 16-bit PCM**. Decide and document whether to resample before
  STFT; keep `sr`, `n_fft`, and `hop_length` in one config, not scattered literals.
- Apply perturbations and mixing in a **reproducible** way (seedable RNG) so a run can be
  replayed, but re-randomize per epoch for augmentation diversity.

## When editing augmentation code
- Verify the resulting (input, target) pair is aligned: the target is the **clean** call, the
  input is that same call plus noise — offsets/stretch applied to the input must not desync it
  from the target beyond what the loss expects.
- If you change any constant above, flag it as a deviation from `SPECS.md`.
