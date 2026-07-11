# Search policy (the human-tuned meta-level)

This file is the **policy** for the architecture search — the autoresearch "program":
the human edits *this*, the search proposes *candidates*, and neither may touch the
**frozen harness**. Keep it short enough to sit in an LLM proposer's context.

## Goal

Maximize the frozen scalar metric: **spectrogram-domain SI-SDR (dB, higher is better)**
of the denoiser's reconstruction of the clean phee spectrogram, on a pinned validation
set, under a **fixed compute budget** (default `max_steps`). The budget is the equalizer:
every candidate gets the same compute, so a bigger model and a smaller one are compared
on one number. Metrics are **seed-averaged**; a challenger must beat the incumbent by more
than the seed-noise band to be kept (the accept rule is code, not your judgment).

## What you may change (the candidate grammar)

Only the knobs in the candidate schema: model family (`n_conv_layers`, `base_channels`,
`channel_mult`, `kernel_size`, `norm`, `act`, `latent_dim`, `residual`, `dropout`),
loss (`beta`, `beta_schedule`, `recon_loss`), and optimisation (`optimizer`, `lr`,
`weight_decay`, `batch_size`).

## What is frozen (never proposes to change)

The clean-call data + colony-noise recipe, the **canonical spectrogram geometry**
(`n_mels`, `n_frames`, `n_fft`, `hop`) so all candidates are scored in the same domain,
the compute budget, the validation set, and the metric definition. `latent_dim` defaults
to 16 (SPECS.md, scientifically validated for marmoset similarity) — search around it, but
treat moves away from 16 as costed by downstream clustering, not just reconstruction.

## How to propose (when you are the LLM proposer)

1. Read the **frontier** (best kept candidates) and **recent experiments** in the context.
2. Propose ONE candidate that is a small, motivated edit of a strong frontier member, or a
   crossover of two — never a repeat of anything already evaluated.
3. Prefer edits with a **hypothesis** ("wider base_channels should help because the current
   frontier is capacity-limited at fixed depth"). State it in one line as `rationale`.
4. **Simplicity criterion**: an equal-or-better metric with fewer parameters is a win; a
   +0.01 dB gain that doubles the model is usually not. The accept rule enforces this within
   the noise band, but bias your proposals the same way.
5. When the frontier stalls, make a **more radical** structural move (depth, residual,
   norm/act family) rather than another tiny lr nudge.

## Loop invariants (enforced by code, listed so you can reason about them)

- Accept if `metric_challenger − metric_incumbent > k_sigma · sqrt(std_c² + std_i²)`.
- Duplicate candidates (same identity hash) are skipped, not re-run.
- Crashed / NaN candidates are logged `crash` and the search moves on.
- The ledger (JSONL) is the persisted state; the run resumes from it.
