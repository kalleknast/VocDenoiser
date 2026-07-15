# Architecture search — conclusions and how to continue

Status as of 2026-07-15, against `search_ledger_v2.jsonl` (25 records, ~26 h GPU, `--max-steps 3000`,
3 seeds). Written as a handoff: the search is finished, and the next move needs labelled data
(call type + caller identity) that is being collected.

**TL;DR** — Stop the search; more `--iters` cannot help, for a measurable reason. Two candidates are
worth carrying forward. But **you cannot train either of them today** — that gap (§3) is the first
thing to fix, and it can be fixed before the data lands.


## 1. The search is done: it plateaued at the metric's noise floor

The incumbent `e7f14ffdd0` (−2.0823) was found at **trial 9, by an ε-greedy random restart**, and has
stood for 15 trials. Evolution has never beaten it. That is not a stuck hill-climb — the search has
run out of *resolving power*:

- 6 of the top 7 are **statistically indistinguishable** from the incumbent: every `|Δ|` falls inside
  its own noise band (`k_sigma=1.0`, band = `sqrt(std_c² + std_i²)`).
- The top-7 **spread is 0.099**, *smaller* than the typical noise band of **0.163**.
- Median `metric_std` at 3 seeds is **0.115** → resolving a 0.05 dB difference needs **~11 seeds**;
  0.03 dB needs **~30**. We run 3.

The accept rule is not rejecting these candidates — it genuinely cannot rank them. Running trials
26–60 at 3 seeds would resample noise at ~1 h/trial. The search still rejects *clearly* worse
candidates (e.g. `9902018084` at −2.1625 is resolvable), so it is working; it has simply converged to
a plateau of equivalent architectures.

**Corollary:** the ledger cannot pick a winner. Something outside the SI-SDR metric has to — which is
what the labelled data is for. See §4.


## 2. The shortlist

All metrics are seed-averaged spectrogram SI-SDR (dB, higher = better). The whole top cluster is one
architecture family — `L4 ch32/16 ×2.0 k4 group l1 adamw lr=1.62e-4 warmup do=0.196 bs=64
latent=32` — differing only in `act`, `beta`, and width.

| id | metric | params | Δ vs incumbent | note |
|---|---|---|---|---|
| `5e0975e121` | −2.0637 ± 0.059 | 4,557,953 | +0.019 (band 0.084) | best raw metric; **within noise** of the incumbent |
| `f4b84629ec` | −2.0764 ± 0.058 | 4,557,953 | +0.006 (band 0.083) | within noise |
| **`e7f14ffdd0`** | **−2.0823 ± 0.060** | 4,557,953 | 0 | **incumbent** (`gelu`, beta 3.049, adamw) |
| **`a60f295172`** | **−2.1164 ± 0.034** | **1,934,945** | −0.034 (band 0.069) | **2.4× smaller, statistically identical, tightest std** |
| `d1fcc68696` | −2.1296 ± 0.140 | 4,557,953 | −0.047 (band 0.152) | within noise |
| `2d4003afe1` | −2.1303 ± 0.058 | 1,934,945 | −0.048 (band 0.083) | within noise |

**Carry forward `e7f14ffdd0` and `a60f295172`.** They bracket the real question: 4.56M vs 1.93M
params for an indistinguishable reconstruction metric. `a60f295172` is `leaky_relu` + beta 1.009 +
`ch16`; `2d4003afe1` is its `gelu` sibling and a reasonable third.

Full config for the two, from the ledger:

```
e7f14ffdd0: n_conv_layers=4 base_channels=32 channel_mult=2.0 kernel_size=4 norm=group act=gelu
            latent_dim=32 residual=False dropout=0.1959602459571401 beta=3.049 beta_schedule=warmup
            recon_loss=l1 optimizer=adamw lr=1.62e-04 weight_decay=2.45e-06 batch_size=64

a60f295172: n_conv_layers=4 base_channels=16 channel_mult=2.0 kernel_size=4 norm=group act=leaky_relu
            latent_dim=32 residual=False dropout=0.1959602459571401 beta=1.009 beta_schedule=warmup
            recon_loss=l1 optimizer=adamw lr=1.62e-04 weight_decay=2.45e-06 batch_size=64
```

Note both want **`latent_dim=32`**, not the SPECS default of 16 — but the search metric is blind to
identity retention, so treat that as a hypothesis for the identity eval to confirm, not a decision.


## 3. BLOCKER: a search winner cannot currently be trained

**This is the first thing to fix, and it does not need the labelled data.** The search's output is
currently unrealisable. Four independent reasons:

1. **The harness never saves weights.** `search/harness.py` sets `enable_checkpointing=False`, so no
   candidate checkpoint exists — only ledger rows. Every candidate is trained for 3000 steps and
   thrown away.
2. **`train` builds the wrong model.** `denoise/train.py:110` builds `BetaVAE(cfg)` — the
   hand-designed model — not the search's `ConfigurableBetaVAE`. `BetaVAE` hardcodes `kernel_size=4`,
   `BatchNorm2d`, `LeakyReLU(0.2)`, its own depth/width, no residual, no dropout.
3. **`eval` also hardcodes the hand model.** `denoise/eval.py:94` calls
   `BetaVAE.load_from_checkpoint(...)`. A `ConfigurableBetaVAE` checkpoint **will not load into it** —
   verified: for the winner, 8 state_dict keys exist only in the candidate model and 29 only in
   `BetaVAE`.
4. **`Config` cannot express the candidate.** `Candidate.to_config_overrides()` emits only **5 of 16**
   knobs (`base_channels`, `latent_dim`, `beta`, `lr`, `batch_size`).

Handing `e7f14ffdd0` to `train` today silently gives you a *different model*: `norm` group→batch,
`act` gelu→leaky_relu, `dropout` 0.196→0.0, `beta_schedule` warmup→const, plus BetaVAE's own
depth/width. The searched architecture is lost and you would be evaluating something the search never
scored.

The 11 lost knobs split by difficulty:

- **(A) Easy — already a `Config` field, just not mapped** by `to_config_overrides()`:
  `optimizer`, `recon_loss`, `weight_decay`, `beta_schedule`. (`BetaVAE` already reads
  `cfg.optimizer` and switches l1/mse.)
- **(B) Real work — not in `Config`, hardcoded in `BetaVAE`**:
  `n_conv_layers`, `channel_mult`, `kernel_size`, `norm`, `act`, `residual`, `dropout`.

**Recommended fix:** don't widen `Config`/`BetaVAE` to absorb (B) — that duplicates
`ConfigurableBetaVAE`, which already implements every knob and is exactly what the ledger scored.
Instead make a ledger candidate trainable and loadable end-to-end:

1. **Make the checkpoint self-describing.** `model_factory.py:65` currently calls
   `save_hyperparameters(ignore=["cand", "cfg"])`, so the candidate is **not** stored in the ckpt and
   the file cannot say what architecture it is. Stop ignoring `cand` (store `cand.to_dict()`), and
   `ConfigurableBetaVAE.load_from_checkpoint(path)` reconstructs itself with no external state.
2. **`train --from-ledger <path> --candidate-id <id>`** — load the row, rebuild via
   `build_search_model(cand, cfg)`, train with full checkpointing. The trained model is then what was
   searched, by construction, with no new architecture code.
3. **Teach `eval` to load it** — pick the class from the ckpt's hparams (once (1) lands, a candidate
   ckpt is self-identifying) instead of assuming `BetaVAE`.

> **Do not verify this with a parameter count.** `BetaVAE(cfg)` and the winner's
> `ConfigurableBetaVAE` both come to **exactly 4,557,953** params while having incompatible
> state_dicts — GroupNorm and BatchNorm2d each carry `2·C` affine params, and activations are
> parameter-free, so the counts coincide while the modules differ. A `num_params` assertion passes on
> the wrong model. **Compare `state_dict()` keys** against a model freshly built from the ledger row.
> (`space.estimate_params()` is still the right cross-check for *size*, which is all it claims.)


## 4. When the labelled data lands

The point of the new data is to **break the §1 tie with a metric that actually matters**. SI-SDR
scores reconstruction only; it cannot see caller-identity retention or call-type structure. The
notebook already flags this ("choose `latent_dim` from the downstream RF/UMAP eval, not this
ledger").

Order of operations:

1. **Fix §3 first** — otherwise there is nothing to evaluate. Do it before the data arrives.
2. **Train the shortlist properly.** `e7f14ffdd0` and `a60f295172` (and `2d4003afe1` if cheap), full
   training with checkpointing — *not* the 3000-step search budget. Use `--resume-from` /
   `FRESH_START` per the notebook; keep `checkpoints/` unpolluted (each run in its own dir — mixing
   runs has bitten this project before).
3. **Run the identity eval on each**, with the new id→identity CSV:

   ```
   python -m vocdenoiser.denoise.eval --ckpt <ckpt> \
       --labels-csv <identity.csv> --labels-key-col id --labels-value-col identity \
       --out-png umap_<candidate-id>.png --out-latents latents_<candidate-id>.npy
   ```

   `eval.py` flags: `--label-from {parent,stem,prefix}` / `--label-sep` if identity is encoded in the
   path rather than a CSV. Note `data/Vocalizations` filenames carry **no** identity — the CSV is
   required.
4. **Rank by identity accuracy, not SI-SDR.** The existing cross-dataset reference point is
   **0.680 ± 0.005** (β-VAE, 16-dim latents, RF proxy, cv=5) on InfantMarmosetsVox vs a 0.178
   majority-class floor. That is a *different* dataset/latent-dim, so it is a sanity anchor, not a
   baseline to beat directly.
5. **Then decide `latent_dim`.** The whole top cluster chose 32 on reconstruction grounds. Whether 32
   beats 16 on identity retention is exactly the open question — and a compression argument favours
   16 if identity holds. Test both on the winning architecture.
6. **Call-type labels** additionally allow re-running the call-agnosticism check
   (`snr validate-types`) on the *denoised* output, and confirm the denoiser is not quietly
   specialising on phees at the expense of other call types.

If identity accuracy also ties between `e7f14ffdd0` and `a60f295172`, **take `a60f295172`** — 2.4×
fewer params for the same performance on both metrics is a decision, not a coin flip.


## 5. What was fixed getting here (all on `main`)

- **`f9a259b`** — the "~4M cap" from `35b2961` was a **no-op**. Bounding the `CHOICES` sets does not
  bound size: params are dominated by the dense bottleneck (`fc_mu`/`fc_logvar`/`fc_dec` over the
  flattened encoder output), which shrinks 4× per conv layer, so the **shallowest** net is the
  **biggest**. 20% of the space still exceeded 4M, max 10.7M, and a 7.59M model reached the ledger.
  Now enforced against a computed count (`space.estimate_params()`, torch-free, reproduces all 87
  `num_params` across both ledgers exactly) with rejection sampling in random/mutate/crossover;
  `--max-params` (0 disables). Also: early divergence abort (`CandidateDiverged` after 25 consecutive
  non-finite steps — 4 crashes had burned 2.7 h training to completion before being scored −inf), and
  `--explore-rate` 0.2→0.35.
- **`9279593`** — residual NaN blowup. **Confirmed fixed in production:** trial 20 `44f68be1cb` is
  `residual=True` **and** `norm=none` (the exact double-fatal combo) and trained to a finite −2.9050.
  Residual crash rate **4/4 → 0/1**; 0 crashes in the last 8 trials. `norm=none` now has surviving
  data too (−2.90 with residual, −4.51 without) — it was only ever a coverage hole because both its
  earlier draws were paired with the residual bug.
- **`f79e704`** — cell 16 now `git pull`s on resume and prints the running revision (see §6).


## 6. Traps that already cost a full analysis pass each

- **`search_ledger*.jsonl` at the repo root is a manual download.** The run writes to Google Drive
  (`$VOCDENOISER_OUTPUT_ROOT/artifacts/`); this box has no Drive mount (pCloud is unrelated). The
  file only changes when re-downloaded. **`stat` it before drawing any conclusion** — a recent mtime
  means "recently downloaded", not "the search is live".
- **Colab "Restart runtime" keeps `/content`**, so the old clone-only guard silently pinned the code
  at whatever revision the runtime first cloned. v2 trials **17–19 ran the pre-cap proposer** this
  way. Fixed in `f79e704`; **check the SHA cell 16 prints against `origin/main`** before trusting a
  resumed run. Only "Disconnect and delete runtime" wipes `/content`. The notebook clones **`main`**,
  so a pushed branch does nothing, and editing the notebook alone ships only CLI flags.
- **Ledger provenance:** trials 0–16 pre-cap, 17–19 stale-code, 20–24 capped. All 25 are
  metric-comparable (`--max-steps` stayed 3000 throughout; only the *proposal space* changed), so the
  §1/§2 analysis uses all of them. Do **not** mix in `search_ledger.jsonl` (v1) — that ran at 1500
  steps and the SI-SDR scale shifts with the budget.
- **A record the current code could not have proposed is the tell that a run is stale.** That is how
  trial 17 (`5e0975e121`, 4.56M params from a `mutate`) exposed the stale-code run: 5,000 mutations
  of its parent under the current code yield 0 over-cap children.
