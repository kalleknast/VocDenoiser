# SNR call-agnosticism validation

**Verdict: PASS**

The **injection test (Check 2)** is the primary evidence — it holds the call fixed and varies only noise + morphology stratum, so a small cross-stratum spread directly means the metric is call-agnostic. Check 1's full-set ρ is confounded by noise (see note); the clean-subset ρ is the fairer guard.

## Check 1 — morphology-correlation guard (want clean-subset |ρ| small)

| morphology proxy | ρ (full set) | ρ (clean subset) | ok? |
|---|---:|---:|:--:|
| dom_freq_hz | +0.234 | -0.147 | ✅ |
| bandwidth_hz | +0.677 | +0.248 | ✅ |
| flatness | +0.676 | +0.265 | ✅ |

Full-set |ρ| is inflated because noisy clips are genuinely both flatter/broader AND lower-SNR — the proxy is contaminated by the very thing we measure. On clean clips the proxy reflects true call shape, so a small clean-subset |ρ| indicates the metric is not ranking call morphology.

## Check 2 — bias-injection across morphology strata

- reference clips: 200 (cleanest by snr_db)
- injected SNR sweep: [-5, 0, 5, 10, 15, 20] dB
- monotonic response (measured rises with injected): **100%** of clips ✅
- max cross-stratum spread of the mean response: **3.65 dB** (tolerance 5.0 dB = 20% of the 25 dB injected sweep) ✅

Per-stratum mean measured snr_db vs injected level:

| stratum | -5dB | 0dB | 5dB | 10dB | 15dB | 20dB |
|---|---|---|---|---|---|---|
| domHi|bwBroad | 15.0 | 16.3 | 17.9 | 19.8 | 21.9 | 24.1 |
| domHi|bwNarrow | 14.8 | 16.0 | 17.6 | 19.5 | 21.5 | 23.6 |
| domLo|bwBroad | 16.8 | 18.3 | 20.0 | 22.0 | 24.1 | 26.2 |
| domLo|bwNarrow | 15.1 | 16.2 | 17.6 | 19.3 | 21.2 | 23.3 |
| domMid|bwBroad | 15.2 | 16.7 | 18.4 | 20.4 | 22.7 | 25.0 |
| domMid|bwNarrow | 14.7 | 15.8 | 17.1 | 18.8 | 20.6 | 22.5 |

Curves that overlap (small spread) across low/high dominant-frequency and narrow/broad-band strata are the direct evidence the metric is call-agnostic.
