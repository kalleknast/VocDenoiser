# SNR call-agnosticism — ground-truth call-type validation

**Verdict: REVIEW**

Labeled set: 321 clips across 11 types (alarm_1, chirp, loud_shrill, phee_2, phee_3, phee_4, seep, trill, tsik, tsik_ek, twitter).

## Controlled injection response per call type (the bias test)

- max cross-type spread of the mean response: **7.85 dB** (tolerance 5.0 dB = 20% of the 25 dB sweep) ⚠️
- phee vs non-phee mean response gap: **+0.79 dB** (phee favored; want |gap| < 2 dB) ✅

| call type | -5dB | 0dB | 5dB | 10dB | 15dB | 20dB | clean median |
|---|---|---|---|---|---|---|---|
| alarm_1 | 13.8 | 13.8 | 13.7 | 13.6 | 13.6 | 13.5 | 13.2 |
| chirp | 14.7 | 15.2 | 16.0 | 17.0 | 18.2 | 19.5 | 24.2 |
| loud_shrill | 15.9 | 15.6 | 15.2 | 14.8 | 14.4 | 14.0 | 13.4 |
| phee_2 *(phee)* | 16.2 | 16.2 | 16.2 | 16.1 | 15.9 | 15.7 | 14.4 |
| phee_3 *(phee)* | 17.2 | 17.2 | 17.2 | 17.0 | 16.8 | 16.6 | 15.0 |
| phee_4 *(phee)* | 16.3 | 16.3 | 16.3 | 16.3 | 16.2 | 16.2 | 16.1 |
| seep | 14.6 | 14.7 | 14.8 | 14.8 | 14.8 | 14.7 | 14.4 |
| trill | 14.2 | 14.2 | 14.2 | 14.1 | 14.1 | 14.1 | 13.4 |
| tsik | 15.5 | 16.2 | 16.9 | 17.6 | 18.2 | 18.7 | 19.5 |
| tsik_ek | 14.7 | 14.9 | 15.2 | 15.6 | 16.2 | 16.9 | 20.3 |
| twitter | 15.5 | 16.3 | 17.3 | 18.5 | 19.9 | 21.4 | 23.2 |

Rows that overlap at each injected level = the metric responds to noise the same way for phees and non-phees. A phee row sitting systematically above the trill / twitter / tsik / ek rows would be the bias we must avoid.
