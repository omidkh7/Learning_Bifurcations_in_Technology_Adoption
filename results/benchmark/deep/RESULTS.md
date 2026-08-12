# Second attempt — results

Matched test N = 1050, natural control N = 12000, chance = 33.3%.

| method | matched test | natural |
|---|---|---|
| t50-only logreg | 33.6% | 50.9% |
| 46-D nearest-centroid | 86.5% | 82.0% |
| 46-D logreg | 99.0% | 88.6% |
| FeatMLP (46-D in) | 99.0% | 94.5% |
| Encoder + feature supervision | 95.6% | 94.5% |
| Encoder ablation (lambda=0) | 89.3% | 74.4% |

## Feature-prediction R2 by group (feature-supervised encoder)

- A:CSD: +0.73
- B:Inflect: +0.43
- C:Phase: +0.85
- D:Catch22: +0.72
- E:SN-fp: +0.77
- F:Transit: +0.85
- G:TC-fp: +0.30
- H:Crit: +0.38

## Seed robustness (3 seeds, same split)

- encoder_fs.matched_test: 92.2% +/- 2.1 (runs: 93.7, 89.2, 93.5)
- encoder_fs.natural: 84.6% +/- 5.8 (runs: 91.2, 77.1, 85.6)
- encoder_0.matched_test: 92.8% +/- 1.5 (runs: 93.0, 90.8, 94.5)
- encoder_0.natural: 84.5% +/- 0.7 (runs: 85.3, 83.6, 84.7)

## Verdict

SEED-ROBUST VERDICT: the raw-curve encoder learns the dynamics on the matched set regardless of feature supervision (with FS 92.2+/-2.1%, without 92.8+/-1.5%, chance 33.3%, old FFT = chance). The decisive ingredients are the t50-MATCHED training data (no inflection shortcut to exploit) and an architecture that sees the noise structure (dx channel + recurrence), not the feature-regression head: feature supervision is performance-neutral here. The best model overall remains the 46-D features as INPUT (FeatMLP, 99.0%).
