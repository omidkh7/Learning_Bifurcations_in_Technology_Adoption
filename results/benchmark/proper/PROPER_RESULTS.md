# FeatMLP proper run (5 seeds, 40000/class pools)

Matched set N = 18603 ([6201, 6201, 6201] per class); matched test ~2793/split; natural N = 18000.
Pool size 40000/class (headline used 15000).

## Matched (overall 99.5 +/- 0.2%)

| class | precision | recall | F1 |
|---|---|---|---|
| SN | 99.3 +/- 0.4 | 99.1 +/- 0.2 | 99.2 +/- 0.2 |
| TC | 99.1 +/- 0.2 | 99.3 +/- 0.4 | 99.2 +/- 0.2 |
| Null | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |

## Natural (overall 93.9 +/- 0.6%)

| class | precision | recall | F1 |
|---|---|---|---|
| SN | 85.2 +/- 1.5 | 99.1 +/- 0.2 | 91.6 +/- 0.8 |
| TC | 99.0 +/- 0.2 | 82.7 +/- 2.0 | 90.1 +/- 1.2 |
| Null | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
