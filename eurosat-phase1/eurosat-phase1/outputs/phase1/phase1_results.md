| Initialisation | Norm | Size | Seeds | Val macro-F1 (%) | Test accuracy (%) | Test macro-F1 (%) | Test balanced acc. (%) | Loop seconds |
|---|---|---|---|---|---|---|---|---|
| imagenet | batch | 64 | 3 | 98.01 ± 0.06 | 98.32 ± 0.11 | 98.27 ± 0.11 | 98.28 ± 0.12 | 530.28 ± 42.36 |
| imagenet | batch | 96 | 3 | 98.50 ± 0.22 | 98.64 ± 0.09 | 98.59 ± 0.08 | 98.58 ± 0.07 | 650.26 ± 155.20 |
| imagenet | batch | 224 | 3 | 98.78 ± 0.20 | 98.94 ± 0.24 | 98.90 ± 0.25 | 98.88 ± 0.25 | 2006.62 ± 530.35 |
| scratch | batch | 96 | 3 | 96.76 ± 0.22 | 96.87 ± 0.24 | 96.76 ± 0.26 | 96.76 ± 0.23 | 1011.52 ± 91.32 |
| scratch | group | 96 | 3 | 94.51 ± 0.29 | 95.31 ± 0.20 | 95.19 ± 0.19 | 95.18 ± 0.20 | 807.92 ± 62.05 |

Mean ± sample standard deviation (ddof=1). Validation-selected checkpoints. Pending means not evaluated.
